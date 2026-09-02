#!/usr/bin/env python3
"""Latency + reliability stats from the NEW stack's loguru log (logs/translator_*.log).

Lines of interest (file format: "YYYY-MM-DD HH:mm:ss.SSS | LEVEL | where | message"):
  [EN] <text> | mode=streaming/sentence | asr=0.312s      -> sentence flushed to translation
  [ES] <text> | e2e=10.86s | translate=0.456s | tts=0.124s | audio=6.20s | queue_was=0
  [HT] ...                                                 -> logged when playback STARTS
  SESSION_END | ...

e2e in the log = speech START of the sentence -> playback start (comparable to the
old parser's "lat start->audio").  We also report playback_start - flush_time, i.e.
how long after the sentence was handed to translation its audio began (this excludes
the silence-timeout wait, so it is a lower bound on "speech end -> audio").

Usage: parse_new_log.py LOG [--from HH:MM[:SS]] [--to HH:MM[:SS]]
"""
from __future__ import annotations

import argparse
import re
import statistics
from datetime import datetime

TS = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})"
FRAG = re.compile(r"^" + TS + r" \| \S+\s*\| [^|]+ \| \[EN\] (.*?) \| mode=streaming/sentence")
EVT = re.compile(r"^" + TS + r" \| \S+\s*\| [^|]+ \| \[(ES|HT|[A-Z]{2})\] (.*?) \| e2e=([\d.]+)s \| translate=([\d.]+)s \| tts=([\d.]+)s \| audio=([\d.]+)s \| queue_was=(\d+)")
ERR = re.compile(r"^" + TS + r" \| (ERROR|CRITICAL)")
DROP = re.compile(r"dropped|Dropping|drop ", re.I)


def pts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")


def pct(v, q):
    if not v:
        return float("nan")
    v = sorted(v)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--from", dest="t_from")
    ap.add_argument("--to", dest="t_to")
    a = ap.parse_args()

    frags = []          # (time, text)
    events = {}         # lang -> list of (time, e2e, since_frag, translate, tts, audio, queue)
    errors = drops = 0
    day = None
    with open(a.log, errors="replace") as f:
        for ln in f:
            m = FRAG.match(ln)
            if m:
                t = pts(m.group(1)); day = day or m.group(1)[:10]
                frags.append((t, m.group(2)))
                continue
            m = EVT.match(ln)
            if m:
                t = pts(m.group(1)); day = day or m.group(1)[:10]
                lang = m.group(2)
                e2e, tr, tts, audio, q = map(float, m.group(4, 5, 6, 7, 8))
                # nearest preceding frag = the sentence this event belongs to (best effort)
                since_frag = None
                for ft, _ in reversed(frags):
                    if ft <= t:
                        since_frag = (t - ft).total_seconds()
                        break
                events.setdefault(lang, []).append((t, e2e, since_frag, tr, tts, audio, int(q)))
                continue
            if ERR.match(ln):
                errors += 1
                if DROP.search(ln):
                    drops += 1

    def in_window(t):
        if day is None:
            return True
        def parse(s):
            for fmt in ("%H:%M:%S", "%H:%M"):
                try:
                    return datetime.strptime(f"{day} {s}", f"%Y-%m-%d {fmt}")
                except ValueError:
                    pass
            raise SystemExit(f"bad time {s}")
        if a.t_from and t < parse(a.t_from):
            return False
        if a.t_to and t > parse(a.t_to):
            return False
        return True

    print(f"sentences committed: {sum(1 for t, _ in frags if in_window(t))}   errors: {errors}   drops: {drops}")
    print(f"{'lang':<5} {'n':>5} | {'e2e start→audio (s)':^24} | {'flush→audio (s)':^16} | {'translate':>9} {'tts':>6} {'audio':>6} {'q>0':>4}")
    print(f"{'':<5} {'':>5} | {'median':>7} {'p95':>7} {'max':>7} | {'median':>7} {'p95':>7} | {'avg':>9} {'avg':>6} {'avg':>6} {'':>4}")
    for lang, evs in sorted(events.items()):
        evs = [e for e in evs if in_window(e[0])]
        if not evs:
            continue
        e2e = [e[1] for e in evs]
        sf = [e[2] for e in evs if e[2] is not None]
        tr = statistics.mean(e[3] for e in evs)
        tts = statistics.mean(e[4] for e in evs)
        au = statistics.mean(e[5] for e in evs)
        qpos = sum(1 for e in evs if e[6] > 0)
        print(f"{lang:<5} {len(evs):>5} | {statistics.median(e2e):>7.1f} {pct(e2e, .95):>7.1f} {max(e2e):>7.1f} | "
              f"{(statistics.median(sf) if sf else float('nan')):>7.1f} {(pct(sf, .95) if sf else float('nan')):>7.1f} | "
              f"{tr:>9.2f} {tts:>6.2f} {au:>6.1f} {qpos:>4}")


if __name__ == "__main__":
    main()
