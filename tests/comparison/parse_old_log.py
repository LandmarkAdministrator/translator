#!/usr/bin/env python3
"""Latency + reliability stats from the OLD translator's log (translate.py v1.1.x).

Per chunk N the log gives:
  "Recording chunk N started: <t>, ended: <t_end> (~Xs)"   -> both timestamps are
        when the chunk was *finalized*; the speech itself spans [t_end - X, t_end].
  "Chunk N Playback started: <p_start>, ended: <p_end> (~Ys)" -> p_start is when
        the blocking write to the USB codec began, i.e. when listeners hear it.
  "Chunk N Recognized: ..." / "[silence]" / "Playback error: ..."

Latency as a listener feels it = p_start - speech_end.
Also reported: p_start - speech_start (how far behind the *start* of the chunk).

Usage: parse_old_log.py LOG [--day YYYY-MM-DD ...]
"""
from __future__ import annotations

import argparse
import re
import statistics
from collections import defaultdict
from datetime import datetime

TS = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)"
REC = re.compile(r"^(\S+ \S+) - Recording chunk (\d+) started: " + TS + r", ended: " + TS + r" \(~([\d.]+)s\)")
PLAY = re.compile(r"^(\S+ \S+) - Chunk (\d+) Playback started: " + TS + r", ended: " + TS + r" \(~([\d.]+)s\)")
RECOG = re.compile(r"^(\S+ \S+) - Chunk (\d+) Recognized: (.*)$")
PERR = re.compile(r"^(\S+ \S+) - Playback error")
INIT = re.compile(r"^(\S+ \S+) - Audio streams initialized")


def pts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")


def pct(vals, q):
    if not vals:
        return float("nan")
    vals = sorted(vals)
    k = (len(vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--day", action="append", help="restrict to these YYYY-MM-DD days")
    args = ap.parse_args()

    # Sessions are delimited by "Audio streams initialized" (chunk numbers restart).
    sessions = []  # list of dicts
    cur = None

    def new_session(day):
        s = {"day": day, "rec": {}, "play": {}, "recog": 0, "silence": 0, "perr": 0, "chunks": 0}
        sessions.append(s)
        return s

    with open(args.log, errors="replace") as f:
        for line in f:
            day = line[:10]
            if args.day and day not in args.day:
                continue
            if INIT.match(line):
                cur = new_session(day)
                continue
            if cur is None or cur["day"] != day:
                cur = new_session(day)
            m = REC.match(line)
            if m:
                n = int(m.group(2))
                t_end = pts(m.group(4))
                dur = float(m.group(5))
                cur["rec"][n] = (t_end, dur)
                cur["chunks"] += 1
                continue
            m = PLAY.match(line)
            if m:
                cur["play"][int(m.group(2))] = pts(m.group(3))
                continue
            m = RECOG.match(line)
            if m:
                if m.group(3).startswith("[silence"):
                    cur["silence"] += 1
                else:
                    cur["recog"] += 1
                continue
            if PERR.match(line):
                cur["perr"] += 1

    print(f"{'session':<22} {'chunks':>6} {'speech':>6} {'played':>6} {'errors':>6} | "
          f"{'lat end→audio (s)':^24} | {'lat start→audio (s)':^20}")
    print(f"{'':<22} {'':>6} {'':>6} {'':>6} {'':>6} | {'median':>7} {'p95':>7} {'max':>7} | {'median':>7} {'p95':>7}")
    all_end, all_start = [], []
    for s in sessions:
        if s["chunks"] == 0:
            continue
        lat_end, lat_start = [], []
        for n, p_start in s["play"].items():
            if n in s["rec"]:
                t_end, dur = s["rec"][n]
                lat_end.append((p_start - t_end).total_seconds())
                lat_start.append((p_start - t_end).total_seconds() + dur)
        first = min(s["rec"].values(), key=lambda x: x[0])[0] if s["rec"] else None
        label = f"{s['day']} {first.strftime('%H:%M') if first else '--:--'}"
        if lat_end:
            print(f"{label:<22} {s['chunks']:>6} {s['recog']:>6} {len(s['play']):>6} {s['perr']:>6} | "
                  f"{statistics.median(lat_end):>7.1f} {pct(lat_end, .95):>7.1f} {max(lat_end):>7.1f} | "
                  f"{statistics.median(lat_start):>7.1f} {pct(lat_start, .95):>7.1f}")
        else:
            print(f"{label:<22} {s['chunks']:>6} {s['recog']:>6} {len(s['play']):>6} {s['perr']:>6} | "
                  f"{'-':>7} {'-':>7} {'-':>7} | {'-':>7} {'-':>7}")
        all_end += lat_end
        all_start += lat_start
    if all_end:
        print("-" * 100)
        print(f"{'ALL':<22} {'':>6} {'':>6} {len(all_end):>6} {'':>6} | "
              f"{statistics.median(all_end):>7.1f} {pct(all_end, .95):>7.1f} {max(all_end):>7.1f} | "
              f"{statistics.median(all_start):>7.1f} {pct(all_start, .95):>7.1f}")


if __name__ == "__main__":
    main()
