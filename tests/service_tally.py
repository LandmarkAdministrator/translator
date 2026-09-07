#!/usr/bin/env python3
"""Hard numbers for a day's services, from ~/translate.log on the Translate PC.

Run it ON the host — it reads journald for translate.service's start/stop
events and the schedule's windows for that weekday:

    python3 tests/service_tally.py                       # today, to stdout
    python3 tests/service_tally.py --date 2026-09-10     # a past day
    python3 tests/service_tally.py --out ~/sermons/logs/service-tally
                                                         # also writes <date>.txt and latest.txt

translate-tally.timer runs the last form nightly at 23:30 and the admin panel
shows latest.txt. Log lines are selected by date, so the service log must
carry one (it has since 2026-09-07; earlier logs need the previous version of
this script, `git show 660ffb5:tests/service_tally.py`).

Per bucket — one per schedule window that day, plus "outside any window" for
tests — it reports: sentences per minute, how many end in a sentence mark /
start with a stray one / hold two sentences, words per segment, the segments
cut by the cap or a timeout, 10-minute bins with first-word-to-audio latency,
per-language latency and queue depth, speed-ups, errors, drops and restarts.
"""
import argparse
import collections
import datetime as dt
import os
import re
import statistics
import subprocess
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS = re.compile(r"^(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d) \|")
EN = re.compile(r"^(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d) \|.*?\[EN\] (.*?) \| mode=streaming/sentence")
TR = re.compile(r"^(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d) \|.*?\[(ES|HT|RU)\] (.*?) \| e2e=([\d.]+)s \| translate=([\d.]+)s "
                r"\| tts=([\d.]+)s \| audio=([\d.]+)s \| queue_was=(\d+)")
SPEED = re.compile(r"^(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d) \|.*?\[(ES|HT|RU)\] queue_depth=(\d+) → speaking at ([\d.]+)x")
END = re.compile(r"[.?!][\"')\]]?\s*$")
TWO = re.compile(r"[.?!][\"')\]]?\s+\S")
LEAD = re.compile(r"^\s*[.?!,;:]")
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def journal_runs(date: dt.date, text: str = None) -> list:
    """translate.service runs that started on `date`: [{start, end, how}]."""
    if text is None:
        since = date.isoformat()
        until = (date + dt.timedelta(days=1)).isoformat()
        text = subprocess.run(["journalctl", "--user", "-u", "translate.service", "--since", since,
                               "--until", until, "--no-pager", "-o", "short-iso"],
                              capture_output=True, text=True).stdout
    runs, cur = [], None
    for l in text.splitlines():
        m = re.match(r"^(\d{4}-\d\d-\d\d)T(\d\d:\d\d:\d\d)\S* .*?: (.*)$", l)
        if not m:
            continue
        d, t, msg = m.groups()
        if msg.startswith("Started") and d == date.isoformat():
            cur = {"start": t, "end": "23:59:59", "how": ""}
            runs.append(cur)
        elif cur and cur["end"] == "23:59:59" and (
                "Deactivated" in msg or "Main process exited" in msg or msg.startswith("Stopping")):
            cur["end"] = t if d == date.isoformat() else "23:59:59"
            cur["how"] = msg[:70]
    return runs


def schedule_windows(path: Path, weekday: int) -> list:
    """[(label, start, end)] for this weekday, from config/schedule.conf."""
    out = []
    try:
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[0] == "window" and parts[1].isdigit() and int(parts[1]) == weekday + 1:
                out.append((f"{DAYS[weekday]} {parts[2]}–{parts[3]} window", parts[2] + ":00", parts[3] + ":00"))
    except OSError:
        pass
    return out


def shift(hms: str, minutes: int) -> str:
    h, m, s = map(int, hms.split(":"))
    total = max(0, min(24 * 60 - 1, h * 60 + m + minutes))
    return f"{total // 60:02d}:{total % 60:02d}:{s:02d}"


def pct(v):
    return f"{100 * v:.1f}%"


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else float("nan")


def report(out, label, group, lines):
    out.append(f"\n=== {label}: {len(group)} start(s): "
               + ", ".join(f"{r['start']}-{r['end'][:5]}" + (f" ({r['how']})" if r['how'] and 'Deactivated' not in r['how'] else "")
                           for r in group)
               + (f"   [{len(group) - 1} restart(s)]" if len(group) > 1 else ""))
    if not group:
        return

    def in_runs(t):
        return any(r["start"] <= t < r["end"] for r in group)

    sents = [(m.group(2), m.group(3).strip()) for m in map(EN.match, lines) if m and in_runs(m.group(2))]
    trs = [m.groups()[1:] for m in map(TR.match, lines) if m and in_runs(m.group(2))]
    speeds = [m.groups()[1:] for m in map(SPEED.match, lines) if m and in_runs(m.group(2))]
    seg = [l for l in lines if (mm := TS.match(l)) and in_runs(mm.group(2))]
    errors = sum(1 for l in seg if "| ERROR " in l or "Playback error" in l)
    drops = sum(1 for l in seg if "DROP |" in l)
    frags = sum(1 for l in seg if "[EN-frag]" in l)
    beats = sum(1 for l in seg if "HEARTBEAT |" in l)
    if not sents:
        out.append(f"    no sentences ({frags} fragments, {beats} heartbeats, {errors} errors)")
        return
    first, last = sents[0][0], sents[-1][0]
    h = lambda t: int(t[:2]) * 60 + int(t[3:5]) + int(t[6:8]) / 60
    mins = max(1e-9, h(last) - h(first))
    texts = [t for _, t in sents]
    words = [len(t.split()) for t in texts]
    out.append(f"    {first} -> {last} ({mins:.0f} min): {len(sents)} sentences from {frags} fragments "
               f"({len(sents) / mins:.1f}/min) | heartbeats {beats} | errors {errors} | drops {drops}")
    out.append(f"    ends with .?!   {pct(sum(bool(END.search(t)) for t in texts) / len(texts))}")
    out.append(f"    leads with mark {pct(sum(bool(LEAD.match(t)) for t in texts) / len(texts))}")
    out.append(f"    two sentences   {pct(sum(bool(TWO.search(t)) for t in texts) / len(texts))}")
    out.append(f"    words/segment   median {statistics.median(words):.0f}  p90 {q(words, .9)}  max {max(words)}"
               f"  | >=60 words (cap-sized) {sum(w >= 60 for w in words)}")
    unp = [w for t, w in zip(texts, words) if not END.search(t)]
    if unp:
        out.append(f"    unpunctuated segments (cut by cap/timeout/silence): {len(unp)} — words median {statistics.median(unp):.0f} p90 {q(unp, .9)}")
    bins = collections.OrderedDict()
    for ts, t in sents:
        bins.setdefault(ts[:4] + "0", []).append(bool(END.search(t)))
    e2e_bins = collections.OrderedDict()
    for r in trs:
        if r[1] == "ES":
            e2e_bins.setdefault(r[0][:4] + "0", []).append(float(r[3]))
    out.append("    10-min bins (sentences | ending with .?! | cut unpunctuated | ES latency first word -> audio, median / p95):")
    for b, v in bins.items():
        e = e2e_bins.get(b, [])
        lat = f"{statistics.median(e):4.1f}s / {q(e, .95):4.1f}s" if e else "   -"
        out.append(f"      {b}  {len(v):3d}  {100 * sum(v) / len(v):3.0f}%  {len(v) - sum(v):3d}   {lat}")
    if trs:
        out.append("    per language (translation events):")
        for lang in ("ES", "HT", "RU"):
            rows = [r for r in trs if r[1] == lang]
            if not rows:
                continue
            e2e = [float(r[3]) for r in rows]; tr = [float(r[4]) for r in rows]
            tts = [float(r[5]) for r in rows]; au = [float(r[6]) for r in rows]; qd = [int(r[7]) for r in rows]
            out.append(f"      {lang}: n={len(rows)}  e2e median {statistics.median(e2e):.1f}s p95 {q(e2e, .95):.1f}s max {max(e2e):.1f}s"
                       f" | translate median {statistics.median(tr):.2f}s | tts median {statistics.median(tts):.2f}s"
                       f" | audio median {statistics.median(au):.1f}s | queue>0 {pct(sum(x > 0 for x in qd) / len(qd))} max {max(qd)}")
    if speeds:
        c = collections.Counter((l, f) for _, l, _, f in speeds)
        out.append("    speed-ups: " + ", ".join(f"{l} {f}x ×{k}" for (l, f), k in sorted(c.items())))
    else:
        out.append("    speed-ups: none (queues never deep enough)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=dt.date.today().isoformat(), help="YYYY-MM-DD (default today)")
    ap.add_argument("--log", default=os.path.expanduser("~/translate.log"))
    ap.add_argument("--schedule", default=os.path.expanduser("~/translator/config/schedule.conf"))
    ap.add_argument("--journal", help="a file of `journalctl -o short-iso` lines instead of live journald (tests)")
    ap.add_argument("--out", help="directory: write <date>.txt and latest.txt there as well")
    a = ap.parse_args()
    date = dt.date.fromisoformat(a.date)

    lines = [ANSI.sub("", l.rstrip("\n")) for l in open(a.log, errors="replace")]
    day = [l for l in lines if l.startswith(date.isoformat() + " ")]
    runs = journal_runs(date, Path(a.journal).read_text() if a.journal else None)
    windows = schedule_windows(Path(a.schedule), date.weekday())

    out = [f"Service tally for {DAYS[date.weekday()]} {date.isoformat()} — generated {dt.datetime.now():%Y-%m-%d %H:%M}",
           f"{len(day)} log lines that day; translate.service started {len(runs)} time(s); "
           f"{len(windows)} scheduled window(s)"]
    unassigned = list(runs)
    for label, start, end in windows:
        grp = [r for r in unassigned if shift(start, -10) <= r["start"] <= shift(end, 5)]
        unassigned = [r for r in unassigned if r not in grp]
        report(out, label, grp, day)
    if unassigned or not windows:
        report(out, "outside any window (manual starts, tests)", unassigned, day)
    text = "\n".join(out) + "\n"
    print(text, end="")
    if a.out:
        d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
        (d / f"{date.isoformat()}.txt").write_text(text)
        (d / "latest.txt").write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
