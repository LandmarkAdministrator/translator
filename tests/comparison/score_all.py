#!/usr/bin/env python3
"""Score every run in a head-to-head manifest: trimmed WER + latency summary.

Reads the orchestrator's manifest.tsv (run, type, config, start, end, log),
finds whole passes of the reference speech inside each run via a marker
phrase (default: the JFK opening), trims to full passes, and prints one row
per run. Old- and new-format logs are handled by type ("old"/"new").

Usage:
  score_all.py MANIFEST_DIR [--reference tests/ab_test/reference_jfk_inaugural.txt]
                            [--marker "vice president johnson"]

MANIFEST_DIR is the headtohead_* directory (containing manifest.tsv and logs).
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent


ANSI = re.compile(r"\x1b\[[0-9;]*m")
CONSOLE = re.compile(r"^(\d{2}:\d{2}:\d{2}) \|\s*(\S+:\d+) \|\s*(\w+) \| (.*)$")


def normalized_new_log(log: Path, run_date: str) -> Path:
    """The orchestrator captures loguru's CONSOLE sink (HH:MM:SS | name:line |
    LEVEL | msg, with ANSI colors); the parsers expect the FILE sink format.
    Convert; pass file-format lines through untouched. Handles midnight
    rollover by bumping the date when the clock goes backwards."""
    out = log.with_suffix(log.suffix + ".norm")
    from datetime import date as _date
    cur = datetime.strptime(run_date, "%Y-%m-%d")
    prev_t = None
    with open(log, errors="replace") as fin, open(out, "w") as fout:
        for ln in fin:
            ln = ANSI.sub("", ln.rstrip("\n"))
            m = CONSOLE.match(ln)
            if m:
                t = datetime.strptime(m.group(1), "%H:%M:%S").time()
                if prev_t and t < prev_t:
                    cur += timedelta(days=1)
                prev_t = t
                fout.write(f"{cur.strftime('%Y-%m-%d')} {m.group(1)}.000 | {m.group(3):<8} | {m.group(2)} | {m.group(4)}\n")
            else:
                fout.write(ln + "\n")
    return out

OLD_RECOG = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - Chunk \d+ Recognized: (.*)$")
NEW_SENT = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+ \| \S+\s*\| [^|]+ \| \[EN\] (.*?) \| mode=streaming/sentence")


def pass_starts(log: Path, kind: str, marker: str) -> list[str]:
    pat = OLD_RECOG if kind == "old" else NEW_SENT
    out, seen_recent = [], None
    for ln in open(log, errors="replace"):
        m = pat.match(ln)
        if not m:
            continue
        if marker in m.group(2).lower():
            ts = m.group(1)
            # ignore duplicate markers within 60s (marker split across lines)
            t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            if seen_recent and (t - seen_recent).total_seconds() < 60:
                continue
            out.append(ts)
            seen_recent = t
    return out


def wer_trimmed(log: Path, kind: str, ref: Path, marker: str) -> str:
    starts = pass_starts(log, kind, marker)
    if len(starts) < 2:
        return f"(need >=2 pass markers, found {len(starts)})"
    n = len(starts) - 1
    end = (datetime.strptime(starts[-1], "%Y-%m-%d %H:%M:%S") - timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
    cmd = [sys.executable, str(HERE / "wer_from_log.py"), str(log), str(ref),
           "--from", starts[0], "--to", end, "--repeats", str(n)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"WER = ([\d.]+)%\s+\(([^)]+)\)", r.stdout)
    return f"{m.group(1)}% over {n} pass(es) ({m.group(2)})" if m else r.stdout.strip().splitlines()[-1]


def latency(log: Path, kind: str) -> str:
    script = "parse_old_log.py" if kind == "old" else "parse_new_log.py"
    r = subprocess.run([sys.executable, str(HERE / script), str(log)], capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    return "\n    ".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest_dir")
    ap.add_argument("--reference", default=str(HERE.parent / "ab_test" / "reference_jfk_inaugural.txt"))
    ap.add_argument("--marker", default="vice president johnson")
    a = ap.parse_args()
    mdir = Path(a.manifest_dir)
    with open(mdir / "manifest.tsv") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for row in rows:
        log = Path(row["log"])
        if not log.exists():
            log = mdir / Path(row["log"]).name
        print("=" * 78)
        print(f"{row['run']}  [{row['type']}]  {row['config']}")
        print(f"  window: {row['start']} -> {row['end']}")
        if not log.exists():
            print("  LOG MISSING")
            continue
        if row["type"] == "new":
            log = normalized_new_log(log, row["start"][:10])
        print(f"  WER: {wer_trimmed(log, row['type'], Path(a.reference), a.marker)}")
        print(f"  latency:\n    {latency(log, row['type'])}")


if __name__ == "__main__":
    main()
