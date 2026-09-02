#!/usr/bin/env python3
"""Per-track scoring for diverse-suite runs.

Reads the suite manifest and a translator log (old translate.py format or
the new stack's, console-captured logs included via score_all's normalizer).
Each loop pass is anchored on the FIRST track's marker; every other track's
window is derived from its manifest offset, so tracks without their own
marker (the singing/stability track) still get windows. Marker matching is
tolerant: any 4 consecutive words of the marker appearing in a line counts
(loop boundaries can clip the first word).

Usage: score_suite.py MANIFEST_DIR LOG --type old|new [--slack 12]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from wer_from_log import normalize  # noqa: E402

OLD_RECOG = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - Chunk \d+ Recognized: (.*)$")
NEW_SENT = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+ \| \S+\s*\| [^|]+ \| \[EN\] (.*?) \| mode=streaming/sentence")
ERRLINE = re.compile(r"Playback error|ERROR|dropped .*audio that never stabilized", re.I)


def lines_of(log: Path, kind: str):
    pat = OLD_RECOG if kind == "old" else NEW_SENT
    for ln in open(log, errors="replace"):
        m = pat.match(ln)
        if m:
            text = m.group(2)
            if text.startswith("[silence"):
                continue
            yield datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), text


def marker_hit(marker_words: list[str], line_words: list[str]) -> bool:
    if len(marker_words) < 4:
        return False
    joined = " ".join(line_words)
    for i in range(len(marker_words) - 3):
        if " ".join(marker_words[i:i + 4]) in joined:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest_dir", type=Path)
    ap.add_argument("log", type=Path)
    ap.add_argument("--type", choices=["old", "new"], required=True)
    ap.add_argument("--slack", type=float, default=12.0)
    a = ap.parse_args()

    manifest = json.loads((a.manifest_dir / "manifest.json").read_text())
    tracks = manifest["tracks"]
    anchor_marker = normalize(tracks[0]["marker"])

    entries = list(lines_of(a.log, a.type))
    anchors = []
    last = None
    for t, text in entries:
        if marker_hit(anchor_marker, normalize(text)):
            if last is None or (t - last).total_seconds() > 120:
                anchors.append(t)
            last = t
    if not anchors:
        sys.exit("no suite passes found (first track's marker never seen)")
    print(f"passes found: {len(anchors)} at {[a0.strftime('%H:%M:%S') for a0 in anchors]}")

    for track in tracks:
        name = track["name"]
        ref_words = None
        if track.get("ref"):
            ref_words = normalize((a.manifest_dir / track["ref"]).read_text())
        for p, anchor in enumerate(anchors, 1):
            w0 = anchor + timedelta(seconds=track["start_sec"] - a.slack)
            w1 = anchor + timedelta(seconds=track["start_sec"] + track["duration_sec"] + a.slack)
            hyp_words, n_lines = [], 0
            for t, text in entries:
                if w0 <= t <= w1:
                    hyp_words += normalize(text)
                    n_lines += 1
            if ref_words:
                from wer_from_log import wer
                if not hyp_words:
                    print(f"  {name:<10} pass {p}: NO OUTPUT in window")
                    continue
                w, s, d, i = wer(ref_words, hyp_words)
                print(f"  {name:<10} pass {p}: WER {w*100:5.2f}%  (sub {s} del {d} ins {i}; {n_lines} lines)")
            else:
                # stability track: report volume of emissions only
                print(f"  {name:<10} pass {p}: {n_lines} lines, {len(hyp_words)} words emitted "
                      f"(stability probe — inspect text manually)")


if __name__ == "__main__":
    main()
