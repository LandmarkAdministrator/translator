#!/usr/bin/env python3
"""WER of a plain-text transcript file against a reference transcript.

Same normalisation and word-level alignment as wer_from_log.py (imported from
it), for scoring batch/offline transcriptions.

Usage: score_text.py HYPOTHESIS.txt REFERENCE.txt [--repeats N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from wer_from_log import normalize, read_reference, wer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hypothesis")
    ap.add_argument("reference")
    ap.add_argument("--repeats", type=int, default=1)
    a = ap.parse_args()
    hyp = normalize(open(a.hypothesis, errors="replace").read())
    ref = normalize(read_reference(a.reference)) * a.repeats
    w, s, d, i = wer(ref, hyp)
    print(f"hyp {len(hyp)} words vs ref {len(ref)} words")
    print(f"WER = {w*100:.2f}%   (sub {s}, del {d}, ins {i})")


if __name__ == "__main__":
    main()
