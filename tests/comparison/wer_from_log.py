#!/usr/bin/env python3
"""Word error rate of a translator log against a reference transcript.

Works for both log formats:
  old translate.py  : "... - Chunk N Recognized: <text>"        (skips [silence...])
  new stack (loguru): lines containing "Transcribed:" or "ASR:" followed by text
                      (pass --pattern to override)

The audio is usually played on a loop, so the reference is repeated --repeats
times to match; by default the script picks the repeat count that minimises
WER (1..8), which is what you want when a window covers a whole number of
passes.  Trim the window with --from/--to (log-local wall-clock, "HH:MM[:SS]"
or full "YYYY-MM-DD HH:MM:SS") so it starts at the beginning of a pass.

Normalisation before scoring: lowercase, strip punctuation, collapse
whitespace, spell out small integers.  Stdlib only.

Example:
  wer_from_log.py ~/translate.log tests/ab_test/reference_jfk_inaugural.txt \
      --from "14:30" --to "15:00"
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime

OLD_RECOG = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - Chunk \d+ Recognized: (.*)$")
# New stack (loguru file format): "YYYY-MM-DD HH:mm:ss.SSS | INFO | where | [EN-frag] <text> | mode=..."
# Use the flushed sentences ("[EN] ... | mode=streaming/sentence"), which are what
# the translators receive; the finer "[EN-frag]" commits can re-emit text after a
# buffer trim when ASR falls behind, which would inflate insertions.
DEFAULT_NEW = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+ \| \S+\s*\| [^|]+ \| \[EN\] (.*?) \| mode=streaming/sentence"

_SMALL = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen",
    "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen",
    "19": "nineteen", "20": "twenty", "30": "thirty", "40": "forty",
    "50": "fifty", "60": "sixty", "70": "seventy", "80": "eighty", "90": "ninety",
    "100": "hundred", "1000": "thousand",
}


def read_reference(path: str) -> str:
    """Read a reference transcript, dropping comment/header lines.

    The reference files carry a '#'-comment header and a ---REFERENCE
    BEGINS--- marker that are not speech; scoring them as deletions inflated
    every WER by ~3 points per pass until 2026-09-01.
    """
    lines = []
    for ln in open(path, errors="replace"):
        s = ln.strip()
        if s.startswith("#") or (s.startswith("---") and s.endswith("---")):
            continue
        lines.append(ln)
    return " ".join(lines)


def normalize(text: str) -> list[str]:
    text = text.lower().replace("-", " ").replace("—", " ").replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    words = []
    for w in text.split():
        w = w.strip("'")
        if not w:
            continue
        words.append(_SMALL.get(w, w))
    return words


def wer(ref: list[str], hyp: list[str]) -> tuple[float, int, int, int]:
    """Return (wer, substitutions, deletions, insertions) via word Levenshtein."""
    n, m = len(ref), len(hyp)
    # dp rows: cost, and we backtrack counts with a parallel table of (S,D,I)
    prev = list(range(m + 1))
    prev_ops = [(0, 0, i) for i in range(m + 1)]
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        cur_ops = [(0, i, 0)] + [(0, 0, 0)] * m
        ri = ref[i - 1]
        for j in range(1, m + 1):
            if ri == hyp[j - 1]:
                cur[j] = prev[j - 1]
                cur_ops[j] = prev_ops[j - 1]
            else:
                sub = prev[j - 1] + 1
                dele = prev[j] + 1
                ins = cur[j - 1] + 1
                best = min(sub, dele, ins)
                cur[j] = best
                if best == sub:
                    s, d, k = prev_ops[j - 1]; cur_ops[j] = (s + 1, d, k)
                elif best == dele:
                    s, d, k = prev_ops[j]; cur_ops[j] = (s, d + 1, k)
                else:
                    s, d, k = cur_ops[j - 1]; cur_ops[j] = (s, d, k + 1)
        prev, prev_ops = cur, cur_ops
    s, d, k = prev_ops[m]
    return (prev[m] / max(n, 1), s, d, k)


def parse_when(s: str, day_hint: str | None) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    if day_hint is None:
        raise SystemExit("--from/--to need a date unless the log has dated lines")
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(f"{day_hint} {s}", f"%Y-%m-%d {fmt}")
        except ValueError:
            pass
    raise SystemExit(f"bad time: {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("reference")
    ap.add_argument("--from", dest="t_from")
    ap.add_argument("--to", dest="t_to")
    ap.add_argument("--repeats", type=int, help="reference repeats (default: best of 1..8)")
    ap.add_argument("--pattern", help="regex with 2 groups (timestamp, text) for the new-stack log")
    ap.add_argument("--dump", help="write the normalised hypothesis to this file")
    args = ap.parse_args()

    pat_new = re.compile(args.pattern or DEFAULT_NEW)
    lines = open(args.log, errors="replace").read().splitlines()
    day_hint = None
    for ln in lines:
        m = re.match(r"^(\d{4}-\d{2}-\d{2})", ln)
        if m:
            day_hint = m.group(1)
            break
    t0 = parse_when(args.t_from, day_hint) if args.t_from else None
    t1 = parse_when(args.t_to, day_hint) if args.t_to else None

    hyp_words: list[str] = []
    n_lines = 0
    for ln in lines:
        m = OLD_RECOG.match(ln) or pat_new.match(ln)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        if (t0 and ts < t0) or (t1 and ts > t1):
            continue
        text = m.group(2).strip()
        if text.startswith("[silence"):
            continue
        hyp_words += normalize(text)
        n_lines += 1

    ref_once = normalize(read_reference(args.reference))
    if args.dump:
        open(args.dump, "w").write(" ".join(hyp_words) + "\n")
    if not hyp_words:
        raise SystemExit("no recognized text in that window")

    candidates = [args.repeats] if args.repeats else range(1, 9)
    best = None
    for r in candidates:
        ref = ref_once * r
        w, s, d, i = wer(ref, hyp_words)
        if best is None or w < best[1]:
            best = (r, w, s, d, i, len(ref))
    r, w, s, d, i, nref = best
    print(f"log lines used: {n_lines}   hypothesis words: {len(hyp_words)}")
    print(f"reference: {len(ref_once)} words x {r} repeat(s) = {nref}")
    print(f"WER = {w*100:.2f}%   (sub {s}, del {d}, ins {i})")


if __name__ == "__main__":
    main()
