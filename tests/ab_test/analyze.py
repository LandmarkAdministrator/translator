#!/usr/bin/env python3
"""
ASR A/B analyzer — parse a translator log file, extract English ASR output
and latency numbers, and report WER against a reference transcript.

Usage:
    python analyze.py <log_file> <reference_file>

The log file is a standard translator log (logs/translator_*.log). The
reference file is a plain-text canonical transcript; lines starting with
"#" and a line matching "---REFERENCE BEGINS---" are stripped.

Both reference and hypothesis are normalized (lowercased, punctuation
stripped, whitespace collapsed) before WER is computed.
"""

from __future__ import annotations

import re
import sys
import statistics
from pathlib import Path


# ---- Log parsing -----------------------------------------------------------

# Matches both batch and streaming [EN] lines.
#   batch:     | [EN] <text> | chunk=7.00s | seg=... | asr=0.521s | ...
#   streaming: | [EN] <text> | mode=streaming | asr=0.310s
EN_LINE_RE = re.compile(r"\|\s*\[EN\]\s+(?P<text>.+?)\s*\|\s*(?P<meta>.*)$")

# Translation events:
#   | [ES] <text> | e2e=9.93s | translate=0.703s | tts=0.157s | audio=4.88s | queue_was=0
XLATE_LINE_RE = re.compile(
    r"\|\s*\[(?P<lang>ES|HT|FR)\]\s+(?P<text>.+?)\s*\|\s*(?P<meta>.*)$"
)

# Extract "key=VALUEs" or "key=VALUE" (numeric) from meta portion.
NUM_FIELD_RE = re.compile(r"(?P<k>\w+)=(?P<v>-?\d+(?:\.\d+)?)s?\b")

# Mode detection from an [EN] meta string.
def _detect_mode(meta: str) -> str:
    if "mode=streaming" in meta:
        return "streaming"
    if "chunk=" in meta:
        return "batch"
    return "unknown"


def _fields(meta: str) -> dict:
    return {m.group("k"): float(m.group("v")) for m in NUM_FIELD_RE.finditer(meta)}


def parse_log(path: Path):
    """Return dict with en_chunks (list of dicts) and xlate_events (list)."""
    en_chunks = []
    xlate_events = []

    with path.open("r", errors="replace") as f:
        for line in f:
            m = EN_LINE_RE.search(line)
            if m:
                meta = m.group("meta")
                en_chunks.append({
                    "text": m.group("text").strip(),
                    "mode": _detect_mode(meta),
                    **_fields(meta),
                })
                continue
            m = XLATE_LINE_RE.search(line)
            if m:
                meta = m.group("meta")
                xlate_events.append({
                    "lang": m.group("lang"),
                    "text": m.group("text").strip(),
                    **_fields(meta),
                })
    return {"en_chunks": en_chunks, "xlate_events": xlate_events}


# ---- Reference handling ---------------------------------------------------

def load_reference(path: Path) -> str:
    lines = []
    started = False
    for raw in path.read_text(errors="replace").splitlines():
        if not started:
            if raw.strip() == "---REFERENCE BEGINS---":
                started = True
            continue
        if raw.lstrip().startswith("#"):
            continue
        lines.append(raw)
    text = "\n".join(lines).strip()
    # If no BEGIN marker was found, just use the whole file minus # lines.
    if not started:
        lines = [ln for ln in path.read_text(errors="replace").splitlines()
                 if not ln.lstrip().startswith("#")]
        text = "\n".join(lines).strip()
    return text


# ---- Normalization ---------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s']")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# ---- WER (fallback if jiwer missing) --------------------------------------

def _wer_levenshtein(ref_words, hyp_words) -> tuple[int, int, int, int]:
    """Return (substitutions, deletions, insertions, ref_len)."""
    n, m = len(ref_words), len(hyp_words)
    if n == 0:
        return (0, 0, m, 0)
    # DP table of edit distance + operation counts
    # Use two rows for memory efficiency.
    prev = list(range(m + 1))
    prev_ops = [(0, 0, i) for i in range(m + 1)]  # (S, D, I)
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        curr_ops = [(0, i, 0)] + [(0, 0, 0)] * m
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                curr[j] = prev[j - 1]
                curr_ops[j] = prev_ops[j - 1]
            else:
                sub = prev[j - 1] + 1
                dele = prev[j] + 1
                ins = curr[j - 1] + 1
                best = min(sub, dele, ins)
                curr[j] = best
                if best == sub:
                    s, d, ii = prev_ops[j - 1]
                    curr_ops[j] = (s + 1, d, ii)
                elif best == dele:
                    s, d, ii = prev_ops[j]
                    curr_ops[j] = (s, d + 1, ii)
                else:
                    s, d, ii = curr_ops[j - 1]
                    curr_ops[j] = (s, d, ii + 1)
        prev, prev_ops = curr, curr_ops
    s, d, ins = prev_ops[m]
    return (s, d, ins, n)


def compute_wer(reference: str, hypothesis: str):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    try:
        import jiwer  # type: ignore
        wer = jiwer.wer(reference, hypothesis)
        try:
            measures = jiwer.compute_measures(reference, hypothesis)
            s = measures.get("substitutions", 0)
            d = measures.get("deletions", 0)
            ins = measures.get("insertions", 0)
        except Exception:
            s, d, ins, _ = _wer_levenshtein(ref_words, hyp_words)
        return {
            "wer": wer,
            "substitutions": s,
            "deletions": d,
            "insertions": ins,
            "ref_words": len(ref_words),
            "hyp_words": len(hyp_words),
            "backend": "jiwer",
        }
    except ImportError:
        s, d, ins, n = _wer_levenshtein(ref_words, hyp_words)
        wer = (s + d + ins) / max(n, 1)
        return {
            "wer": wer,
            "substitutions": s,
            "deletions": d,
            "insertions": ins,
            "ref_words": n,
            "hyp_words": len(hyp_words),
            "backend": "builtin",
        }


# ---- Stats helpers --------------------------------------------------------

def _pct(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(name, values, unit="s"):
    if not values:
        print(f"  {name:14s}: (no data)")
        return
    print(f"  {name:14s}: n={len(values):<4d} "
          f"mean={statistics.mean(values):.3f}{unit}  "
          f"median={statistics.median(values):.3f}{unit}  "
          f"p95={_pct(values, 95):.3f}{unit}  "
          f"max={max(values):.3f}{unit}")


# ---- Main ------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <log_file> <reference_file>", file=sys.stderr)
        sys.exit(2)

    log_path = Path(sys.argv[1])
    ref_path = Path(sys.argv[2])

    if not log_path.is_file():
        print(f"ERROR: log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)
    if not ref_path.is_file():
        print(f"ERROR: reference file not found: {ref_path}", file=sys.stderr)
        sys.exit(1)

    parsed = parse_log(log_path)
    en_chunks = parsed["en_chunks"]
    xlate_events = parsed["xlate_events"]

    modes = {c["mode"] for c in en_chunks}
    mode_str = "/".join(sorted(modes)) if modes else "unknown"

    hyp = " ".join(c["text"] for c in en_chunks)
    hyp_norm = normalize(hyp)
    ref_norm = normalize(load_reference(ref_path))

    if not ref_norm:
        print("ERROR: reference file is empty after stripping comments.",
              file=sys.stderr)
        print("       Paste the canonical transcript into the reference file.",
              file=sys.stderr)
        sys.exit(1)

    wer_info = compute_wer(ref_norm, hyp_norm)

    print("=" * 72)
    print(f"Log file   : {log_path}")
    print(f"Reference  : {ref_path}")
    print(f"ASR mode   : {mode_str}")
    print(f"WER backend: {wer_info['backend']}")
    print("=" * 72)
    print()
    print("English ASR accuracy (vs reference):")
    print(f"  WER           : {wer_info['wer']*100:.2f}%")
    print(f"  substitutions : {wer_info['substitutions']}")
    print(f"  deletions     : {wer_info['deletions']}")
    print(f"  insertions    : {wer_info['insertions']}")
    print(f"  ref words     : {wer_info['ref_words']}")
    print(f"  hyp words     : {wer_info['hyp_words']}")
    print()

    # ASR latency
    asr_times = [c["asr"] for c in en_chunks if "asr" in c]
    chunk_durs = [c["chunk"] for c in en_chunks if "chunk" in c]
    print("ASR latency:")
    summarize("asr_time", asr_times)
    summarize("chunk_dur", chunk_durs)
    if chunk_durs and asr_times and len(chunk_durs) == len(asr_times):
        rtfs = [a / c for a, c in zip(asr_times, chunk_durs) if c > 0]
        summarize("RTF", rtfs, unit="")
    print()

    # Translation latency (end-to-end)
    by_lang: dict[str, list[float]] = {}
    translate_by_lang: dict[str, list[float]] = {}
    tts_by_lang: dict[str, list[float]] = {}
    queue_by_lang: dict[str, list[float]] = {}
    for ev in xlate_events:
        if "e2e" in ev:
            by_lang.setdefault(ev["lang"], []).append(ev["e2e"])
        if "translate" in ev:
            translate_by_lang.setdefault(ev["lang"], []).append(ev["translate"])
        if "tts" in ev:
            tts_by_lang.setdefault(ev["lang"], []).append(ev["tts"])
        if "queue_was" in ev:
            queue_by_lang.setdefault(ev["lang"], []).append(ev["queue_was"])

    for lang in sorted(by_lang):
        print(f"[{lang}] pipeline latency:")
        summarize("e2e",       by_lang.get(lang, []))
        summarize("translate", translate_by_lang.get(lang, []))
        summarize("tts",       tts_by_lang.get(lang, []))
        summarize("queue_was", queue_by_lang.get(lang, []), unit="")
        print()

    # Diff preview (first 400 chars of hyp vs ref for sanity check)
    print("-" * 72)
    print("Normalized hypothesis (first 300 chars):")
    print(f"  {hyp_norm[:300]}{'...' if len(hyp_norm) > 300 else ''}")
    print()
    print("Normalized reference  (first 300 chars):")
    print(f"  {ref_norm[:300]}{'...' if len(ref_norm) > 300 else ''}")
    print("=" * 72)


if __name__ == "__main__":
    main()
