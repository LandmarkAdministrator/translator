#!/usr/bin/env python3
"""Sweep flush settings against captured fragments with real arrival times.

capture_fragments.py pays for ASR once; this replays that stream through the
production SentenceBuffer at many settings, so the flush policy can be chosen
from real timing rather than from a log replay that cannot reconstruct
sub-second gaps.

    python3 tests/sweep_flush.py frags_sermon.json
"""
from __future__ import annotations

import json
import re
import statistics
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/src")
from pipeline.sentence_buffer import SentenceBuffer  # noqa: E402

LEAD = re.compile(r"^\s*[.?!,;:]")
END = re.compile(r"[.?!][\"')\]]?\s*$")


def replay(frags, **kw):
    """Feed the captured stream, ticking between fragments so the silence
    timeout can fire in the gaps exactly as it would live."""
    b = SentenceBuffer(**kw)
    out = []
    prev_t = frags[0]["t"]
    for f in frags:
        # Tick through the gap at the cadence the audio callback would.
        t = prev_t
        while t < f["t"]:
            t = min(t + 0.1, f["t"])
            r = b.tick(now=t)
            if r:
                out.append(r[0])
        r = b.feed(f["text"], start_wall=f["audio_t"], asr_time=0.0, now=f["t"])
        if r:
            out.append(r[0])
        prev_t = f["t"]
    r = b.flush()
    if r:
        out.append(r[0])
    return out


def stats(sents):
    w = sorted(len(s.split()) for s in sents)
    return {
        "n": len(sents),
        "median_w": statistics.median(w),
        "p90_w": w[int(len(w) * 0.9)],
        "max_w": max(w),
        "ends": 100 * sum(bool(END.search(s)) for s in sents) / len(sents),
        "leads": 100 * sum(bool(LEAD.match(s)) for s in sents) / len(sents),
        "tiny": 100 * sum(1 for x in w if x <= 3) / len(sents),
    }


def main() -> int:
    path = sys.argv[1]
    frags = json.load(open(path))["fragments"]
    print(f"  {len(frags)} captured fragments, real arrival times\n")

    base = dict(min_emit_words=3, max_buffer_chars=800)
    print(f"  {'silence':>7} {'hard':>5} {'maxw':>5} | {'sents':>5} {'med':>4} {'p90':>4} "
          f"{'max':>4} | {'ends.?!':>8} {'leads':>6} {'<=3w':>5}")
    print("  " + "-" * 74)

    rows = []
    for sil in (1.0, 1.5, 2.0, 2.5):
        for hard in (6.0, 10.0, 15.0):
            s = replay(frags, silence_timeout=sil, hard_timeout=hard, max_emit_words=40,
                       silence_min_words=1, strip_lead_punct=True, **base)
            st = stats(s)
            rows.append(((sil, hard), 40, st))
            print(f"  {sil:>7.1f} {hard:>5.0f} {40:>5} | {st['n']:>5} "
                  f"{st['median_w']:>4.0f} {st['p90_w']:>4} {st['max_w']:>4} | "
                  f"{st['ends']:>7.1f}% {st['leads']:>5.1f}% {st['tiny']:>4.1f}%")

    # Same sweep with the strip disabled, to show what the marks cost visually.
    print()
    for sil in (1.0, 2.0):
        s = replay(frags, silence_timeout=sil, hard_timeout=6.0, max_emit_words=40,
                   silence_min_words=1, strip_lead_punct=False, **base)
        st = stats(s)
        print(f"  {sil:>7.1f} {40:>5} {'no':>6} | {st['n']:>5} "
              f"{st['median_w']:>4.0f} {st['p90_w']:>4} {st['max_w']:>4} | "
              f"{st['ends']:>7.1f}% {st['leads']:>5.1f}% {st['tiny']:>4.1f}%")

    print("\n  === sentences at silence=1.0 (today) ===")
    for s in replay(frags, silence_timeout=1.0, hard_timeout=6.0, max_emit_words=0,
                    silence_min_words=3, strip_lead_punct=False, **base)[20:26]:
        print(f"    {s[:94]}")
    print("\n  === sentences at the best-looking setting ===")
    best = max(rows, key=lambda r: r[2]["ends"])
    (bsil, bhard) = best[0]
    print(f"    (silence={bsil}, hard={bhard}, max_words=40)")
    for s in replay(frags, silence_timeout=bsil, hard_timeout=bhard, max_emit_words=40,
                    silence_min_words=1, strip_lead_punct=True, **base)[20:26]:
        print(f"    {s[:94]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
