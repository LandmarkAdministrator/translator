#!/usr/bin/env python3
"""Sentence-buffer flush policy.

The rules, in priority order:
  1. punctuation flushes early, but only once the unit is worth translating
  2. a word cap bounds how late an unpunctuated run can arrive
  3. silence flushes even very short utterances, so "Amen." is not held
  4. nothing is ever dropped for being short
  5. a mark opening a fragment belongs to the sentence already sent
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pipeline.sentence_buffer import SentenceBuffer  # noqa: E402

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"          got:  {got!r}")
        print(f"          want: {want!r}")
        FAIL.append(name)


def feed_all(buf, frags, t0=0.0, step=0.1):
    """Feed fragments at `step` apart; collect whatever is emitted."""
    out, t = [], t0
    for f in frags:
        t += step
        r = buf.feed(f, start_wall=t, asr_time=0.01, now=t)
        if r:
            out.append(r[0])
    return out, t


print("1. punctuation flushes early, once past the min-words floor")
b = SentenceBuffer(silence_timeout=99, hard_timeout=99, min_emit_words=3, max_emit_words=0)
got, _ = feed_all(b, ["The word of God", " is living", " and powerful.", " And so"])
check("flushes at the period", got, ["The word of God is living and powerful."])

print("\n2. a short clause + period does NOT split mid-thought")
b = SentenceBuffer(silence_timeout=99, hard_timeout=99, min_emit_words=3, max_emit_words=0)
got, _ = feed_all(b, ["Amen.", " And so it is"])
check("held below min_emit_words", got, [])

print("\n3. word cap bounds an unpunctuated run")
b = SentenceBuffer(silence_timeout=99, hard_timeout=99, min_emit_words=3, max_emit_words=6)
got, _ = feed_all(b, ["one two", " three four", " five six seven"])
check("flushed at the cap", len(got), 1)
check("cap respected (>=6 words)", len(got[0].split()) >= 6 if got else False, True)

print("\n4. silence flushes a SHORT utterance (the latency fix)")
b = SentenceBuffer(silence_timeout=1.0, hard_timeout=99, min_emit_words=3,
                   silence_min_words=1, max_emit_words=0)
_, t = feed_all(b, ["Amen."])
got = b.tick(now=t + 1.5)
check("short line released on silence", got[0] if got else None, "Amen.")

print("\n5. silence_min_words=0 releases even a bare word")
b = SentenceBuffer(silence_timeout=1.0, hard_timeout=99, min_emit_words=3,
                   silence_min_words=0, max_emit_words=0)
_, t = feed_all(b, ["Hallelujah"])
got = b.tick(now=t + 1.5)
check("bare word released", got[0] if got else None, "Hallelujah")

print("\n6. a mark opening a fragment is dropped, not shown to congregants")
b = SentenceBuffer(silence_timeout=99, hard_timeout=99, min_emit_words=3, max_emit_words=0)
got, _ = feed_all(b, [". The grain shall then be", " swallowed up in victory."])
check("no leading period", got, ["The grain shall then be swallowed up in victory."])

print("\n7. punctuation INSIDE a buffer still attaches correctly")
b = SentenceBuffer(silence_timeout=99, hard_timeout=99, min_emit_words=3, max_emit_words=0)
got, _ = feed_all(b, ["pierce through the sky", ". The grain shall then be swallowed."])
check("joined across the boundary", got,
      ["pierce through the sky. The grain shall then be swallowed."])

print("\n8. nothing is dropped for being short")
b = SentenceBuffer(silence_timeout=99, hard_timeout=99, min_emit_words=3, max_emit_words=0)
feed_all(b, ["Amen"])
got = b.flush()
check("shutdown flush releases it", got[0] if got else None, "Amen")

print("\n9. abbreviations do not end a sentence")
b = SentenceBuffer(silence_timeout=99, hard_timeout=99, min_emit_words=2, max_emit_words=0)
got, _ = feed_all(b, ["Turn to Mr.", " Smith and read"])
check("held at 'Mr.'", got, [])

print("\n10. subword joining is preserved (no 'just ice')")
b = SentenceBuffer(silence_timeout=99, hard_timeout=99, min_emit_words=2, max_emit_words=0)
got, _ = feed_all(b, [" just", "ice", " for all."])
check("subwords joined", got, ["justice for all."])

print()
if FAIL:
    print(f"{len(FAIL)} FAILED: {FAIL}")
    sys.exit(1)
print("ALL SENTENCE BUFFER TESTS PASSED")
