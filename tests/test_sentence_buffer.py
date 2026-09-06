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

print("\n11. two sentences in one fragment split at the FIRST mark, one per call")
b = SentenceBuffer(silence_timeout=99, hard_timeout=99, min_emit_words=3, max_emit_words=0,
                   punct_boundary=True)
got, t = feed_all(b, ["First sentence here. Second one too. Third begins"])
check("feed releases only the first", got, ["First sentence here."])
check("reason is punct_internal", b.last_reason, "punct_internal")
got = b.tick(now=t + 0.1)
check("tick releases the second without waiting on a timeout",
      got[0] if got else None, "Second one too.")
check("remainder stays buffered", "".join(b._frags).strip(), "Third begins")

print("\n12. a split remainder from feed() takes that fragment's clock")
b = SentenceBuffer(silence_timeout=99, hard_timeout=5.0, min_emit_words=3, max_emit_words=0,
                   punct_boundary=True)
b.feed("one two three four", start_wall=0.0, asr_time=0.0, now=100.0)
got = b.feed(" five six seven. eight nine", start_wall=1.0, asr_time=0.0, now=101.0)
check("head released", got[0] if got else None, "one two three four five six seven.")
check("not timed out 4.9 s after the remainder arrived", b.tick(now=105.9), None)
got = b.tick(now=106.0)
check("hard timeout runs from the remainder's arrival", got[0] if got else None, "eight nine")
check("reason recorded by tick", b.last_reason, "hard_timeout")

print("\n13. a split remainder from tick() keeps its arrival time (no fresh timeout)")
b = SentenceBuffer(silence_timeout=99, hard_timeout=5.0, min_emit_words=3, max_emit_words=0,
                   punct_boundary=True)
got = b.feed("alpha beta gamma delta. epsilon zeta eta. theta iota",
             start_wall=0.0, asr_time=0.0, now=100.0)
check("feed releases the first", got[0] if got else None, "alpha beta gamma delta.")
got = b.tick(now=101.5)
check("tick releases the second", got[0] if got else None, "epsilon zeta eta.")
check("remainder not timed out at 104.9", b.tick(now=104.9), None)
got = b.tick(now=105.0)
check("remainder times out 5 s after it ARRIVED (t=100), not after the split",
      got[0] if got else None, "theta iota")

print()
if FAIL:
    print(f"{len(FAIL)} FAILED: {FAIL}")
    sys.exit(1)
print("ALL SENTENCE BUFFER TESTS PASSED")
