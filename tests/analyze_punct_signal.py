#!/usr/bin/env python3
"""How usable is the ASR's punctuation as a complete-thought signal?

Two questions:
  1. How often does a definitive sentence mark actually arrive?
  2. What would it cost in latency to always wait for one?

The mark arrives at the head of the fragment *after* the sentence it closes, so
"waiting for it" means holding the finished words until the speaker has said
enough of the next sentence for the ASR to commit that fragment. This measures
that wait against the audio clock, which is what a listener experiences.

    python3 tests/analyze_punct_signal.py frags15.json
"""
from __future__ import annotations

import json
import re
import statistics
import sys

SENT_MARK = re.compile(r"^\s*([.?!])")
ANY_MARK = re.compile(r"^\s*([.?!,;:])")


def main() -> int:
    d = json.load(open(sys.argv[1]))
    chunks = d["chunks"]
    push = d.get("push_secs", 1.5)
    audio_secs = d["seconds"]
    withtext = [c for c in chunks if c["text"]]

    # Flatten to a word stream carrying the audio time each word was spoken and
    # the wall-clock moment its chunk became available.
    words = []
    for c in chunks:
        toks, times = c.get("tokens") or [], c.get("times") or []
        for j, tk in enumerate(toks):
            words.append({
                "tok": tk,
                "audio_t": times[j] if j < len(times) else None,
                "avail_t": c["t"],           # when we could first have acted on it
            })
    have_times = sum(1 for w in words if w["audio_t"] is not None)
    print(f"  {audio_secs:.0f}s of audio, {len(chunks)} chunks of {push}s, "
          f"{len(withtext)} produced text")
    print(f"  {len(words)} tokens, {have_times} with acoustic timestamps\n")

    # --- 1. how often does a definitive mark arrive? ---
    sent_marks = [i for i, w in enumerate(words) if SENT_MARK.match(w["tok"])]
    any_marks = [i for i, w in enumerate(words) if ANY_MARK.match(w["tok"])]
    print("  === 1. how often is there a definitive sentence mark? ===")
    print(f"    sentence marks (. ? !):   {len(sent_marks)}  "
          f"= one per {len(words)/max(1,len(sent_marks)):.1f} tokens, "
          f"{60*len(sent_marks)/audio_secs:.1f}/min")
    print(f"    any mark (incl. , ; :):   {len(any_marks)}  "
          f"= one per {len(words)/max(1,len(any_marks)):.1f} tokens, "
          f"{60*len(any_marks)/audio_secs:.1f}/min")
    if len(sent_marks) > 1:
        gaps = [b - a for a, b in zip(sent_marks, sent_marks[1:])]
        gs = sorted(gaps)
        print(f"    tokens between sentence marks: median {statistics.median(gs):.0f}  "
              f"p90 {gs[int(len(gs)*0.9)]}  max {max(gs)}")
    if len(any_marks) > 1:
        g2 = sorted(b - a for a, b in zip(any_marks, any_marks[1:]))
        print(f"    tokens between any mark:       median {statistics.median(g2):.0f}  "
              f"p90 {g2[int(len(g2)*0.9)]}  max {max(g2)}")

    # --- 2. what does waiting for the mark cost? ---
    # The sentence closed by a mark ends with the previous token. Its last word
    # was spoken at audio_t; we may only send it once the mark's chunk arrives.
    print("\n  === 2. latency cost of always waiting for a sentence mark ===")
    waits, spans = [], []
    for idx in sent_marks:
        if idx == 0:
            continue
        last = words[idx - 1]
        mark = words[idx]
        if last["audio_t"] is None:
            continue
        # Wall-clock availability of the mark, minus when the closing word was
        # spoken. Capture was realtime-paced so these clocks are comparable.
        waits.append(mark["avail_t"] - last["audio_t"])
        spans.append(mark["avail_t"] - last["avail_t"])
    if waits:
        ws = sorted(waits)
        print(f"    wait from last word spoken -> mark available:")
        print(f"      median {statistics.median(ws):.2f}s  p90 {ws[int(len(ws)*0.9)]:.2f}s  "
              f"max {max(ws):.2f}s")
        ss = sorted(spans)
        print(f"    extra delay vs sending as soon as the words were available:")
        print(f"      median {statistics.median(ss):.2f}s  p90 {ss[int(len(ss)*0.9)]:.2f}s  "
              f"max {max(ss):.2f}s")
        over = [x for x in ss if x > 5.0]
        print(f"      waits over 5s: {len(over)} of {len(ss)} ({100*len(over)/len(ss):.1f}%)")

    # --- 3. how long would a listener wait with no fallback at all? ---
    print("\n  === 3. worst case with NO fallback (mark-only flushing) ===")
    if len(sent_marks) > 1:
        secs = []
        for a, b in zip(sent_marks, sent_marks[1:]):
            ta, tb = words[a]["avail_t"], words[b]["avail_t"]
            secs.append(tb - ta)
        se = sorted(secs)
        print(f"    seconds between consecutive marks: median {statistics.median(se):.1f}s  "
              f"p90 {se[int(len(se)*0.9)]:.1f}s  max {max(se):.1f}s")
        print(f"    stretches over 15s: {sum(1 for x in se if x > 15)} "
              f"(these are where a fallback is unavoidable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
