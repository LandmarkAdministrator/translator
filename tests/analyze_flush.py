#!/usr/bin/env python3
"""Compare sentence-flush strategies against a real service transcript.

The streaming ASR emits sentence-final punctuation at the *head* of the
following fragment ("...through the sky" then ". The grain shall then be"), so
the buffer's "ends with .?!" flush trigger almost never fires and sentences get
cut wherever the speaker happened to pause. This replays the recorded fragment
stream through several strategies so the choice is made on evidence rather than
intuition.

    python3 tests/analyze_flush.py service_text.log
"""
from __future__ import annotations

import re
import statistics
import sys
from typing import Iterator

FRAG = re.compile(r"\[EN-frag\]\s(.*?)(?:\s\|\smode=|$)")
LEAD_PUNCT = re.compile(r"^\s*([.?!,;:]+)")
SENT_END = re.compile(r"[.?!][\"')\]]?\s*$")
ABBREV = re.compile(
    r"(?:^|[\s\"(\[\'])"
    r"(?:Mr|Mrs|Ms|Mx|Dr|St|Jr|Sr|Mt|Ft|Fr|Rev|Hon|vs|etc|i\.e|e\.g|cf|No|Vol)\.\s*$",
    re.IGNORECASE,
)


def fragments(path: str) -> Iterator[str]:
    for line in open(path, errors="replace"):
        m = FRAG.search(line)
        if m and m.group(1).strip():
            yield m.group(1)


def join(buf: str, frag: str) -> str:
    """Mirror the buffer's space-aware join: leading space = new word."""
    if not buf:
        return frag.lstrip()
    return buf + frag if frag.startswith(" ") else buf + frag


def flush_current(frags, min_words=3):
    """What runs today: flush only when the buffer ends in .?! (or on timeout,
    which we cannot replay from a log, so this shows the punctuation path only)."""
    out, buf = [], ""
    for f in frags:
        buf = join(buf, f)
        if SENT_END.search(buf) and not ABBREV.search(buf) and len(buf.split()) >= min_words:
            out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out


def flush_punct_moved(frags, min_words=3):
    """Proposed: a fragment starting with sentence-final punctuation closes the
    sentence already in the buffer, and that mark goes where it belongs."""
    out, buf = [], ""
    for f in frags:
        m = LEAD_PUNCT.match(f)
        if m and buf:
            mark = m.group(1)
            candidate = buf.rstrip() + mark
            rest = f[m.end():]
            if (any(c in mark for c in ".?!")
                    and not ABBREV.search(candidate)
                    and len(candidate.split()) >= min_words):
                out.append(candidate.strip())
                buf = rest.lstrip()
                continue
            buf = candidate + rest
            continue
        buf = join(buf, f)
        if SENT_END.search(buf) and not ABBREV.search(buf) and len(buf.split()) >= min_words:
            out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out


def report(name: str, sents: list[str]) -> dict:
    if not sents:
        print(f"  {name}: no sentences")
        return {}
    words = [len(s.split()) for s in sents]
    lead = sum(1 for s in sents if LEAD_PUNCT.match(s))
    end = sum(1 for s in sents if SENT_END.search(s))
    short = sum(1 for w in words if w <= 3)
    long_ = sum(1 for w in words if w > 40)
    print(f"  {name}")
    print(f"    sentences        {len(sents)}")
    print(f"    words/sentence   median {statistics.median(words):.0f}  "
          f"mean {statistics.mean(words):.1f}  max {max(words)}")
    print(f"    ends with .?!    {end} ({100*end/len(sents):.1f}%)")
    print(f"    starts w/ punct  {lead} ({100*lead/len(sents):.1f}%)")
    print(f"    <=3 words        {short} ({100*short/len(sents):.1f}%)  "
          f"(NLLB hallucinates on these)")
    print(f"    >40 words        {long_} ({100*long_/len(sents):.1f}%)  "
          f"(late audio, high latency)")
    return {"n": len(sents), "end": end, "lead": lead, "median": statistics.median(words)}


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "service_text.log"
    frags = list(fragments(path))
    print(f"replaying {len(frags)} recorded ASR fragments\n")

    lead_frags = sum(1 for f in frags if LEAD_PUNCT.match(f))
    end_frags = sum(1 for f in frags if SENT_END.search(f))
    print(f"  fragment stream: {100*lead_frags/len(frags):.1f}% start with punctuation, "
          f"{100*end_frags/len(frags):.1f}% end with it")
    print("  (this is why the 'buffer ends in .?!' trigger rarely fires)\n")

    report("CURRENT (punctuation trigger only)", flush_current(frags))
    print()
    report("PROPOSED (move leading mark onto previous sentence)", flush_punct_moved(frags))

    print("\n  === sample output, proposed ===")
    for s in flush_punct_moved(frags)[40:48]:
        print(f"    {s[:96]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
