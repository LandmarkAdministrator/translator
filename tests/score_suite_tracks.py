#!/usr/bin/env python3
"""Per-track WER and segmentation for the diverse-input suite.

suite_v4 is five very different tracks in one file — a political speech, a
sermon, a commencement address, congregational singing, and a LibriSpeech
mixture. A single WER over the whole thing hides the one that matters: singing
is far harder than speech and would be averaged away.

Sentences are assigned to a track by the audio time of their first fragment,
which the capture records, so this needs no alignment guesswork.

    python3 tests/score_suite_tracks.py frags_suite.json
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pipeline.sentence_buffer import SentenceBuffer  # noqa: E402

SUITE = Path(__file__).resolve().parent / "comparison" / "suite"
END = re.compile(r"[.?!][\"')\]]?\s*$")
LEAD = re.compile(r"^\s*[.?!,;:]")

OLD = dict(silence_timeout=1.0, hard_timeout=6.0, min_emit_words=3,
           max_buffer_chars=800, max_emit_words=0, silence_min_words=3,
           strip_lead_punct=False, punct_boundary=False)
NEW = dict(silence_timeout=30.0, hard_timeout=15.0, min_emit_words=3,
           max_buffer_chars=800, max_emit_words=60, silence_min_words=1,
           strip_lead_punct=True, punct_boundary=True)


def replay(chunks, **kw):
    """Return (sentence, audio_time_of_first_fragment)."""
    b = SentenceBuffer(**kw)
    out = []
    for c in chunks:
        r = (b.feed(c["text"], start_wall=c["audio_t"], asr_time=0.0, now=c["t"])
             if c["text"] else b.tick(now=c["t"]))
        if r:
            out.append((r[0], r[1]))
    r = b.flush()
    if r:
        out.append((r[0], r[1]))
    return out


def words(text: str) -> list[str]:
    return [w for w in re.sub(r"[^\w\s']", " ", text.lower()).split() if w]


def wer(ref: list[str], hyp: list[str]) -> float:
    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        cur = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return 100.0 * prev[len(hyp)] / max(1, len(ref))


def read_ref(name: str) -> list[str]:
    lines = [l for l in (SUITE / name).read_text().splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    return words(" ".join(lines))


def main() -> int:
    data = json.load(open(sys.argv[1]))
    chunks = data["chunks"]
    man = json.load(open(SUITE / "manifest.json"))
    tracks = man["tracks"]
    print(f"  {data['seconds']/60:.1f} min captured, {len(chunks)} chunks "
          f"of {data['push_secs']}s\n")

    def bucket(sents):
        out = {t["name"]: [] for t in tracks}
        for text, at in sents:
            for t in tracks:
                if t["start_sec"] <= at < t["start_sec"] + t["duration_sec"]:
                    out[t["name"]].append(text)
                    break
        return out

    old_b, new_b = bucket(replay(chunks, **OLD)), bucket(replay(chunks, **NEW))

    print(f"  {'track':11} {'ref w':>6} | {'OLD wer':>8} {'ends':>6} | "
          f"{'NEW wer':>8} {'ends':>6} {'leads':>6} | {'sents':>11}")
    print("  " + "-" * 78)
    tot_old = tot_new = tot_ref = 0
    for t in tracks:
        ref = read_ref(t["ref"])
        o, n = old_b[t["name"]], new_b[t["name"]]
        if not o or not n:
            print(f"  {t['name']:11} {len(ref):>6} | (no sentences fell in this track)")
            continue
        ow, nw = wer(ref, words(" ".join(o))), wer(ref, words(" ".join(n)))
        oe = 100 * sum(bool(END.search(s)) for s in o) / len(o)
        ne = 100 * sum(bool(END.search(s)) for s in n) / len(n)
        nl = 100 * sum(bool(LEAD.match(s)) for s in n) / len(n)
        print(f"  {t['name']:11} {len(ref):>6} | {ow:>7.2f}% {oe:>5.1f}% | "
              f"{nw:>7.2f}% {ne:>5.1f}% {nl:>5.1f}% | {len(o):>4} -> {len(n):<4}")
        tot_ref += len(ref)
        tot_old += ow * len(ref)
        tot_new += nw * len(ref)
    if tot_ref:
        print("  " + "-" * 78)
        print(f"  {'weighted':11} {tot_ref:>6} | {tot_old/tot_ref:>7.2f}%        | "
              f"{tot_new/tot_ref:>7.2f}%")

    print("\n  === sentence shape, whole suite ===")
    for name, b in (("old", old_b), ("new", new_b)):
        allw = sorted(len(s.split()) for v in b.values() for s in v)
        print(f"    {name}: {len(allw)} sentences, median {statistics.median(allw):.0f}w, "
              f"p90 {allw[int(len(allw)*0.9)]}w, max {max(allw)}w")

    json.dump({k: v for k, v in new_b.items()}, open("/tmp/suite_new_sents.json", "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
