#!/usr/bin/env python3
"""Per-track scoring for diverse-suite runs, with wrap-around reconstruction.

Each run is longer than one suite pass, so the loop wraps: the track that was
playing when the run began is split — its later portion appears at the start of
the run, its beginning re-appears at the end. This scorer finds every window
for a track, works out which fragment holds the track's opening, and stitches
the fragments back together on their overlapping text before scoring, so the
whole played body is measured once.

Windows are anchored on the first track's spoken marker; virtual anchors one
period back/forward let runs that start mid-suite score every track.

Usage: score_suite.py MANIFEST_DIR LOG --type old|new [--slack 12] [--verbose]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from wer_from_log import normalize, wer  # noqa: E402

OLD_RECOG = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - Chunk \d+ Recognized: (.*)$")
NEW_SENT = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+ \| \S+\s*\| [^|]+ \| \[EN\] (.*?) \| mode=streaming/sentence")

TS = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)"
OLD_REC_T = re.compile(r"^\S+ \S+ - Recording chunk (\d+) started: " + TS
                       + r", ended: " + TS + r" \(~([\d.]+)s\)")
OLD_PLAY_T = re.compile(r"^\S+ \S+ - Chunk (\d+) Playback started: " + TS)
NEW_EVT_T = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+ \| \S+\s*\| [^|]+ \| "
                       r"\[ES\] .*? \| e2e=([\d.]+)s")

MIN_OVERLAP = 8       # words that must line up to accept a stitch
MAX_OVERLAP = 600     # cap the search window
MATCH_RATIO = 0.65    # fraction of positions that must agree (ASR differs)


def lines_of(log: Path, kind: str):
    pat = OLD_RECOG if kind == "old" else NEW_SENT
    for ln in open(log, errors="replace"):
        m = pat.match(ln)
        if m:
            text = m.group(2)
            if text.startswith("[silence"):
                continue
            yield datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), text


def latency_events(log: Path, kind: str) -> list[tuple[datetime, float]]:
    """(speech_start_time, speech_start -> audio_start seconds) per utterance.

    Keyed by WHEN THE SPEECH HAPPENED, not when its audio played: with a large
    latency those land in different tracks, and bucketing by playback time
    charges one track's backlog to the next one.
    """
    out: list[tuple[datetime, float]] = []
    if kind == "new":
        for ln in open(log, errors="replace"):
            m = NEW_EVT_T.match(ln)
            if m:
                play = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                e2e = float(m.group(2))
                out.append((play - timedelta(seconds=e2e), e2e))
        return out
    rec: dict[str, tuple[datetime, float]] = {}
    for ln in open(log, errors="replace"):
        m = OLD_REC_T.match(ln)
        if m:
            rec[m.group(1)] = (datetime.strptime(m.group(3), "%Y-%m-%d %H:%M:%S.%f"),
                               float(m.group(4)))
            continue
        m = OLD_PLAY_T.match(ln)
        if m and m.group(1) in rec:
            end_t, dur = rec[m.group(1)]
            play = datetime.strptime(m.group(2), "%Y-%m-%d %H:%M:%S.%f")
            speech_start = end_t - timedelta(seconds=dur)
            out.append((speech_start, (play - speech_start).total_seconds()))
    return out


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def marker_hit(marker_words: list[str], line_words: list[str]) -> bool:
    if len(marker_words) < 4:
        return False
    joined = " ".join(line_words)
    return any(" ".join(marker_words[i:i + 4]) in joined
               for i in range(len(marker_words) - 3))


def find_join(a: list[str], b: list[str], probe: int = 40) -> int | None:
    """Index i in a where b's content starts, so a[:i] + b reconstructs it.

    b may begin anywhere inside a, not only at a's tail: a run can capture the
    end of a track and then, after the loop wraps, more of that track's
    beginning than the first fragment was missing. Matching only tail-to-head
    then finds nothing and the fragments get concatenated, duplicating text.
    """
    probe_words = b[:probe]
    if len(probe_words) < MIN_OVERLAP:
        return None
    best_i, best_score = None, 0.0
    for i in range(len(a)):
        seg = a[i:i + len(probe_words)]
        if len(seg) < MIN_OVERLAP:
            break
        same = sum(1 for x, y in zip(seg, probe_words) if x == y)
        score = same / len(probe_words)
        if score > best_score:
            best_score, best_i = score, i
    return best_i if best_score >= MATCH_RATIO else None


def head_similarity(words: list[str], ref: list[str], n: int = 25) -> float:
    """How well a fragment's opening matches the reference's opening."""
    if not words:
        return -1.0
    probe, target = words[:n], ref[:n]
    best = 0
    for start in range(0, min(len(probe), n)):
        same = sum(1 for x, y in zip(probe[start:], target) if x == y)
        best = max(best, same)
    return best / max(len(target), 1)


def stitch(fragments: list[list[str]]) -> tuple[list[str], list[str]]:
    """Join fragments in order, merging on overlap. Returns (words, notes)."""
    merged, notes = list(fragments[0]), []
    for i, frag in enumerate(fragments[1:], 2):
        j = find_join(merged, frag)
        if j is not None:
            overlap = len(merged) - j
            merged = merged[:j] + frag
            notes.append(f"joined fragment {i} at word {j} "
                         f"({overlap}-word overlap absorbed)")
            continue
        # The other fragment may instead start earlier — e.g. a wrap fragment
        # that begins a few words into the track, so the first fragment's
        # opening cannot be found inside it. Try the reverse direction.
        j = find_join(frag, merged)
        if j is not None:
            overlap = len(frag) - j
            merged = frag[:j] + merged
            notes.append(f"fragment {i} prepended at word {j} "
                         f"({overlap}-word overlap absorbed)")
            continue
        # No alignment either way: concatenating would fabricate duplicates,
        # so keep the longer fragment and say so.
        if len(frag) > len(merged):
            merged = list(frag)
        notes.append(f"fragment {i} did NOT align — kept the longer fragment "
                     f"only (no concatenation)")
    return merged, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest_dir", type=Path)
    ap.add_argument("log", type=Path)
    ap.add_argument("--type", choices=["old", "new"], required=True)
    ap.add_argument("--slack", type=float, default=12.0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    manifest = json.loads((a.manifest_dir / "manifest.json").read_text())
    tracks = manifest["tracks"]
    anchor_marker = normalize(tracks[0]["marker"])

    entries = list(lines_of(a.log, a.type))
    if not entries:
        sys.exit("no recognized text in this log")
    lat_events = latency_events(a.log, a.type)

    anchors, last = [], None
    for t, text in entries:
        if marker_hit(anchor_marker, normalize(text)):
            if last is None or (t - last).total_seconds() > 120:
                anchors.append(t)
            last = t
    if not anchors:
        sys.exit("no suite passes found (first track's marker never seen)")

    period = max(t["start_sec"] + t["duration_sec"] for t in tracks) \
        + manifest.get("track_gap_sec", 20)
    virtual = []
    for anchor in anchors:
        for k in (-1, 0, 1):
            va = anchor + timedelta(seconds=k * period)
            if not any(abs((va - v).total_seconds()) < 60 for v in virtual):
                virtual.append(va)
    virtual.sort()
    t_lo, t_hi = entries[0][0], entries[-1][0]
    used = [v for v in virtual if t_lo - timedelta(seconds=period) <= v <= t_hi]
    print(f"run {t_lo:%H:%M:%S}–{t_hi:%H:%M:%S} · markers at "
          f"{[x.strftime('%H:%M:%S') for x in anchors]} · windows from "
          f"{[x.strftime('%H:%M:%S') for x in used]}")

    for track in tracks:
        name = track["name"]
        ref_words = normalize((a.manifest_dir / track["ref"]).read_text()) \
            if track.get("ref") else None

        frags = []  # (window_start, words, n_lines)
        lats: list[float] = []
        for anchor in used:
            w0 = anchor + timedelta(seconds=track["start_sec"] - a.slack)
            w1 = anchor + timedelta(seconds=track["start_sec"]
                                    + track["duration_sec"] + a.slack)
            words, n_lines = [], 0
            for t, text in entries:
                if w0 <= t <= w1:
                    words += normalize(text)
                    n_lines += 1
            lats += [v for t, v in lat_events if w0 <= t <= w1]
            if words:
                frags.append((w0, words, n_lines))

        if not frags:
            print(f"  {name:<10} no output in any window")
            continue

        notes = []
        if len(frags) == 1:
            words, n_lines = frags[0][1], frags[0][2]
        else:
            # Order fragments so the one holding the track's opening comes
            # first, then the rest in run order (circular), and stitch.
            if ref_words:
                scores = [head_similarity(f[1], ref_words) for f in frags]
                head_i = max(range(len(frags)), key=lambda i: scores[i])
            else:
                head_i = 0
            order = frags[head_i:] + frags[:head_i]
            words, notes = stitch([f[1] for f in order])
            n_lines = sum(f[2] for f in frags)
            notes.insert(0, f"{len(frags)} fragments (wrap-around reconstructed)")

        lat = ""
        if lats:
            lat = (f" | latency speech-start→audio: med {pct(lats, .5):4.1f}s "
                   f"p95 {pct(lats, .95):5.1f}s (n={len(lats)})")
        if ref_words:
            w, s, d, i = wer(ref_words, words)
            print(f"  {name:<10} WER {w * 100:5.2f}%  (sub {s} del {d} ins {i}; "
                  f"{n_lines} lines, {len(words)}/{len(ref_words)} words){lat}")
        else:
            print(f"  {name:<10} {n_lines} lines, {len(words)} words "
                  f"(stability probe — inspect text manually){lat}")
        for note in notes:
            print(f"             · {note}")
        if a.verbose and ref_words:
            print(f"             · hyp head: {' '.join(words[:12])}")


if __name__ == "__main__":
    main()
