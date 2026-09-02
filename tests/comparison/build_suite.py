#!/usr/bin/env python3
"""Assemble the diverse-input test suite: one WAV, multiple tracks, each with
a reference transcript and a spoken marker (its opening words) so per-track
windows can be recovered from any translator log.

v1 tracks:
  1. JFK inaugural (continuity with all prior numbers)
  2. LibriSpeech test-other mix — 6 speakers, 3 female / 3 male, ~100 s each
     ("hard" benchmark audio with exact published transcripts)
Slots for a sermon excerpt and a singing segment are added when the user
picks the recordings (see plan Phase 2).

Usage:
  build_suite.py --libri-dir DIR --jfk JFK.mp3 --out OUTDIR
Outputs: OUTDIR/suite_v1.wav (16 kHz mono PCM16), per-track .ref.txt,
manifest.json (track name, marker words, offsets).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from wer_from_log import read_reference  # noqa: E402

SR = 16000
UTT_GAP = 0.7      # between utterances within a speaker
SPK_GAP = 3.0      # between speakers within the LibriSpeech track
TRACK_GAP = 20.0   # between tracks
SPEAKERS = ["1998", "3331", "6070", "1688", "2609", "8188"]  # 3F + 3M
PER_SPEAKER_SECS = 100.0


def ffmpeg_to_pcm(path: Path) -> np.ndarray:
    out = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-ar", str(SR),
         "-ac", "1", "-f", "s16le", "-"],
        capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.int16)


def silence(secs: float) -> np.ndarray:
    return np.zeros(int(secs * SR), dtype=np.int16)


def libri_track(libri_dir: Path):
    audio_parts, ref_parts = [], []
    for spk in SPEAKERS:
        spk_dir = libri_dir / "test-other" / spk
        got = 0.0
        for chap_dir in sorted(spk_dir.iterdir()):
            trans = next(chap_dir.glob("*.trans.txt"), None)
            if trans is None:
                continue
            lines = dict(
                l.split(" ", 1) for l in trans.read_text().splitlines() if " " in l)
            for utt_id in sorted(lines):
                if got >= PER_SPEAKER_SECS:
                    break
                flac = chap_dir / f"{utt_id}.flac"
                if not flac.exists():
                    continue
                pcm = ffmpeg_to_pcm(flac)
                audio_parts += [pcm, silence(UTT_GAP)]
                ref_parts.append(lines[utt_id].strip())
                got += len(pcm) / SR + UTT_GAP
            if got >= PER_SPEAKER_SECS:
                break
        audio_parts.append(silence(SPK_GAP))
    return np.concatenate(audio_parts), " ".join(ref_parts)


def marker_words(ref_text: str, n: int = 6) -> str:
    words = re.sub(r"[^A-Za-z' ]+", " ", ref_text).lower().split()
    return " ".join(words[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libri-dir", required=True, type=Path)
    ap.add_argument("--jfk", required=True, type=Path)
    ap.add_argument("--jfk-ref", required=True, type=Path)
    ap.add_argument("--sermon", type=Path, help="sermon excerpt audio (track 2)")
    ap.add_argument("--sermon-ref", type=Path, help="human-corrected transcript")
    ap.add_argument("--singing", type=Path,
                    help="talk/song/talk segment (stability track, no reference)")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    tracks = []
    parts = []
    cursor = 0.0

    # Track 1 — JFK
    jfk = ffmpeg_to_pcm(a.jfk)
    jfk_ref = " ".join(read_reference(str(a.jfk_ref)).split())
    (a.out / "track1_jfk.ref.txt").write_text(jfk_ref + "\n")
    tracks.append({"name": "jfk", "start_sec": round(cursor, 1),
                   "duration_sec": round(len(jfk) / SR, 1),
                   "marker": marker_words(jfk_ref),
                   "ref": "track1_jfk.ref.txt"})
    parts += [jfk, silence(TRACK_GAP)]
    cursor += len(jfk) / SR + TRACK_GAP

    # Track 2 — sermon excerpt (the domain that decides everything)
    if a.sermon and a.sermon_ref:
        sermon = ffmpeg_to_pcm(a.sermon)
        sermon_ref = " ".join(read_reference(str(a.sermon_ref)).split())
        (a.out / "track_sermon.ref.txt").write_text(sermon_ref + "\n")
        tracks.append({"name": "sermon", "start_sec": round(cursor, 1),
                       "duration_sec": round(len(sermon) / SR, 1),
                       "marker": marker_words(sermon_ref),
                       "ref": "track_sermon.ref.txt"})
        parts += [sermon, silence(TRACK_GAP)]
        cursor += len(sermon) / SR + TRACK_GAP

    # Singing / transition track — stability probe, no reference
    if a.singing:
        sing = ffmpeg_to_pcm(a.singing)
        tracks.append({"name": "singing", "start_sec": round(cursor, 1),
                       "duration_sec": round(len(sing) / SR, 1),
                       "marker": None, "ref": None})
        parts += [sing, silence(TRACK_GAP)]
        cursor += len(sing) / SR + TRACK_GAP

    # Track — LibriSpeech mix
    ls_audio, ls_ref = libri_track(a.libri_dir)
    (a.out / "track2_libri.ref.txt").write_text(ls_ref + "\n")
    tracks.append({"name": "libri-mix", "start_sec": round(cursor, 1),
                   "duration_sec": round(len(ls_audio) / SR, 1),
                   "marker": marker_words(ls_ref),
                   "ref": "track2_libri.ref.txt"})
    parts += [ls_audio, silence(TRACK_GAP)]
    cursor += len(ls_audio) / SR + TRACK_GAP

    suite = np.concatenate(parts)
    with wave.open(str(a.out / "suite_v1.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(suite.tobytes())
    (a.out / "manifest.json").write_text(json.dumps(
        {"sample_rate": SR, "track_gap_sec": TRACK_GAP, "tracks": tracks}, indent=2) + "\n")
    print(f"suite_v1.wav: {len(suite)/SR/60:.1f} min, {len(tracks)} tracks")
    for t in tracks:
        print(f"  {t['name']:<10} start {t['start_sec']:>7.1f}s  "
              f"dur {t['duration_sec']:>6.1f}s  marker: \"{t['marker']}\"")


if __name__ == "__main__":
    main()
