# Live multilingual HLS packager

Takes the ATEM's RTMP and produces a delayed adaptive-bitrate HLS stream
with selectable audio and subtitle languages, for embedding on the church
site. Runs on the Translate PC's **Radeon 890M** while the RTX 3060 keeps
carrying the live translation — different silicon, different driver, no
contention.

**Status: phases 1–2.** Ingest, rendition ladder, delayed publisher and
sliding window, English audio. Subtitles are phase 3; the dub tracks are
phase 4. The manifest shape is already final, so adding them is
configuration rather than surgery.

## Shape

```
ATEM ──RTMP──> ingest (ffmpeg, 890M) ──> staging/ ──> publisher (T-45s) ──> public/ ──> Caddy VM
```

Encoding runs in real time; publishing runs 45 seconds behind it. That gap
is not overhead — it is what makes the translated tracks possible. Placing
sentence *N* in a dub timeline needs `t0` of sentence *N+1* to know the
slot width, and running behind is how you already have it.

Video variants are **video-only**; every language including English lives
in the HLS `AUDIO` group. That way the dub tracks are peers of English
rather than a bolt-on, and variants carry no duplicate audio.

## Running

```bash
# Live, from config/livestream.yaml
./venv/bin/python -m livestream.run

# The same pipeline over a recording — how this is developed, and how a
# regression is reproduced without waiting for a Sunday.
./venv/bin/python -m livestream.run --input-file service.mp4 --encoder x264

# Show the resolved config and the exact ffmpeg command, run nothing
./venv/bin/python -m livestream.run --check
```

## Clock calibration

Before phase 3, the fixed offset between the translator's `t0` wall clock
and the ATEM's media timeline has to be measured once. Everything that
places a subtitle cue or a dub sentence depends on it.

```bash
# With the ingest running and the ATEM sending. Clap once, hard.
./venv/bin/python -m livestream.calibrate
./venv/bin/python -m livestream.calibrate --repeat 5 --write
```

It records from the capture device, reads the same wall-clock window back
out of the ingest's audio, and cross-correlates onset envelopes. It
reports a confidence with every measurement and refuses to write an
offset it does not trust — a good match scores 0.75+, unrelated audio
scores below 0.1.

It is fixed hardware latency, so it holds. Re-run it only if you change
the ATEM's streaming profile or the capture device.

## Tests

```bash
./venv/bin/python tests/test_livestream.py            # unit, instant
./venv/bin/python tests/test_livestream.py --e2e      # + a real encode, ~1 min
./venv/bin/python tests/test_livestream_calibrate.py  # correlation maths
```

The unit tests drive the publisher with a fixed clock, so the delay, the
sliding window and the media sequence are covered without encoding
anything. `--e2e` generates a test pattern, runs the real pipeline and
checks the bundle decodes and that the three video renditions are
segment-aligned.

## Notes that will save you time

**Audio segments do not align to video segments, and must not be forced
to.** An AAC frame is 1024 samples — 21.33 ms at 48 kHz — so a 4.000 s
video segment is 187.5 audio frames. HLS does not require alternate
renditions to share a segment count. The publisher advances each track
independently against wall-clock using `EXT-X-PROGRAM-DATE-TIME`, never by
segment index. Forcing 1:1 is how you get accumulating A/V drift.

**`mp4a.40.2` stays in `CODECS` on the video variants** even though their
audio lives in an alternate group. hls.js needs the audio codec declared
up front to allocate the audio SourceBuffer; without it the MediaSource is
torn down. Multi-Bitrate-Sermons learned this the hard way — see the
`inject_audio_groups` docstring in its `scripts/pipeline/hls.py`.

**A segment with no `PROGRAM-DATE-TIME` stalls that track rather than
being published.** Without it there is no way to place the segment in wall
clock, and a visible stall beats silent drift.

**`staging_dir` and `public_dir` want to be on tmpfs** and on the *same*
filesystem — segments are hardlinked between them, so publishing moves no
bytes. A cross-filesystem layout still works, it just falls back to
copying.

## Configuration

`config/livestream.yaml`, with per-box env overrides:
`LIVESTREAM_ENCODER`, `LIVESTREAM_VAAPI_DEVICE`, `LIVESTREAM_DELAY`,
`LIVESTREAM_INPUT_FILE`. Same convention as `scripts/run_production.sh`.

## Still to do

- Phase 3 — subtitles: extend the relay to carry cues, incremental VTT
  segmenter, `SUBTITLES` group
- Phase 4 — dub audio: per-language timelines, English bed ducked ~18 dB
  under the synthesized voice, duration matching against the next `t0`
- Phase 5 — WordPress live player, standby outside service windows,
  service-window scheduling, alerting, a cgroup cap on the packager so it
  can never starve the translator's audio callbacks
