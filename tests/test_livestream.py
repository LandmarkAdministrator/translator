"""Tests for the livestream packager (phases 1-2).

    ./venv/bin/python tests/test_livestream.py          # unit only, ~instant
    ./venv/bin/python tests/test_livestream.py --e2e    # + a real ffmpeg run

The unit tests drive the publisher with a fixed clock and a hand-written
staging playlist, so the delay logic, the sliding window and the media
sequence are covered without encoding anything.

--e2e generates a test pattern, runs the real pipeline over it with x264,
and checks the output is a coherent multi-rendition HLS bundle.  It takes
about a minute and needs ffmpeg on PATH.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livestream import m3u8  # noqa: E402
from livestream.config import Config, Rendition, AudioTrack  # noqa: E402
from livestream.ingest import build_command  # noqa: E402
from livestream.publisher import Publisher  # noqa: E402

FAILURES: list[str] = []
T0 = datetime(2026, 9, 21, 10, 0, 0, tzinfo=timezone.utc)


def check(label: str, got, want) -> None:
    if got != want:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


def ok(label: str, cond: bool, detail: str = "") -> None:
    if not cond:
        FAILURES.append(f"{label}{': ' + detail if detail else ''}")


# --- fixtures ---------------------------------------------------------------

def write_staging(dir_: Path, n: int, *, start: datetime = T0,
                  duration: float = 4.0, with_init: bool = True) -> None:
    """Write a staging playlist and its segment files, as ffmpeg would."""
    dir_.mkdir(parents=True, exist_ok=True)
    lines = ["#EXTM3U", "#EXT-X-VERSION:7",
             f"#EXT-X-TARGETDURATION:{int(duration)}",
             "#EXT-X-MEDIA-SEQUENCE:0"]
    if with_init:
        (dir_ / "init.mp4").write_bytes(b"\0" * 64)
        lines.append('#EXT-X-MAP:URI="init.mp4"')
    for i in range(n):
        pdt = start + timedelta(seconds=i * duration)
        name = f"seg_{i:05d}.m4s"
        (dir_ / name).write_bytes(b"\0" * 128)
        stamp = pdt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        lines += [f"#EXT-X-PROGRAM-DATE-TIME:{stamp}",
                  f"#EXTINF:{duration:.6f},", name]
    (dir_ / "playlist.m3u8").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_config(root: Path, **kw) -> Config:
    cfg = Config(
        source="file", input_file="x.mp4", encoder="x264",
        renditions=[Rendition("720p", 720, 2500, 2800, 5000)],
        audio_tracks=[AudioTrack("en", "English", default=True, active=True)],
        staging_dir=root / "staging", public_dir=root / "public",
        segment_seconds=4.0, delay_seconds=45.0, window_segments=6,
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    cfg.validate()
    return cfg


# --- playlist parsing / writing ---------------------------------------------

def test_parse_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "video_720p"
        write_staging(d, 3)
        parsed = m3u8.parse_media_playlist(d / "playlist.m3u8")
        check("segment count", len(parsed.segments), 3)
        check("init uri", parsed.init_uri, "init.mp4")
        check("first pdt", parsed.segments[0].program_date_time, T0)
        check("third pdt", parsed.segments[2].program_date_time,
              T0 + timedelta(seconds=8))
        check("duration", parsed.segments[0].duration, 4.0)


def test_parse_tolerates_truncated_tail() -> None:
    """ffmpeg is writing this file while we read it."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "playlist.m3u8"
        p.write_text(
            "#EXTM3U\n#EXT-X-VERSION:7\n#EXT-X-TARGETDURATION:4\n"
            "#EXT-X-MEDIA-SEQUENCE:0\n"
            "#EXT-X-PROGRAM-DATE-TIME:2026-09-21T10:00:00.000Z\n"
            "#EXTINF:4.000000,\nseg_00000.m4s\n"
            "#EXT-X-PROGRAM-DATE-TIME:2026-09-21T10:00:04.000Z\n"
            "#EXTINF:4.000000,\n",          # <- no URI yet
            encoding="utf-8")
        parsed = m3u8.parse_media_playlist(p)
        check("dropped the incomplete tail", len(parsed.segments), 1)


def test_parse_missing_file() -> None:
    parsed = m3u8.parse_media_playlist(Path("/nonexistent/playlist.m3u8"))
    check("empty on missing", parsed.segments, [])


def test_pdt_offset_format() -> None:
    """ffmpeg can emit +0000 without the colon."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "playlist.m3u8"
        p.write_text(
            "#EXTM3U\n#EXT-X-PROGRAM-DATE-TIME:2026-09-21T10:00:00.000+0000\n"
            "#EXTINF:4.000000,\nseg_00000.m4s\n", encoding="utf-8")
        parsed = m3u8.parse_media_playlist(p)
        check("parsed +0000", parsed.segments[0].program_date_time, T0)


def test_master_shape() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "master.m3u8"
        m3u8.write_master(
            out,
            [{"height": 1080, "width": 1920, "bandwidth": 5128000,
              "uri": "video_1080p/playlist.m3u8"},
             {"height": 720, "width": 1280, "bandwidth": 2928000,
              "uri": "video_720p/playlist.m3u8"}],
            audio_tracks=[
                {"name": "English", "language": "en",
                 "uri": "audio_en/playlist.m3u8", "default": True},
                {"name": "Kreyòl Ayisyen", "language": "ht",
                 "uri": "audio_ht/playlist.m3u8", "default": False},
            ],
            frame_rate=30.0,
        )
        text = out.read_text(encoding="utf-8")
        ok("independent segments", "#EXT-X-INDEPENDENT-SEGMENTS" in text)
        ok("1080 gets level 4.0", 'avc1.640028' in text)
        ok("720 gets level 3.1", 'avc1.64001f' in text)
        # The bug the archive project had to re-learn: alternate audio does
        # not mean you drop the audio codec from CODECS.
        ok("audio codec kept in CODECS", text.count("mp4a.40.2") >= 2, text)
        ok("audio group referenced", 'AUDIO="audio"' in text)
        ok("default audio marked", "DEFAULT=YES" in text)
        ok("endonym preserved", "Kreyòl Ayisyen" in text)
        ok("closed captions suppressed", "CLOSED-CAPTIONS=NONE" in text)
        ok("frame rate advertised", "FRAME-RATE=30.000" in text)
        ok("no ENDLIST in a master", "#EXT-X-ENDLIST" not in text)


# --- the delay --------------------------------------------------------------

def test_nothing_published_before_the_delay() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = make_config(root)
        write_staging(cfg.staging_dir / "video_720p", 5)
        write_staging(cfg.staging_dir / "audio_en", 5)
        pub = Publisher(cfg)
        for t in pub.tracks:
            t.public.mkdir(parents=True, exist_ok=True)
        # 20 s after the first segment: well inside a 45 s delay.
        n = pub.tick(now=T0 + timedelta(seconds=20))
        check("published nothing yet", n, 0)
        ok("no master yet", not (cfg.public_dir / "master.m3u8").exists())


def test_publishes_once_past_the_delay() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = make_config(root)
        write_staging(cfg.staging_dir / "video_720p", 20)
        write_staging(cfg.staging_dir / "audio_en", 20)
        pub = Publisher(cfg)
        for t in pub.tracks:
            t.public.mkdir(parents=True, exist_ok=True)
        # At T0+53 s the deadline is T0+8 s, so segments ending at 4 s and
        # 8 s qualify — two per track, four in total.
        n = pub.tick(now=T0 + timedelta(seconds=53))
        check("published both tracks", n, 4)
        pl = m3u8.parse_media_playlist(
            cfg.public_dir / "video_720p" / "playlist.m3u8")
        check("two segments live", len(pl.segments), 2)
        check("init copied", pl.init_uri, "init.mp4")
        ok("init file present",
           (cfg.public_dir / "video_720p" / "init.mp4").is_file())
        ok("master written", (cfg.public_dir / "master.m3u8").is_file())


def test_sliding_window_and_media_sequence() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = make_config(root, window_segments=3)
        write_staging(cfg.staging_dir / "video_720p", 40)
        write_staging(cfg.staging_dir / "audio_en", 40)
        pub = Publisher(cfg)
        for t in pub.tracks:
            t.public.mkdir(parents=True, exist_ok=True)
        # Far enough ahead that ten segments qualify.
        pub.tick(now=T0 + timedelta(seconds=45 + 40))
        track = pub.tracks[0]
        check("window held at 3", len(track.published), 3)
        check("media sequence advanced", track.media_sequence, 7)
        pl_text = (cfg.public_dir / "video_720p" / "playlist.m3u8").read_text()
        ok("sequence in playlist", "#EXT-X-MEDIA-SEQUENCE:7" in pl_text, pl_text)
        ok("dropped file removed",
           not (cfg.public_dir / "video_720p" / "seg_00000.m4s").exists())
        ok("kept file present",
           (cfg.public_dir / "video_720p" / "seg_00009.m4s").is_file())
        ok("live playlist has no ENDLIST", "#EXT-X-ENDLIST" not in pl_text)


def test_publishing_is_monotonic_across_ticks() -> None:
    """Repeated ticks must not republish or skip."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = make_config(root, window_segments=100)
        write_staging(cfg.staging_dir / "video_720p", 30)
        write_staging(cfg.staging_dir / "audio_en", 30)
        pub = Publisher(cfg)
        for t in pub.tracks:
            t.public.mkdir(parents=True, exist_ok=True)
        seen = 0
        for extra in (50, 60, 60, 70):
            seen += pub.tick(now=T0 + timedelta(seconds=extra))
        track = pub.tracks[0]
        uris = [s.uri for s in track.published]
        check("no duplicates", len(uris), len(set(uris)))
        check("contiguous", uris, [f"seg_{i:05d}.m4s" for i in range(len(uris))])
        # A repeated tick at the same clock must add nothing.
        before = len(track.published)
        pub.tick(now=T0 + timedelta(seconds=70))
        check("idempotent at a fixed clock", len(track.published), before)


def test_missing_pdt_stalls_rather_than_drifts() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = make_config(root)
        d = cfg.staging_dir / "video_720p"
        d.mkdir(parents=True, exist_ok=True)
        (d / "seg_00000.m4s").write_bytes(b"\0" * 32)
        (d / "playlist.m3u8").write_text(
            "#EXTM3U\n#EXT-X-VERSION:7\n#EXTINF:4.000000,\nseg_00000.m4s\n",
            encoding="utf-8")
        write_staging(cfg.staging_dir / "audio_en", 3)
        pub = Publisher(cfg)
        for t in pub.tracks:
            t.public.mkdir(parents=True, exist_ok=True)
        pub.tick(now=T0 + timedelta(seconds=200))
        check("video stalled", len(pub.tracks[0].published), 0)
        ok("audio unaffected", len(pub.tracks[1].published) > 0)


def test_finish_closes_playlists() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = make_config(root)
        write_staging(cfg.staging_dir / "video_720p", 10)
        write_staging(cfg.staging_dir / "audio_en", 10)
        pub = Publisher(cfg)
        for t in pub.tracks:
            t.public.mkdir(parents=True, exist_ok=True)
        pub.tick(now=T0 + timedelta(seconds=60))
        pub.finish()
        text = (cfg.public_dir / "video_720p" / "playlist.m3u8").read_text()
        ok("ENDLIST appended", "#EXT-X-ENDLIST" in text)


# --- config -----------------------------------------------------------------

def test_config_rejects_bad_values() -> None:
    def rejects(label: str, **kw) -> None:
        try:
            with tempfile.TemporaryDirectory() as td:
                make_config(Path(td), **kw)
        except ValueError:
            return
        FAILURES.append(f"config should have rejected {label}")

    rejects("delay shorter than a segment", delay_seconds=1.0)
    rejects("window below 3", window_segments=2)
    rejects("unknown encoder", encoder="quicksync")
    rejects("no active audio",
            audio_tracks=[AudioTrack("en", "English", default=True, active=False)])
    rejects("two defaults", audio_tracks=[
        AudioTrack("en", "English", default=True),
        AudioTrack("es", "Español", default=True),
    ])


def test_ffmpeg_command_shape() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = make_config(root, renditions=[
            Rendition("1080p", 1080, 4500, 5000, 10000),
            Rendition("480p", 480, 1200, 1400, 2400),
        ])
        for enc, expect_codec, expect_scale in (
            ("x264", "libx264", "scale=w=-2:h=1080"),
            ("vaapi", "h264_vaapi", "scale_vaapi=w=-2:h=1080"),
            ("nvenc", "h264_nvenc", "scale_cuda=w=-2:h=1080"),
        ):
            cfg.encoder = enc
            cmd = build_command(cfg)
            joined = " ".join(cmd)
            ok(f"{enc}: codec", expect_codec in joined)
            ok(f"{enc}: scaler", expect_scale in joined)
            ok(f"{enc}: split for 2 renditions", "split=2" in joined)
            ok(f"{enc}: forced keyframes",
               "expr:gte(t,n_forced*4)" in joined)
            ok(f"{enc}: fmp4", "-hls_segment_type fmp4" in joined)
            ok(f"{enc}: program date time",
               "+program_date_time" in joined)
            # Video variants carry no audio; audio is its own rendition.
            check(f"{enc}: -an per video rendition", cmd.count("-an"), 2)
            check(f"{enc}: one audio output", cmd.count("-vn"), 1)
        cfg.encoder = "vaapi"
        ok("vaapi uploads to the GPU",
           "hwupload" in " ".join(build_command(cfg)))


# --- end to end -------------------------------------------------------------

def run_e2e() -> None:
    """Encode a real test pattern and validate the published bundle."""
    if shutil.which("ffmpeg") is None:
        print("  (skipped --e2e: no ffmpeg)")
        return
    root = Path(tempfile.mkdtemp(prefix="livestream-e2e-"))
    src = root / "source.mp4"
    print(f"  generating a 40 s test pattern in {root} ...")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30:duration=40",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=40",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(src)],
        check=True, timeout=300)

    # A short delay so the run finishes quickly; the logic is identical.
    print("  running the packager for 45 s at a 10 s delay ...")
    proc = subprocess.run(
        [sys.executable, "-m", "livestream.run",
         "--input-file", str(src), "--encoder", "x264",
         "--delay", "10", "--duration", "45", "--clean",
         "--staging-dir", str(root / "staging"),
         "--public-dir", str(root / "public")],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True, text=True, timeout=400)
    if proc.returncode != 0:
        FAILURES.append(f"e2e: runner exited {proc.returncode}\n{proc.stderr[-3000:]}")
        return

    public = root / "public"
    master = public / "master.m3u8"
    if not master.is_file():
        FAILURES.append(f"e2e: no master.m3u8\n{proc.stderr[-3000:]}")
        return

    text = master.read_text(encoding="utf-8")
    ok("e2e: three variants", text.count("#EXT-X-STREAM-INF") == 3, text)
    ok("e2e: audio group", 'TYPE=AUDIO' in text)

    # Every rendition must actually have playable segments.
    counts = {}
    for name in ("video_1080p", "video_720p", "video_480p", "audio_en"):
        pl = m3u8.parse_media_playlist(public / name / "playlist.m3u8")
        counts[name] = len(pl.segments)
        ok(f"e2e: {name} has segments", pl.segments != [], "none published")
        ok(f"e2e: {name} has an init segment",
           (public / name / (pl.init_uri or "")).is_file())
        for seg in pl.segments:
            ok(f"e2e: {name}/{seg.uri} exists",
               (public / name / seg.uri).is_file())

    # The three video renditions share forced keyframes, so their segment
    # counts must match exactly.  Audio is deliberately free to differ.
    vids = {k: v for k, v in counts.items() if k.startswith("video_")}
    ok("e2e: video renditions aligned", len(set(vids.values())) == 1, str(vids))

    # And each rendition must decode.
    for name in ("video_1080p", "audio_en"):
        pl = m3u8.parse_media_playlist(public / name / "playlist.m3u8")
        if not pl.segments:
            continue
        blob = root / f"{name}-probe.mp4"
        blob.write_bytes(
            (public / name / pl.init_uri).read_bytes()
            + (public / name / pl.segments[0].uri).read_bytes())
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_name,width,height", "-of", "csv=p=0", str(blob)],
            capture_output=True, text=True, timeout=60)
        ok(f"e2e: {name} decodes", r.returncode == 0 and bool(r.stdout.strip()),
           r.stderr[-500:])
        print(f"    {name}: {counts[name]} segments, probe -> {r.stdout.strip()}")

    shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    e2e = "--e2e" in sys.argv
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    label = f"{len(tests)} unit groups"
    if e2e:
        run_e2e()
        label += " + end-to-end"
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print(f"OK — {label} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
