"""The ingest half: one ffmpeg turning the ATEM's RTMP into staging segments.

It produces four independent HLS outputs into `staging/`:

    video_1080p/  video_720p/  video_480p/   video-only
    audio_en/                                AAC, no video

Video-only variants plus a separate audio rendition is the target manifest
shape — the AUDIO group is where every language lives, English included.
It also means the dub tracks added in phase 4 are peers of English rather
than a bolt-on, and the variants carry no wasted duplicate audio.

Nothing here publishes.  The publisher reads these directories on a delay.

Keyframes
---------
All three video outputs get `-force_key_frames expr:gte(t,n_forced*N)` and
a matching `-g`, so every rendition cuts at exactly the same source frames
and a player can switch variants at any segment boundary.  Audio is not
aligned to them and does not need to be (see m3u8.py).
"""
from __future__ import annotations

import logging
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from .config import Config, Rendition

log = logging.getLogger("livestream.ingest")

# ffmpeg restarts: the ATEM dropping its connection is routine (someone
# power-cycles the switcher mid-week), so the supervisor retries forever
# with a bounded backoff rather than exiting.
RESTART_MIN, RESTART_MAX = 1.0, 15.0


def _scale_filter(encoder: str, height: int) -> str:
    """Per-encoder scaler.  VAAPI and NVENC scale on the GPU, so frames
    never leave device memory between decode and encode."""
    if encoder == "vaapi":
        return f"scale_vaapi=w=-2:h={height}"
    if encoder == "nvenc":
        return f"scale_cuda=w=-2:h={height}"
    return f"scale=w=-2:h={height}:flags=bicubic"


def _video_codec_args(cfg: Config, r: Rendition) -> list[str]:
    common = [
        "-b:v", f"{r.bitrate_kbps}k",
        "-maxrate", f"{r.maxrate_kbps}k",
        "-bufsize", f"{r.bufsize_kbps}k",
    ]
    if cfg.encoder == "vaapi":
        # VCN's H.264 encoder has no B-frames; quality at these bitrates on
        # a static scene is fine.  `-rc_mode CBR` keeps segment sizes even,
        # which matters more for a live sliding window than peak quality.
        return ["-c:v", "h264_vaapi", "-profile:v", "high", "-rc_mode", "CBR", *common]
    if cfg.encoder == "nvenc":
        return [
            "-c:v", "h264_nvenc", "-preset", "p4", "-tune", "ll",
            "-profile:v", "high", "-rc", "cbr", *common,
        ]
    return [
        "-c:v", "libx264", "-preset", "veryfast", "-profile:v", "high",
        "-sc_threshold", "0", *common,
    ]


def _hls_output_args(out_dir: Path, seg_seconds: float, prefix: str) -> list[str]:
    """HLS muxer options shared by every output.

    fMP4 rather than TS: exact baseMediaDecodeTime keeps audio frame jitter
    per-fragment instead of letting it accumulate into A/V drift across a
    90-minute service, and it leaves LL-HLS open later.

    `-hls_list_size 0` keeps the full history in the staging playlist; the
    publisher needs to see segments the live window has already passed, and
    prunes staging itself.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        "-f", "hls",
        "-hls_time", f"{seg_seconds:g}",
        "-hls_list_size", "0",
        "-hls_segment_type", "fmp4",
        "-hls_fmp4_init_filename", "init.mp4",
        "-hls_segment_filename", str(out_dir / f"{prefix}_%05d.m4s"),
        "-hls_flags", "+program_date_time+independent_segments+temp_file",
        str(out_dir / "playlist.m3u8"),
    ]


def build_command(cfg: Config) -> list[str]:
    """Assemble the full ffmpeg argv.  Pure — unit-testable without running
    anything, which is how the encoder variants are covered."""
    fps = cfg.framerate or 30.0
    gop = max(1, int(round(fps * cfg.segment_seconds)))
    keyframe_expr = f"expr:gte(t,n_forced*{cfg.segment_seconds:g})"

    cmd: list[str] = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "warning"]

    # --- input --------------------------------------------------------
    if cfg.encoder == "nvenc":
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    elif cfg.encoder == "vaapi":
        cmd += ["-vaapi_device", cfg.vaapi_device]

    if cfg.source == "rtmp":
        # -listen 1 makes ffmpeg the RTMP server, so there is no separate
        # media server to run, supervise or keep patched.
        cmd += ["-listen", "1", "-timeout", "0", "-i", cfg.rtmp_url]
    else:
        if cfg.loop_input:
            cmd += ["-stream_loop", "-1"]
        # -re paces a file at real time, which is what makes a recording a
        # faithful stand-in for the live feed.
        cmd += ["-re", "-i", str(cfg.input_file)]

    # --- filter graph -------------------------------------------------
    n = len(cfg.renditions)
    labels = [f"v{i}" for i in range(n)]
    if cfg.encoder == "vaapi":
        head = "[0:v]format=nv12,hwupload," + f"split={n}"
    else:
        head = f"[0:v]split={n}"
    chain = head + "".join(f"[s{i}]" for i in range(n))
    for i, r in enumerate(cfg.renditions):
        chain += f";[s{i}]{_scale_filter(cfg.encoder, r.height)}[{labels[i]}]"
    cmd += ["-filter_complex", chain]

    # --- one output per video rendition -------------------------------
    for label, r in zip(labels, cfg.renditions):
        cmd += [
            "-map", f"[{label}]", "-an",
            *_video_codec_args(cfg, r),
            "-g", str(gop), "-keyint_min", str(gop),
            "-force_key_frames", keyframe_expr,
            *_hls_output_args(
                cfg.staging_dir / r.dir_name, cfg.segment_seconds, "seg"
            ),
        ]

    # --- the English audio rendition ----------------------------------
    en_dir = cfg.staging_dir / "audio_en"
    cmd += [
        "-map", "0:a:0", "-vn",
        "-c:a", "aac", "-b:a", f"{cfg.audio_bitrate_kbps}k",
        "-ac", "2", "-ar", "48000",
        *_hls_output_args(en_dir, cfg.segment_seconds, "seg"),
    ]
    return cmd


class Ingest:
    """Supervises the ingest ffmpeg: starts it, restarts it, stops it cleanly."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._proc: Optional[subprocess.Popen] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "Ingest":
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is not on PATH")
        self._thread = threading.Thread(target=self._run, name="ingest", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        proc = self._proc
        if proc and proc.poll() is None:
            # SIGINT lets ffmpeg finalize the segment it is writing.
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                log.warning("ffmpeg did not exit on SIGINT; killing")
                proc.kill()
        if self._thread:
            self._thread.join(timeout=timeout)

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _run(self) -> None:
        backoff = RESTART_MIN
        while not self._stop.is_set():
            cmd = build_command(self.cfg)
            log.info("starting ingest: %s", " ".join(cmd))
            started = time.monotonic()
            try:
                self._proc = subprocess.Popen(
                    cmd, stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in self._proc.stdout:  # type: ignore[union-attr]
                    line = line.rstrip()
                    if line:
                        log.warning("[ffmpeg] %s", line)
                self._proc.wait()
            except Exception as e:  # noqa: BLE001
                log.error("ingest failed to start: %s", e)

            if self._stop.is_set():
                return
            # A process that ran for a while then died is a dropped feed;
            # one that dies instantly is a config error, so back off harder.
            if time.monotonic() - started > 30:
                backoff = RESTART_MIN
            log.warning("ingest exited; restarting in %.1fs", backoff)
            self._stop.wait(backoff)
            backoff = min(backoff * 2, RESTART_MAX)
