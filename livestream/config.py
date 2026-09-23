"""Livestream packager configuration.

Loaded from `config/livestream.yaml`, with env-var overrides for the few
knobs an operator changes per box (the encoder, the VAAPI device).  Same
convention as `config/settings.yaml` and `scripts/run_production.sh`:
the file is the source of truth, env is the per-machine escape hatch.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Rendition:
    """One video rendition.  `height` drives the scale filter and the
    codec tag; width is derived from the source aspect at runtime so 4:3
    sources are not advertised as 16:9."""
    name: str
    height: int
    bitrate_kbps: int
    maxrate_kbps: int
    bufsize_kbps: int

    @property
    def dir_name(self) -> str:
        return f"video_{self.height}p"


# Defaults sized for a locked-off camera on a mostly static scene, against
# the 2 Gbps WAN.  150 viewers all on 1080p is ~694 Mbps — 35% of it.
DEFAULT_RENDITIONS = [
    Rendition("1080p", 1080, 4500, 5000, 10000),
    Rendition("720p", 720, 2500, 2800, 5000),
    Rendition("480p", 480, 1200, 1400, 2400),
]


@dataclass(frozen=True)
class AudioTrack:
    """One entry in the HLS AUDIO group.

    Phase 1-2 ships `en` only.  The dub languages are declared here from
    the start so the manifest shape is final and adding them later is
    configuration rather than surgery.
    """
    language: str
    name: str                  # endonym, shown in the player menu
    default: bool = False
    # False until the dub pipeline fills it (phase 4).  A declared-but-
    # inactive track is simply left out of the master.
    active: bool = True

    @property
    def dir_name(self) -> str:
        return f"audio_{self.language}"


# Endonyms, matching Multi-Bitrate-Sermons' _LANG_DISPLAY.
LANG_DISPLAY = {
    "en": "English",
    "es": "Español",
    "ht": "Kreyòl Ayisyen",
    "ru": "Русский",
}


@dataclass
class Config:
    # --- source -------------------------------------------------------
    # "rtmp"  — listen for the ATEM's push
    # "file"  — loop a recording; how you develop and test without a service
    source: str = "rtmp"
    rtmp_url: str = "rtmp://0.0.0.0:1935/live/atem"
    input_file: Optional[str] = None
    loop_input: bool = True          # file source only

    # --- encode -------------------------------------------------------
    # "vaapi" (Radeon 890M), "nvenc" (RTX 3060), "x264" (CPU, quality target)
    encoder: str = "vaapi"
    vaapi_device: str = "/dev/dri/renderD128"
    renditions: list[Rendition] = field(default_factory=lambda: list(DEFAULT_RENDITIONS))
    framerate: Optional[float] = None    # None = follow the source
    audio_bitrate_kbps: int = 128

    # --- timing -------------------------------------------------------
    segment_seconds: float = 4.0
    # How far behind real time the public playlists run.  This is the
    # budget the dub stage spends on duration matching: when placing
    # sentence N we need t0 of sentence N+1, so the delay must exceed the
    # translator's end-to-end latency (~8 s) plus one sentence.  45 s is
    # comfortable; below ~30 s the lookahead gets thin.
    delay_seconds: float = 45.0
    # Segments kept in the live playlist.  Six at 4 s = a 24 s window,
    # which is three segments of player buffer plus headroom.
    window_segments: int = 6

    # --- layout -------------------------------------------------------
    staging_dir: Path = REPO_ROOT / "run" / "staging"
    public_dir: Path = REPO_ROOT / "run" / "public"
    # Keep every segment ever produced, for the post-service VOD bundle.
    # Off by default: the live window is small and tmpfs-friendly.
    archive_segments: bool = False

    # --- tracks -------------------------------------------------------
    audio_tracks: list[AudioTrack] = field(default_factory=lambda: [
        AudioTrack("en", LANG_DISPLAY["en"], default=True, active=True),
        AudioTrack("es", LANG_DISPLAY["es"], active=False),
        AudioTrack("ht", LANG_DISPLAY["ht"], active=False),
        AudioTrack("ru", LANG_DISPLAY["ru"], active=False),
    ])

    def active_audio(self) -> list[AudioTrack]:
        return [t for t in self.audio_tracks if t.active]

    def validate(self) -> None:
        if self.source not in ("rtmp", "file"):
            raise ValueError(f"unknown source {self.source!r} (rtmp|file)")
        if self.source == "file" and not self.input_file:
            raise ValueError("source=file needs input_file")
        if self.encoder not in ("vaapi", "nvenc", "x264"):
            raise ValueError(f"unknown encoder {self.encoder!r} (vaapi|nvenc|x264)")
        if not self.renditions:
            raise ValueError("at least one rendition is required")
        if self.segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive")
        if self.delay_seconds < self.segment_seconds:
            raise ValueError("delay_seconds must exceed one segment")
        if self.window_segments < 3:
            # Apple's guidance: a player needs three segments to start.
            raise ValueError("window_segments must be at least 3")
        active = self.active_audio()
        if not active:
            raise ValueError("at least one audio track must be active")
        if sum(1 for t in active if t.default) != 1:
            raise ValueError("exactly one active audio track must be DEFAULT=YES")


def load(path: Optional[Path] = None) -> Config:
    """Read config/livestream.yaml if present, then apply env overrides.

    A missing file is not an error — the defaults above are a working
    configuration for the reference host.
    """
    cfg = Config()
    path = path or (REPO_ROOT / "config" / "livestream.yaml")

    raw = {}
    if path.is_file():
        try:
            import yaml
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as e:  # noqa: BLE001 — config must never hard-fail here
            raise ValueError(f"could not read {path}: {e}") from e

    for key in (
        "source", "rtmp_url", "input_file", "loop_input", "encoder",
        "vaapi_device", "framerate", "audio_bitrate_kbps",
        "segment_seconds", "delay_seconds", "window_segments",
        "archive_segments",
    ):
        if key in raw and raw[key] is not None:
            setattr(cfg, key, raw[key])

    for key in ("staging_dir", "public_dir"):
        if raw.get(key):
            setattr(cfg, key, Path(str(raw[key])).expanduser())

    if raw.get("renditions"):
        cfg.renditions = [
            Rendition(
                name=r.get("name") or f"{r['height']}p",
                height=int(r["height"]),
                bitrate_kbps=int(r["bitrate_kbps"]),
                maxrate_kbps=int(r.get("maxrate_kbps", r["bitrate_kbps"] * 11 // 10)),
                bufsize_kbps=int(r.get("bufsize_kbps", r["bitrate_kbps"] * 2)),
            )
            for r in raw["renditions"]
        ]

    if raw.get("audio_tracks"):
        cfg.audio_tracks = [
            AudioTrack(
                language=t["language"],
                name=t.get("name") or LANG_DISPLAY.get(t["language"], t["language"].upper()),
                default=bool(t.get("default", False)),
                active=bool(t.get("active", True)),
            )
            for t in raw["audio_tracks"]
        ]

    # Env overrides — per-box, matching LBC_VIDEO_ENCODER in the archive project.
    if os.environ.get("LIVESTREAM_ENCODER"):
        cfg.encoder = os.environ["LIVESTREAM_ENCODER"].strip().lower()
    if os.environ.get("LIVESTREAM_VAAPI_DEVICE"):
        cfg.vaapi_device = os.environ["LIVESTREAM_VAAPI_DEVICE"].strip()
    if os.environ.get("LIVESTREAM_DELAY"):
        cfg.delay_seconds = float(os.environ["LIVESTREAM_DELAY"])
    if os.environ.get("LIVESTREAM_INPUT_FILE"):
        cfg.source = "file"
        cfg.input_file = os.environ["LIVESTREAM_INPUT_FILE"]

    cfg.staging_dir = Path(cfg.staging_dir)
    cfg.public_dir = Path(cfg.public_dir)
    cfg.validate()
    return cfg
