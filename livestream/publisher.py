"""The publishing half: move staging segments into public/ on a delay.

One `Track` per rendition — three video, one audio today, the dub and
subtitle tracks later.  Each advances independently against wall-clock
using EXT-X-PROGRAM-DATE-TIME, never by segment index, because audio and
video segment boundaries genuinely differ (see m3u8.py).

The rule the whole design rests on: a segment is published only once its
wall-clock end is older than `delay_seconds`.  That gap is what lets the
dub stage in phase 4 know the next sentence's t0 before it has to place
the current one.

Segments are hardlinked rather than copied.  Staging and public live on
the same filesystem, so this is a directory entry — no bytes move, and
pruning either side is independent of the other.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from . import m3u8
from .config import Config

log = logging.getLogger("livestream.publisher")


@dataclass
class Track:
    """One rendition's journey from staging to public."""
    name: str                       # directory name, e.g. "video_1080p"
    kind: str                       # "video" | "audio" | "subtitles"
    staging: Path
    public: Path
    language: Optional[str] = None

    media_sequence: int = 0
    published: list[m3u8.Segment] = field(default_factory=list)
    init_copied: bool = False
    init_uri: Optional[str] = None
    # How far into the staging playlist we have consumed.
    cursor: int = 0

    def playlist_path(self) -> Path:
        return self.public / "playlist.m3u8"


class Publisher:
    """Runs the delayed publish loop across every track."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tracks: list[Track] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._master_written = False
        self._build_tracks()

    def _build_tracks(self) -> None:
        cfg = self.cfg
        for r in cfg.renditions:
            self.tracks.append(Track(
                name=r.dir_name, kind="video",
                staging=cfg.staging_dir / r.dir_name,
                public=cfg.public_dir / r.dir_name,
            ))
        for t in cfg.active_audio():
            self.tracks.append(Track(
                name=t.dir_name, kind="audio", language=t.language,
                staging=cfg.staging_dir / t.dir_name,
                public=cfg.public_dir / t.dir_name,
            ))

    # -- lifecycle ------------------------------------------------------

    def start(self) -> "Publisher":
        self.cfg.public_dir.mkdir(parents=True, exist_ok=True)
        for t in self.tracks:
            t.public.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="publisher", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def finish(self) -> None:
        """Append EXT-X-ENDLIST everywhere.

        The live bundle then *is* a valid VOD bundle in the format the
        lbc-sermons plugin already plays — a watchable recording seconds
        after the closing prayer, while the archive pipeline does its
        proper x264 two-pass encode from source in its own time.
        """
        for t in self.tracks:
            m3u8.write_media_playlist(
                t.playlist_path(), t.published,
                media_sequence=t.media_sequence,
                init_uri=t.init_uri,
                target_duration=self.cfg.segment_seconds,
                ended=True,
            )
        log.info("playlists closed with ENDLIST")

    def _run(self) -> None:
        # Poll at a fraction of the segment duration: fast enough that a
        # segment is never more than a moment late, cheap enough to ignore.
        interval = max(0.25, self.cfg.segment_seconds / 4.0)
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 — a live service never dies on one bad pass
                log.exception("publish pass failed; continuing")
            self._stop.wait(interval)

    # -- the work -------------------------------------------------------

    def tick(self, now: Optional[datetime] = None) -> int:
        """One publish pass.  Returns how many segments were published.

        Separated from the loop so tests can drive it with a fixed clock.
        """
        now = now or datetime.now(timezone.utc)
        deadline = now - timedelta(seconds=self.cfg.delay_seconds)
        total = 0
        for track in self.tracks:
            total += self._advance(track, deadline)
        if total and not self._master_written:
            self._write_master()
        return total

    def _advance(self, track: Track, deadline: datetime) -> int:
        parsed = m3u8.parse_media_playlist(track.staging / "playlist.m3u8")
        if not parsed.segments:
            return 0

        if parsed.init_uri and not track.init_copied:
            src = track.staging / parsed.init_uri
            if src.is_file():
                shutil.copy2(src, track.public / parsed.init_uri)
                track.init_uri = parsed.init_uri
                track.init_copied = True
            else:
                # Init segment not flushed yet; nothing can be published
                # without it, so wait for the next pass.
                return 0

        published = 0
        while track.cursor < len(parsed.segments):
            seg = parsed.segments[track.cursor]
            end = seg.end_time()
            if end is None:
                # No PDT means we cannot place this segment in wall-clock.
                # Refusing to publish is the safe failure: the stream
                # stalls visibly rather than drifting silently.
                log.error("%s: segment %s has no PROGRAM-DATE-TIME; stalling",
                          track.name, seg.uri)
                break
            if end > deadline:
                break
            if not self._link_segment(track, seg):
                break
            track.published.append(seg)
            track.cursor += 1
            published += 1

        if not published:
            return 0

        self._trim(track)
        m3u8.write_media_playlist(
            track.playlist_path(), track.published,
            media_sequence=track.media_sequence,
            init_uri=track.init_uri,
            target_duration=max(self.cfg.segment_seconds, parsed.target_duration),
        )
        return published

    def _link_segment(self, track: Track, seg: m3u8.Segment) -> bool:
        src = track.staging / seg.uri
        dst = track.public / seg.uri
        if dst.exists():
            return True
        if not src.is_file():
            # ffmpeg listed it but has not renamed it into place yet.
            return False
        try:
            os.link(src, dst)
        except OSError:
            # Different filesystems, or a hardlink-hostile mount — fall
            # back to a copy rather than failing the whole pass.
            try:
                shutil.copy2(src, dst)
            except OSError as e:
                log.error("%s: could not publish %s: %s", track.name, seg.uri, e)
                return False
        return True

    def _trim(self, track: Track) -> None:
        """Slide the window: drop old entries and, unless archiving, their
        files.  Media sequence advances by exactly what was dropped."""
        excess = len(track.published) - self.cfg.window_segments
        if excess <= 0:
            return
        dropped, track.published = track.published[:excess], track.published[excess:]
        track.media_sequence += len(dropped)
        if self.cfg.archive_segments:
            return
        for seg in dropped:
            try:
                (track.public / seg.uri).unlink(missing_ok=True)
            except OSError:
                pass
            try:
                (track.staging / seg.uri).unlink(missing_ok=True)
            except OSError:
                pass

    def _write_master(self) -> None:
        """Written once the first segments exist, so the master never
        points at a playlist that is not there yet."""
        cfg = self.cfg
        video = []
        for r, track in zip(cfg.renditions, self.tracks[:len(cfg.renditions)]):
            width = self._width_for(track, r.height)
            audio_kbps = cfg.audio_bitrate_kbps
            video.append({
                "height": r.height,
                "width": width,
                "bandwidth": (r.maxrate_kbps + audio_kbps) * 1000,
                "uri": f"{r.dir_name}/playlist.m3u8",
            })

        audio = [
            {
                "name": t.name,
                "language": t.language,
                "uri": f"{t.dir_name}/playlist.m3u8",
                "default": t.default,
                "autoselect": True,
            }
            for t in cfg.active_audio()
        ]

        m3u8.write_master(
            cfg.public_dir / "master.m3u8", video,
            audio_tracks=audio,
            frame_rate=cfg.framerate,
        )
        self._master_written = True
        log.info("master.m3u8 written: %d video, %d audio", len(video), len(audio))

    def _width_for(self, track: Track, height: int) -> int:
        """Derive the advertised width from a real segment rather than
        assuming 16:9 — the ATEM is 16:9, but this is cheap insurance and
        the archive project was bitten by exactly this with 4:3 sources."""
        width = int(round(height * 16 / 9))
        seg = track.published[-1] if track.published else None
        if seg is not None:
            probed = _probe_width(track.public / seg.uri, track.public / (track.init_uri or ""))
            if probed:
                width = probed
        return width - (width % 2)


def _probe_width(segment: Path, init: Path) -> Optional[int]:
    """ffprobe a fragment for its coded width.  Best-effort: a failure
    just means we fall back to the 16:9 assumption."""
    import subprocess
    if not segment.is_file():
        return None
    target = segment
    # A bare .m4s has no moov; concatenate the init in front to probe it.
    tmp = None
    try:
        if init.is_file():
            tmp = segment.with_suffix(".probe.mp4")
            tmp.write_bytes(init.read_bytes() + segment.read_bytes())
            target = tmp
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width", "-of", "csv=p=0", str(target)],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        return int(out.splitlines()[0]) if out else None
    except Exception:  # noqa: BLE001
        return None
    finally:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
