"""Reading ffmpeg's staging playlists and writing the public live ones.

Two directions:

  parse_media_playlist()  — read what the ingest ffmpeg has produced so far
  write_media_playlist()  — write the sliding-window playlist viewers fetch
  write_master()          — write the master, with audio and subtitle groups

On segment alignment
--------------------
The three video renditions share forced keyframes, so their segments are
IDR-aligned and identical in duration — that is what ABR switching needs.
Audio is NOT aligned to them and cannot be: an AAC frame is 1024 samples,
21.33 ms at 48 kHz, so a 4.000 s video segment is 187.5 audio frames.

HLS does not require alternate renditions to share a segment count, and
trying to force it is how you get accumulating A/V drift.  So each
rendition here keeps its own playlist, its own media sequence and its own
durations, and the player syncs on timestamps.  The publisher advances
each one independently against wall-clock, never by segment index.

The EXT-X-MEDIA writers below are deliberately the same shape as
Multi-Bitrate-Sermons' `scripts/pipeline/hls.py`, including keeping
`mp4a.40.2` in CODECS on variants that point at an alternate audio group
(see that file's `inject_audio_groups` docstring — dropping it tears down
the MediaSource in hls.js).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SUBTITLE_GROUP_ID = "subs"
AUDIO_GROUP_ID = "audio"

GENERIC_AUDIO_CODEC_TAG = "mp4a.40.2"


def video_codec_tag(height: int) -> str:
    """H.264 High profile, level by height.  1080p30 needs level 4.0;
    advertising 3.1 (caps ~720p30) is a lie strict validators act on."""
    return "avc1.640028" if height >= 1080 else "avc1.64001f"


@dataclass(frozen=True)
class Segment:
    """One segment as ffmpeg's staging playlist describes it."""
    uri: str
    duration: float
    program_date_time: Optional[datetime]

    def end_time(self) -> Optional[datetime]:
        if self.program_date_time is None:
            return None
        return self.program_date_time + _seconds(self.duration)


def _seconds(x: float):
    from datetime import timedelta
    return timedelta(seconds=x)


@dataclass
class ParsedPlaylist:
    segments: list[Segment]
    init_uri: Optional[str]      # #EXT-X-MAP URI, for fMP4
    target_duration: float


_EXTINF = re.compile(r"^#EXTINF:([\d.]+)")
_PDT = re.compile(r"^#EXT-X-PROGRAM-DATE-TIME:(.+)$")
_MAP = re.compile(r'^#EXT-X-MAP:URI="([^"]+)"')
_TARGET = re.compile(r"^#EXT-X-TARGETDURATION:(\d+)")


def _parse_pdt(raw: str) -> Optional[datetime]:
    raw = raw.strip()
    # ffmpeg writes e.g. 2026-09-21T14:03:12.000+0000 — fromisoformat in
    # 3.11+ handles the Z form and the offset form, but not a bare "+0000"
    # without a colon on every version we might run on.
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    m = re.match(r"^(.*[+-]\d{2})(\d{2})$", raw)
    if m:
        raw = f"{m.group(1)}:{m.group(2)}"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_media_playlist(path: Path) -> ParsedPlaylist:
    """Parse one of ffmpeg's staging playlists.

    Tolerant by design: the ingest process is writing this file while we
    read it, so a truncated tail is normal.  A trailing #EXTINF with no URI
    line yet is simply dropped — it will be there on the next pass.
    """
    segments: list[Segment] = []
    init_uri: Optional[str] = None
    target = 0.0
    pending_dur: Optional[float] = None
    pending_pdt: Optional[datetime] = None

    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ParsedPlaylist([], None, 0.0)

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = _EXTINF.match(line)
            if m:
                pending_dur = float(m.group(1))
                continue
            m = _PDT.match(line)
            if m:
                pending_pdt = _parse_pdt(m.group(1))
                continue
            m = _MAP.match(line)
            if m:
                init_uri = m.group(1)
                continue
            m = _TARGET.match(line)
            if m:
                target = float(m.group(1))
            continue
        # A URI line closes the pending segment.
        if pending_dur is not None:
            segments.append(Segment(line, pending_dur, pending_pdt))
            pending_dur = None
            # ffmpeg emits PDT on every segment with +program_date_time, but
            # if it ever emits only the first, carry it forward by duration.
            if pending_pdt is not None:
                pending_pdt = pending_pdt + _seconds(segments[-1].duration)

    if not target and segments:
        target = max(s.duration for s in segments)
    return ParsedPlaylist(segments, init_uri, target)


def write_media_playlist(
    path: Path,
    segments: list[Segment],
    *,
    media_sequence: int,
    init_uri: Optional[str] = None,
    target_duration: Optional[float] = None,
    ended: bool = False,
) -> None:
    """Write a live (or, with `ended`, a finished) media playlist.

    Written to a temp file and renamed, because viewers poll this path
    continuously and a half-written playlist is a client-side error.
    """
    if target_duration is None:
        target_duration = max((s.duration for s in segments), default=4.0)

    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:7",
        f"#EXT-X-TARGETDURATION:{max(1, math.ceil(target_duration))}",
        f"#EXT-X-MEDIA-SEQUENCE:{media_sequence}",
    ]
    if init_uri:
        lines.append(f'#EXT-X-MAP:URI="{init_uri}"')

    for seg in segments:
        if seg.program_date_time is not None:
            stamp = seg.program_date_time.astimezone(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            )[:-3] + "Z"
            lines.append(f"#EXT-X-PROGRAM-DATE-TIME:{stamp}")
        lines.append(f"#EXTINF:{seg.duration:.6f},")
        lines.append(seg.uri)

    if ended:
        lines.append("#EXT-X-ENDLIST")

    body = "\n".join(lines) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)


def _quote(s: str) -> str:
    return f'"{s}"'


def media_audio_line(track: dict) -> str:
    parts = [
        "TYPE=AUDIO",
        f"GROUP-ID={_quote(AUDIO_GROUP_ID)}",
        f"NAME={_quote(track['name'])}",
    ]
    if track.get("default", False):
        parts.append("DEFAULT=YES")
    parts.append("AUTOSELECT=" + ("YES" if track.get("autoselect", True) else "NO"))
    parts.append(f"LANGUAGE={_quote(track['language'])}")
    parts.append(f"URI={_quote(track['uri'])}")
    return "#EXT-X-MEDIA:" + ",".join(parts)


def media_subtitle_line(track: dict) -> str:
    parts = [
        "TYPE=SUBTITLES",
        f"GROUP-ID={_quote(SUBTITLE_GROUP_ID)}",
        f"NAME={_quote(track['name'])}",
    ]
    if track.get("default", False):
        parts.append("DEFAULT=YES")
    parts.append("AUTOSELECT=" + ("YES" if track.get("autoselect", True) else "NO"))
    parts.append("FORCED=" + ("YES" if track.get("forced", False) else "NO"))
    parts.append(f"LANGUAGE={_quote(track['language'])}")
    parts.append(f"URI={_quote(track['uri'])}")
    return "#EXT-X-MEDIA:" + ",".join(parts)


def write_master(
    path: Path,
    video_variants: list[dict],
    *,
    audio_tracks: list[dict] | None = None,
    subtitle_tracks: list[dict] | None = None,
    frame_rate: Optional[float] = None,
) -> None:
    """Write master.m3u8.

    `video_variants`: {height, width, bandwidth, uri}
    `audio_tracks` / `subtitle_tracks`: see the media line helpers above.

    The variants are video-only — their audio lives in the AUDIO group —
    but CODECS still advertises mp4a.40.2, which is required and which
    hls.js needs up front to allocate the audio SourceBuffer.
    """
    lines = ["#EXTM3U", "#EXT-X-VERSION:7", "#EXT-X-INDEPENDENT-SEGMENTS", ""]

    if audio_tracks:
        lines.extend(media_audio_line(t) for t in audio_tracks)
        lines.append("")
    if subtitle_tracks:
        lines.extend(media_subtitle_line(t) for t in subtitle_tracks)
        lines.append("")

    for v in video_variants:
        codecs = f"{video_codec_tag(v['height'])},{GENERIC_AUDIO_CODEC_TAG}"
        attrs = (
            f"BANDWIDTH={int(v['bandwidth'])},"
            f"RESOLUTION={int(v['width'])}x{int(v['height'])},"
            f"CODECS={_quote(codecs)}"
        )
        if audio_tracks:
            attrs += f",AUDIO={_quote(AUDIO_GROUP_ID)}"
        if subtitle_tracks:
            attrs += f",SUBTITLES={_quote(SUBTITLE_GROUP_ID)}"
        # Suppress phantom in-band CC detection on Roku and AVPlayer.
        attrs += ",CLOSED-CAPTIONS=NONE"
        if frame_rate:
            attrs += f",FRAME-RATE={frame_rate:.3f}"
        lines.append(f"#EXT-X-STREAM-INF:{attrs}")
        lines.append(v["uri"])
        lines.append("")

    body = "\n".join(lines).rstrip() + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)
