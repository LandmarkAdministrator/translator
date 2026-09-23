"""Live multilingual HLS packager.

Takes the ATEM's RTMP, produces a delayed adaptive-bitrate HLS stream with
selectable audio and subtitle languages, and hands the segments to an
origin.  Designed to run on the Translate PC's Radeon 890M while the RTX
3060 carries the live translation.

Phases 1-2 (here): ingest, rendition ladder, delayed publisher, sliding
window, English audio.  Phase 3 adds subtitles, phase 4 the dub tracks.
"""
