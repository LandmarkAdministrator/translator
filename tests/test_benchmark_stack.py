"""Tests for tests/benchmark_stack.py — the analysis, not the running.

    ./venv/bin/python tests/test_benchmark_stack.py

The harness itself needs models and a GPU. What is tested here is the
maths that produces the headline number: overlap. If that is wrong, the
whole benchmark tells a confident lie, which is worse than no benchmark.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from benchmark_stack import (  # noqa: E402
    analyse_spans, max_concurrency, union_duration,
)

FAILURES: list[str] = []


def check(label: str, got, want, tol: float = 1e-6) -> None:
    same = (abs(got - want) <= tol
            if isinstance(got, (int, float)) and isinstance(want, (int, float))
            else got == want)
    if not same:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


def ok(label: str, cond: bool, detail: str = "") -> None:
    if not cond:
        FAILURES.append(f"{label}{': ' + detail if detail else ''}")


def test_union_basic() -> None:
    check("disjoint", union_duration([(0, 1), (2, 3)]), 2.0)
    check("identical", union_duration([(0, 1), (0, 1)]), 1.0)
    check("partial", union_duration([(0, 2), (1, 3)]), 3.0)
    check("nested", union_duration([(0, 10), (2, 3)]), 10.0)
    check("touching", union_duration([(0, 1), (1, 2)]), 2.0)
    check("empty", union_duration([]), 0.0)
    check("unsorted input", union_duration([(2, 3), (0, 1)]), 2.0)


def test_concurrency() -> None:
    peak, mean = max_concurrency([(0, 10), (0, 10), (0, 10)])
    check("peak of 3 identical", peak, 3)
    check("mean of 3 identical", mean, 3.0)
    peak, mean = max_concurrency([(0, 1), (2, 3), (4, 5)])
    check("peak of 3 serial", peak, 1)
    check("mean of 3 serial", mean, 1.0)
    check("empty", max_concurrency([]), (0, 0.0))


def _spans_file(spans: list[tuple[str, str, float, float]]) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "spans.jsonl"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "meta", "t0_wall": 0}) + "\n")
        for stage, lang, start, end in spans:
            fh.write(json.dumps({
                "kind": "span", "stage": stage, "lang": lang,
                "thread": f"t-{lang}", "start": start, "end": end,
                "dur": round(end - start, 6),
            }) + "\n")
    return tmp


def test_perfectly_parallel_reads_as_3x() -> None:
    """Three languages doing 1 s of work at the same instant."""
    spans = [("translate", lang, 0.0, 1.0) for lang in ("es", "ht", "ru")]
    got = analyse_spans(_spans_file(spans))
    st = got["stages"]["translate"]
    check("parallel overlap", st["overlap"], 3.0)
    check("parallel union", st["union_s"], 1.0)
    check("parallel total", st["total_s"], 3.0)
    check("parallel peak", st["peak_concurrent"], 3)


def test_fully_serial_reads_as_1x() -> None:
    """The same work, one language after another."""
    spans = [("translate", lang, i * 1.0, i * 1.0 + 1.0)
             for i, lang in enumerate(("es", "ht", "ru"))]
    got = analyse_spans(_spans_file(spans))
    st = got["stages"]["translate"]
    check("serial overlap", st["overlap"], 1.0)
    check("serial union", st["union_s"], 3.0)
    check("serial peak", st["peak_concurrent"], 1)


def test_partial_overlap() -> None:
    """Half-overlapping: the realistic case, and the one that must not
    round to either extreme."""
    spans = [("translate", "es", 0.0, 1.0),
             ("translate", "ht", 0.5, 1.5),
             ("translate", "ru", 1.0, 2.0)]
    got = analyse_spans(_spans_file(spans))
    st = got["stages"]["translate"]
    check("partial union", st["union_s"], 2.0)
    check("partial total", st["total_s"], 3.0)
    check("partial overlap", st["overlap"], 1.5)
    check("partial peak", st["peak_concurrent"], 2)


def test_pipeline_combines_translate_and_tts() -> None:
    spans = [("translate", "es", 0.0, 1.0), ("tts", "es", 1.0, 2.0),
             ("translate", "ht", 0.0, 1.0), ("tts", "ht", 1.0, 2.0)]
    got = analyse_spans(_spans_file(spans))
    pl = got["pipeline"]
    check("pipeline work", pl["work_s"], 4.0)
    check("pipeline busy", pl["busy_s"], 2.0)
    check("pipeline overlap", pl["overlap"], 2.0)
    check("pipeline peak", pl["peak_concurrent"], 2)
    check("duty cycle saturated", pl["duty_cycle"], 1.0)


def test_duty_cycle_below_one_when_idle() -> None:
    """A gap between sentences must show up as idle time, or a machine
    with plenty of headroom looks saturated."""
    spans = [("translate", "es", 0.0, 1.0), ("translate", "es", 3.0, 4.0)]
    got = analyse_spans(_spans_file(spans))
    check("duty cycle", got["pipeline"]["duty_cycle"], 0.5)


def test_per_language_means_are_separated() -> None:
    spans = [("tts", "es", 0.0, 0.5), ("tts", "ht", 0.0, 1.5),
             ("tts", "ru", 0.0, 1.0)]
    got = analyse_spans(_spans_file(spans))
    per = got["stages"]["tts"]["per_language_mean_s"]
    check("es mean", per["es"], 0.5)
    check("ht mean", per["ht"], 1.5)
    check("ru mean", per["ru"], 1.0)


def test_queue_depth_trend_detects_falling_behind() -> None:
    tmp = Path(tempfile.mkdtemp()) / "spans.jsonl"
    with open(tmp, "w", encoding="utf-8") as fh:
        for i in range(20):
            fh.write(json.dumps({
                "kind": "span", "stage": "translate", "lang": "es",
                "thread": "t", "start": i, "end": i + 0.5, "dur": 0.5,
            }) + "\n")
            fh.write(json.dumps({
                "kind": "sentence", "t": i, "lang": "es",
                # Depth climbs through the run: the signature of a box
                # that cannot keep up.
                "queue_depth": i // 2, "audio_seconds": 2.0,
                "speed": 1.2 if i > 10 else 1.0,
            }) + "\n")
    got = analyse_spans(tmp)
    sd = got["sentences_detail"]
    check("count", sd["count"], 20)
    ok = sd["queue_depth_second_half"] > sd["queue_depth_first_half"] + 1
    if not ok:
        FAILURES.append(f"rising queue not detected: {sd}")
    if not sd["sped_up_fraction"] > 0.3:
        FAILURES.append(f"speed-ups not counted: {sd['sped_up_fraction']}")


def test_missing_and_malformed_file() -> None:
    tmp = Path(tempfile.mkdtemp()) / "spans.jsonl"
    tmp.write_text("not json\n{\"kind\":\"span\"\n", encoding="utf-8")
    got = analyse_spans(tmp)
    check("malformed lines skipped", got["spans"], 0)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print(f"OK — {len(tests)} test groups passed")
    return 0



# --- track extraction (appended 2026-09-22) ---------------------------------

def test_track_extraction_cuts_and_caches() -> None:
    """The suite is 56 minutes; a comparison uses one track. Every run must
    get byte-identical audio, so the cut is cached, not redone."""
    import shutil as _sh
    import subprocess as _sp
    import benchmark_stack as bs

    if not _sh.which("ffmpeg"):
        print("  (skipped track extraction: no ffmpeg)")
        return

    root = Path(tempfile.mkdtemp())
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"sample_rate": 16000, "tracks": [
        {"name": "sermon", "start_sec": 10.0, "duration_sec": 5.0},
        {"name": "singing", "start_sec": 20.0, "duration_sec": 3.0},
    ]}), encoding="utf-8")

    src = root / "suite.wav"
    _sp.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=30",
             "-ar", "16000", "-ac", "1", str(src)], check=True, timeout=120)

    old = bs.SUITE_MANIFEST
    bs.SUITE_MANIFEST = manifest
    try:
        tracks = bs.load_tracks()
        check("manifest parsed", sorted(tracks), ["sermon", "singing"])

        out = root / "out"
        cut = bs.extract_track(src, "sermon", out)
        ok("cut exists", cut.is_file())

        dur = float(_sp.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(cut)],
            capture_output=True, text=True, timeout=60).stdout.strip())
        ok("cut is the right length", abs(dur - 5.0) < 0.3, f"got {dur:.2f}s")

        # Cached: the second call must return the same file untouched.
        mtime = cut.stat().st_mtime_ns
        again = bs.extract_track(src, "sermon", out)
        check("same path", again, cut)
        check("not re-cut", again.stat().st_mtime_ns, mtime)

        # 'all' passes the original through.
        check("all passes through", bs.extract_track(src, "all", out), src)

        try:
            bs.extract_track(src, "nope", out)
            FAILURES.append("unknown track should have raised")
        except SystemExit:
            pass
    finally:
        bs.SUITE_MANIFEST = old
        _sh.rmtree(root, ignore_errors=True)

if __name__ == "__main__":
    raise SystemExit(main())
