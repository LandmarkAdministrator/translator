"""End-to-end benchmark of the live translation stack.

Answers three questions with measurements instead of arithmetic:

  1. How much does each additional language actually cost?
     Run with 1 language and with 3; compare. Because each language has
     its own thread, queue, translator and TTS instance, they overlap —
     so three languages is NOT three times one, and the point of this
     harness is to find out what it actually is.

  2. Can this machine keep up in real time?
     In realtime mode the queue depth and the end-to-end latency either
     stay flat or they grow. Growth means the box is over-subscribed, and
     no amount of average-latency reporting hides it.

  3. How does the 890M compare with the RTX 3060?
     Same recording, same languages, same script, both machines.

Usage
-----
    # The comparison set, on whichever machine you are on:
    ./venv/bin/python tests/benchmark_stack.py --audio sermon.wav --suite compare

    # One configuration:
    ./venv/bin/python tests/benchmark_stack.py --audio sermon.wav \\
        --languages es ht ru --mode realtime --label "3060-3lang"

    # Compare two finished runs:
    ./venv/bin/python tests/benchmark_stack.py --report out/*.json

Modes
-----
  realtime    Feeds the file at wall-clock speed, as a service would.
              Measures whether the machine KEEPS UP. Watch queue_depth.
  throughput  Uses run.py --no-realtime to feed as fast as possible.
              Measures the raw ceiling, independent of pacing.

Run both. Realtime tells you if it works; throughput tells you by how
much, and how much headroom a fourth language would need.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent

# The mixed-content suite built by tests/comparison/build_suite.py: JFK,
# a sermon, Steve Jobs, congregational singing, and a LibriSpeech mix,
# separated by 20 s gaps. 56 minutes in total.
DEFAULT_AUDIO = REPO / "tests" / "ab_test" / "audio" / "suite_v4.wav"
SUITE_MANIFEST = REPO / "tests" / "comparison" / "suite" / "manifest.json"

# Four configurations over the whole 56-minute file is about four hours in
# realtime mode. The sermon track alone is the representative workload and
# makes a full comparison an hour, so it is the default.
DEFAULT_TRACK = "sermon"


def load_tracks() -> dict:
    """Track name -> (start_sec, duration_sec) from the suite manifest."""
    if not SUITE_MANIFEST.is_file():
        return {}
    try:
        data = json.loads(SUITE_MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {t["name"]: (float(t["start_sec"]), float(t["duration_sec"]))
            for t in data.get("tracks", [])}


def extract_track(audio: Path, track: str, out_dir: Path) -> Path:
    """Cut one track out of the suite, once, and reuse it thereafter.

    Every configuration must chew byte-identical audio or the comparison
    is meaningless, so the cut is cached rather than redone per run.
    """
    if track in ("all", "full", ""):
        return audio
    tracks = load_tracks()
    if track not in tracks:
        raise SystemExit(
            f"unknown track {track!r}; manifest has: "
            f"{', '.join(sorted(tracks)) or '(manifest not found)'}")
    start, dur = tracks[track]
    out_dir.mkdir(parents=True, exist_ok=True)
    cut = out_dir / f"{audio.stem}.{track}.wav"
    if cut.is_file():
        return cut
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(audio), "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
         "-ac", "1", "-ar", "16000", str(cut)],
        check=True, timeout=600)
    return cut


# --------------------------------------------------------------------------
# GPU sampling
# --------------------------------------------------------------------------

def _detect_gpu_tool() -> tuple[Optional[str], Optional[list[str]]]:
    """Return (vendor, command) for whichever GPU tool is present."""
    if shutil.which("nvidia-smi"):
        return "nvidia", [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    if shutil.which("rocm-smi"):
        return "amd", ["rocm-smi", "--showuse", "--showmemuse", "--showtemp",
                       "--csv"]
    return None, None


def _parse_nvidia(line: str) -> Optional[dict]:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return None
    try:
        return {"gpu_util": float(parts[0]), "mem_util": float(parts[1]),
                "mem_used_mb": float(parts[2]), "temp_c": float(parts[3])}
    except ValueError:
        return None


def _parse_rocm(out: str) -> Optional[dict]:
    """rocm-smi --csv output varies by version, so pick columns by name."""
    lines = [l for l in out.splitlines() if l.strip()]
    if len(lines) < 2:
        return None
    header = [h.strip().lower() for h in lines[0].split(",")]
    row = [c.strip() for c in lines[1].split(",")]
    got: dict = {}
    for name, value in zip(header, row):
        try:
            v = float(value.replace("%", "").replace("c", "").strip())
        except ValueError:
            continue
        if "gpu use" in name or "gpu_use" in name or "gpu%" in name:
            got["gpu_util"] = v
        elif "memory use" in name or "mem use" in name or "vram%" in name:
            got["mem_util"] = v
        elif "temp" in name and "temp_c" not in got:
            got["temp_c"] = v
    return got or None


class GPUSampler(threading.Thread):
    """Polls the GPU while the run is in flight.

    Utilization here is a coarse, sampled number — it tells you whether
    the GPU is pinned or idle, not precisely how busy. The span overlap
    from the profiler is the finer instrument; this is the sanity check
    that sits beside it.
    """

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True, name="gpu-sampler")
        self.interval = interval
        self.vendor, self.cmd = _detect_gpu_tool()
        self.samples: list[dict] = []
        self._stop = threading.Event()

    def run(self) -> None:
        if not self.cmd:
            return
        t0 = time.perf_counter()
        while not self._stop.is_set():
            try:
                out = subprocess.run(self.cmd, capture_output=True, text=True,
                                     timeout=10).stdout
                s = (_parse_nvidia(out.strip().splitlines()[0])
                     if self.vendor == "nvidia" and out.strip()
                     else _parse_rocm(out))
                if s:
                    s["t"] = round(time.perf_counter() - t0, 3)
                    self.samples.append(s)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=5)

    def summary(self) -> dict:
        if not self.samples:
            return {"available": False, "vendor": self.vendor}
        def stat(key: str) -> dict:
            vals = [s[key] for s in self.samples if key in s]
            if not vals:
                return {}
            vals_sorted = sorted(vals)
            return {
                "mean": round(statistics.fmean(vals), 1),
                "p50": round(vals_sorted[len(vals) // 2], 1),
                "p95": round(vals_sorted[int(len(vals) * 0.95) - 1], 1),
                "max": round(max(vals), 1),
            }
        return {
            "available": True, "vendor": self.vendor,
            "samples": len(self.samples),
            "gpu_util": stat("gpu_util"),
            "mem_used_mb": stat("mem_used_mb"),
            "mem_util": stat("mem_util"),
            "temp_c": stat("temp_c"),
        }


# --------------------------------------------------------------------------
# Span analysis — the part that answers the parallelism question
# --------------------------------------------------------------------------

def union_duration(spans: list[tuple[float, float]]) -> float:
    """Total wall-clock time covered by these spans, counting overlap once."""
    if not spans:
        return 0.0
    merged: list[list[float]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(e - s for s, e in merged)


def max_concurrency(spans: list[tuple[float, float]]) -> tuple[int, float]:
    """Peak simultaneous spans, and the mean concurrency while any is active."""
    if not spans:
        return 0, 0.0
    events = []
    for s, e in spans:
        events.append((s, 1))
        events.append((e, -1))
    events.sort()
    cur = peak = 0
    weighted = 0.0
    last_t = events[0][0]
    for t, delta in events:
        if cur > 0:
            weighted += cur * (t - last_t)
        last_t = t
        cur += delta
        peak = max(peak, cur)
    active = union_duration(spans)
    return peak, round(weighted / active, 2) if active else 0.0


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * p))]


def analyse_spans(path: Path) -> dict:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    spans = [r for r in rows if r.get("kind") == "span"]
    sentences = [r for r in rows if r.get("kind") == "sentence"]

    out: dict = {"spans": len(spans), "sentences": len(sentences)}
    if not spans:
        return out

    by_stage: dict[str, list[dict]] = {}
    for s in spans:
        by_stage.setdefault(s["stage"], []).append(s)

    stages: dict = {}
    for stage, items in by_stage.items():
        durs = [i["dur"] for i in items]
        pairs = [(i["start"], i["end"]) for i in items]
        total = sum(durs)
        union = union_duration(pairs)
        peak, mean_conc = max_concurrency(pairs)
        per_lang = {}
        for i in items:
            per_lang.setdefault(i.get("lang") or "-", []).append(i["dur"])
        stages[stage] = {
            "calls": len(items),
            "mean_s": round(statistics.fmean(durs), 4),
            "p50_s": round(_pct(durs, 0.5), 4),
            "p95_s": round(_pct(durs, 0.95), 4),
            "total_s": round(total, 3),
            "union_s": round(union, 3),
            # The headline number: how many languages' worth of work the
            # hardware actually did at once.
            "overlap": round(total / union, 2) if union else 0.0,
            "peak_concurrent": peak,
            "mean_concurrent": mean_conc,
            "per_language_mean_s": {
                k: round(statistics.fmean(v), 4) for k, v in sorted(per_lang.items())
            },
        }
    out["stages"] = stages

    # Combined GPU-bearing work: translate and tts together.
    gpu_spans = [(s["start"], s["end"]) for s in spans
                 if s["stage"] in ("translate", "tts")]
    if gpu_spans:
        total = sum(e - s for s, e in gpu_spans)
        union = union_duration(gpu_spans)
        peak, mean_conc = max_concurrency(gpu_spans)
        span_wall = max(e for _, e in gpu_spans) - min(s for s, _ in gpu_spans)
        out["pipeline"] = {
            "busy_s": round(union, 3),
            "work_s": round(total, 3),
            "wall_s": round(span_wall, 3),
            # Fraction of the run during which the pipeline had something
            # in flight. Near 1.0 means saturated.
            "duty_cycle": round(union / span_wall, 3) if span_wall else 0.0,
            "overlap": round(total / union, 2) if union else 0.0,
            "peak_concurrent": peak,
            "mean_concurrent": mean_conc,
        }

    if sentences:
        qd = [s.get("queue_depth", 0) for s in sentences]
        audio = [s.get("audio_seconds", 0.0) for s in sentences]
        # Queue depth trend: the clearest signal of falling behind. If the
        # back half of the run queues deeper than the front half, the
        # machine is not keeping up, whatever the averages say.
        half = max(1, len(qd) // 2)
        out["sentences_detail"] = {
            "count": len(sentences),
            "by_language": {
                lang: sum(1 for s in sentences if s.get("lang") == lang)
                for lang in sorted({s.get("lang", "?") for s in sentences})
            },
            "queue_depth_mean": round(statistics.fmean(qd), 2),
            "queue_depth_max": max(qd),
            "queue_depth_first_half": round(statistics.fmean(qd[:half]), 2),
            "queue_depth_second_half": round(statistics.fmean(qd[half:]), 2),
            "audio_seconds_total": round(sum(audio), 1),
            "sped_up_fraction": round(
                sum(1 for s in sentences if s.get("speed", 1.0) > 1.0) / len(sentences), 3),
        }
    return out


# --------------------------------------------------------------------------
# Running one configuration
# --------------------------------------------------------------------------

@dataclass
class RunSpec:
    label: str
    languages: list[str]
    mode: str                       # realtime | throughput
    audio: Path
    timeout: float = 3600.0
    extra_env: dict = field(default_factory=dict)


def machine_info() -> dict:
    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if line.lower().startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    for cmd, key in ((["nvidia-smi", "--query-gpu=name,memory.total",
                       "--format=csv,noheader"], "gpu"),
                     (["rocm-smi", "--showproductname"], "gpu")):
        if shutil.which(cmd[0]):
            try:
                out = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=15).stdout.strip()
                if out:
                    info[key] = out.splitlines()[0].strip()
                    break
            except Exception:
                pass
    try:
        mem = Path("/proc/meminfo").read_text()
        for line in mem.splitlines():
            if line.startswith("MemTotal"):
                info["ram_gb"] = round(int(line.split()[1]) / 1024 / 1024, 1)
                break
    except Exception:
        pass
    return info


def run_one(spec: RunSpec, out_dir: Path, python: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    spans_path = out_dir / f"{spec.label}.spans.jsonl"
    log_path = out_dir / f"{spec.label}.log"
    spans_path.unlink(missing_ok=True)

    env = dict(os.environ)
    env["TRANSLATE_PROFILE"] = str(spans_path)
    env.update(spec.extra_env)

    cmd = [python, "run.py", "--input-file", str(spec.audio),
           "--languages", *spec.languages]
    if spec.mode == "throughput":
        cmd.append("--no-realtime")

    print(f"\n=== {spec.label} : {' '.join(spec.languages)} / {spec.mode} ===")
    print(f"    {' '.join(cmd)}")

    sampler = GPUSampler()
    sampler.start()
    started = time.perf_counter()
    rc = -1
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.Popen(cmd, cwd=str(REPO), env=env,
                                    stdout=logf, stderr=subprocess.STDOUT)
            rc = proc.wait(timeout=spec.timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"    TIMED OUT after {spec.timeout:.0f}s")
    finally:
        wall = time.perf_counter() - started
        sampler.stop()

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from device_report import collect as collect_devices, verdicts
        devices = collect_devices()
        device_verdicts = [{"level": l, "message": m} for l, m in verdicts(devices)]
    except Exception as e:  # noqa: BLE001
        devices, device_verdicts = {"error": str(e)}, []

    result = {
        "label": spec.label,
        "languages": spec.languages,
        "mode": spec.mode,
        "audio": str(spec.audio),
        "returncode": rc,
        "wall_s": round(wall, 2),
        "machine": machine_info(),
        "gpu": sampler.summary(),
        "devices": devices,
        "device_verdicts": device_verdicts,
        "log": str(log_path),
    }
    if spans_path.is_file():
        result["profile"] = analyse_spans(spans_path)
        result["spans_file"] = str(spans_path)
    else:
        result["profile"] = {"error": "no spans recorded — is profiling.py in "
                                      "src/pipeline and imported by coordinator.py?"}

    (out_dir / f"{spec.label}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(f"    wall {wall:.1f}s, rc={rc} -> {out_dir / (spec.label + '.json')}")
    return result


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_report(results: list[dict]) -> None:
    print()
    print("=" * 78)
    print("BENCHMARK REPORT")
    print("=" * 78)

    for r in results:
        m = r.get("machine", {})
        print(f"\n{r['label']}  ({', '.join(r['languages'])}, {r['mode']})")
        print(f"  machine     : {m.get('cpu', '?')}")
        print(f"                {m.get('gpu', 'no GPU tool')}, "
              f"{m.get('ram_gb', '?')} GB RAM")
        print(f"  wall        : {r['wall_s']}s   rc={r['returncode']}")

        for v in r.get("device_verdicts", []):
            if v["level"] in ("FAIL", "WARN"):
                print(f"  [{v['level']:<4}]     {v['message']}")

        prof = r.get("profile", {})
        if "error" in prof:
            print(f"  PROFILE     : {prof['error']}")
            continue

        pl = prof.get("pipeline", {})
        if pl:
            print(f"  pipeline    : work {pl['work_s']}s over {pl['busy_s']}s busy "
                  f"({pl['wall_s']}s span)")
            print(f"  OVERLAP     : {pl['overlap']}x   "
                  f"(peak {pl['peak_concurrent']} concurrent, "
                  f"mean {pl['mean_concurrent']})")
            print(f"  duty cycle  : {pl['duty_cycle']}  "
                  f"{'<- SATURATED' if pl['duty_cycle'] > 0.9 else ''}")

        for stage, st in sorted(prof.get("stages", {}).items()):
            langs = ", ".join(f"{k} {v:.2f}s"
                              for k, v in st["per_language_mean_s"].items())
            print(f"  {stage:<10}: mean {st['mean_s']:.3f}s  p95 {st['p95_s']:.3f}s  "
                  f"overlap {st['overlap']}x  [{langs}]")

        sd = prof.get("sentences_detail", {})
        if sd:
            trend = sd["queue_depth_second_half"] - sd["queue_depth_first_half"]
            verdict = ("KEEPING UP" if trend <= 0.5
                       else f"FALLING BEHIND (+{trend:.1f} queue depth)")
            print(f"  sentences   : {sd['count']}  {sd['by_language']}")
            print(f"  queue depth : mean {sd['queue_depth_mean']}  "
                  f"max {sd['queue_depth_max']}  "
                  f"({sd['queue_depth_first_half']} -> "
                  f"{sd['queue_depth_second_half']})  {verdict}")
            if sd["sped_up_fraction"] > 0.05:
                print(f"  speed-ups   : {sd['sped_up_fraction']:.0%} of sentences "
                      f"were sped up to catch up")

        g = r.get("gpu", {})
        if g.get("available"):
            u = g.get("gpu_util", {})
            mem = g.get("mem_used_mb", {})
            print(f"  gpu util    : mean {u.get('mean', '?')}%  "
                  f"p95 {u.get('p95', '?')}%  max {u.get('max', '?')}%")
            if mem:
                print(f"  gpu memory  : mean {mem.get('mean', '?')} MB  "
                      f"max {mem.get('max', '?')} MB")

    # Per-language cost, derived from a 1-language and a 3-language run of
    # the same mode. This is the number the whole exercise exists for.
    print()
    print("-" * 78)
    print("COST OF EACH ADDITIONAL LANGUAGE")
    print("-" * 78)
    by_mode: dict[str, list[dict]] = {}
    for r in results:
        if "error" not in r.get("profile", {}):
            by_mode.setdefault(r["mode"], []).append(r)

    for mode, group in sorted(by_mode.items()):
        group = sorted(group, key=lambda r: len(r["languages"]))
        if len(group) < 2:
            continue
        base = group[0]
        bp = base["profile"].get("pipeline", {})
        print(f"\n  {mode}:")
        print(f"    {len(base['languages'])} language "
              f"-> busy {bp.get('busy_s', 0)}s, overlap {bp.get('overlap', 0)}x")
        for r in group[1:]:
            p = r["profile"].get("pipeline", {})
            n_base, n = len(base["languages"]), len(r["languages"])
            if not bp.get("busy_s"):
                continue
            busy_ratio = p.get("busy_s", 0) / bp["busy_s"]
            added = n - n_base
            marginal = (busy_ratio - 1.0) / added if added else 0.0
            print(f"    {n} languages -> busy {p.get('busy_s', 0)}s, "
                  f"overlap {p.get('overlap', 0)}x")
            print(f"      {busy_ratio:.2f}x the busy time for {n}x the languages")
            print(f"      each extra language adds ~{marginal:.0%} "
                  f"of a single-language run")
            if marginal < 0.35:
                print("      => languages are largely free; the GPU is not the limit")
            elif marginal > 0.8:
                print("      => languages are nearly additive; the GPU IS the limit")

    print()
    print("-" * 78)
    print("HOW TO READ THIS")
    print("-" * 78)
    print("  overlap 3.0x with 3 languages = perfectly parallel, a language is free")
    print("  overlap 1.0x with 3 languages = fully serialized, a language costs full")
    print("  duty cycle near 1.0           = the pipeline never idles: saturated")
    print("  queue depth rising            = not keeping up, whatever the averages say")
    print()


# --------------------------------------------------------------------------

SUITE = [
    ("1lang-realtime", ["es"], "realtime"),
    ("3lang-realtime", ["es", "ht", "ru"], "realtime"),
    ("1lang-throughput", ["es"], "throughput"),
    ("3lang-throughput", ["es", "ht", "ru"], "throughput"),
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="benchmark_stack", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", type=Path, default=None,
                   help=f"recording to drive the pipeline "
                        f"(default: {DEFAULT_AUDIO.name})")
    p.add_argument("--track", default=DEFAULT_TRACK,
                   help="which track of the suite to use, or 'all' for the "
                        "whole 56-minute file (default: sermon)")
    p.add_argument("--languages", nargs="+", default=None)
    p.add_argument("--mode", choices=("realtime", "throughput"), default="realtime")
    p.add_argument("--suite", choices=("compare", "quick"), default=None,
                   help="compare: 1 and 3 languages, realtime and throughput")
    p.add_argument("--label", default=None)
    p.add_argument("--tag", default=None,
                   help="prefix for labels, e.g. 3060 or 890m")
    p.add_argument("--out", type=Path, default=REPO / "benchmarks")
    p.add_argument("--python", default=str(REPO / "venv" / "bin" / "python"))
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--report", nargs="+", type=Path, default=None,
                   help="print a report from existing .json results and exit")
    args = p.parse_args(argv)

    if args.report:
        results = []
        for path in args.report:
            for f in ([path] if path.is_file() else sorted(path.glob("*.json"))):
                try:
                    results.append(json.loads(f.read_text(encoding="utf-8")))
                except Exception as e:
                    print(f"skipping {f}: {e}")
        if not results:
            print("no results found")
            return 1
        print_report(results)
        return 0

    audio = args.audio or DEFAULT_AUDIO
    if not audio.is_file():
        print(f"audio not found: {audio}")
        print("Pass --audio, or check the suite exists in tests/ab_test/audio/")
        return 2
    try:
        audio = extract_track(audio, args.track, args.out / "audio")
    except subprocess.CalledProcessError as e:
        print(f"could not extract track {args.track!r}: {e}")
        return 2
    args.audio = audio
    python = args.python if Path(args.python).exists() else sys.executable

    tag = args.tag or platform.node()
    specs: list[RunSpec] = []
    if args.suite:
        chosen = SUITE if args.suite == "compare" else SUITE[:2]
        for name, langs, mode in chosen:
            specs.append(RunSpec(f"{tag}-{name}", langs, mode, args.audio,
                                 timeout=args.timeout))
    else:
        langs = args.languages or ["es", "ht", "ru"]
        label = args.label or f"{tag}-{len(langs)}lang-{args.mode}"
        specs.append(RunSpec(label, langs, args.mode, args.audio,
                             timeout=args.timeout))

    print(f"machine: {json.dumps(machine_info(), indent=2)}")
    print(f"python : {python}")
    print(f"audio  : {args.audio}  (track: {args.track})")
    print(f"runs   : {len(specs)}")

    results = [run_one(s, args.out, python) for s in specs]
    print_report(results)
    print(f"\nresults written to {args.out}")
    print("Compare two machines with:")
    print(f"  python tests/benchmark_stack.py --report {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
