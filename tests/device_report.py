"""What is actually running on the GPU, and what quietly is not.

    ./venv/bin/python tests/device_report.py
    ./venv/bin/python tests/device_report.py --json

Why this exists
---------------
`run_production.sh` exports `NLLB_DEVICE=cuda`, `KOKORO_DEVICE=cuda` and
`MMS_DEVICE=cuda`.  On ROCm that is correct — torch maps `cuda` to HIP,
and the archive project's own note ("HIP maps cuda->hip; 2,700+ ASR jobs
prove it") confirms it works.  So translation and both TTS backends do
use the GPU on an AMD box.

ASR on CPU is **not** a fault.  `tests/benchmarks/REPORT_20260515.md`
measured Parakeet TDT v3 at RTF 0.027 on CPU — 37x faster than realtime —
against 0.005 on CUDA, and concluded "ASR will never be the bottleneck".
The production recommendation in that same report deliberately puts
Parakeet on CPU to keep the GPU free for translation.  So a CPU-only
onnxruntime is the intended configuration, not a regression.

What *does* invalidate a comparison is the two boxes running different
ASR **models**.  The
NeMo unified Parakeet (`PARAKEET_MODEL=unified-remote`) needs a second
Python 3.11 venv, and `TODO.md` item 3 records that on ROCm hosts that
venv is still built by hand.  If the laptop falls back to the onnx-asr
TDT model, it is running a different model with lower accuracy and no
punctuation-driven sentence boundaries — which changes how many
sentences reach translation, and therefore changes every number
downstream.

So: run this on both machines before benchmarking, and keep the output
next to the results.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent


def _probe_torch() -> dict:
    out: dict[str, Any] = {}
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": str(e)}

    out["available"] = True
    out["version"] = torch.__version__
    out["cuda_is_available"] = bool(torch.cuda.is_available())
    out["cuda_build"] = getattr(torch.version, "cuda", None)
    out["hip_build"] = getattr(torch.version, "hip", None)
    out["flavor"] = ("rocm" if getattr(torch.version, "hip", None)
                     else "cuda" if getattr(torch.version, "cuda", None)
                     else "cpu")
    try:
        out["device_count"] = torch.cuda.device_count()
        out["devices"] = [torch.cuda.get_device_name(i)
                          for i in range(torch.cuda.device_count())]
    except Exception:
        out["device_count"] = 0
        out["devices"] = []

    # Placement is not the same as speed. A HIP build that silently runs on
    # the host will still *report* a cuda device, so time a real matmul on
    # both and compare. A GPU that is genuinely working is many times faster
    # than the CPU at this size; a ratio near 1 means something is wrong.
    if out["cuda_is_available"]:
        try:
            n = 2048
            a = torch.randn(n, n)
            b = torch.randn(n, n)
            t0 = time.perf_counter()
            for _ in range(3):
                a @ b
            cpu_s = (time.perf_counter() - t0) / 3

            # fp16 on the GPU: that is what the benchmarks and production
            # load models in. An integrated GPU (Radeon 890M) is only ~1.5x
            # a 12-core CPU at fp32 but 4-7x at fp16, so an fp32 test flags a
            # healthy iGPU as a "silent fallback". A real fallback runs fp16
            # on the host and is far slower than CPU fp32, so this still
            # catches it.
            ga, gb = a.half().to("cuda"), b.half().to("cuda")
            torch.cuda.synchronize()
            (ga @ gb)          # warm up kernels and allocator
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                ga @ gb
            torch.cuda.synchronize()
            gpu_s = (time.perf_counter() - t0) / 3

            out["matmul_cpu_s"] = round(cpu_s, 4)
            out["matmul_gpu_s"] = round(gpu_s, 4)
            out["gpu_speedup"] = round(cpu_s / gpu_s, 1) if gpu_s else None
            out["gpu_really_working"] = bool(gpu_s and cpu_s / gpu_s > 3.0)
        except Exception as e:  # noqa: BLE001
            out["matmul_error"] = str(e)
            out["gpu_really_working"] = False
    else:
        out["gpu_really_working"] = False
    return out


def _probe_onnxruntime() -> dict:
    """The onnx-asr Parakeet path — the documented silent CPU fallback."""
    try:
        import onnxruntime as ort
    except Exception as e:  # noqa: BLE001
        return {"installed": False, "error": str(e)}
    providers = list(ort.get_available_providers())
    accel = [p for p in providers
             if p not in ("CPUExecutionProvider", "AzureExecutionProvider")]
    return {
        "installed": True,
        "version": getattr(ort, "__version__", "?"),
        "providers": providers,
        "accelerated_providers": accel,
        # This is the line that matters on a ROCm box.
        "cpu_only": not accel,
    }


def _probe_asr_choice() -> dict:
    """Which ASR model this box would actually use."""
    model = os.environ.get("PARAKEET_MODEL", "").strip()
    unified_python = os.environ.get("UNIFIED_PYTHON", "").strip()
    out = {
        "PARAKEET_MODEL": model or "(unset)",
        "UNIFIED_PYTHON": unified_python or "(unset)",
        "using_nemo_unified": model == "unified-remote",
    }
    if unified_python:
        out["unified_python_exists"] = Path(unified_python).exists()
    # The conventional location from the README.
    guess = Path.home() / "nemo-venv" / "bin" / "python"
    out["nemo_venv_present"] = guess.exists()
    if not out["using_nemo_unified"]:
        out["note"] = (
            "Falling back to the onnx-asr TDT model: lower accuracy and no "
            "punctuation-driven sentence boundaries. Benchmarks against a "
            "box running unified-remote are NOT comparable."
        )
    return out


def _probe_env() -> dict:
    keys = ["NLLB_MODEL", "NLLB_DEVICE", "NLLB_DTYPE", "ES_TTS", "HT_TTS",
            "RU_TTS", "KOKORO_DEVICE", "MMS_DEVICE", "HSA_OVERRIDE_GFX_VERSION",
            "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
            "TRANSLATE_PROFILE"]
    return {k: os.environ.get(k, "(unset)") for k in keys}


def _probe_tools() -> dict:
    tools = {}
    for name in ("ffmpeg", "ffprobe", "espeak-ng", "nvidia-smi", "rocm-smi",
                 "rocminfo", "vainfo"):
        tools[name] = shutil.which(name) or None
    gpu_name = None
    for cmd in (["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader"],
                ["rocm-smi", "--showproductname", "--csv"]):
        if tools.get(cmd[0]):
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
                if r.returncode == 0 and r.stdout.strip():
                    gpu_name = r.stdout.strip().splitlines()[-1].strip()
                    break
            except Exception:
                pass
    tools["gpu_name"] = gpu_name
    return tools


def _probe_machine() -> dict:
    info = {"hostname": platform.node(), "platform": platform.platform(),
            "python": sys.version.split()[0], "cpu_count": os.cpu_count()}
    try:
        for line in subprocess.run(["lscpu"], capture_output=True, text=True,
                                   timeout=10).stdout.splitlines():
            if line.lower().startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                info["ram_gb"] = round(int(line.split()[1]) / 1024 / 1024, 1)
                break
    except Exception:
        pass
    return info


def collect() -> dict:
    return {
        "machine": _probe_machine(),
        "torch": _probe_torch(),
        "onnxruntime": _probe_onnxruntime(),
        "asr": _probe_asr_choice(),
        "env": _probe_env(),
        "tools": _probe_tools(),
    }


# Reported alongside the device checks because it explains the shape of
# every benchmark result on this stack.
TRANSLATION_NOTE = (
    "Translation is serialized across languages by design: the NLLB weights "
    "are shared between language pipelines and generate() is guarded by one "
    "shared lock (translation.py line ~400), because concurrent generate() on "
    "a shared model segfaults on ROCm. TTS is NOT serialized \u2014 each language "
    "owns its own TTSService. So expect translate overlap near 1.0x and tts "
    "overlap above 1.0x."
)


def verdicts(rep: dict) -> list[tuple[str, str]]:
    """(level, message). level is OK, WARN or FAIL."""
    out: list[tuple[str, str]] = []
    t = rep["torch"]
    if not t.get("available"):
        out.append(("FAIL", f"torch did not import: {t.get('error')}"))
        return out

    out.append(("OK", f"torch {t['version']} ({t['flavor']} build)"))
    if not t.get("cuda_is_available"):
        out.append(("FAIL", "no GPU visible to torch — everything will run on CPU"))
    elif t.get("gpu_really_working"):
        out.append(("OK", f"GPU compute confirmed: {t.get('gpu_speedup')}x "
                          f"faster than CPU on a 2048x2048 matmul (GPU fp16 vs CPU fp32)"))
    else:
        out.append(("FAIL", f"a GPU is reported but is not actually faster "
                            f"({t.get('gpu_speedup')}x) — suspect a silent "
                            f"CPU fallback"))

    ort = rep["onnxruntime"]
    if ort.get("installed"):
        if ort.get("cpu_only"):
            # Measured at RTF 0.027 = 37x realtime in REPORT_20260515.md,
            # and chosen deliberately there to keep the GPU for translation.
            out.append(("OK", "onnxruntime is CPU-only — expected; Parakeet on "
                              "CPU measured 37x realtime and is the "
                              "recommended production placement"))
        else:
            out.append(("OK", f"onnxruntime accelerated: "
                              f"{', '.join(ort['accelerated_providers'])}"))

    asr = rep["asr"]
    if asr.get("using_nemo_unified"):
        out.append(("OK", "ASR is the NeMo unified Parakeet (unified-remote)"))
    else:
        out.append(("WARN", "ASR is the onnx-asr TDT model, not unified-remote. "
                            "Fine on its own, but NOT comparable with a box "
                            "running unified-remote: different model, no "
                            "punctuation-driven sentence boundaries, so a "
                            "different number of sentences reaches translation"))

    for key in ("NLLB_DEVICE", "KOKORO_DEVICE", "MMS_DEVICE"):
        val = rep["env"].get(key)
        if val == "(unset)":
            out.append(("WARN", f"{key} unset — did you source run_production.sh?"))
        elif val != "cuda":
            out.append(("WARN", f"{key}={val} (cuda maps to HIP on ROCm; "
                                f"anything else means CPU)"))
    if not rep["tools"].get("ffmpeg"):
        out.append(("WARN", "ffmpeg not found — the benchmark needs it to "
                            "prepare audio"))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="device_report", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", type=Path, default=None, help="also write JSON here")
    args = p.parse_args(argv)

    rep = collect()
    rep["verdicts"] = [{"level": l, "message": m} for l, m in verdicts(rep)]

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(rep, indent=2))
        return 0

    m, t = rep["machine"], rep["torch"]
    print("=" * 70)
    print(f"DEVICE REPORT — {m.get('hostname')}")
    print("=" * 70)
    print(f"  cpu      : {m.get('cpu', '?')} ({m.get('cpu_count')} threads)")
    print(f"  ram      : {m.get('ram_gb', '?')} GB")
    print(f"  gpu      : {rep['tools'].get('gpu_name') or '(no smi tool)'}")
    if t.get("devices"):
        print(f"  torch see: {', '.join(t['devices'])}")
    print()
    for level, msg in verdicts(rep):
        mark = {"OK": "  ok  ", "WARN": " WARN ", "FAIL": " FAIL "}[level]
        print(f"[{mark}] {msg}")
    print()
    print("  note:")
    for chunk in TRANSLATION_NOTE.split(". "):
        if chunk.strip():
            print(f"    {chunk.strip().rstrip('.')}.")
    print()
    print("  env:")
    for k, v in rep["env"].items():
        if v != "(unset)":
            print(f"    {k}={v}")
    print()
    fails = sum(1 for l, _ in verdicts(rep) if l == "FAIL")
    if fails:
        print(f"  {fails} problem(s) that would invalidate a benchmark.")
        return 1
    print("  Ready to benchmark. Keep this output beside the results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
