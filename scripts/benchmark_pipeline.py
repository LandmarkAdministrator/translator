"""
Per-component pipeline benchmark.

Measures how fast each stage in the live-translator pipeline runs against
a single fixed audio file (default: tests/ab_test/audio/stitched_test.wav).
Each model is loaded standalone — no streaming wrappers, no sentence
buffers, no audio playback — so the numbers reflect raw inference speed
rather than pipeline overhead.

Tested components:

  * ASR — every Parakeet variant onnx-asr officially supports:
        nemo-parakeet-ctc-0.6b
        nemo-parakeet-rnnt-0.6b
        nemo-parakeet-tdt-0.6b-v2
        nemo-parakeet-tdt-0.6b-v3   (current production)
    Each is run on CPU; if a CUDA / ROCm GPU is available, also on GPU
    (onnx-asr's ROCM/CUDA provider where applicable).

  * Translation — facebook/nllb-200-distilled-1.3B and facebook/nllb-200-3.3B
    in fp16 on GPU.  Translates every ASR sentence to es + ht.

  * TTS — Kokoro 82M (es voice em_alex) and facebook/mms-tts-hat (Haitian)
    each on CPU and on GPU.

  * Simultaneous load — load the production stack at once and report peak
    VRAM so we know if it fits.

For each component we report:
    load time, total inference time, peak VRAM (when on GPU), peak RAM,
    raw output (saved to disk for inspection), and RTF (real-time factor)
    relative to the input audio duration.

Output is written to tests/benchmarks/<timestamp>/:
    report.md                      summary table
    results.json                   raw timings
    asr_<id>_<device>.txt          ASR transcript per (model, device)
    translate_<size>_<lang>.txt    translation output per (model, target)

Designed to run unattended overnight. Failures in any one component are
caught, logged, and don't abort the rest of the run.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import resource
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, List, Optional


# ----------------------------------------------------------------- helpers

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIO = PROJECT_ROOT / "tests" / "ab_test" / "audio" / "stitched_test.wav"


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def gpu_summary() -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return "no GPU"
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total_mb = torch.cuda.get_device_properties(idx).total_memory / 1024**2
        return f"{name} ({total_mb:.0f} MiB total)"
    except Exception as e:
        return f"GPU probe failed: {e}"


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def peak_vram_mib() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
    except Exception:
        pass
    return 0.0


def reset_vram_stats() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except Exception:
        pass


@dataclass
class StageResult:
    """One row in the results table."""
    component: str          # e.g. "asr", "translate", "tts"
    model_id: str
    device: str             # "cpu", "cuda"
    load_time_s: float = 0.0
    inference_time_s: float = 0.0
    audio_duration_s: float = 0.0   # for ASR
    output_seconds: float = 0.0     # for TTS (audio produced)
    rtf: float = 0.0                # inference / reference; for ASR vs audio in, for TTS vs audio out
    peak_vram_mib: float = 0.0
    peak_rss_mib: float = 0.0
    n_items: int = 0                # sentences translated, tokens decoded, etc.
    ok: bool = True
    error: str = ""
    extras: dict = field(default_factory=dict)


# ----------------------------------------------------------------- ASR

PARAKEET_MODELS = [
    "nemo-parakeet-ctc-0.6b",
    "nemo-parakeet-rnnt-0.6b",
    "nemo-parakeet-tdt-0.6b-v2",
    "nemo-parakeet-tdt-0.6b-v3",
]


def benchmark_parakeet(model_id: str, audio_path: Path, out_dir: Path,
                       device: str, providers: Optional[List[str]] = None,
                       chunk_seconds: float = 30.0) -> StageResult:
    """Run a single Parakeet model over the audio in 30s chunks.

    onnx-asr's recognize() processes a non-streaming buffer; calling it on
    the entire 60-min audio at once allocates >16 GiB and OOMs on the 3060
    box. Chunking matches how production uses the model (streaming wrapper
    feeds a rolling buffer) and gives a realistic RTF.
    """
    import numpy as np
    import soundfile as sf

    r = StageResult(component="asr", model_id=model_id, device=device)
    reset_vram_stats()

    t0 = time.monotonic()
    try:
        import onnx_asr
        # Default providers per device
        if providers is None:
            if device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif device == "rocm":
                providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

        base = onnx_asr.load_model(
            model_id,
            providers=providers,
        )
        model = base.with_timestamps()
        r.load_time_s = time.monotonic() - t0

        # Load audio (resample to 16k if needed)
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if sr != 16000:
            from src.audio.resample import resample_audio
            data = resample_audio(data, sr, 16000)
            sr = 16000
        r.audio_duration_s = len(data) / sr

        # Chunked decode. We accumulate inference time (excluding sleep/IO)
        # and concatenate text. Memory peaks within a single chunk window,
        # not across the whole file.
        chunk_samples = int(chunk_seconds * sr)
        total_inference = 0.0
        all_text_parts: List[str] = []
        all_tokens_count = 0
        n_chunks = 0
        offset = 0
        while offset < len(data):
            end = min(offset + chunk_samples, len(data))
            chunk = data[offset:end]
            t1 = time.monotonic()
            result = model.recognize(chunk, sample_rate=sr)
            total_inference += time.monotonic() - t1
            piece = getattr(result, "text", "") or ""
            if not piece:
                tokens = list(getattr(result, "tokens", []) or [])
                piece = " ".join(tokens)
            all_text_parts.append(piece)
            all_tokens_count += len(list(getattr(result, "tokens", []) or []))
            n_chunks += 1
            offset = end

        r.inference_time_s = total_inference
        r.n_items = all_tokens_count
        r.rtf = r.inference_time_s / r.audio_duration_s if r.audio_duration_s else 0.0
        r.extras["chunks"] = n_chunks
        r.extras["chunk_seconds"] = chunk_seconds
        text = "\n".join(p for p in all_text_parts if p)

        # Save transcript
        out_file = out_dir / f"asr_{model_id.replace('/', '_')}_{device}.txt"
        out_file.write_text(text + "\n")
        r.extras["output_path"] = str(out_file.relative_to(out_dir.parent))
        r.extras["output_chars"] = len(text)
    except Exception as e:
        r.ok = False
        r.error = f"{type(e).__name__}: {e}"
        r.extras["traceback"] = traceback.format_exc()
    finally:
        r.peak_vram_mib = peak_vram_mib()
        r.peak_rss_mib = peak_rss_mib()
    return r


def split_into_sentences(text: str) -> list[str]:
    """Naive sentence segmentation by [.!?] terminators, abbrev-aware."""
    import re
    # Roughly match what SentenceBuffer's punctuation rule does.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    return [p for p in parts if p.strip()]


# --------------------------------------------------------- Translation NLLB

NLLB_MODELS = [
    "facebook/nllb-200-distilled-1.3B",
    "facebook/nllb-200-3.3B",
]
NLLB_LANG_CODES = {"en": "eng_Latn", "es": "spa_Latn", "ht": "hat_Latn"}


def benchmark_nllb(model_id: str, source_text: str, out_dir: Path,
                   targets: List[str], device: str = "cuda",
                   dtype_str: str = "float16") -> List[StageResult]:
    """Translate every sentence in source_text to each target language.

    Loads the model once and runs both target languages back-to-back.
    Returns one StageResult per target lang.
    """
    results = []
    reset_vram_stats()
    t0 = time.monotonic()
    model = tokenizer = None
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}.get(dtype_str, torch.float16)
        # tokenizer needs src_lang to add the correct BOS for NLLB
        tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=dtype)
        model = model.to(device)
        model.eval()
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.max_length = None
        load_time = time.monotonic() - t0

        sentences = split_into_sentences(source_text)
        for tgt in targets:
            tgt_code = NLLB_LANG_CODES[tgt]
            forced_bos = tokenizer.convert_tokens_to_ids(tgt_code)
            r = StageResult(
                component="translate",
                model_id=model_id,
                device=device,
                load_time_s=load_time,
                n_items=len(sentences),
            )
            reset_vram_stats()
            outputs = []
            t1 = time.monotonic()
            try:
                with torch.no_grad():
                    for sentence in sentences:
                        if not sentence.strip():
                            outputs.append("")
                            continue
                        inputs = tokenizer(
                            sentence, return_tensors="pt", truncation=True, max_length=512
                        ).to(device)
                        max_new = max(32, min(512, int(inputs["input_ids"].shape[-1] * 2.5) + 32))
                        out = model.generate(
                            **inputs,
                            forced_bos_token_id=forced_bos,
                            max_new_tokens=max_new,
                            num_beams=1,
                            repetition_penalty=1.3,
                            no_repeat_ngram_size=3,
                        )
                        outputs.append(tokenizer.decode(out[0], skip_special_tokens=True))
                r.inference_time_s = time.monotonic() - t1
                short = model_id.split("/")[-1]
                out_file = out_dir / f"translate_{short}_{tgt}.txt"
                out_file.write_text("\n".join(outputs))
                r.extras["output_path"] = str(out_file.relative_to(out_dir.parent))
                r.extras["target_lang"] = tgt
                r.extras["sentences_per_sec"] = len(sentences) / r.inference_time_s if r.inference_time_s else 0
            except Exception as e:
                r.ok = False
                r.error = f"{type(e).__name__}: {e}"
                r.extras["traceback"] = traceback.format_exc()
            finally:
                r.peak_vram_mib = peak_vram_mib()
                r.peak_rss_mib = peak_rss_mib()
            results.append(r)
    except Exception as e:
        # Loading failed: emit one stub result per target lang
        for tgt in targets:
            results.append(StageResult(
                component="translate",
                model_id=model_id,
                device=device,
                ok=False,
                error=f"load failed: {type(e).__name__}: {e}",
                extras={"target_lang": tgt, "traceback": traceback.format_exc()},
            ))
    finally:
        # Free this big boy before the next stage
        try:
            del model
        except Exception:
            pass
        try:
            del tokenizer
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return results


# --------------------------------------------------------------------- TTS

def benchmark_kokoro(translations_path: Path, out_dir: Path,
                     device: str) -> StageResult:
    """Synthesize every Spanish sentence with Kokoro on the requested device."""
    import numpy as np
    r = StageResult(component="tts", model_id="hexgrad/Kokoro-82M:em_alex", device=device)
    reset_vram_stats()
    t0 = time.monotonic()
    pipe = None
    try:
        from kokoro import KPipeline
        pipe = KPipeline(lang_code="e", device=device)
        r.load_time_s = time.monotonic() - t0

        sentences = [ln for ln in translations_path.read_text().splitlines() if ln.strip()]
        r.n_items = len(sentences)

        total_audio_samples = 0
        SR = 24000
        t1 = time.monotonic()
        for sentence in sentences:
            for _g, _p, audio in pipe(sentence, voice="em_alex"):
                if audio is None:
                    continue
                if hasattr(audio, "cpu"):
                    audio = audio.cpu().numpy()
                total_audio_samples += len(audio)
        r.inference_time_s = time.monotonic() - t1
        r.output_seconds = total_audio_samples / SR
        r.rtf = r.inference_time_s / r.output_seconds if r.output_seconds else 0.0
    except Exception as e:
        r.ok = False
        r.error = f"{type(e).__name__}: {e}"
        r.extras["traceback"] = traceback.format_exc()
    finally:
        r.peak_vram_mib = peak_vram_mib()
        r.peak_rss_mib = peak_rss_mib()
        try:
            del pipe
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return r


def benchmark_mms_tts(translations_path: Path, out_dir: Path,
                      device: str) -> StageResult:
    """Synthesize every Haitian sentence with facebook/mms-tts-hat."""
    import numpy as np
    r = StageResult(component="tts", model_id="facebook/mms-tts-hat", device=device)
    reset_vram_stats()
    t0 = time.monotonic()
    model = tokenizer = None
    try:
        import torch
        from transformers import VitsModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hat")
        model = VitsModel.from_pretrained("facebook/mms-tts-hat").to(device)
        model.eval()
        sr = int(model.config.sampling_rate)
        r.load_time_s = time.monotonic() - t0

        sentences = [ln for ln in translations_path.read_text().splitlines() if ln.strip()]
        r.n_items = len(sentences)

        total_samples = 0
        t1 = time.monotonic()
        with torch.no_grad():
            for sentence in sentences:
                inputs = tokenizer(sentence, return_tensors="pt").to(device)
                out = model(**inputs).waveform
                total_samples += int(out.numel())
        r.inference_time_s = time.monotonic() - t1
        r.output_seconds = total_samples / sr
        r.rtf = r.inference_time_s / r.output_seconds if r.output_seconds else 0.0
    except Exception as e:
        r.ok = False
        r.error = f"{type(e).__name__}: {e}"
        r.extras["traceback"] = traceback.format_exc()
    finally:
        r.peak_vram_mib = peak_vram_mib()
        r.peak_rss_mib = peak_rss_mib()
        try:
            del model
        except Exception:
            pass
        try:
            del tokenizer
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return r


# ---------------------------------------------------------- simultaneous load

def benchmark_simultaneous_load(out_dir: Path,
                                nllb_model_id: str = "facebook/nllb-200-distilled-1.3B"
                                ) -> StageResult:
    """Try to load the production stack (Parakeet + NLLB + Kokoro + MMS) all at once.

    Reports peak VRAM and whether load succeeded — answers "does the
    production stack fit on this GPU". Defaults to the 1.3B NLLB we
    actually ship; pass facebook/nllb-200-3.3B to ask whether the bigger
    quality model would also fit.
    """
    r = StageResult(component="simultaneous_load", model_id=f"prod_stack_with_{nllb_model_id.split('/')[-1]}",
                    device="mixed")
    reset_vram_stats()
    handles = []
    try:
        import torch
        # 1. Parakeet (CPU via onnx-asr — keep CPU here so VRAM numbers
        # only reflect the torch-backed models)
        import onnx_asr
        parakeet = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v3",
            providers=["CPUExecutionProvider"],
        ).with_timestamps()
        handles.append(parakeet)

        # 2. NLLB on GPU (fp16)
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        nllb_tok = AutoTokenizer.from_pretrained(nllb_model_id, src_lang="eng_Latn")
        nllb = AutoModelForSeq2SeqLM.from_pretrained(
            nllb_model_id, dtype=torch.float16
        ).to("cuda")
        nllb.eval()
        handles.append((nllb_tok, nllb))

        # 3. Kokoro (CPU; Kokoro's LSTM kernels run fine on CPU and
        # production keeps it there to leave the GPU for NLLB)
        from kokoro import KPipeline
        kokoro = KPipeline(lang_code="e", device="cpu")
        handles.append(kokoro)

        # 4. MMS-TTS-hat on GPU
        from transformers import VitsModel
        mms_tok = AutoTokenizer.from_pretrained("facebook/mms-tts-hat")
        mms = VitsModel.from_pretrained("facebook/mms-tts-hat").to("cuda")
        mms.eval()
        handles.append((mms_tok, mms))

        r.peak_vram_mib = peak_vram_mib()
        r.peak_rss_mib = peak_rss_mib()
        r.extras["loaded_models"] = [
            "nemo-parakeet-tdt-0.6b-v3 (CPU via onnxruntime)",
            f"{nllb_model_id} (cuda, fp16)",
            "hexgrad/Kokoro-82M (cpu)",
            "facebook/mms-tts-hat (cuda)",
        ]
    except Exception as e:
        r.ok = False
        r.error = f"{type(e).__name__}: {e}"
        r.extras["traceback"] = traceback.format_exc()
        r.peak_vram_mib = peak_vram_mib()
        r.peak_rss_mib = peak_rss_mib()
    finally:
        del handles
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return r


# -------------------------------------------------------------- report

def emit_report(results: List[StageResult], out_dir: Path,
                audio_path: Path, machine_info: dict) -> None:
    """Markdown summary at out_dir/report.md."""
    md = [
        f"# Pipeline benchmark — {now_stamp()}",
        "",
        f"- Audio: `{audio_path}` ({machine_info.get('audio_duration_s', 0):.1f}s)",
        f"- Host: `{machine_info.get('hostname', '?')}`",
        f"- Python: `{machine_info.get('python', '?')}`",
        f"- Torch: `{machine_info.get('torch', '?')}`",
        f"- GPU: `{machine_info.get('gpu', '?')}`",
        f"- CPUs: {machine_info.get('cpu_count', '?')}",
        "",
    ]

    by_component: dict[str, list] = {}
    for r in results:
        by_component.setdefault(r.component, []).append(r)

    for component, rows in by_component.items():
        md.append(f"## {component}")
        md.append("")
        md.append("| Model | Device | Load (s) | Infer (s) | RTF | Items | Peak VRAM (MiB) | Peak RSS (MiB) | OK | Note |")
        md.append("|---|---|---:|---:|---:|---:|---:|---:|:---:|---|")
        for r in rows:
            note = r.error if not r.ok else (r.extras.get("target_lang", "") or r.extras.get("output_chars", "") or "")
            md.append(
                f"| `{r.model_id}` | {r.device} | {r.load_time_s:.2f} | "
                f"{r.inference_time_s:.2f} | {r.rtf:.3f} | {r.n_items} | "
                f"{r.peak_vram_mib:.0f} | {r.peak_rss_mib:.0f} | "
                f"{'✓' if r.ok else '✗'} | {note} |"
            )
        md.append("")

    (out_dir / "report.md").write_text("\n".join(md) + "\n")


# -------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", nargs="?", default=str(DEFAULT_AUDIO),
                        help="Path to test audio (default: stitched_test.wav)")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-translate", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--skip-simultaneous", action="store_true")
    parser.add_argument("--reuse-asr-from", type=Path,
                        help="Skip ASR; use this transcript file as source for translation")
    parser.add_argument("--reuse-translate-from", type=Path,
                        help="Directory containing translate_*_es.txt + translate_*_ht.txt")
    parser.add_argument("--out", type=Path,
                        help="Output directory (default: tests/benchmarks/<timestamp>/)")
    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        print(f"audio not found: {audio_path}", file=sys.stderr)
        sys.exit(2)

    out_dir = args.out or (PROJECT_ROOT / "tests" / "benchmarks" / now_stamp())
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Machine info for the header
    import platform, multiprocessing, soundfile as sf
    info = soundfile_info = sf.info(str(audio_path))
    machine_info = {
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "cpu_count": multiprocessing.cpu_count(),
        "audio_duration_s": info.duration,
    }
    try:
        import torch
        machine_info["torch"] = torch.__version__
        machine_info["gpu"] = gpu_summary()
    except Exception as e:
        machine_info["torch"] = "import failed"
        machine_info["gpu"] = str(e)

    print(f"=== Pipeline benchmark — out_dir={out_dir} ===")
    print(json.dumps(machine_info, indent=2))

    results: List[StageResult] = []

    def save_partial():
        """Persist results.json after every stage so a later crash doesn't
        lose what we've already measured."""
        try:
            (out_dir / "results.json").write_text(json.dumps({
                "machine": machine_info,
                "audio": str(audio_path),
                "results": [asdict(r) for r in results],
            }, indent=2))
        except Exception as e:
            print(f"  WARN: could not save results.json: {e}", file=sys.stderr)

    # ASR phase ---------------------------------------------------
    canonical_transcript_path: Optional[Path] = None
    if not args.skip_asr:
        # Decide whether GPU is plausible for onnx-asr. We try CUDA first.
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
        except Exception:
            providers = []
        gpu_provider = None
        if "CUDAExecutionProvider" in providers:
            gpu_provider = "cuda"
        elif "ROCMExecutionProvider" in providers:
            gpu_provider = "rocm"

        for model_id in PARAKEET_MODELS:
            for dev in ["cpu", gpu_provider]:
                if dev is None:
                    continue
                print(f"  asr: {model_id}  device={dev}")
                r = benchmark_parakeet(model_id, audio_path, out_dir, dev)
                results.append(r)
                print(f"    -> ok={r.ok}  rtf={r.rtf:.3f}  vram={r.peak_vram_mib:.0f}MiB")
                if not r.ok:
                    print(f"    ERROR: {r.error}\n{r.extras.get('traceback', '')}", file=sys.stderr)
                save_partial()
        # Use v3-CPU transcript as canonical translation input if available
        for r in results:
            if r.component == "asr" and r.ok and "v3" in r.model_id:
                canonical_transcript_path = out_dir / Path(r.extras["output_path"]).name
                break
        if canonical_transcript_path is None:
            # fall back to any successful ASR output
            for r in results:
                if r.component == "asr" and r.ok:
                    canonical_transcript_path = out_dir / Path(r.extras["output_path"]).name
                    break
    elif args.reuse_asr_from:
        canonical_transcript_path = args.reuse_asr_from

    # Translation phase -------------------------------------------
    es_translation_path: Optional[Path] = None
    ht_translation_path: Optional[Path] = None
    if not args.skip_translate and canonical_transcript_path and canonical_transcript_path.exists():
        source_text = canonical_transcript_path.read_text()
        for model_id in NLLB_MODELS:
            print(f"  translate: {model_id}  -> es, ht")
            stage_results = benchmark_nllb(model_id, source_text, out_dir, targets=["es", "ht"])
            for r in stage_results:
                results.append(r)
                print(f"    -> {r.extras.get('target_lang','?')} ok={r.ok} rtf={r.inference_time_s:.1f}s "
                      f"sps={r.extras.get('sentences_per_sec',0):.2f}")
                if not r.ok:
                    print(f"    ERROR: {r.error}\n{r.extras.get('traceback', '')}", file=sys.stderr)
            save_partial()
            # Pick 1.3B's outputs as canonical for TTS (lighter model, faster)
            if "1.3B" in model_id:
                for r in stage_results:
                    if r.ok:
                        path = PROJECT_ROOT / "tests" / "benchmarks" / out_dir.name / Path(r.extras["output_path"]).name
                        if r.extras.get("target_lang") == "es":
                            es_translation_path = out_dir / Path(r.extras["output_path"]).name
                        if r.extras.get("target_lang") == "ht":
                            ht_translation_path = out_dir / Path(r.extras["output_path"]).name
    elif args.reuse_translate_from:
        cand_es = list(args.reuse_translate_from.glob("translate_*_es.txt"))
        cand_ht = list(args.reuse_translate_from.glob("translate_*_ht.txt"))
        if cand_es: es_translation_path = cand_es[0]
        if cand_ht: ht_translation_path = cand_ht[0]

    # TTS phase ---------------------------------------------------
    if not args.skip_tts:
        try:
            import torch
            tts_devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        except Exception:
            tts_devices = ["cpu"]
        if es_translation_path and es_translation_path.exists():
            for dev in tts_devices:
                print(f"  tts (es/Kokoro): device={dev}")
                r = benchmark_kokoro(es_translation_path, out_dir, dev)
                results.append(r)
                print(f"    -> ok={r.ok}  rtf={r.rtf:.3f}  vram={r.peak_vram_mib:.0f}MiB")
                if not r.ok:
                    print(f"    ERROR: {r.error}\n{r.extras.get('traceback', '')}", file=sys.stderr)
                save_partial()
        if ht_translation_path and ht_translation_path.exists():
            for dev in tts_devices:
                print(f"  tts (ht/MMS): device={dev}")
                r = benchmark_mms_tts(ht_translation_path, out_dir, dev)
                results.append(r)
                print(f"    -> ok={r.ok}  rtf={r.rtf:.3f}  vram={r.peak_vram_mib:.0f}MiB")
                if not r.ok:
                    print(f"    ERROR: {r.error}\n{r.extras.get('traceback', '')}", file=sys.stderr)
                save_partial()

    # Simultaneous-load phase ------------------------------------
    # Run BOTH variants: the production stack (1.3B) AND the quality-target
    # stack (3.3B). User wants to know whether each fits on this GPU.
    if not args.skip_simultaneous:
        for nllb in ("facebook/nllb-200-distilled-1.3B", "facebook/nllb-200-3.3B"):
            print(f"  simultaneous load: Parakeet + {nllb} + Kokoro + MMS-TTS-hat")
            r = benchmark_simultaneous_load(out_dir, nllb_model_id=nllb)
            results.append(r)
            print(f"    -> ok={r.ok}  peak_vram={r.peak_vram_mib:.0f}MiB")
            if not r.ok:
                print(f"    ERROR: {r.error}", file=sys.stderr)
            save_partial()

    # Persist results & emit report -------------------------------
    raw = [asdict(r) for r in results]
    (out_dir / "results.json").write_text(json.dumps({
        "machine": machine_info,
        "audio": str(audio_path),
        "results": raw,
    }, indent=2))
    emit_report(results, out_dir, audio_path, machine_info)
    print(f"\nDone. Report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
