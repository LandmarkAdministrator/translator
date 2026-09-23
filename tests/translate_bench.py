"""Benchmark and compare translation models, across models and devices.

Three questions:

  1. Is a permissively licensed model good enough to replace NLLB-200?
     NLLB-200 and MMS-TTS are CC-BY-NC-4.0, which blocks any commercial
     appliance. M2M-100 (MIT) and MADLAD-400 (Apache 2.0) are the
     candidates. This dumps every model's output side by side so the
     difference can actually be read rather than assumed.

  2. Where does each one run best — CPU or GPU?
     Note that on an AMD box "cuda" IS ROCm: torch maps cuda to HIP, so
     `--device cuda` on the laptop measures ROCm. Real CUDA needs the
     RTX 3060 box. One machine cannot answer CUDA vs ROCm; two can.

  3. Do per-language model instances beat one shared model?
     This is the interesting one. `translation.py` shares NLLB weights
     across languages and serializes generate() behind one lock, because
     concurrent generate() on a shared model segfaults on ROCm. So today
     three languages translate in series. Three SEPARATE smaller models
     have no shared lock and can run in parallel — and 3x M2M-100 418M is
     about the memory of one NLLB-1.3B. `--threads` measures whether that
     trade actually pays.

Usage
-----
    # One model, one device, dump translations and timings
    ./venv/bin/python tests/translate_bench.py --model m2m100-418m \\
        --device cuda --langs es ht ru --sentences 100

    # Concurrency: 3 separate instances vs 1 shared
    ./venv/bin/python tests/translate_bench.py --model m2m100-418m \\
        --device cuda --threads 3

    # Read the quality comparison across finished runs
    ./venv/bin/python tests/translate_bench.py --compare out/*.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pipeline import number_guard      # noqa: E402  (needs the path above)

# English source text: the sermon track's reference transcript, which is
# the register that matters. Falls back to the Jobs transcript.
SOURCE_CANDIDATES = [
    REPO / "tests" / "comparison" / "suite" / "track_sermon.ref.txt",
    REPO / "tests" / "ab_test" / "reference_sermon.txt",
    REPO / "tests" / "comparison" / "suite" / "track_jobs.ref.txt",
]

# Shorthand -> (hf id, family). Family decides how the target language is
# signalled, which differs per architecture and is the usual source of
# silently-wrong output.
MODELS = {
    "nllb-1.3b":        ("facebook/nllb-200-distilled-1.3B", "nllb"),
    "nllb-600m":        ("facebook/nllb-200-distilled-600M", "nllb"),
    "m2m100-418m":      ("facebook/m2m100_418M", "m2m100"),
    "m2m100-1.2b":      ("facebook/m2m100_1.2B", "m2m100"),
    "madlad-3b":        ("google/madlad400-3b-mt", "madlad"),
    "madlad-7b":        ("google/madlad400-7b-mt", "madlad"),
    "aya-101":          ("CohereLabs/aya-101", "aya"),
    # One small model per pair, which is how Opus-MT is actually deployed.
    "opus":             ("Helsinki-NLP/opus-mt-en-{tgt}", "opus"),
    "opus-es":          ("Helsinki-NLP/opus-mt-en-es", "marian"),
    "opus-ht":          ("Helsinki-NLP/opus-mt-en-ht", "marian"),
    "opus-ru":          ("Helsinki-NLP/opus-mt-en-ru", "marian"),
    # General-purpose LLMs, prompted to translate.
    "qwen3-8b":         ("Qwen/Qwen3-8B", "causal"),
    "qwen3.6-27b":      ("Qwen/Qwen3.6-27B", "causal"),
    "mistral-nemo-12b": ("mistralai/Mistral-Nemo-Instruct-2407", "causal"),
}

LICENSES = {
    "nllb": "CC-BY-NC-4.0  (NOT usable commercially)",
    "m2m100": "MIT",
    "madlad": "Apache-2.0",
    "aya": "Apache-2.0",
    "marian": "CC-BY-4.0",
    "opus": "CC-BY-4.0",
    "causal": "see the model card",
}

# Per-model overrides, where the family does not settle it.
MODEL_LICENSES = {
    "qwen3-8b": "Apache-2.0",
    "qwen3.6-27b": "Apache-2.0",
    "mistral-nemo-12b": "Apache-2.0",
}


def license_of(model_key: str, family: str) -> str:
    return MODEL_LICENSES.get(model_key) or LICENSES.get(family, "?")


# Instruction-following models are told the language by name, not by a code.
LANG_NAMES = {"en": "English", "es": "Spanish", "ht": "Haitian Creole",
              "ru": "Russian", "fr": "French", "pt": "Portuguese"}

NLLB_CODES = {"en": "eng_Latn", "es": "spa_Latn", "ht": "hat_Latn",
              "ru": "rus_Cyrl", "fr": "fra_Latn", "pt": "por_Latn"}


SUITE_DIR = REPO / "tests" / "comparison" / "suite"
SUITE_TRACKS = [("jfk", "track1_jfk.ref.txt"), ("sermon", "track_sermon.ref.txt"),
                ("jobs", "track_jobs.ref.txt"), ("singing", "track_singing.ref.txt"),
                ("libri", "track2_libri.ref.txt")]


def _split(text: str, min_words: int, max_words: int = 45) -> list[str]:
    """Sentences, with run-ons cut down.

    The LibriSpeech reference carries no punctuation at all — 1,400 words in
    one block — so anything longer than a spoken sentence is chopped into
    pieces a translator would actually be handed.
    """
    out: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+", " ".join(text.split())):
        words = part.split()
        if len(words) < min_words:
            continue
        if len(words) <= max_words:
            out.append(part.strip())
            continue
        for i in range(0, len(words), 30):
            chunk = words[i:i + 30]
            if len(chunk) >= min_words:
                out.append(" ".join(chunk))
    return out


def _clean(path: Path) -> str:
    lines = [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
             if not ln.startswith("#") and not ln.startswith("---")]
    return " ".join(" ".join(lines).split())


def load_sentences(limit: int, min_words: int = 4,
                   source: str = "sermon") -> tuple[list[str], list[str]]:
    """English to translate, and the track each piece came from.

    `sermon` is the 10-minute sermon transcript: the register that matters but
    only ~114 sentences. `suite` is all five tracks of the 56-minute test
    suite — sermon, JFK, a Stanford commencement, congregational singing and
    read speech — which is four times the material and far more varied.
    """
    if source == "suite":
        pairs: list[tuple[str, str]] = []
        for name, fname in SUITE_TRACKS:
            f = SUITE_DIR / fname
            if not f.is_file():
                continue
            pairs += [(s, name) for s in _split(_clean(f), min_words)]
        if not pairs:
            raise SystemExit(f"no suite transcripts under {SUITE_DIR}")
        if limit > 0:
            pairs = pairs[:limit]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    src = Path(source) if source not in ("sermon", "auto") else None
    if src is None:
        src = next((p for p in SOURCE_CANDIDATES if p.is_file()), None)
    if src is None or not src.is_file():
        raise SystemExit("no source transcript found; looked for:\n  " +
                         "\n  ".join(str(p) for p in SOURCE_CANDIDATES))
    out = _split(_clean(src), min_words)
    if limit > 0:
        out = out[:limit]
    return out, [src.stem] * len(out)


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

class Translator:
    """One loaded model, able to translate into one or more targets.

    Each instance owns its own weights — deliberately, since measuring
    per-language instances against a shared one is half the point.

    Families differ in how the target language is signalled and in which
    model class loads them, which is the usual source of silently-wrong
    output: NLLB and M2M-100 force a language token, MADLAD prefixes the
    input, Aya is told in words, Opus-MT has one model per pair, and a
    general LLM is asked in a chat prompt.
    """

    def __init__(self, hf_id: str, family: str, device: str,
                 dtype: str = "auto", cache_dir: Optional[str] = None,
                 langs: Optional[list[str]] = None, number_guard: bool = False):
        self.hf_id, self.family, self.device = hf_id, family, device
        self.dtype_name = dtype
        self.cache_dir = cache_dir
        self.langs = langs or []
        self.number_guard = number_guard
        self.guard_counts: dict[str, int] = {}
        self.model = None
        self.tokenizer = None
        self.models: dict[str, object] = {}       # opus: one per target
        self.tokenizers: dict[str, object] = {}
        self.load_seconds = 0.0

    # -- loading ---------------------------------------------------------
    def _dtype(self):
        import torch
        if self.dtype_name in ("int8", "int4"):
            return self._compute_dtype()  # weights are quantized; this is the maths
        if self.dtype_name == "bf16":
            return torch.bfloat16
        if self.dtype_name == "fp16":
            return torch.float16
        if self.dtype_name == "fp32":
            return torch.float32
        if self.device != "cuda":
            return torch.float32
        return torch.bfloat16 if self.family in self.T5_FAMILIES else torch.float16

    # MADLAD-400 and Aya-101 are T5 and mT5: trained in bfloat16, they
    # overflow in fp16 and emit one repeated character. Measured on the RTX
    # 3060, 2026-09-22: madlad-3b in fp16 scored a 90% loop rate and lost
    # every number. Anything T5-shaped gets bf16 maths.
    T5_FAMILIES = ("madlad", "aya")

    def _compute_dtype(self):
        import torch
        if self.family in self.T5_FAMILIES:
            return torch.bfloat16
        return torch.float16 if self.dtype_name != "fp32" else torch.float32

    def _quant(self):
        """8-bit / 4-bit weights, so a 13B model fits a 12 GB card."""
        if self.dtype_name not in ("int8", "int4"):
            return None
        import torch
        from transformers import BitsAndBytesConfig
        if self.dtype_name == "int8":
            return BitsAndBytesConfig(load_in_8bit=True)
        skip = None
        if self.family == "aya":
            # Aya-101 is mT5-XXL: a 256k-token vocabulary whose output layer
            # is ~2 GB in fp16. bitsandbytes leaves that layer unquantized by
            # default, which is what pushed a 13B model past a 12 GB card
            # (OOM on the RTX 3060, 2026-09-22). Quantize it too, on both
            # machines, so the two runs stay comparable.
            skip = []
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=self._compute_dtype(),
                                  bnb_4bit_use_double_quant=True,
                                  llm_int8_skip_modules=skip)

    def _load_model(self, hf_id: str, cls):
        kwargs = {"cache_dir": self.cache_dir}
        q = self._quant()
        if q is not None:
            if self.device != "cuda":
                raise RuntimeError("int8/int4 need a GPU (bitsandbytes)")
            kwargs["quantization_config"] = q
            kwargs["device_map"] = {"": 0}
        else:
            kwargs["dtype"] = self._dtype()
        model = cls.from_pretrained(hf_id, **kwargs)
        self._repair_embeddings(model)
        if q is None:
            model = model.to(self.device)
        model = model.eval()
        if getattr(model, "generation_config", None) is not None:
            # NLLB ships max_length=200, which fights max_new_tokens.
            model.generation_config.max_length = None
        return model

    @staticmethod
    def _repair_embeddings(model) -> None:
        """Undo a mis-tied encoder embedding.

        MADLAD-400's checkpoint contains only `decoder.embed_tokens` (the real
        input table) and `lm_head` (the real output head) — no `shared` and no
        `encoder.embed_tokens`. transformers 5 declines to tie what the config
        asks for and fills the encoder's embedding from `lm_head`, so the
        encoder embeds words with the output head and the model emits one
        repeated character. Found 2026-09-22: every MADLAD translation was
        garbage until the encoder was pointed back at the real table. The test
        below is specific enough to leave healthy models alone.
        """
        import torch
        import torch.nn as nn
        enc = getattr(getattr(model, "encoder", None), "embed_tokens", None)
        dec = getattr(getattr(model, "decoder", None), "embed_tokens", None)
        head = getattr(model, "lm_head", None)
        if enc is None or dec is None or head is None:
            return
        try:
            broken = (torch.equal(enc.weight, head.weight)
                      and not torch.equal(enc.weight, dec.weight))
        except Exception:      # quantized or sharded weights: leave it alone
            return
        if not broken:
            return
        emb = dec.weight.data.clone()
        out = head.weight.data.clone()
        shared = nn.Embedding(emb.size(0), emb.size(1))
        shared.weight.data = emb
        model.shared = shared
        model.encoder.embed_tokens = shared
        model.decoder.embed_tokens = shared
        lm = nn.Linear(emb.size(1), emb.size(0), bias=False)
        lm.weight.data = out
        model.lm_head = lm
        print("  [repair] encoder embedding was tied to lm_head; "
              "restored from the checkpoint's own input table")

    def load(self) -> None:
        from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                                  AutoModelForCausalLM)
        t0 = time.perf_counter()
        if self.family == "opus":
            for lang in self.langs:
                hf_id = self.hf_id.format(tgt=lang)
                self.tokenizers[lang] = AutoTokenizer.from_pretrained(
                    hf_id, cache_dir=self.cache_dir)
                self.models[lang] = self._load_model(hf_id, AutoModelForSeq2SeqLM)
        else:
            cls = AutoModelForCausalLM if self.family == "causal" \
                else AutoModelForSeq2SeqLM
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hf_id, cache_dir=self.cache_dir)
            if self.family == "causal" and self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = self._load_model(self.hf_id, cls)
        self.load_seconds = time.perf_counter() - t0

    # -- prompting -------------------------------------------------------
    def _prepare(self, text: str, tgt: str):
        """Family-specific target-language signalling."""
        tok = self.tokenizers.get(tgt, self.tokenizer)
        if self.family == "nllb":
            tok.src_lang = NLLB_CODES["en"]
            enc = tok(text, return_tensors="pt")
            bos = tok.convert_tokens_to_ids(NLLB_CODES[tgt])
            return enc, {"forced_bos_token_id": bos}
        if self.family == "m2m100":
            tok.src_lang = "en"
            enc = tok(text, return_tensors="pt")
            return enc, {"forced_bos_token_id": tok.get_lang_id(tgt)}
        if self.family == "madlad":
            # T5-style: the target is a prefix token in the input itself.
            return tok(f"<2{tgt}> {text}", return_tensors="pt"), {}
        if self.family == "aya":
            # The model card's own format: "Translate to X: <text>".
            return tok(f"Translate to {LANG_NAMES[tgt]}: {text}",
                       return_tensors="pt"), {}
        # Marian and Opus are one model per pair; the target is implicit.
        return tok(text, return_tensors="pt"), {}

    _THINK = re.compile(r"<think>.*?</think>", re.S)

    def _chat_prompt(self, text: str, tgt: str) -> str:
        msgs = [{"role": "user", "content":
                 f"Translate the following English sentence into "
                 f"{LANG_NAMES[tgt]}. Reply with the translation only, "
                 f"with no explanation, no quotes and no notes.\n\n{text}"}]
        try:      # Qwen models reason unless told not to; that is not a translator
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

    def _clean(self, out: str) -> str:
        out = self._THINK.sub("", out).strip()
        for line in out.splitlines():
            line = line.strip().strip('"').strip()
            if line:
                return line
        return ""

    # -- translating -----------------------------------------------------
    def translate(self, text: str, tgt: str, max_new_tokens: int = 256) -> str:
        """Translate, optionally protecting the numbers (see number_guard)."""
        if not self.number_guard:
            return self._translate_raw(text, tgt, max_new_tokens)
        out, how = number_guard.guard(
            text, lambda t: self._translate_raw(t, tgt, max_new_tokens), tgt)
        self.guard_counts[how] = self.guard_counts.get(how, 0) + 1
        return out

    def _translate_raw(self, text: str, tgt: str, max_new_tokens: int = 256) -> str:
        import torch
        if self.family == "causal":
            prompt = self._chat_prompt(text, tgt)
            enc = self.tokenizer(prompt, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id)
            new = out[0][enc["input_ids"].shape[-1]:]
            return self._clean(self.tokenizer.decode(new, skip_special_tokens=True))
        model = self.models.get(tgt, self.model)
        enc, extra = self._prepare(text, tgt)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 num_beams=1, **extra)
        tok = self.tokenizers.get(tgt, self.tokenizer)
        return tok.batch_decode(out, skip_special_tokens=True)[0].strip()

    def unload(self) -> None:
        import torch
        self.model = None
        self.tokenizer = None
        self.models.clear()
        self.tokenizers.clear()
        if self.device == "cuda":
            torch.cuda.empty_cache()


TRACKS: list[str] = []      # filled by execute(); one entry per sentence


# --------------------------------------------------------------------------
# Quality signals (no reference translation needed)
# --------------------------------------------------------------------------

_DIGITS = re.compile(r"\d+")


def quality_signals(src: str, out: str) -> dict:
    """Cheap checks that catch the ways MT actually fails in production.

    None of these need a gold translation. They catch the failures that
    matter for sermons: dropped scripture references, repetition spirals,
    and the model simply echoing English back.
    """
    src_nums = _DIGITS.findall(src)
    out_nums = _DIGITS.findall(out)
    sig = {
        "empty": not out,
        # Every number in the source should survive. A dropped chapter or
        # verse is the single most visible MT failure in this domain.
        "numbers_in": len(src_nums),
        "numbers_kept": sum(1 for n in src_nums if n in out_nums),
        "numbers_lost": max(0, len(src_nums) - sum(1 for n in src_nums if n in out_nums)),
        # Output identical to input = the model did not translate.
        "copied_source": out.strip().lower() == src.strip().lower(),
        "len_ratio": round(len(out) / len(src), 3) if src else 0.0,
    }
    try:
        from pipeline.text_filters import looks_like_loop
        sig["looks_like_loop"] = bool(looks_like_loop(out))
    except Exception:
        # Fallback: the same token repeated 4+ times running.
        words = out.lower().split()
        sig["looks_like_loop"] = any(
            len(set(words[i:i + 4])) == 1 for i in range(max(0, len(words) - 3)))
    return sig


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------

@dataclass
class Run:
    model: str
    hf_id: str
    family: str
    device: str
    dtype: str
    langs: list[str]
    threads: int
    sentences: int
    source: str = "sermon"
    load_seconds: float = 0.0
    wall_seconds: float = 0.0
    per_call: list[float] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    peak_vram_mb: Optional[float] = None
    error: Optional[str] = None
    number_guard: bool = False
    guard_counts: dict = field(default_factory=dict)


def _peak_vram_mb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    except Exception:
        pass
    return None


def run_serial(tr: Translator, sents: list[str], langs: list[str],
               run: Run) -> None:
    """One model, every language in turn — today's shape."""
    t0 = time.perf_counter()
    for i, s in enumerate(sents):
        for lang in langs:
            c0 = time.perf_counter()
            try:
                out = tr.translate(s, lang)
            except Exception as e:  # noqa: BLE001
                out = f"<<ERROR: {e}>>"
            dt = time.perf_counter() - c0
            run.per_call.append(dt)
            run.rows.append({"i": i, "lang": lang, "src": s, "out": out,
                             "seconds": round(dt, 4),
                             **quality_signals(s, out)})
    run.wall_seconds = time.perf_counter() - t0


def run_threaded(model_key: str, hf_id: str, family: str, device: str,
                 sents: list[str], langs: list[str], run: Run,
                 cache_dir: Optional[str], dtype: str = "auto") -> None:
    """One SEPARATE model instance per language, all at once.

    The comparison that matters for language count: no shared weights, so
    no shared lock, so the languages can genuinely overlap.
    """
    translators: dict[str, Translator] = {}
    load_total = 0.0
    for lang in langs:
        tr = Translator(hf_id, family, device, dtype=dtype,
                        cache_dir=cache_dir, langs=[lang])
        tr.load()
        load_total += tr.load_seconds
        translators[lang] = tr
    run.load_seconds = load_total

    lock = threading.Lock()
    t0 = time.perf_counter()

    def worker(lang: str) -> None:
        tr = translators[lang]
        for i, s in enumerate(sents):
            c0 = time.perf_counter()
            try:
                out = tr.translate(s, lang)
            except Exception as e:  # noqa: BLE001
                out = f"<<ERROR: {e}>>"
            dt = time.perf_counter() - c0
            with lock:
                run.per_call.append(dt)
                run.rows.append({"i": i, "lang": lang, "src": s, "out": out,
                                 "track": TRACKS[i] if i < len(TRACKS) else "?",
                                 "seconds": round(dt, 4),
                                 **quality_signals(s, out)})

    threads = [threading.Thread(target=worker, args=(l,), name=f"tr-{l}")
               for l in langs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    run.wall_seconds = time.perf_counter() - t0
    run.peak_vram_mb = _peak_vram_mb()
    for tr in translators.values():
        tr.unload()


def execute(model_key: str, device: str, langs: list[str], n_sentences: int,
            threaded: bool, cache_dir: Optional[str], dtype: str = "auto",
            source: str = "sermon", number_guard_on: bool = False) -> Run:
    hf_id, family = MODELS.get(model_key, (model_key, "m2m100"))
    sents, tracks = load_sentences(n_sentences, source=source)
    TRACKS.clear(); TRACKS.extend(tracks)
    run = Run(model=model_key, hf_id=hf_id, family=family, device=device,
              dtype=dtype, langs=langs,
              threads=len(langs) if threaded else 1, sentences=len(sents))
    run.source = source
    run.number_guard = number_guard_on
    try:
        import torch
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        if threaded:
            run_threaded(model_key, hf_id, family, device, sents, langs,
                         run, cache_dir, dtype)
        else:
            tr = Translator(hf_id, family, device, dtype=dtype,
                            cache_dir=cache_dir, langs=langs,
                            number_guard=number_guard_on)
            tr.load()
            run.load_seconds = tr.load_seconds
            run_serial(tr, sents, langs, run)
            run.guard_counts = dict(tr.guard_counts)
            run.peak_vram_mb = _peak_vram_mb()
            tr.unload()
    except Exception as e:  # noqa: BLE001
        run.error = f"{type(e).__name__}: {e}"
    return run


def run_to_dict(run: Run) -> dict:
    calls = run.per_call
    d = {
        "model": run.model, "hf_id": run.hf_id, "family": run.family,
        "license": license_of(run.model, run.family),
        "device": run.device, "dtype": run.dtype, "langs": run.langs, "threads": run.threads,
        "sentences": run.sentences, "translations": len(run.rows),
        "source": run.source, "number_guard": run.number_guard,
        "guard_counts": run.guard_counts,
        "load_seconds": round(run.load_seconds, 2),
        "wall_seconds": round(run.wall_seconds, 2),
        "peak_vram_mb": run.peak_vram_mb,
        "error": run.error,
        "host": platform.node(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if calls:
        s = sorted(calls)
        d["per_call"] = {
            "mean_s": round(statistics.fmean(calls), 4),
            "p50_s": round(s[len(s) // 2], 4),
            "p95_s": round(s[min(len(s) - 1, int(len(s) * 0.95))], 4),
            "max_s": round(max(calls), 4),
        }
        d["throughput_per_s"] = round(len(calls) / run.wall_seconds, 3) \
            if run.wall_seconds else 0.0
    if run.rows:
        n = len(run.rows)
        d["quality"] = {
            "empty_rate": round(sum(r["empty"] for r in run.rows) / n, 4),
            "copied_source_rate": round(
                sum(r["copied_source"] for r in run.rows) / n, 4),
            "loop_rate": round(sum(r["looks_like_loop"] for r in run.rows) / n, 4),
            "numbers_total": sum(r["numbers_in"] for r in run.rows),
            "numbers_lost": sum(r["numbers_lost"] for r in run.rows),
            "number_retention": round(
                1 - sum(r["numbers_lost"] for r in run.rows) /
                max(1, sum(r["numbers_in"] for r in run.rows)), 4),
            "len_ratio_mean": round(
                statistics.fmean([r["len_ratio"] for r in run.rows]), 3),
        }
        d["rows"] = run.rows
    return d


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

def compare(paths: list[Path], samples: int = 8) -> None:
    runs, gpu_of = [], {}
    for p in paths:
        files = [p] if p.is_file() else sorted(p.glob("*.json"))
        for f in files:
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"skip {f}: {e}")
                continue
            if not isinstance(d, dict):
                continue
            if "model" in d and "device" in d:
                runs.append(d)
            elif "torch" in d and "machine" in d:
                # device_report.json sits beside the runs: it names the GPU
                # that "cuda" meant on that machine.
                devs = d["torch"].get("devices") or []
                name = (devs[0].get("name") if isinstance(devs[0], dict) else str(devs[0])) if devs else None
                gpu_of[d["machine"].get("hostname", "?")] = name or d["torch"].get("flavor", "?")
    if not runs:
        print("no runs found")
        return

    # Results from several machines are reported together, so every lookup
    # below is keyed by host as well as model, device and thread count —
    # otherwise one machine's "cuda" run silently stands in for the other's.
    hosts = sorted({r.get("host", "?") for r in runs})
    multi = len(hosts) > 1
    hw = (max(len(h) for h in hosts) + 2) if multi else 0
    def H(r): return r.get("host", "?")
    def hc(r): return f"{H(r):<{hw}}" if multi else ""
    hh = f"{'host':<{hw}}" if multi else ""
    width = 104 + hw
    key = lambda r: (r["model"], r["device"], r["threads"], H(r))
    ok = [r for r in runs if not r.get("error")]
    models = sorted({r["model"] for r in ok})
    def find(host, model, device, threaded):
        return next((r for r in ok if H(r) == host and r["model"] == model
                     and r["device"] == device
                     and ((r["threads"] > 1) if threaded else (r["threads"] == 1))), None)

    print("=" * width)
    print("TRANSLATION BENCHMARK")
    for h in hosts:
        print(f"  {h}: cuda = {gpu_of.get(h, '?')}")
    print("=" * width)
    print(f"{hh}{'model':<18}{'prec':<6}{'device':<8}{'thr':<5}{'load':>7}{'wall':>8}"
          f"{'mean':>8}{'p95':>8}{'tr/s':>8}{'VRAM':>9}  license")
    print("-" * width)
    for r in sorted(runs, key=key):
        if r.get("error"):
            print(f"{hc(r)}{r['model']:<18}{r.get('dtype', '?'):<6}"
                  f"{r['device']:<8}{r['threads']:<5}  ERROR: {str(r['error'])[:70]}")
            continue
        pc = r.get("per_call", {})
        print(f"{hc(r)}{r['model']:<18}{r.get('dtype', '?'):<6}"
              f"{r['device']:<8}{r['threads']:<5}"
              f"{r['load_seconds']:>7.1f}{r['wall_seconds']:>8.1f}"
              f"{pc.get('mean_s', 0):>8.3f}{pc.get('p95_s', 0):>8.3f}"
              f"{r.get('throughput_per_s', 0):>8.2f}"
              f"{(str(r.get('peak_vram_mb')) + 'M') if r.get('peak_vram_mb') else '—':>9}"
              f"  {r.get('license', '?')}")

    print()
    print("QUALITY SIGNALS (no gold reference — these catch real failures)")
    print("-" * width)
    print(f"{hh}{'model':<18}{'prec':<6}{'device':<8}{'thr':<5}{'empty':>8}{'copied':>9}"
          f"{'loops':>8}{'num kept':>10}{'len ratio':>11}")
    for r in sorted(runs, key=key):
        q = r.get("quality")
        if not q:
            continue
        print(f"{hc(r)}{r['model']:<18}{r.get('dtype', '?'):<6}"
              f"{r['device']:<8}{r['threads']:<5}"
              f"{q['empty_rate']:>8.1%}{q['copied_source_rate']:>9.1%}"
              f"{q['loop_rate']:>8.1%}{q['number_retention']:>10.1%}"
              f"{q['len_ratio_mean']:>11.2f}")

    print()
    print("WHERE EACH MODEL RUNS BEST")
    print("-" * width)
    for host in hosts:
        if multi:
            print(f"  [{host}]")
        for model in models:
            cpu, gpu = find(host, model, "cpu", False), find(host, model, "cuda", False)
            if cpu and gpu:
                cpu_t = cpu.get("per_call", {}).get("mean_s", 0)
                gpu_t = gpu.get("per_call", {}).get("mean_s", 0)
                if gpu_t:
                    note = ("GPU barely helps — run it on CPU and keep the GPU free"
                            if cpu_t / gpu_t < 2.0 else "GPU is worth it")
                    print(f"  {model:<18} CPU {cpu_t:.3f}s vs GPU {gpu_t:.3f}s "
                          f"= {cpu_t / gpu_t:.1f}x  -> {note}")
            else:
                have = ", ".join(n for n, r in (("cpu", cpu), ("cuda", gpu)) if r) or "nothing"
                print(f"  {model:<18} only measured on: {have}")

    print()
    print("SHARED MODEL vs PER-LANGUAGE INSTANCES")
    print("-" * width)
    for host in hosts:
        if multi:
            print(f"  [{host}]")
        for model in models:
            for device in ("cpu", "cuda"):
                one, many = find(host, model, device, False), find(host, model, device, True)
                if one and many and one["wall_seconds"] and many["wall_seconds"]:
                    # Both did the same number of translations.
                    speedup = one["wall_seconds"] / many["wall_seconds"]
                    print(f"  {model:<18} {device:<6} serial {one['wall_seconds']:.1f}s "
                          f"vs {many['threads']} instances {many['wall_seconds']:.1f}s "
                          f"= {speedup:.2f}x"
                          f"{'  <- parallelism pays' if speedup > 1.4 else ''}")
                    if many.get("peak_vram_mb") and one.get("peak_vram_mb"):
                        print(f"                 VRAM {one['peak_vram_mb']}M -> "
                              f"{many['peak_vram_mb']}M")

    if multi:
        print()
        print("SAME CONFIGURATION ON EACH MACHINE (mean seconds per translation, lower is faster)")
        print("-" * width)
        print(f"  {'model':<18}{'device':<8}{'thr':<5}" + "".join(f"{h[:16]:>18}" for h in hosts))
        for model in models:
            for device in ("cpu", "cuda"):
                for threaded in (False, True):
                    cells = [find(h, model, device, threaded) for h in hosts]
                    if sum(c is not None for c in cells) < 2:
                        continue
                    thr = next(c["threads"] for c in cells if c)
                    print(f"  {model:<18}{device:<8}{thr:<5}" + "".join(
                        f"{c['per_call']['mean_s']:>18.3f}" if c and c.get("per_call")
                        else f"{'—':>18}" for c in cells))

    # Side-by-side output, for reading: one run per model (the first
    # machine's GPU run where there is one), not every run of it.
    print()
    print("SIDE BY SIDE (read these — the metrics above cannot judge fluency)")
    print("=" * width)
    pick: dict[str, dict] = {}
    for r in sorted(ok, key=lambda r: (H(r) != hosts[0], r["device"] != "cuda")):
        if r.get("rows") and r["threads"] == 1:
            pick.setdefault(r["model"], r)
    with_rows = [pick[m] for m in sorted(pick)]
    if not with_rows:
        return
    print("  from: " + ", ".join(f"{r['model']} ({H(r)}, {r['device']})" for r in with_rows))
    langs = sorted({row["lang"] for r in with_rows for row in r["rows"]})
    for lang in langs:
        print(f"\n### {lang.upper()}")
        base = with_rows[0]
        idx = sorted({row["i"] for row in base["rows"] if row["lang"] == lang})
        step = max(1, len(idx) // samples)
        for i in idx[::step][:samples]:
            src = next((row["src"] for row in base["rows"]
                        if row["i"] == i and row["lang"] == lang), None)
            if not src:
                continue
            print(f"\n  EN   {src[:150]}")
            for r in with_rows:
                out = next((row["out"] for row in r["rows"]
                            if row["i"] == i and row["lang"] == lang), None)
                if out is not None:
                    print(f"  {r['model'][:16]:<16} {out[:150]}")


# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="translate_bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="m2m100-418m",
                   help=f"one of {', '.join(MODELS)} or a raw HF id")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                   help="on AMD, 'cuda' is ROCm via HIP")
    p.add_argument("--langs", nargs="+", default=["es", "ht", "ru"])
    p.add_argument("--dtype", default="auto",
                   choices=("auto", "fp16", "bf16", "fp32", "int8", "int4"),
                   help="auto = fp16 on GPU, fp32 on CPU; int8/int4 need a GPU "
                        "and bitsandbytes, and are how a 13B model fits 12 GB")
    p.add_argument("--sentences", type=int, default=60,
                   help="0 or less means every sentence in the source")
    p.add_argument("--number-guard", action="store_true",
                   help="protect numbers across translation (src/pipeline/number_guard.py)")
    p.add_argument("--source", default="sermon",
                   help="'sermon' (the 10-minute transcript), 'suite' (all five "
                        "tracks of the 56-minute test suite), or a path")
    p.add_argument("--threads", action="store_true",
                   help="one model instance per language, run concurrently")
    p.add_argument("--out", type=Path, default=REPO / "benchmarks" / "translate")
    p.add_argument("--label", default=None)
    p.add_argument("--cache-dir", default=str(REPO / "models" / "translation"))
    p.add_argument("--compare", nargs="+", type=Path, default=None)
    p.add_argument("--samples", type=int, default=8)
    args = p.parse_args(argv)

    if args.compare:
        compare(args.compare, args.samples)
        return 0

    print(f"model   : {args.model} -> {MODELS.get(args.model, (args.model,))[0]}")
    print(f"device  : {args.device}"
          f"{'   (on AMD this is ROCm)' if args.device == 'cuda' else ''}")
    print(f"langs   : {' '.join(args.langs)}")
    print(f"mode    : {'per-language instances' if args.threads else 'one shared model'}")
    print(f"dtype   : {args.dtype}")
    print(f"source  : {args.source}")
    print(f"guard   : {'on' if args.number_guard else 'off'}")
    print(f"sentences: {args.sentences}")

    run = execute(args.model, args.device, args.langs, args.sentences,
                  args.threads, args.cache_dir, args.dtype, args.source,
                  args.number_guard)
    d = run_to_dict(run)

    args.out.mkdir(parents=True, exist_ok=True)
    label = args.label or (f"{args.model}-{args.device}"
                           f"{'-x' + str(len(args.langs)) if args.threads else ''}")
    path = args.out / f"{label}.json"
    path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

    if run.error:
        print(f"\nERROR: {run.error}")
        return 1
    pc = d.get("per_call", {})
    q = d.get("quality", {})
    print(f"\n  load      {d['load_seconds']}s")
    print(f"  wall      {d['wall_seconds']}s for {d['translations']} translations")
    print(f"  per call  mean {pc.get('mean_s')}s  p95 {pc.get('p95_s')}s")
    print(f"  vram      {d.get('peak_vram_mb')} MB")
    print(f"  quality   numbers kept {q.get('number_retention', 0):.1%}, "
          f"loops {q.get('loop_rate', 0):.1%}, "
          f"copied {q.get('copied_source_rate', 0):.1%}")
    print(f"\n  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
