"""
Translation Service

Default: Helsinki-NLP Opus-MT via Transformers (MarianMT), CPU.
Optional: Meta NLLB-200 via Transformers when NLLB_MODEL env var is set
(e.g. facebook/nllb-200-distilled-600M / 1.3B / facebook/nllb-200-3.3B).
"""

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re


# Shared NLLB weights across target languages. NLLB is multilingual — we load
# the model once and each TranslationService instance only keeps its own
# tokenizer + forced_bos_token_id. Keyed by (model_name, device, dtype).
# Value is [model, lock, refcount] (list so refcount is mutable): the lock
# serializes generate() calls so two language pipelines can't hit the same
# GPU kernel concurrently (PyTorch forward/generate on a shared model is not
# thread-safe — 1.3B on ROCm reliably segfaults without this). Refcount
# lets unload() drop the entry when the last user goes, actually freeing
# VRAM — without it the module-level dict kept pinning the model forever.
_NLLB_CACHE: Dict[Tuple[str, str, str], list] = {}
_NLLB_CACHE_LOCK = threading.Lock()


# NLLB uses BCP-47-ish FLORES codes, not our 2-letter internal codes.
NLLB_LANG_CODES = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "ht": "hat_Latn",
    "fr": "fra_Latn",
    "pt": "por_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "ru": "rus_Cyrl",
}


@dataclass
class TranslationResult:
    """Result of translation."""
    source_text: str
    translated_text: str
    source_language: str
    target_language: str
    processing_time: float

    @property
    def is_empty(self) -> bool:
        """Check if translation is empty."""
        return not self.translated_text.strip()


class TranslationService:
    """
    Translation service using Helsinki-NLP Opus-MT models.

    Uses Hugging Face Transformers for model loading and inference.
    Optimized for CPU inference with efficient batching.
    """

    # Model mapping for language pairs
    MODEL_MAP = {
        ('en', 'es'): 'Helsinki-NLP/opus-mt-en-es',
        ('en', 'ht'): 'Helsinki-NLP/opus-mt-en-ht',
        ('en', 'fr'): 'Helsinki-NLP/opus-mt-en-fr',
        ('en', 'de'): 'Helsinki-NLP/opus-mt-en-de',
        ('en', 'pt'): 'Helsinki-NLP/opus-mt-en-pt',
        ('en', 'it'): 'Helsinki-NLP/opus-mt-en-it',
        ('en', 'zh'): 'Helsinki-NLP/opus-mt-en-zh',
        ('en', 'ja'): 'Helsinki-NLP/opus-mt-en-jap',
        ('en', 'ko'): 'Helsinki-NLP/opus-mt-en-ko',
        ('en', 'ar'): 'Helsinki-NLP/opus-mt-en-ar',
        ('en', 'ru'): 'Helsinki-NLP/opus-mt-en-ru',
    }

    def __init__(
        self,
        source_language: str = "en",
        target_language: str = "es",
        model_name: Optional[str] = None,
        device: str = "cpu",
        max_length: int = 512,
        download_root: Optional[str] = None,
    ):
        """
        Initialize the translation service.

        Args:
            source_language: Source language code (e.g., 'en')
            target_language: Target language code (e.g., 'es', 'ht')
            model_name: Explicit model name (overrides language pair lookup)
            device: Device to use ('cpu' or 'cuda')
            max_length: Maximum output length
            download_root: Directory for downloaded models
        """
        self.source_language = source_language
        self.target_language = target_language
        self.max_length = max_length

        # NLLB takes precedence via env var; if set, we use it regardless of
        # any explicit Opus-MT model_name. This keeps the switch a single
        # environment variable with no CLI plumbing through coordinator.
        nllb_env = os.environ.get("NLLB_MODEL", "").strip()
        if nllb_env:
            self._backend = "nllb"
            self._model_name = nllb_env
            self._nllb_src = NLLB_LANG_CODES.get(source_language)
            self._nllb_tgt = NLLB_LANG_CODES.get(target_language)
            if not self._nllb_src or not self._nllb_tgt:
                raise ValueError(
                    f"NLLB: no FLORES code for {source_language} -> {target_language}. "
                    f"Known: {sorted(NLLB_LANG_CODES)}"
                )
        else:
            self._backend = "opus"
            if model_name:
                self._model_name = model_name
            else:
                key = (source_language, target_language)
                if key in self.MODEL_MAP:
                    self._model_name = self.MODEL_MAP[key]
                else:
                    raise ValueError(
                        f"No model found for {source_language} -> {target_language}. "
                        f"Available pairs: {list(self.MODEL_MAP.keys())}"
                    )

        # Device selection:
        #   Opus-MT is fast on CPU and that's the long-standing default.
        #   NLLB (especially 1.3B / 3.3B) is much happier on GPU — we honor
        #   NLLB_DEVICE=cpu|cuda and otherwise auto-detect cuda availability
        #   for NLLB only. Opus-MT stays pinned to CPU.
        if self._backend == "nllb":
            env_dev = os.environ.get("NLLB_DEVICE", "").strip().lower()
            if env_dev == "":
                try:
                    import torch
                    self._device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    self._device = "cpu"
            elif env_dev in ("cpu", "cuda"):
                self._device = env_dev
            else:
                raise ValueError(
                    f"NLLB_DEVICE={env_dev!r} is not recognized. "
                    f"Valid values: 'cpu', 'cuda', or unset for auto-detect."
                )
        else:
            self._device = "cpu"

        # Model storage
        if download_root is None:
            download_root = str(Path(__file__).parent.parent.parent / "models" / "translation")

        self._download_root = download_root
        Path(self._download_root).mkdir(parents=True, exist_ok=True)

        self._model = None
        self._tokenizer = None
        self._loaded = False
        # Serializes generate() calls. For NLLB this is replaced at load()
        # time with the cache's shared lock so two language pipelines can't
        # run generate() concurrently on the same GPU model (that segfaults
        # PyTorch/ROCm). MarianMT uses a per-instance lock that has no
        # contention (each instance owns its own model).
        self._gen_lock: threading.Lock = threading.Lock()

    def load(self) -> None:
        """Load the translation model."""
        if self._loaded:
            return

        print(f"Loading translation model '{self._model_name}' (backend={self._backend})...")

        if self._backend == "nllb":
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            # Tokenizer is small and per-instance (each language pins its own
            # src_lang). Cache it alongside the model for speed on 2nd+ load.
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                cache_dir=self._download_root,
                src_lang=self._nllb_src,
            )
            # Transformers 5.x removed tokenizer.lang_code_to_id; this works
            # across versions.
            self._nllb_tgt_id = self._tokenizer.convert_tokens_to_ids(self._nllb_tgt)

            # bf16 on GPU keeps the 3.3B model inside a 16 GB VRAM budget and
            # is significantly faster on RDNA3.5. On CPU we keep fp32.
            dtype_env = os.environ.get("NLLB_DTYPE", "").strip().lower()
            if dtype_env in ("fp16", "float16"):
                dtype = torch.float16
            elif dtype_env in ("bf16", "bfloat16"):
                dtype = torch.bfloat16
            elif dtype_env in ("fp32", "float32"):
                dtype = torch.float32
            else:
                dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

            cache_key = (self._model_name, self._device, str(dtype))
            self._nllb_cache_key = cache_key  # remembered so unload() can decref
            with _NLLB_CACHE_LOCK:
                cached = _NLLB_CACHE.get(cache_key)
                if cached is not None:
                    # Share weights with an earlier-loaded pipeline (e.g. Spanish).
                    # NLLB routes to the target language at generate() time, and
                    # the shared lock serializes generate() across languages.
                    self._model = cached[0]
                    self._gen_lock = cached[1]
                    cached[2] += 1  # bump refcount
                    print(
                        f"NLLB weights shared from cache (key={cache_key}, "
                        f"refcount={cached[2]}); not reloading to GPU"
                    )
                else:
                    self._model = AutoModelForSeq2SeqLM.from_pretrained(
                        self._model_name,
                        cache_dir=self._download_root,
                        dtype=dtype,
                    )
                    self._model = self._model.to(self._device)
                    self._model.eval()
                    # NLLB's generation_config ships max_length=200, which
                    # conflicts with our per-call max_new_tokens and causes a
                    # warning on every generate(). max_new_tokens is the correct
                    # bound for us (output length scales with input), so drop
                    # max_length from the model default.
                    if hasattr(self._model, "generation_config") and self._model.generation_config is not None:
                        self._model.generation_config.max_length = None
                    self._gen_lock = threading.Lock()
                    _NLLB_CACHE[cache_key] = [self._model, self._gen_lock, 1]
        else:
            from transformers import MarianMTModel, MarianTokenizer
            self._tokenizer = MarianTokenizer.from_pretrained(
                self._model_name,
                cache_dir=self._download_root,
            )
            self._model = MarianMTModel.from_pretrained(
                self._model_name,
                cache_dir=self._download_root,
            )
            self._model = self._model.to(self._device)
            self._model.eval()

        self._loaded = True
        print(f"Translation model loaded: {self.source_language} -> {self.target_language} ({self._backend}, device={self._device})")

    def unload(self) -> None:
        """Unload the model to free memory.

        For NLLB (shared-weight path) we decrement the cache refcount and
        only drop the cache entry — and actually free GPU memory — when the
        last TranslationService using it unloads. Without this, the
        module-level _NLLB_CACHE would pin the model across the process
        lifetime even when both language pipelines have been unloaded.
        """
        if self._backend == "nllb" and self._model is not None:
            key = getattr(self, "_nllb_cache_key", None)
            if key is not None:
                with _NLLB_CACHE_LOCK:
                    cached = _NLLB_CACHE.get(key)
                    if cached is not None:
                        cached[2] -= 1
                        if cached[2] <= 0:
                            del _NLLB_CACHE[key]
            # Drop our own reference either way. If we were the last user
            # the cache entry is gone above; the model is now dereferenced
            # and CUDA memory will be reclaimed on next torch.cuda.empty_cache.
            self._model = None
            self._gen_lock = threading.Lock()  # harmless placeholder
        elif self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        # Best-effort VRAM release when we actually dropped the cache.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @property
    def device(self) -> str:
        """Get the device being used."""
        return self._device

    def _build_gen_kwargs(self, input_token_len: int) -> dict:
        """Build generate() kwargs per backend.

        Opus-MT keeps the original 4-beam search — it's short, cheap, and was
        proven good in prior runs. NLLB on this hardware needs much tighter
        constraints: greedy (beams=1) is 3-4x faster and, combined with
        anti-repetition settings and a bounded max_new_tokens, prevents the
        "Mr. Mr. Mr. ... " hallucination spirals we saw with 3.3B.
        """
        if self._backend == "nllb":
            # Output length bounded by input length. NLLB output can be ~1.3-
            # 1.5x input tokens for en→ht in particular (Creole is verbose),
            # and the SentenceBuffer occasionally emits long hard-timeout
            # spans. 2.5x + 32 floor gives headroom without letting the model
            # run all the way to its old 200-token default.
            max_new = max(32, min(self.max_length, int(input_token_len * 2.5) + 32))
            return dict(
                forced_bos_token_id=self._nllb_tgt_id,
                max_new_tokens=max_new,
                num_beams=1,
                do_sample=False,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )
        # Opus-MT (MarianMT): preserve the original beam-search config.
        return dict(
            max_length=self.max_length,
            num_beams=4,
            early_stopping=True,
        )

    def translate(self, text: str) -> TranslationResult:
        """
        Translate text from source to target language.

        Args:
            text: Text to translate

        Returns:
            TranslationResult with translated text
        """
        if not self._loaded:
            self.load()

        start_time = time.time()

        # Handle empty input
        if not text or not text.strip():
            return TranslationResult(
                source_text=text,
                translated_text="",
                source_language=self.source_language,
                target_language=self.target_language,
                processing_time=0.0,
            )

        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self._device)

        # Generate translation — the lock serializes concurrent calls from
        # different language pipelines that share a model (NLLB case).
        import torch
        gen_kwargs = self._build_gen_kwargs(inputs["input_ids"].shape[-1])
        with torch.no_grad(), self._gen_lock:
            outputs = self._model.generate(**inputs, **gen_kwargs)

        # Warn if NLLB hit its token budget — output was likely truncated.
        # outputs.shape[-1] counts forced-BOS, so compare to max_new_tokens+1.
        if self._backend == "nllb":
            max_new = gen_kwargs.get("max_new_tokens", 0)
            if max_new and outputs.shape[-1] >= max_new + 1:
                print(
                    f"NLLB {self.source_language}->{self.target_language}: "
                    f"output hit max_new_tokens={max_new} — likely truncated "
                    f"(input tokens={inputs['input_ids'].shape[-1]})"
                )

        # Decode
        translated_text = self._tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        processing_time = time.time() - start_time

        return TranslationResult(
            source_text=text,
            translated_text=translated_text,
            source_language=self.source_language,
            target_language=self.target_language,
            processing_time=processing_time,
        )

    def translate_batch(self, texts: List[str]) -> List[TranslationResult]:
        """
        Translate multiple texts efficiently.

        Args:
            texts: List of texts to translate

        Returns:
            List of TranslationResult objects
        """
        if not self._loaded:
            self.load()

        start_time = time.time()

        # Filter empty texts
        non_empty_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        non_empty_texts = [texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            return [
                TranslationResult(
                    source_text=t,
                    translated_text="",
                    source_language=self.source_language,
                    target_language=self.target_language,
                    processing_time=0.0,
                )
                for t in texts
            ]

        # Tokenize batch
        inputs = self._tokenizer(
            non_empty_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self._device)

        # Generate translations — serialized across language pipelines when
        # sharing a model (NLLB).
        import torch
        gen_kwargs = self._build_gen_kwargs(inputs["input_ids"].shape[-1])
        with torch.no_grad(), self._gen_lock:
            outputs = self._model.generate(**inputs, **gen_kwargs)

        # Decode
        translated_texts = self._tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )

        processing_time = time.time() - start_time
        per_item_time = processing_time / len(non_empty_texts)

        # Build results, inserting empty results where needed
        results = []
        translated_idx = 0

        for i, text in enumerate(texts):
            if i in non_empty_indices:
                results.append(TranslationResult(
                    source_text=text,
                    translated_text=translated_texts[translated_idx],
                    source_language=self.source_language,
                    target_language=self.target_language,
                    processing_time=per_item_time,
                ))
                translated_idx += 1
            else:
                results.append(TranslationResult(
                    source_text=text,
                    translated_text="",
                    source_language=self.source_language,
                    target_language=self.target_language,
                    processing_time=0.0,
                ))

        return results

    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
        return False


class MultiTargetTranslator:
    """
    Translates to multiple target languages simultaneously.

    Manages multiple TranslationService instances for efficient
    multi-language translation.
    """

    def __init__(
        self,
        source_language: str = "en",
        target_languages: List[str] = None,
        download_root: Optional[str] = None,
    ):
        """
        Initialize the multi-target translator.

        Args:
            source_language: Source language code
            target_languages: List of target language codes
            download_root: Directory for downloaded models
        """
        self.source_language = source_language
        self.target_languages = target_languages or ['es', 'ht']

        self._services = {}
        self._download_root = download_root

    def load(self) -> None:
        """Load all translation models."""
        for target in self.target_languages:
            if target not in self._services:
                service = TranslationService(
                    source_language=self.source_language,
                    target_language=target,
                    download_root=self._download_root,
                )
                service.load()
                self._services[target] = service

    def unload(self) -> None:
        """Unload all models."""
        for service in self._services.values():
            service.unload()
        self._services.clear()

    def translate(self, text: str) -> dict[str, TranslationResult]:
        """
        Translate text to all target languages.

        Args:
            text: Text to translate

        Returns:
            Dictionary mapping target language to TranslationResult
        """
        results = {}
        for target, service in self._services.items():
            results[target] = service.translate(text)
        return results

    def translate_to(self, text: str, target_language: str) -> TranslationResult:
        """
        Translate text to a specific target language.

        Args:
            text: Text to translate
            target_language: Target language code

        Returns:
            TranslationResult
        """
        if target_language not in self._services:
            raise ValueError(f"Language {target_language} not loaded")
        return self._services[target_language].translate(text)

    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
        return False
