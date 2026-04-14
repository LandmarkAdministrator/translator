"""
Translation Service

Uses Helsinki-NLP Opus-MT models with CTranslate2 for efficient translation.
Falls back to Hugging Face Transformers if CTranslate2 is not available.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import re

# Set GPU environment
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')


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

        # Resolve model name
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

        # Use CPU (CTranslate2 doesn't support ROCm)
        self._device = "cpu"
        if device != "cpu":
            print("Note: Translation using CPU mode (CTranslate2 requires NVIDIA CUDA)")

        # Model storage
        if download_root is None:
            download_root = str(Path(__file__).parent.parent.parent / "models" / "translation")

        self._download_root = download_root
        Path(self._download_root).mkdir(parents=True, exist_ok=True)

        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load the translation model."""
        if self._loaded:
            return

        from transformers import MarianMTModel, MarianTokenizer

        print(f"Loading translation model '{self._model_name}'...")

        self._tokenizer = MarianTokenizer.from_pretrained(
            self._model_name,
            cache_dir=self._download_root,
        )

        self._model = MarianMTModel.from_pretrained(
            self._model_name,
            cache_dir=self._download_root,
        )

        # Move to device
        self._model = self._model.to(self._device)
        self._model.eval()

        self._loaded = True
        print(f"Translation model loaded: {self.source_language} -> {self.target_language}")

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @property
    def device(self) -> str:
        """Get the device being used."""
        return self._device

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

        # Generate translation
        import torch
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True,
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

        # Generate translations
        import torch
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True,
            )

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
