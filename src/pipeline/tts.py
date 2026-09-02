"""
TTS (Text-to-Speech) Service

Default backend: Piper (ONNX, fast neural VITS) — one voice per language.
Optional MMS backend: Meta's facebook/mms-tts-<lang> (VITS via transformers),
enabled per-language by env var e.g. `HT_TTS=mms` to use mms-tts-hat for
Haitian Creole. MMS is the only viable natively-trained Haitian voice —
Piper has no ht model, so the default Piper wiring falls back to French.
"""

import os
import time
import wave
import io
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import numpy as np


# MMS-TTS model ids by our 2-letter language code.
MMS_MODELS = {
    "ht": "facebook/mms-tts-hat",
    "es": "facebook/mms-tts-spa",
    "fr": "facebook/mms-tts-fra",
}


# Kokoro 82M (`hexgrad/Kokoro-82M`) per-language config:
# (lang_code passed to KPipeline, voice id, output sample rate).
# CPU-only — Kokoro relies on PyTorch LSTM kernels that don't have a ROCm
# implementation today (would silently spend minutes recompiling MIOpen
# kernels on the AMD iGPU). Validated CPU RTF ~7.95x on Ryzen AI 9 HX 370
# in the sister batch project.
KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_VOICES = {
    # Spanish: 'em_alex' is a neutral male voice; the leading 'e' is the
    # Kokoro language code for Spanish.
    "es": ("e", "em_alex", 24000),
}


@dataclass
class SpeechResult:
    """Result of TTS synthesis."""
    text: str
    audio: np.ndarray
    sample_rate: int
    language: str
    processing_time: float

    @property
    def duration(self) -> float:
        """Duration of audio in seconds."""
        return len(self.audio) / self.sample_rate

    @property
    def is_empty(self) -> bool:
        """Check if audio is empty."""
        return len(self.audio) == 0


class TTSService:
    """
    Text-to-speech service using Piper.

    Piper provides fast, high-quality neural TTS using ONNX Runtime.
    Supports multiple voices and languages.
    """

    # Voice models for different languages
    # Format: (language_code, voice_name): (model_name, sample_rate)
    VOICE_MAP = {
        ('es', 'default'): ('es_ES-davefx-medium', 22050),
        ('es', 'davefx'): ('es_ES-davefx-medium', 22050),
        ('es', 'sharvard'): ('es_ES-sharvard-medium', 22050),
        ('es', 'mls'): ('es_ES-mls_10246-low', 16000),
        ('ht', 'default'): ('fr_FR-upmc-medium', 22050),  # Fallback to French
        ('fr', 'default'): ('fr_FR-upmc-medium', 22050),
        ('fr', 'upmc'): ('fr_FR-upmc-medium', 22050),
        ('en', 'default'): ('en_US-lessac-medium', 22050),
        ('en', 'lessac'): ('en_US-lessac-medium', 22050),
        ('en', 'amy'): ('en_US-amy-medium', 22050),
    }

    def __init__(
        self,
        language: str = "es",
        voice: str = "default",
        model_path: Optional[str] = None,
        sample_rate: int = 22050,
        speed: float = 1.0,
        download_root: Optional[str] = None,
    ):
        """
        Initialize the TTS service.

        Args:
            language: Target language code (e.g., 'es', 'ht', 'fr')
            voice: Voice name or 'default'
            model_path: Explicit path to model file (overrides voice lookup)
            sample_rate: Output sample rate
            speed: Speech speed multiplier (1.0 = normal)
            download_root: Directory for downloaded models
        """
        self.language = language
        self.voice = voice
        self.speed = speed
        self._sample_rate = sample_rate

        # Model storage
        if download_root is None:
            download_root = str(Path(__file__).parent.parent.parent / "models" / "tts")

        self._download_root = Path(download_root)
        self._download_root.mkdir(parents=True, exist_ok=True)

        # Per-language TTS backend selection via env var `{LANG_UPPER}_TTS`:
        #   "mms"    → facebook/mms-tts-<lang> (Meta's multilingual VITS)
        #   "kokoro" → hexgrad/Kokoro-82M (English/Spanish, CPU-only, very fast)
        #   unset    → Piper (default, ONNX, CPU-only)
        # Sample rate is read from the model config at load() time; we init
        # with a plausible default here.
        env_var = f"{language.upper()}_TTS"
        backend_choice = os.environ.get(env_var, "").strip().lower()
        self._use_mms = (backend_choice == "mms")
        self._use_kokoro = (backend_choice == "kokoro")

        if self._use_mms:
            if language not in MMS_MODELS:
                raise ValueError(
                    f"MMS-TTS requested for '{language}' but no model id is known. "
                    f"Known: {sorted(MMS_MODELS)}"
                )
            self._model_name = MMS_MODELS[language]
            self._model_path = None
            # Typical MMS-TTS sampling rate; overridden in load() from config.
            self._sample_rate = 16000
        elif self._use_kokoro:
            if language not in KOKORO_VOICES:
                raise ValueError(
                    f"Kokoro requested for '{language}' but no voice is configured. "
                    f"Known: {sorted(KOKORO_VOICES)}"
                )
            kk_lang_code, kk_voice, kk_sr = KOKORO_VOICES[language]
            self._kokoro_lang_code = kk_lang_code
            self._kokoro_voice = kk_voice
            self._sample_rate = kk_sr
            self._model_name = f"{KOKORO_REPO}:{kk_voice}"
            self._model_path = None
        elif model_path:
            self._model_path = Path(model_path)
            self._model_name = self._model_path.stem
        else:
            key = (language, voice)
            if key in self.VOICE_MAP:
                self._model_name, self._sample_rate = self.VOICE_MAP[key]
            else:
                # Try default for language
                default_key = (language, 'default')
                if default_key in self.VOICE_MAP:
                    self._model_name, self._sample_rate = self.VOICE_MAP[default_key]
                else:
                    raise ValueError(
                        f"No voice found for language '{language}'. "
                        f"Available: {list(set(k[0] for k in self.VOICE_MAP.keys()))}"
                    )
            self._model_path = None

        self._voice = None
        # MMS-specific handles
        self._mms_model = None
        self._mms_tokenizer = None
        self._mms_device = "cpu"
        # Kokoro-specific handle (KPipeline)
        self._kokoro_pipeline = None
        self._kokoro_device = "cpu"
        self._loaded = False

    def _get_model_path(self) -> Path:
        """Get the path to the model file, downloading if needed."""
        if self._model_path:
            return self._model_path

        model_dir = self._download_root / self._model_name
        model_file = model_dir / f"{self._model_name}.onnx"
        config_file = model_dir / f"{self._model_name}.onnx.json"

        if model_file.exists() and config_file.exists():
            return model_file

        # Download model using piper command
        print(f"Downloading TTS model '{self._model_name}'...")
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Use piper-tts to download the model
            result = subprocess.run(
                [
                    "piper",
                    "--model", self._model_name,
                    "--download-dir", str(self._download_root),
                    "--update-voices",
                ],
                capture_output=True,
                text=True,
                input="",  # Empty input to just trigger download
                timeout=300,
            )

            # Check if model was downloaded
            if model_file.exists():
                print(f"Model downloaded: {model_file}")
                return model_file

        except subprocess.TimeoutExpired:
            print("Model download timed out")
        except Exception as e:
            print(f"Error downloading model: {e}")

        # If piper download failed, try direct download
        return self._download_model_direct()

    def _download_model_direct(self) -> Path:
        """Download model directly from Hugging Face."""
        import urllib.request

        model_dir = self._download_root / self._model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        model_file = model_dir / f"{self._model_name}.onnx"
        config_file = model_dir / f"{self._model_name}.onnx.json"

        # Parse model name to construct URL
        # Format: lang_REGION-speaker-quality
        parts = self._model_name.split('-')
        if len(parts) >= 3:
            lang_region = parts[0]  # e.g., es_ES
            lang = lang_region.split('_')[0]  # e.g., es

            base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang}/{lang_region}"

            # Try to download
            onnx_url = f"{base_url}/{parts[1]}/{parts[2]}/{self._model_name}.onnx"
            json_url = f"{base_url}/{parts[1]}/{parts[2]}/{self._model_name}.onnx.json"

            try:
                print(f"Downloading from {onnx_url}...")
                urllib.request.urlretrieve(onnx_url, model_file)
                urllib.request.urlretrieve(json_url, config_file)
                print("Download complete")
                return model_file
            except Exception as e:
                print(f"Direct download failed: {e}")

        raise RuntimeError(f"Could not download model '{self._model_name}'")

    def load(self) -> None:
        """Load the TTS model."""
        if self._loaded:
            return

        if self._use_mms:
            import torch
            from transformers import VitsModel, AutoTokenizer
            # Keep MMS models alongside other TTS assets so they share the
            # gitignore and backup story.
            mms_cache = str(self._download_root / "mms")
            Path(mms_cache).mkdir(parents=True, exist_ok=True)
            # Device: env var override, else auto (cuda when available).
            env_dev = os.environ.get("MMS_DEVICE", "").strip().lower()
            if env_dev in ("cpu", "cuda"):
                mms_device = env_dev
            else:
                mms_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._mms_device = mms_device
            print(f"Loading TTS model '{self._model_name}' (MMS/VITS, device={mms_device})...")
            self._mms_tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, cache_dir=mms_cache
            )
            self._mms_model = VitsModel.from_pretrained(
                self._model_name, cache_dir=mms_cache
            ).to(mms_device)
            self._mms_model.eval()
            self._sample_rate = int(self._mms_model.config.sampling_rate)
            self._loaded = True
            print(f"TTS model loaded: {self.language} ({self._model_name}) @ {self._sample_rate}Hz [MMS, {mms_device}]")
            return

        if self._use_kokoro:
            try:
                from kokoro import KPipeline
            except ImportError as e:
                raise RuntimeError(
                    "Kokoro requested but the `kokoro` package is not installed. "
                    "Install with: pip install --no-deps kokoro misaki num2words "
                    "espeakng-loader phonemizer-fork (full deps tree breaks on "
                    "Python 3.13 because misaki[en] pins numpy==1.26.4)."
                ) from e
            import torch
            kokoro_cache = str(self._download_root / "kokoro")
            Path(kokoro_cache).mkdir(parents=True, exist_ok=True)
            # Kokoro pulls its weights from HF the first time; cache_dir is
            # honored via the HF_HOME env var. Set if caller didn't already.
            os.environ.setdefault("HF_HOME", kokoro_cache)
            # Device: env var override, else auto. Kokoro's LSTM kernels have
            # no ROCm implementation today (silently recompiles MIOpen on AMD),
            # so on AMD GPU systems force CPU by setting KOKORO_DEVICE=cpu.
            env_dev = os.environ.get("KOKORO_DEVICE", "").strip().lower()
            if env_dev in ("cpu", "cuda"):
                kokoro_device = env_dev
            else:
                kokoro_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._kokoro_device = kokoro_device
            print(
                f"Loading TTS model '{KOKORO_REPO}' voice={self._kokoro_voice} "
                f"lang={self._kokoro_lang_code} (Kokoro on {kokoro_device})..."
            )
            self._kokoro_pipeline = KPipeline(
                lang_code=self._kokoro_lang_code,
                device=kokoro_device,
            )
            self._loaded = True
            print(
                f"TTS model loaded: {self.language} ({KOKORO_REPO}:{self._kokoro_voice}) "
                f"@ {self._sample_rate}Hz [Kokoro, {kokoro_device}]"
            )
            return

        from piper import PiperVoice

        model_path = self._get_model_path()
        config_path = model_path.with_suffix('.onnx.json')

        print(f"Loading TTS model '{self._model_name}'...")

        self._voice = PiperVoice.load(
            str(model_path),
            config_path=str(config_path) if config_path.exists() else None,
        )

        self._loaded = True
        print(f"TTS model loaded: {self.language} ({self._model_name})")

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._voice is not None:
            del self._voice
            self._voice = None
        if self._mms_model is not None:
            del self._mms_model
            self._mms_model = None
        if self._mms_tokenizer is not None:
            del self._mms_tokenizer
            self._mms_tokenizer = None
        if self._kokoro_pipeline is not None:
            del self._kokoro_pipeline
            self._kokoro_pipeline = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @property
    def sample_rate(self) -> int:
        """Get the output sample rate."""
        return self._sample_rate

    def synthesize(self, text: str, speed: Optional[float] = None) -> SpeechResult:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            speed: Optional per-call speed multiplier (adaptive playback:
                   the coordinator passes >1.0 when the playback queue backs
                   up so live audio catches back up to the speaker). Falls
                   back to the service-level default.

        Returns:
            SpeechResult with audio data
        """
        if not self._loaded:
            self.load()

        effective_speed = speed if speed and speed > 0 else self.speed
        start_time = time.time()

        # Handle empty input
        if not text or not text.strip():
            return SpeechResult(
                text=text,
                audio=np.array([], dtype=np.float32),
                sample_rate=self._sample_rate,
                language=self.language,
                processing_time=0.0,
            )

        if self._use_mms:
            import torch
            inputs = self._mms_tokenizer(text, return_tensors="pt").to(self._mms_device)
            # VITS exposes speaking_rate as a model attribute; scale duration
            # without a pitch shift, restore afterwards.
            prev_rate = getattr(self._mms_model, "speaking_rate", None)
            if prev_rate is not None and effective_speed != 1.0:
                self._mms_model.speaking_rate = prev_rate * effective_speed
            try:
                with torch.no_grad():
                    out = self._mms_model(**inputs).waveform
            finally:
                if prev_rate is not None:
                    self._mms_model.speaking_rate = prev_rate
            # VitsModel returns (batch, samples); we synth one text at a time.
            audio = out[0].cpu().numpy().astype(np.float32)
        elif self._use_kokoro:
            # Kokoro yields (graphemes, phonemes, audio) per phrase. We
            # concatenate all yielded audio chunks into one waveform.
            audio_chunks = []
            for _graphemes, _phonemes, chunk in self._kokoro_pipeline(
                    text, voice=self._kokoro_voice, speed=effective_speed):
                if chunk is not None:
                    if hasattr(chunk, "cpu"):  # torch tensor
                        chunk = chunk.cpu().numpy()
                    audio_chunks.append(np.asarray(chunk, dtype=np.float32))
            if audio_chunks:
                audio = np.concatenate(audio_chunks).astype(np.float32)
            else:
                audio = np.array([], dtype=np.float32)
        else:
            # Synthesize using Piper.
            # Use length_scale (native VITS parameter) instead of numpy resampling —
            # length_scale adjusts phoneme duration without changing pitch.
            # length_scale = 1/speed: speed=0.9 → length_scale≈1.11 (10% slower)
            from piper.config import SynthesisConfig
            syn_config = SynthesisConfig(length_scale=1.0 / effective_speed) if effective_speed != 1.0 else None

            audio_chunks = []
            for chunk in self._voice.synthesize(text, syn_config=syn_config):
                if hasattr(chunk, 'audio_float_array'):
                    audio_chunks.append(chunk.audio_float_array)

            if audio_chunks:
                audio = np.concatenate(audio_chunks).astype(np.float32)
            else:
                audio = np.array([], dtype=np.float32)

        processing_time = time.time() - start_time

        return SpeechResult(
            text=text,
            audio=audio,
            sample_rate=self._sample_rate,
            language=self.language,
            processing_time=processing_time,
        )

    def synthesize_to_file(self, text: str, output_path: str) -> SpeechResult:
        """
        Synthesize speech and save to WAV file.

        Args:
            text: Text to synthesize
            output_path: Path to output WAV file

        Returns:
            SpeechResult with audio data
        """
        result = self.synthesize(text)

        if len(result.audio) > 0:
            # Convert to int16 for WAV
            audio_int16 = (result.audio * 32767).astype(np.int16)

            # Write WAV file
            with wave.open(output_path, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(self._sample_rate)
                wav.writeframes(audio_int16.tobytes())

        return result

    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
        return False


class MultiLanguageTTS:
    """
    TTS service supporting multiple languages simultaneously.

    Manages multiple TTSService instances for efficient
    multi-language speech synthesis.
    """

    def __init__(
        self,
        languages: List[str] = None,
        download_root: Optional[str] = None,
    ):
        """
        Initialize the multi-language TTS service.

        Args:
            languages: List of language codes
            download_root: Directory for downloaded models
        """
        self.languages = languages or ['es', 'ht']
        self._services = {}
        self._download_root = download_root

    def load(self) -> None:
        """Load all TTS models."""
        for lang in self.languages:
            if lang not in self._services:
                service = TTSService(
                    language=lang,
                    download_root=self._download_root,
                )
                service.load()
                self._services[lang] = service

    def unload(self) -> None:
        """Unload all models."""
        for service in self._services.values():
            service.unload()
        self._services.clear()

    def synthesize(self, text: str, language: str) -> SpeechResult:
        """
        Synthesize speech in a specific language.

        Args:
            text: Text to synthesize
            language: Target language code

        Returns:
            SpeechResult with audio data
        """
        if language not in self._services:
            raise ValueError(f"Language {language} not loaded")
        return self._services[language].synthesize(text)

    def get_sample_rate(self, language: str) -> int:
        """Get the sample rate for a language."""
        if language not in self._services:
            raise ValueError(f"Language {language} not loaded")
        return self._services[language].sample_rate

    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
        return False
