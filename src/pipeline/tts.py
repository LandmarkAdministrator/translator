"""
TTS (Text-to-Speech) Service

Uses Piper for fast, high-quality neural text-to-speech synthesis.
Piper uses ONNX Runtime for efficient inference.
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

        # Resolve model
        if model_path:
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
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @property
    def sample_rate(self) -> int:
        """Get the output sample rate."""
        return self._sample_rate

    def synthesize(self, text: str) -> SpeechResult:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize

        Returns:
            SpeechResult with audio data
        """
        if not self._loaded:
            self.load()

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

        # Synthesize using Piper
        audio_chunks = []

        for chunk in self._voice.synthesize(text):
            # AudioChunk has audio_float_array attribute
            if hasattr(chunk, 'audio_float_array'):
                audio_chunks.append(chunk.audio_float_array)

        if audio_chunks:
            audio = np.concatenate(audio_chunks).astype(np.float32)
        else:
            audio = np.array([], dtype=np.float32)

        # Apply speed adjustment if needed
        if self.speed != 1.0 and len(audio) > 0:
            new_length = int(len(audio) / self.speed)
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(indices, np.arange(len(audio)), audio)

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
