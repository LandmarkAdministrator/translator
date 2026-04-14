"""
ASR (Automatic Speech Recognition) Service

Uses faster-whisper for speech-to-text transcription.

faster-whisper uses CTranslate2 for inference, which runs efficiently on CPU
on any platform. The ASR service intentionally runs on CPU — this is a design
choice, not a limitation. CTranslate2 int8 CPU inference is fast enough for
real-time use and avoids GPU memory contention with the translation and TTS
models that also need GPU resources.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Generator
import numpy as np


@dataclass
class TranscriptionSegment:
    """A segment of transcribed speech."""
    text: str
    start: float
    end: float
    confidence: float = 1.0
    words: List[dict] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end - self.start


@dataclass
class TranscriptionResult:
    """Result of transcription."""
    text: str
    segments: List[TranscriptionSegment]
    language: str
    language_probability: float
    duration: float
    processing_time: float

    @property
    def is_empty(self) -> bool:
        """Check if transcription is empty."""
        return not self.text.strip()


class ASRService:
    """
    Speech recognition service using faster-whisper (CTranslate2 backend).

    Runs on CPU intentionally — CTranslate2 int8 is fast enough for real-time
    transcription and avoids contention with GPU memory used by translation/TTS.
    """

    def __init__(
        self,
        model_size: str = "base.en",
        language: str = "en",
        device: str = "cpu",  # Default to CPU for faster-whisper
        compute_type: str = "auto",
        vad_threshold: float = 0.6,  # Higher = stricter voice detection
        min_speech_duration: float = 0.5,  # Minimum 500ms of speech
        max_speech_duration: float = 30.0,
        min_audio_energy: float = 0.01,  # Minimum audio level to process
        no_speech_threshold: float = 0.6,  # Filter low-confidence transcriptions
        download_root: Optional[str] = None,
    ):
        """
        Initialize the ASR service.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
            language: Source language code (e.g., 'en' for English)
            device: Device to use ('cpu' recommended for AMD/ROCm)
            compute_type: Computation type (auto, float32, int8)
            vad_threshold: Voice activity detection threshold (higher = stricter)
            min_speech_duration: Minimum speech duration in seconds
            max_speech_duration: Maximum speech duration before segmentation
            min_audio_energy: Minimum RMS energy to process (filters silence)
            no_speech_threshold: Confidence threshold to filter hallucinations
            download_root: Directory for downloaded models
        """
        self.model_size = model_size
        self.language = language
        self.vad_threshold = vad_threshold
        self.min_speech_duration = min_speech_duration
        self.max_speech_duration = max_speech_duration
        self.min_audio_energy = min_audio_energy
        self.no_speech_threshold = no_speech_threshold

        # ASR runs on CPU — intentional design choice, not a limitation.
        # See module docstring for rationale.
        self._device = "cpu"

        # Set compute type
        if compute_type == "auto":
            self._compute_type = "int8"  # int8 is fastest on CPU
        else:
            self._compute_type = compute_type

        # Set model download location
        if download_root is None:
            download_root = str(Path(__file__).parent.parent.parent / "models" / "asr")

        self._download_root = download_root
        Path(self._download_root).mkdir(parents=True, exist_ok=True)

        self._model = None
        self._loaded = False

    def load(self) -> None:
        """Load the Whisper model."""
        if self._loaded:
            return

        from faster_whisper import WhisperModel

        print(f"Loading Whisper model '{self.model_size}' on {self._device}...")
        print(f"Compute type: {self._compute_type}")

        self._model = WhisperModel(
            self.model_size,
            device=self._device,
            compute_type=self._compute_type,
            download_root=self._download_root,
            cpu_threads=os.cpu_count() or 4,
        )

        self._loaded = True
        print("Whisper model loaded successfully")

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @property
    def device(self) -> str:
        """Get the device being used."""
        return self._device

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (mono, float32)
            sample_rate: Sample rate of the audio

        Returns:
            TranscriptionResult with transcription and metadata
        """
        if not self._loaded:
            self.load()

        start_time = time.time()

        # Ensure audio is correct format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Resample if needed
        if sample_rate != 16000:
            ratio = 16000 / sample_rate
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(indices, np.arange(len(audio)), audio)

        audio_duration = len(audio) / 16000

        # Check minimum audio energy (filter silence/noise)
        rms_energy = np.sqrt(np.mean(audio ** 2))
        if rms_energy < self.min_audio_energy:
            # Audio too quiet, skip processing
            return TranscriptionResult(
                text="",
                segments=[],
                language=self.language,
                language_probability=1.0,
                duration=audio_duration,
                processing_time=time.time() - start_time,
            )

        # Normalize audio
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # Transcribe with faster-whisper
        segments_gen, info = self._model.transcribe(
            audio,
            language=self.language if self.language != "auto" else None,
            beam_size=5,
            temperature=0.0,  # Deterministic output (no sampling randomness)
            vad_filter=True,
            vad_parameters={
                "threshold": self.vad_threshold,
                "min_speech_duration_ms": int(self.min_speech_duration * 1000),
                "max_speech_duration_s": self.max_speech_duration,
                "min_silence_duration_ms": 500,  # Require 500ms silence between segments
            },
            no_speech_threshold=self.no_speech_threshold,
        )

        # Collect segments, filtering low-confidence ones
        segments = []
        full_text_parts = []

        for segment in segments_gen:
            # Get confidence (avg_logprob is negative, closer to 0 = more confident)
            confidence = segment.avg_logprob if hasattr(segment, 'avg_logprob') else 0.0

            # Filter very low confidence segments (likely hallucinations)
            # avg_logprob typically ranges from -1 to 0, with -0.5 being decent
            if confidence < -1.0:
                continue  # Skip low-confidence transcription

            # Skip very short segments (likely noise or fragments)
            if len(segment.text.strip()) < 10:
                continue

            seg = TranscriptionSegment(
                text=segment.text.strip(),
                start=segment.start,
                end=segment.end,
                confidence=confidence,
            )
            segments.append(seg)
            full_text_parts.append(segment.text)

        full_text = " ".join(full_text_parts).strip()
        processing_time = time.time() - start_time

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            language=info.language if info.language else self.language,
            language_probability=info.language_probability if info.language_probability else 1.0,
            duration=audio_duration,
            processing_time=processing_time,
        )

    def transcribe_stream(
        self,
        audio_chunks: Generator[np.ndarray, None, None],
        sample_rate: int = 16000,
    ) -> Generator[TranscriptionResult, None, None]:
        """
        Transcribe a stream of audio chunks.

        Args:
            audio_chunks: Generator yielding audio chunks
            sample_rate: Sample rate of the audio

        Yields:
            TranscriptionResult for each chunk
        """
        if not self._loaded:
            self.load()

        for chunk in audio_chunks:
            result = self.transcribe(chunk, sample_rate)
            if not result.is_empty:
                yield result

    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
        return False


class WhisperTorchService:
    """
    Alternative ASR service using OpenAI Whisper with PyTorch directly.

    This is an optional alternative to ASRService. It supports GPU acceleration
    via PyTorch (AMD ROCm or NVIDIA CUDA) but is slower than faster-whisper and
    requires installing the openai-whisper package separately:

        pip install openai-whisper

    This package is NOT installed by default. Use ASRService (faster-whisper)
    for production. This class exists for experimentation only.
    """

    def __init__(
        self,
        model_size: str = "base.en",
        language: str = "en",
        device: str = "auto",
        download_root: Optional[str] = None,
    ):
        """
        Initialize the Whisper service with PyTorch backend.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            language: Source language code
            device: Device to use ('cuda', 'cpu', or 'auto')
            download_root: Directory for downloaded models
        """
        self.model_size = model_size
        self.language = language

        # Detect device
        if device == "auto":
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        if download_root is None:
            download_root = str(Path(__file__).parent.parent.parent / "models" / "asr")

        self._download_root = download_root
        Path(self._download_root).mkdir(parents=True, exist_ok=True)

        self._model = None
        self._loaded = False

    def load(self) -> None:
        """Load the Whisper model."""
        if self._loaded:
            return

        try:
            import whisper
        except ImportError:
            raise ImportError(
                "openai-whisper is not installed. "
                "Install it with: pip install openai-whisper\n"
                "Or use ASRService (faster-whisper) which is installed by default."
            )

        print(f"Loading Whisper model '{self.model_size}' on {self._device} (PyTorch)...")

        self._model = whisper.load_model(
            self.model_size,
            device=self._device,
            download_root=self._download_root,
        )

        self._loaded = True
        print("Whisper model loaded successfully")

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @property
    def device(self) -> str:
        """Get the device being used."""
        return self._device

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (mono, float32)
            sample_rate: Sample rate of the audio

        Returns:
            TranscriptionResult with transcription and metadata
        """
        if not self._loaded:
            self.load()

        import whisper  # Optional dependency — see class docstring

        start_time = time.time()

        # Ensure audio is correct format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Resample if needed
        if sample_rate != 16000:
            ratio = 16000 / sample_rate
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(indices, np.arange(len(audio)), audio)

        audio_duration = len(audio) / 16000

        # Pad or trim to expected length
        audio = whisper.pad_or_trim(audio)

        # Compute mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(self._device)

        # Decode
        options = whisper.DecodingOptions(
            language=self.language if self.language != "auto" else None,
            fp16=(self._device == "cuda"),
        )
        result = whisper.decode(self._model, mel, options)

        processing_time = time.time() - start_time

        # Create segment
        segments = [
            TranscriptionSegment(
                text=result.text.strip(),
                start=0.0,
                end=audio_duration,
                confidence=1.0,
            )
        ] if result.text.strip() else []

        return TranscriptionResult(
            text=result.text.strip(),
            segments=segments,
            language=result.language if hasattr(result, 'language') else self.language,
            language_probability=1.0,
            duration=audio_duration,
            processing_time=processing_time,
        )

    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
        return False
