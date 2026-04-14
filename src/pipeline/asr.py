"""
ASR (Automatic Speech Recognition) Service

Uses faster-whisper for speech-to-text transcription.

faster-whisper uses CTranslate2 for inference. By default the service runs on
CPU (int8) which is fast enough for real-time use with small models (base, small).
For larger models (medium, large-v3), GPU acceleration is recommended — pass
device="cuda" to enable it. On AMD ROCm systems CTranslate2 uses the HIP CUDA
compatibility layer; make sure HSA_OVERRIDE_GFX_VERSION is set (handled by
.env.rocm / the launcher script).

GPU compute types:
  float16      — full precision, good accuracy, ~2GB VRAM for large-v3
  int8_float16 — fastest, minimal quality loss
CPU compute types:
  int8         — fastest on CPU (default)
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Generator, Tuple
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
        device: str = "cpu",  # "cpu" or "cuda" (ROCm uses cuda compat layer)
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

        self._device = device

        # Set compute type based on device if auto
        if compute_type == "auto":
            self._compute_type = "float16" if device == "cuda" else "int8"
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

        device_label = "GPU (ROCm/CUDA)" if self._device == "cuda" else "CPU"
        print(f"Loading Whisper model '{self.model_size}' on {device_label}...")
        print(f"Compute type: {self._compute_type}")

        try:
            self._model = WhisperModel(
                self.model_size,
                device=self._device,
                compute_type=self._compute_type,
                download_root=self._download_root,
                cpu_threads=os.cpu_count() or 4,
            )
        except RuntimeError as e:
            if self._device == "cuda":
                print(f"GPU load failed ({e}), falling back to CPU...")
                self._device = "cpu"
                self._compute_type = "int8"
                self._model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    download_root=self._download_root,
                    cpu_threads=os.cpu_count() or 4,
                )
            else:
                raise

        self._loaded = True
        print(f"Whisper model loaded successfully on {self._device}")

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


class WhisperTransformersService:
    """
    GPU-accelerated Whisper using HuggingFace transformers + PyTorch.

    Uses the same PyTorch backend that runs MarianMT and Piper, so it works
    correctly with AMD ROCm (unlike ASRService which uses CTranslate2 with
    NVIDIA-only CUDA binaries).

    Requires: pip install transformers (already installed)
    Model is downloaded from HuggingFace on first load (~3GB for large-v3).
    """

    MODEL_MAP = {
        "large-v3": "openai/whisper-large-v3",
        "medium.en": "openai/whisper-medium.en",
        "small.en": "openai/whisper-small.en",
        "base.en": "openai/whisper-base.en",
        "tiny.en": "openai/whisper-tiny.en",
        "medium": "openai/whisper-medium",
        "small": "openai/whisper-small",
        "base": "openai/whisper-base",
        "tiny": "openai/whisper-tiny",
    }

    def __init__(
        self,
        model_size: str = "large-v3",
        language: str = "en",
        device: str = "cuda",
        min_audio_energy: float = 0.01,
        no_speech_threshold: float = 0.6,
        download_root: Optional[str] = None,
    ):
        self.model_size = model_size
        self.language = language
        self.min_audio_energy = min_audio_energy
        self.no_speech_threshold = no_speech_threshold

        self._device = device
        self._model_id = self.MODEL_MAP.get(model_size, f"openai/whisper-{model_size}")
        self._download_root = download_root
        self._pipe = None
        self._loaded = False

    def load(self) -> None:
        """Load the Whisper model via transformers pipeline."""
        if self._loaded:
            return

        import torch
        from transformers import pipeline as hf_pipeline, AutoProcessor

        device_label = "GPU (ROCm/CUDA)" if self._device == "cuda" else "CPU"
        print(f"Loading Whisper model '{self.model_size}' on {device_label} (transformers)...")

        torch_dtype = torch.float16 if self._device == "cuda" else torch.float32

        kwargs = dict(
            model=self._model_id,
            device=self._device,
            torch_dtype=torch_dtype,
            model_kwargs={"attn_implementation": "eager"},  # ROCm: skip flash-attn
        )
        if self._download_root:
            kwargs["model_kwargs"]["cache_dir"] = self._download_root

        self._pipe = hf_pipeline("automatic-speech-recognition", **kwargs)
        self._loaded = True
        print(f"Whisper model loaded successfully on {self._device}")

    def unload(self) -> None:
        """Unload model and free GPU memory."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            self._loaded = False
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return self._device

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe audio using transformers Whisper pipeline."""
        if not self._loaded:
            self.load()

        start_time = time.time()

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if sample_rate != 16000:
            ratio = 16000 / sample_rate
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

        audio_duration = len(audio) / 16000

        rms_energy = np.sqrt(np.mean(audio ** 2))
        if rms_energy < self.min_audio_energy:
            return TranscriptionResult(
                text="", segments=[], language=self.language,
                language_probability=1.0, duration=audio_duration,
                processing_time=time.time() - start_time,
            )

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # English-only models (*.en) don't accept language/task params
        if self.model_size.endswith(".en"):
            generate_kwargs = {}
        else:
            generate_kwargs = {"language": self.language, "task": "transcribe"}

        try:
            result = self._pipe(
                {"raw": audio, "sampling_rate": 16000},
                return_timestamps=True,
                generate_kwargs=generate_kwargs,
            )
        except Exception as e:
            print(f"  [ASR ERROR] WhisperTransformers failed: {e}")
            processing_time = time.time() - start_time
            return TranscriptionResult(
                text="", segments=[], language=self.language,
                language_probability=1.0, duration=audio_duration,
                processing_time=processing_time,
            )

        full_text = result.get("text", "").strip()

        # Hallucination filter: discard results where a single word or trigram repeats
        # excessively (model loops on applause/noise instead of producing real speech)
        if full_text and self._is_hallucination(full_text):
            preview = full_text[:80] + ("..." if len(full_text) > 80 else "")
            print(f"  [HALLUCINATION FILTERED] {preview}")
            processing_time = time.time() - start_time
            return TranscriptionResult(
                text="", segments=[], language=self.language,
                language_probability=1.0, duration=audio_duration,
                processing_time=processing_time,
            )

        chunks = result.get("chunks", [])

        segments = []
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            ts = chunk.get("timestamp", (0.0, audio_duration))
            start = ts[0] if ts[0] is not None else 0.0
            end = ts[1] if ts[1] is not None else audio_duration
            if len(text) >= 3:
                segments.append(TranscriptionSegment(
                    text=text, start=start, end=end, confidence=-0.3,
                ))

        processing_time = time.time() - start_time
        return TranscriptionResult(
            text=full_text, segments=segments, language=self.language,
            language_probability=1.0, duration=audio_duration,
            processing_time=processing_time,
        )

    @staticmethod
    def _is_hallucination(text: str) -> bool:
        """
        Return True if the text looks like a Whisper hallucination loop.

        Two heuristics:
        - Word dominance: a single word makes up >40% of output OR appears >6 times.
          (catches "oh oh oh oh..." and "thank you thank you...")
        - Trigram loop: any 3-word phrase repeats more than 3 times.
          (catches "out of the road out of the road...")
        """
        from collections import Counter
        words = text.lower().split()
        if not words:
            return False

        counts = Counter(words)
        top_word, top_count = counts.most_common(1)[0]
        if top_count > 6 or top_count > len(words) * 0.40:
            return True

        if len(words) >= 9:
            trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
            tg_counts = Counter(trigrams)
            if tg_counts.most_common(1)[0][1] > 3:
                return True

        return False

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False


class StreamingASRBuffer:
    """
    Rolling audio buffer with stable-prefix streaming transcription.

    Instead of waiting for silence to emit a chunk, this runs Whisper on a
    growing audio buffer every time new audio arrives (~1-2s intervals).
    Text is only emitted once it appears identically in two consecutive
    transcriptions (stable prefix), so Whisper can correct itself with
    growing context before we commit to translating.

    This solves the sentence-spanning problem: a sentence that starts in one
    chunk and ends in the next is held in the buffer until Whisper sees the
    complete sentence and the stable prefix catches up to include it.

    Algorithm:
        1. Accumulate small audio chunks into a rolling buffer (max 25s)
        2. After each chunk, run Whisper on the whole buffer
        3. Compare current transcription to previous: find longest word-prefix
           that matches (the "stable" portion)
        4. Words that are stable AND not in the unstable tail (last TAIL_WORDS)
           are confirmed and emitted
        5. Trim confirmed audio from buffer (keeping a short overlap for context)
        6. Force-emit if no progress after MAX_STALL_S seconds
    """

    TAIL_WORDS = 3       # Never confirm the last N words (may still change with context)
    MAX_STALL_S = 8.0    # Force-emit if no new confirmed text after this many seconds
    OVERLAP_S = 0.5      # Audio overlap to keep after trimming (context for next call)

    def __init__(self, asr_service: "ASRService", max_buffer_s: float = 25.0):
        self._asr = asr_service
        self._max_buffer_s = max_buffer_s
        self._sample_rate = 16000

        self._audio_buffer: np.ndarray = np.array([], dtype=np.float32)
        self._buffer_start_wall: float = 0.0   # Wall time of first sample in buffer
        self._prev_words: List[str] = []
        self._committed_words: int = 0          # Words in buffer already emitted
        self._last_commit_wall: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(
        self, audio: np.ndarray, chunk_start_wall: float = 0.0
    ) -> Optional[Tuple[str, float, float]]:
        """
        Feed a small audio chunk. Returns (confirmed_text, start_wall_time, asr_time)
        when new text is confirmed, or None if nothing is ready yet.
        """
        if self._buffer_start_wall == 0.0:
            self._buffer_start_wall = chunk_start_wall or time.time()
            self._last_commit_wall = self._buffer_start_wall

        # Append and enforce hard cap
        self._audio_buffer = np.concatenate([self._audio_buffer, audio])
        max_samples = int(self._max_buffer_s * self._sample_rate)
        if len(self._audio_buffer) > max_samples:
            excess = len(self._audio_buffer) - max_samples
            self._audio_buffer = self._audio_buffer[excess:]
            self._buffer_start_wall += excess / self._sample_rate

        # Run Whisper on full buffer
        asr_start = time.time()
        result = self._asr.transcribe(self._audio_buffer, self._sample_rate)
        asr_time = time.time() - asr_start

        if result.is_empty:
            if time.time() - self._last_commit_wall > self.MAX_STALL_S:
                self._reset()
            return None

        curr_words = result.text.strip().split()
        if not curr_words:
            return None

        # Stable prefix: how many words match between previous and current transcription
        stable = self._stable_prefix_len(self._prev_words, curr_words)

        # Confirm words that are stable AND not in the unstable tail
        # Force-emit everything (minus tail) if stalled too long
        stalled = (time.time() - self._last_commit_wall) > self.MAX_STALL_S
        confirm_up_to = min(stable, len(curr_words) - self.TAIL_WORDS)
        if stalled:
            confirm_up_to = max(confirm_up_to, len(curr_words) - self.TAIL_WORDS)

        self._prev_words = curr_words

        if confirm_up_to <= self._committed_words:
            return None

        # Newly confirmed words
        new_words = curr_words[self._committed_words:confirm_up_to]
        new_text = " ".join(new_words).strip()
        if not new_text:
            return None

        # Wall time of the start of the newly confirmed text
        text_start_s = self._word_start_time(result, self._committed_words)
        text_start_wall = self._buffer_start_wall + text_start_s

        # Trim buffer to end of confirmed text, keeping OVERLAP_S for context
        confirm_end_s = self._word_end_time(result, confirm_up_to - 1)
        trim_s = max(0.0, confirm_end_s - self.OVERLAP_S)
        trim_samples = int(trim_s * self._sample_rate)
        self._audio_buffer = self._audio_buffer[trim_samples:]
        self._buffer_start_wall += trim_s

        # After trimming, Whisper will re-index from 0, so reset word tracking
        # Keep the tail words as context so stable-prefix still works next round
        self._prev_words = curr_words[confirm_up_to:]
        self._committed_words = 0
        self._last_commit_wall = time.time()

        return (new_text, text_start_wall, asr_time)

    def flush(self) -> Optional[Tuple[str, float, float]]:
        """
        Emit all remaining uncommitted text in the buffer (called on shutdown).
        """
        if len(self._audio_buffer) < int(0.2 * self._sample_rate):
            return None

        asr_start = time.time()
        result = self._asr.transcribe(self._audio_buffer, self._sample_rate)
        asr_time = time.time() - asr_start

        if result.is_empty:
            return None

        curr_words = result.text.strip().split()
        if len(curr_words) <= self._committed_words:
            return None

        new_words = curr_words[self._committed_words:]
        new_text = " ".join(new_words).strip()
        if not new_text:
            return None

        text_start_s = self._word_start_time(result, self._committed_words)
        text_start_wall = self._buffer_start_wall + text_start_s
        self._reset()
        return (new_text, text_start_wall, asr_time)

    def reset(self) -> None:
        """Public reset — call between sessions."""
        self._reset()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self._audio_buffer = np.array([], dtype=np.float32)
        self._buffer_start_wall = 0.0
        self._prev_words = []
        self._committed_words = 0
        self._last_commit_wall = 0.0

    @staticmethod
    def _stable_prefix_len(prev: List[str], curr: List[str]) -> int:
        """Length of longest common word prefix, ignoring punctuation."""
        n = min(len(prev), len(curr))
        for i in range(n):
            if prev[i].lower().rstrip(".,!?;:'\"") != curr[i].lower().rstrip(".,!?;:'\""):
                return i
        return n

    @staticmethod
    def _word_start_time(result: "TranscriptionResult", word_idx: int) -> float:
        """Segment start time that contains the word at word_idx."""
        count = 0
        for seg in result.segments:
            words_in_seg = len(seg.text.strip().split())
            if count + words_in_seg > word_idx:
                return seg.start
            count += words_in_seg
        return result.segments[0].start if result.segments else 0.0

    @staticmethod
    def _word_end_time(result: "TranscriptionResult", word_idx: int) -> float:
        """Segment end time that contains the word at word_idx."""
        count = 0
        for seg in result.segments:
            words_in_seg = len(seg.text.strip().split())
            count += words_in_seg
            if count > word_idx:
                return seg.end
        return result.segments[-1].end if result.segments else 0.0


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
