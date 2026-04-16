"""
Streaming ASR adapter backed by whisper_streaming (UFAL).

Wraps OnlineASRProcessor + WhisperTimestampedASR so the coordinator's
existing streaming callback can feed audio and receive *committed* text
(LocalAgreement-2 stable-prefix policy).

Public shape matches the old StreamingASRBuffer so the coordinator
doesn't have to change: feed(audio, chunk_start_wall) -> Optional
(text, start_wall, asr_time); flush() -> same.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .vendor.whisper_online import OnlineASRProcessor, WhisperTimestampedASR

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class _FP16WhisperTimestampedASR(WhisperTimestampedASR):
    """
    Load openai-whisper in float16 to fit large-v3 on the 7.6 GiB iGPU.

    The default `whisper.load_model` keeps weights in float32 during the
    load → .to(device) path, which OOMs on the Radeon 890M. We load on
    CPU, cast to half, then move to GPU.
    """

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        import torch
        import whisper
        import whisper_timestamped
        from whisper_timestamped import transcribe_timestamped

        self.transcribe_timestamped = transcribe_timestamped

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(modelsize, device="cpu", download_root=cache_dir)
        if device == "cuda":
            model = model.half().to(device)
            # Force FP16 inference so mel input types match model weights;
            # whisper's auto-detect doesn't cover whisper_timestamped's path.
            self.transcribe_kargs["fp16"] = True
        return model


class LocalAgreementASRBuffer:
    """
    Streaming ASR using UFAL's whisper_streaming LocalAgreement-2 policy.

    The underlying OnlineASRProcessor runs whisper on a growing buffer on
    each feed() and only emits tokens that have been seen in two
    consecutive decodes. Completed segments are trimmed from the buffer.
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        language: str = "en",
        download_root: Optional[str] = None,
        buffer_trim_seconds: float = 15.0,
    ):
        self._model_size = model_size
        self._language = language
        self._download_root = download_root
        self._buffer_trim_seconds = buffer_trim_seconds

        self._asr: Optional[WhisperTimestampedASR] = None
        self._online: Optional[OnlineASRProcessor] = None
        self._session_start_wall: float = 0.0
        self._stream_started: bool = False

    def load(self) -> None:
        """Load whisper model. Must be called before feed()."""
        cache_dir = self._download_root
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._asr = _FP16WhisperTimestampedASR(
            lan=self._language,
            modelsize=self._model_size,
            cache_dir=cache_dir,
        )
        self._online = OnlineASRProcessor(
            self._asr,
            buffer_trimming=("segment", self._buffer_trim_seconds),
        )

    def feed(
        self,
        audio: np.ndarray,
        chunk_start_wall: float = 0.0,
    ) -> Optional[Tuple[str, float, float]]:
        """Feed audio samples; return (text, start_wall, asr_time) when committed."""
        if self._online is None:
            raise RuntimeError("LocalAgreementASRBuffer.load() not called")

        if not self._stream_started:
            self._session_start_wall = chunk_start_wall or time.time()
            self._stream_started = True

        # Normalize to float32 mono 16 kHz (caller is responsible for rate/channel).
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)

        self._online.insert_audio_chunk(audio)

        asr_start = time.time()
        beg, end, text = self._online.process_iter()
        asr_time = time.time() - asr_start

        if not text or beg is None:
            return None

        text_start_wall = self._session_start_wall + float(beg)
        return (text.strip(), text_start_wall, asr_time)

    def flush(self) -> Optional[Tuple[str, float, float]]:
        """Emit any remaining uncommitted text at shutdown."""
        if self._online is None:
            return None
        asr_start = time.time()
        beg, end, text = self._online.finish()
        asr_time = time.time() - asr_start
        if not text or beg is None:
            return None
        text_start_wall = self._session_start_wall + float(beg)
        return (text.strip(), text_start_wall, asr_time)
