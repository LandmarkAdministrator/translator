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

from .vendor.whisper_online import OnlineASRProcessor, ASRBase

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Hard safety valve: if OnlineASRProcessor fails to commit anything for a long
# stretch (LocalAgreement-2 keeps disagreeing), its audio_buffer grows without
# bound because "segment"-based trimming requires at least one commit. Force-
# trim when the buffer exceeds this, keeping the tail as fresh context.
MAX_BUFFER_SEC = 15.0
TRIM_KEEP_SEC = 5.0


class _OpenAIWhisperASR(ASRBase):
    """
    Whisper backend using openai-whisper's native word-timestamp support
    (word_timestamps=True), avoiding the extra DTW pass in whisper-
    timestamped. Significantly faster per decode on the iGPU while still
    providing the word-level timing OnlineASRProcessor needs.
    """

    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        import torch
        import whisper

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._fp16 = (device == "cuda")
        return whisper.load_model(modelsize, device=device, download_root=cache_dir)

    def transcribe(self, audio, init_prompt=""):
        # temperature=0 disables fallback retries: without this, a single
        # low-confidence window can trigger five re-decodes at T=0.2..1.0,
        # multiplying latency 5x. On a busy streaming buffer this is fatal.
        # no_speech_threshold=0.6 lets whisper skip silent windows instead
        # of hallucinating through them.
        result = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt or None,
            word_timestamps=True,
            condition_on_previous_text=False,
            temperature=0.0,
            no_speech_threshold=0.6,
            fp16=self._fp16,
            verbose=None,
            **self.transcribe_kargs,
        )
        return result

    def ts_words(self, r):
        out = []
        for s in r["segments"]:
            for w in s.get("words", []) or []:
                out.append((w["start"], w["end"], w["word"]))
        return out

    def segments_end_ts(self, res):
        return [s["end"] for s in res["segments"]]

    def use_vad(self):
        self.transcribe_kargs["no_speech_threshold"] = 0.6

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


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
        self._asr = _OpenAIWhisperASR(
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

        # Pre-trim BEFORE decoding: if LocalAgreement-2 has failed to commit
        # recently, the buffer will be large and transcribe() will spend
        # proportional time on it. A single runaway decode blocks the audio
        # callback thread indefinitely, so cap the buffer before we feed it
        # to whisper, not after.
        pre_buffer_sec = len(self._online.audio_buffer) / SAMPLE_RATE
        if pre_buffer_sec > MAX_BUFFER_SEC:
            drop = pre_buffer_sec - TRIM_KEEP_SEC
            self._online.chunk_at(self._online.buffer_time_offset + drop)
            logger.warning(
                "streaming ASR buffer at %.1fs pre-trimmed to %.1fs (no recent commit)",
                pre_buffer_sec, TRIM_KEEP_SEC,
            )

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
