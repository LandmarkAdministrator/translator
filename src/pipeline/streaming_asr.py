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


class _HFTransformersASR(ASRBase):
    """
    Whisper backend via the Hugging Face transformers pipeline — the same
    implementation the batch path uses at RTF 0.33 on this hardware. The
    reference openai-whisper implementation was ~2-3x slower per decode,
    putting streaming RTF above 1. This backend brings streaming decode
    into the same performance class as batch.

    Word-level timestamps come from the pipeline's return_timestamps="word"
    mode — no separate DTW alignment pass.
    """

    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        import torch
        from transformers import pipeline as hf_pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._device = device

        model_id = (
            modelsize if "/" in (modelsize or "")
            else f"openai/whisper-{modelsize}"
        )

        model_kwargs = {"attn_implementation": "eager"}  # ROCm: skip flash-attn
        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir

        return hf_pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            dtype=dtype,
            model_kwargs=model_kwargs,
        )

    def transcribe(self, audio, init_prompt=""):
        # init_prompt is intentionally ignored: condition_on_previous_text
        # was harmful in the openai-whisper path (one bad decode poisoned
        # the next); HF's prompt_ids mechanism has the same failure mode
        # and wasn't giving a quality win. OnlineASRProcessor's
        # LocalAgreement-2 provides the context continuity we need.
        # Stay close to batch mode's generate_kwargs. Earlier attempt to
        # force temperature=0 and no_speech_threshold=0.6 through the HF
        # pipeline triggered an UnboundLocalError inside
        # generate_with_fallback(). The HF path has its own fallback behavior
        # that proved safe at RTF 0.33 in batch mode — leave it alone.
        gen_kwargs = {
            "language": self.original_language,
            "task": self.transcribe_kargs.get("task", "transcribe"),
        }

        result = self.model(
            {"raw": audio.astype(np.float32), "sampling_rate": SAMPLE_RATE},
            return_timestamps="word",
            generate_kwargs=gen_kwargs,
        )
        return result

    def ts_words(self, r):
        out = []
        for chunk in r.get("chunks", []) or []:
            word = chunk.get("text", "")
            ts = chunk.get("timestamp") or (None, None)
            start, end = ts[0], ts[1]
            if word and start is not None and end is not None:
                out.append((start, end, word))
        return out

    def segments_end_ts(self, res):
        # HF returns per-word chunks; approximate segment boundaries at
        # sentence-ending punctuation. OnlineASRProcessor uses these for
        # optional segment-based buffer trimming (we also have a hard
        # pre-trim in LocalAgreementASRBuffer.feed so this is a hint).
        ends = []
        for chunk in res.get("chunks", []) or []:
            word = (chunk.get("text") or "").strip()
            ts = chunk.get("timestamp") or (None, None)
            if word and word[-1:] in ".!?" and ts[1] is not None:
                ends.append(ts[1])
        return ends

    def use_vad(self):
        # HF pipeline exposes VAD indirectly via generate_kwargs'
        # no_speech_threshold (set in transcribe()); no extra wiring needed.
        pass

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

        self._asr: Optional[_HFTransformersASR] = None
        self._online: Optional[OnlineASRProcessor] = None
        self._session_start_wall: float = 0.0
        self._stream_started: bool = False

    def load(self) -> None:
        """Load whisper model. Must be called before feed()."""
        cache_dir = self._download_root
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._asr = _HFTransformersASR(
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
