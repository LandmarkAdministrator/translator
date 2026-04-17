"""
Parakeet streaming ASR adapter.

Uses onnx-asr to run NVIDIA's Parakeet TDT/RNN-T model through ONNX Runtime.
Parakeet is natively a batch model here (true streaming would require NeMo's
stateful decoder), so we approximate streaming by running the full buffer on
every feed() and applying LocalAgreement-2 at the token level to decide what
has stabilized and can be emitted.

Fed by the coordinator's streaming audio callback (1.5s chunks); emits
(text, start_wall, asr_time) when LocalAgreement-2 commits new tokens.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Same safety-valve idea as the Whisper path: if LocalAgreement fails to commit
# for a long stretch, transcribe time grows with buffer length. Cap it.
MAX_BUFFER_SEC = 15.0
HARD_DROP_BUFFER_SEC = 22.0
TRIM_KEEP_SEC = 5.0


def _longest_common_prefix(a: Sequence[str], b: Sequence[str]) -> int:
    """Return length of the longest common prefix of two token lists."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _tokens_to_text(tokens: Sequence[str]) -> str:
    """Join Parakeet tokens into a readable string.

    onnx-asr returns tokens with SentencePiece-style leading spaces already
    expanded (▁ → space) by the time they land in TimestampedResult.tokens,
    so plain concatenation plus whitespace cleanup is sufficient.
    """
    return "".join(tokens).strip()


class ParakeetASRBuffer:
    """
    Streaming ASR using Parakeet via onnx-asr with LocalAgreement-2.

    On each feed(), the entire rolling buffer is re-transcribed. Tokens in
    the common prefix of the last two hypotheses are considered "stable" and
    emitted. When the last emitted token is far enough back in the buffer we
    trim the audio up to that point, matching how OnlineASRProcessor handles
    the Whisper path.
    """

    def __init__(
        self,
        model_name: str = "nemo-parakeet-tdt-0.6b-v3",
        cache_dir: Optional[str] = None,
        providers: Optional[List[str]] = None,
        quantization: Optional[str] = None,
    ):
        self._model_name = model_name
        # onnx_asr.load_model() doesn't accept cache_dir directly — HF downloads
        # honor the HF_HOME env var, so we export it before importing if the
        # caller asked for a specific cache location.
        self._cache_dir = cache_dir
        self._providers = providers
        self._quantization = quantization

        self._model = None  # TimestampedResultsAsrAdapter
        self._buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        self._buffer_start_wall: float = 0.0
        self._session_started: bool = False
        self._session_start_wall: float = 0.0

        # LocalAgreement-2 state
        self._prev_tokens: List[str] = []
        self._prev_timestamps: List[float] = []
        self._committed_count: int = 0  # tokens already emitted from current buffer

    def load(self) -> None:
        """Load Parakeet model. Must be called before feed()."""
        import os
        if self._cache_dir:
            Path(self._cache_dir).mkdir(parents=True, exist_ok=True)
            # onnx_asr delegates downloads to huggingface_hub, which respects
            # HF_HOME. Set it only if the caller didn't already.
            os.environ.setdefault("HF_HOME", self._cache_dir)

        try:
            import onnx_asr
        except ImportError as e:
            raise RuntimeError(
                "onnx-asr is not installed. Run scripts/install_parakeet.sh on "
                "the target machine to install onnx-asr + onnxruntime-rocm."
            ) from e

        # Try ROCm first and fall back to CPU if its shared libraries aren't
        # resolvable. On this hardware (Radeon 890M / gfx1150) the shipped
        # onnxruntime-rocm wheel expects specific hipblas/amdhip SONAME
        # versions; if they don't match the installed ROCm, the provider
        # silently falls back to CPU anyway — we do it explicitly here so
        # the choice ends up in the log.
        providers = self._providers
        if providers is None:
            import onnxruntime as ort
            avail = ort.get_available_providers()
            providers = []
            if "ROCMExecutionProvider" in avail:
                providers.append("ROCMExecutionProvider")
            providers.append("CPUExecutionProvider")

        kwargs = {"providers": providers}
        if self._quantization:
            kwargs["quantization"] = self._quantization

        base = onnx_asr.load_model(self._model_name, **kwargs)
        # with_timestamps() returns an adapter whose recognize() yields
        # TimestampedResult(text, timestamps, tokens, logprobs). We use the
        # timestamps for buffer trimming and commit-boundary reporting.
        self._model = base.with_timestamps()
        logger.info("Parakeet loaded: model=%s providers=%s", self._model_name, providers)

    def feed(
        self,
        audio: np.ndarray,
        chunk_start_wall: float = 0.0,
    ) -> Optional[Tuple[str, float, float]]:
        """Feed audio samples; return (text, start_wall, asr_time) when committed."""
        if self._model is None:
            raise RuntimeError("ParakeetASRBuffer.load() not called")

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)

        if not self._session_started:
            self._session_start_wall = chunk_start_wall or time.time()
            self._buffer_start_wall = self._session_start_wall
            self._session_started = True

        self._buffer = np.concatenate([self._buffer, audio])

        asr_start = time.time()
        result = self._model.recognize(self._buffer, sample_rate=SAMPLE_RATE)
        asr_time = time.time() - asr_start

        tokens = list(result.tokens or [])
        timestamps = list(result.timestamps or [])
        # Defensive: if the model returned no timestamps, length mismatch would
        # break our trim logic. Treat as "no commit possible this step".
        if len(timestamps) != len(tokens):
            self._prev_tokens = tokens
            self._prev_timestamps = []
            self._maybe_hard_drop()
            return None

        # LocalAgreement-2: the new stable prefix is the LCP of the last two
        # hypotheses. Anything beyond the previous committed count is fresh
        # text to emit.
        stable_len = _longest_common_prefix(self._prev_tokens, tokens)
        newly_count = stable_len - self._committed_count

        self._prev_tokens = tokens
        self._prev_timestamps = timestamps

        if newly_count <= 0:
            self._maybe_trim(tokens, timestamps, stable_len)
            self._maybe_hard_drop()
            return None

        new_tokens = tokens[self._committed_count:stable_len]
        new_text = _tokens_to_text(new_tokens)

        first_ts = timestamps[self._committed_count]
        text_start_wall = self._buffer_start_wall + float(first_ts)

        self._committed_count = stable_len
        self._maybe_trim(tokens, timestamps, stable_len)
        self._maybe_hard_drop()

        if not new_text:
            return None
        return (new_text, text_start_wall, asr_time)

    def flush(self) -> Optional[Tuple[str, float, float]]:
        """Emit any remaining uncommitted text at shutdown."""
        if self._model is None or len(self._buffer) == 0:
            return None
        asr_start = time.time()
        result = self._model.recognize(self._buffer, sample_rate=SAMPLE_RATE)
        asr_time = time.time() - asr_start

        tokens = list(result.tokens or [])
        timestamps = list(result.timestamps or [])
        # At shutdown we trust the final decode without waiting for agreement.
        if len(tokens) <= self._committed_count:
            return None
        remaining_tokens = tokens[self._committed_count:]
        text = _tokens_to_text(remaining_tokens)
        if not text:
            return None
        if len(timestamps) == len(tokens):
            first_ts = timestamps[self._committed_count]
            text_start_wall = self._buffer_start_wall + float(first_ts)
        else:
            text_start_wall = self._buffer_start_wall
        return (text, text_start_wall, asr_time)

    def _maybe_trim(self, tokens, timestamps, stable_len: int) -> None:
        """Trim audio buffer up to the end of the last committed token."""
        buffer_sec = len(self._buffer) / SAMPLE_RATE
        if buffer_sec <= MAX_BUFFER_SEC:
            return
        if stable_len <= 0:
            return
        cut_ts = float(timestamps[stable_len - 1])
        cut_samples = int(cut_ts * SAMPLE_RATE)
        if cut_samples <= 0 or cut_samples >= len(self._buffer):
            return
        self._buffer = self._buffer[cut_samples:]
        self._buffer_start_wall += cut_ts
        # After trim, the new buffer shares no tokens with the pre-trim decode,
        # so both the previous-hypothesis and committed-count trackers reset.
        self._prev_tokens = []
        self._prev_timestamps = []
        self._committed_count = 0

    def _maybe_hard_drop(self) -> None:
        """Last-resort trim when LocalAgreement keeps disagreeing."""
        buffer_sec = len(self._buffer) / SAMPLE_RATE
        if buffer_sec <= HARD_DROP_BUFFER_SEC:
            return
        drop = buffer_sec - TRIM_KEEP_SEC
        keep_samples = int(TRIM_KEEP_SEC * SAMPLE_RATE)
        self._buffer = self._buffer[-keep_samples:]
        self._buffer_start_wall += drop
        self._prev_tokens = []
        self._prev_timestamps = []
        self._committed_count = 0
        logger.warning(
            "parakeet buffer hit %.1fs without progress — hard-trimmed to %.1fs",
            buffer_sec, TRIM_KEEP_SEC,
        )
