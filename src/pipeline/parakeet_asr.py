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

import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# loguru, like the rest of the pipeline: stdlib logging was never routed
# anywhere, so this module's messages — including the one that says the
# ASR server was killed for stalling — did not reach the service log.
from loguru import logger

SAMPLE_RATE = 16000

# Same safety-valve idea as the Whisper path: if LocalAgreement fails to commit
# for a long stretch, transcribe time grows with buffer length. Cap it.
MAX_BUFFER_SEC = 15.0
HARD_DROP_BUFFER_SEC = 22.0
TRIM_KEEP_SEC = 5.0
# Evict committed audio as soon as this much of it sits at the buffer head
# (2026-09-01: waiting for MAX_BUFFER_SEC let buffers grow until the panic
# path re-emitted/lost words — 187-486 inserted words per 45-min run).
EVICT_COMMITTED_SEC = 3.0


class _RemoteResult:
    """Duck-typed stand-in for onnx-asr's TimestampedResult (plus word ends).

    A protocol-2 server sends only the tokens that are NEW since its previous
    reply, plus `count`, its running total for the stream, so the client can
    tell if it ever missed a reply. A protocol-1 server sends the whole stream
    and no count; `count is None` is how the client tells them apart.
    """

    __slots__ = ("text", "tokens", "timestamps", "ends", "count")

    def __init__(self, tokens: List[str], timestamps: List[float],
                 ends: Optional[List[float]] = None,
                 count: Optional[int] = None):
        self.tokens = tokens
        self.timestamps = timestamps
        self.ends = ends
        self.count = count
        self.text = "".join(tokens)


class _RemoteUnifiedModel:
    """parakeet-unified-en-0.6b behind a subprocess (see unified_asr_server.py).

    NeMo needs its own venv (Python 3.11), so the model runs out-of-process
    and this adapter mirrors just enough of onnx-asr's timestamped-recognize
    interface for ParakeetASRBuffer — which then contributes LocalAgreement-2,
    committed-audio eviction, and the sentence buffer unchanged. Env overrides:
    UNIFIED_PYTHON (default ~/nemo-venv/bin/python), UNIFIED_SERVER (default:
    unified_asr_server.py next to this file), UNIFIED_REPLY_TIMEOUT (seconds,
    default 30).

    Every reply is read against a deadline. A dead or wedged server never
    answers; silence is NOT that case — the server replies to every push,
    with nothing new when nothing was said, so a quiet room still yields one
    line per chunk. Decode takes ~165 ms per 1.5 s chunk on the RTX 3060, so
    30 s is a stall, not a slow one. The unbounded readline() this replaces
    left the audio thread hung until the scheduler's 900 s watchdog noticed.
    """

    is_streaming = True  # server owns the stream: labels are final once emitted

    def __init__(self):
        import os
        self._python = os.environ.get(
            "UNIFIED_PYTHON", os.path.expanduser("~/nemo-venv/bin/python"))
        self._script = os.environ.get(
            "UNIFIED_SERVER", str(Path(__file__).parent / "unified_asr_server.py"))
        self.reply_timeout = float(os.environ.get("UNIFIED_REPLY_TIMEOUT", "30"))
        self._proc = None
        self._rbuf = b""      # read from the pipe, not yet consumed
        self.protocol = 1

    def start(self) -> None:
        import json
        import subprocess
        self._proc = subprocess.Popen(
            [self._python, self._script],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
        )
        # ~40 s to load warm; a first-time download can take far longer.
        msg = json.loads(self._read_reply(timeout=900.0) or b"{}")
        if not msg.get("ready"):
            raise RuntimeError(f"unified ASR server failed to start: {msg}")
        self.protocol = int(msg.get("protocol", 1))
        logger.info("unified ASR server ready: protocol {} ({})", self.protocol, self._python)

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _read_reply(self, timeout: float) -> bytes:
        """One line from the server, or raise once `timeout` passes without one.

        Reads the raw descriptor rather than the BufferedReader so select()
        can never be fooled by bytes already sitting in Python's buffer.
        """
        import os
        import select
        fd = self._proc.stdout.fileno()
        deadline = time.monotonic() + timeout
        while b"\n" not in self._rbuf:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                logger.error("unified ASR server: no reply in {:.0f}s — killing it", timeout)
                self.stop()
                raise RuntimeError(f"unified ASR server stalled (no reply in {timeout:.0f}s)")
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                raise RuntimeError("unified ASR server closed the pipe")
            self._rbuf += chunk
        line, _, self._rbuf = self._rbuf.partition(b"\n")
        return line

    def recognize(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> "_RemoteResult":
        import json
        import struct
        if not self.alive():
            raise RuntimeError("unified ASR server is not running")
        data = np.ascontiguousarray(audio, dtype=np.float32).tobytes()
        self._proc.stdin.write(struct.pack("<I", len(data)))
        self._proc.stdin.write(data)
        self._proc.stdin.flush()
        msg = json.loads(self._read_reply(self.reply_timeout))
        if "error" in msg:
            raise RuntimeError(f"unified ASR server: {msg['error']}")
        return _RemoteResult(list(msg["tokens"]),
                             [float(t) for t in msg["timestamps"]],
                             [float(t) for t in msg.get("ends", [])] or None,
                             msg.get("count"))

    def flush(self) -> Optional["_RemoteResult"]:
        """Signal end of stream; returns whatever the server had left to say."""
        import json
        import struct
        if not self.alive():
            return None
        try:
            self._proc.stdin.write(struct.pack("<I", 0xFFFFFFFF))
            self._proc.stdin.flush()
            msg = json.loads(self._read_reply(self.reply_timeout))
            if "error" in msg:
                return None
            return _RemoteResult(list(msg["tokens"]),
                                 [float(t) for t in msg["timestamps"]],
                                 None, msg.get("count"))
        except Exception:
            return None

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5.0)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass


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
    expanded (▁ → space) by the time they land in TimestampedResult.tokens.
    We *preserve* leading whitespace on the result because downstream
    consumers (the SentenceBuffer) need it to tell word continuations from
    new words. For example, when commit N emits `" just"` and commit N+1
    emits `"ice"`, the SentenceBuffer must concatenate them directly to get
    `"justice"`. If we stripped here, both fragments would lose their
    boundary signal and the buffer's space-join would produce `"just ice"`.
    """
    return "".join(tokens)


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
        model_path: Optional[str] = None,
    ):
        self._model_name = model_name
        # onnx_asr.load_model() doesn't accept cache_dir directly — HF downloads
        # honor the HF_HOME env var, so we export it before importing if the
        # caller asked for a specific cache location.
        self._cache_dir = cache_dir
        self._providers = providers
        self._quantization = quantization
        # Local-path mode: when model_path is set, onnx-asr loads from disk
        # rather than downloading from HF. model_name then has to be the
        # *adapter type* (e.g. "nemo-conformer-tdt") not a HF repo id. Useful
        # for community exports that need a hand-written config.json or
        # checkpoints that aren't published on HF.
        self._model_path = model_path

        self._model = None  # TimestampedResultsAsrAdapter
        self._buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        self._buffer_start_wall: float = 0.0
        self._session_started: bool = False
        self._session_start_wall: float = 0.0

        # LocalAgreement-2 state
        self._prev_tokens: List[str] = []
        self._prev_timestamps: List[float] = []
        self._prev_ends: List[float] = []  # word END times (unified-remote only)
        self._committed_count: int = 0  # tokens already emitted from current buffer

    def load(self) -> None:
        """Load Parakeet model. Must be called before feed()."""
        import os
        if self._cache_dir:
            Path(self._cache_dir).mkdir(parents=True, exist_ok=True)
            # onnx_asr delegates downloads to huggingface_hub, which respects
            # HF_HOME. Set it only if the caller didn't already.
            os.environ.setdefault("HF_HOME", self._cache_dir)

        # "unified-remote" = nvidia/parakeet-unified-en-0.6b via the NeMo
        # subprocess server; everything downstream (LocalAgreement, eviction,
        # sentence buffer) is shared with the onnx-asr path.
        if self._model_name == "unified-remote":
            remote = _RemoteUnifiedModel()
            remote.start()
            self._model = remote
            logger.info("Parakeet loaded: unified-remote (parakeet-unified-en-0.6b, server={})",
                        remote._python)
            return

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
            import os
            import onnxruntime as ort
            avail = ort.get_available_providers()
            providers = []
            # PARAKEET_DEVICE=cuda puts ASR on the NVIDIA GPU (onnxruntime-gpu).
            # Default stays CPU: on the RTX 3060 box the GPU was historically
            # reserved for the backlog worker, and CPU was fast enough for the
            # 0.6B model. The 1.1B model on a laptop-class CPU is not — the
            # 2026-09-01 live smoke test averaged 1.6 s ASR per 1.5 s chunk.
            want = os.environ.get("PARAKEET_DEVICE", "cpu").strip().lower()
            if want == "cuda" and "CUDAExecutionProvider" in avail:
                providers.append("CUDAExecutionProvider")
            elif want == "cuda":
                logger.warning("PARAKEET_DEVICE=cuda but CUDAExecutionProvider is not available ({}); using CPU", avail)
            if "ROCMExecutionProvider" in avail:
                providers.append("ROCMExecutionProvider")
            providers.append("CPUExecutionProvider")

        kwargs = {"providers": providers}
        if self._quantization:
            kwargs["quantization"] = self._quantization
        if self._model_path:
            kwargs["path"] = self._model_path

        base = onnx_asr.load_model(self._model_name, **kwargs)
        # with_timestamps() returns an adapter whose recognize() yields
        # TimestampedResult(text, timestamps, tokens, logprobs). We use the
        # timestamps for buffer trimming and commit-boundary reporting.
        self._model = base.with_timestamps()
        logger.info(
            "Parakeet loaded: model={}{} providers={}",
            self._model_name,
            f" path={self._model_path}" if self._model_path else "",
            providers,
        )

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

        # Streaming-native backend (unified-remote): the server owns the audio
        # stream and its emitted labels are final. Send only the NEW chunk,
        # take the token-count diff as committed text, and skip LocalAgreement
        # and eviction entirely — there is no rolling buffer on this side.
        if getattr(self._model, "is_streaming", False):
            asr_start = time.time()
            result = self._model.recognize(audio, sample_rate=SAMPLE_RATE)
            asr_time = time.time() - asr_start
            new_tokens, first_ts = self._streaming_delta(result)
            new_text = _tokens_to_text(new_tokens) if new_tokens else ""
            if not new_text:
                return None
            return (new_text, self._session_start_wall + first_ts, asr_time)

        self._buffer = np.concatenate([self._buffer, audio])

        asr_start = time.time()
        result = self._model.recognize(self._buffer, sample_rate=SAMPLE_RATE)
        asr_time = time.time() - asr_start

        tokens = list(result.tokens or [])
        timestamps = list(result.timestamps or [])
        ends = list(getattr(result, "ends", None) or [])
        if len(ends) != len(tokens):
            ends = []
        # Defensive: if the model returned no timestamps, length mismatch would
        # break our trim logic. Treat as "no commit possible this step".
        if len(timestamps) != len(tokens):
            self._prev_tokens = tokens
            self._prev_timestamps = []
            self._prev_ends = []
            self._maybe_hard_drop()
            return None

        # LocalAgreement-2: the new stable prefix is the LCP of the last two
        # hypotheses. Anything beyond the previous committed count is fresh
        # text to emit.
        stable_len = _longest_common_prefix(self._prev_tokens, tokens)
        newly_count = stable_len - self._committed_count

        self._prev_tokens = tokens
        self._prev_timestamps = timestamps
        self._prev_ends = ends

        if newly_count <= 0:
            self._maybe_trim(tokens, timestamps, ends, stable_len)
            self._maybe_hard_drop()
            return None

        new_tokens = tokens[self._committed_count:stable_len]
        new_text = _tokens_to_text(new_tokens)

        first_ts = timestamps[self._committed_count]
        text_start_wall = self._buffer_start_wall + float(first_ts)

        self._committed_count = stable_len
        self._maybe_trim(tokens, timestamps, ends, stable_len)
        self._maybe_hard_drop()

        if not new_text:
            return None
        return (new_text, text_start_wall, asr_time)

    def flush(self) -> Optional[Tuple[str, float, float]]:
        """Emit any remaining uncommitted text at shutdown."""
        if self._model is None:
            return None
        # Streaming-native backend: ask the server to finalize the stream and
        # emit whatever tokens arrived after the last push. (The local buffer
        # is always empty in this mode.)
        if getattr(self._model, "is_streaming", False):
            asr_start = time.time()
            result = self._model.flush()
            asr_time = time.time() - asr_start
            if result is None:
                return None
            new_tokens, first_ts = self._streaming_delta(result)
            text = _tokens_to_text(new_tokens) if new_tokens else ""
            if not text:
                return None
            return (text, self._session_start_wall + first_ts, asr_time)
        if len(self._buffer) == 0:
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

    def alive(self) -> bool:
        """False once the ASR backend is gone for good — the server died, or
        was killed for stalling. A transient decode error leaves this True."""
        if self._model is None:
            return False
        check = getattr(self._model, "alive", None)
        return True if check is None else bool(check())

    def _streaming_delta(self, result) -> Tuple[List[str], float]:
        """New tokens, and the start time of the first, from a streaming reply
        of either protocol (see _RemoteResult)."""
        toks = list(result.tokens or [])
        times = list(result.timestamps or [])
        if result.count is None:
            # Protocol 1: the whole stream; the diff is ours to take.
            if len(toks) <= self._committed_count:
                return [], 0.0
            first = times[self._committed_count] if self._committed_count < len(times) else 0.0
            new = toks[self._committed_count:]
            self._committed_count = len(toks)
            return new, float(first)
        # Protocol 2: only what is new, plus the server's running total.
        if not toks:
            return [], 0.0
        first = float(times[0]) if times else 0.0
        self._committed_count += len(toks)
        if result.count != self._committed_count:
            logger.warning("unified ASR: token count drifted (server {}, client {}); resyncing",
                           result.count, self._committed_count)
            self._committed_count = int(result.count)
        return toks, first

    def _maybe_trim(self, tokens, timestamps, ends, stable_len: int) -> None:
        """Evict audio whose words are already committed.

        Runs after every decode, not only when the buffer is huge: as soon as
        >= EVICT_COMMITTED_SEC of committed audio sits at the head of the
        buffer, cut it at the start of the first *uncommitted* token (per-token
        timestamps are starts, so cutting at `timestamps[stable_len]` removes
        exactly the committed words and nothing else; with no uncommitted token
        to anchor on we step past the last committed one by a word-duration
        margin). Only committed audio ever leaves, so duplicate re-emission is
        structurally impossible on this path.

        Crucially, the running hypothesis is *shifted*, not discarded: the
        uncommitted tokens survive with their timestamps moved left, so
        LocalAgreement keeps its continuity and can commit again on the very
        next decode. (The old code reset the trackers, costing a decode of
        re-stabilization — and only trimmed past MAX_BUFFER_SEC, which let
        buffers grow until the panic path duplicated or lost words; observed
        2026-09-01 as 187-486 inserted words per 45-minute run.)
        """
        if stable_len <= 0:
            return
        cut_idx = stable_len  # tokens [0, cut_idx) leave the buffer
        if ends and stable_len <= len(ends):
            # Unified-remote path. Word-boundary cuts clip the next word even
            # at the previous word's reported END (co-articulation + RNNT
            # timestamp lag; observed 2026-09-02 as one lost word per eviction,
            # 222 dels/pass, still 95 after cutting at ends). This model emits
            # punctuation, so cut at the last committed *sentence-final* word
            # instead — a real pause lives there and nothing co-articulates
            # across it. Only if no sentence end is committed and the buffer
            # is oversized do we fall back to a word-end cut.
            punct_idx = None
            for k in range(stable_len - 1, -1, -1):
                if tokens[k].rstrip().endswith((".", "?", "!")):
                    punct_idx = k
                    break
            buffer_sec = len(self._buffer) / SAMPLE_RATE
            if punct_idx is not None and float(ends[punct_idx]) >= EVICT_COMMITTED_SEC:
                cut_idx = punct_idx + 1
                cut_ts = float(ends[punct_idx]) + 0.05
            elif buffer_sec > MAX_BUFFER_SEC:
                cut_ts = float(ends[stable_len - 1])
            else:
                return
            cut_samples = int(cut_ts * SAMPLE_RATE)
            if cut_samples <= 0 or cut_samples >= len(self._buffer):
                return
            self._buffer = self._buffer[cut_samples:]
            self._buffer_start_wall += cut_ts
            # Shift the hypothesis; committed-but-kept tokens (those between
            # cut_idx and stable_len when we cut at punctuation) remain in
            # prev so LocalAgreement continuity holds, and committed_count
            # marks them as already emitted.
            self._prev_tokens = list(tokens[cut_idx:])
            self._prev_timestamps = [t - cut_ts for t in timestamps[cut_idx:]]
            self._prev_ends = [t - cut_ts for t in ends[cut_idx:]]
            self._committed_count = stable_len - cut_idx
            return
        if stable_len < len(timestamps):
            # Cut point: start of the first *uncommitted* token when it exists.
            cut_ts = float(timestamps[stable_len])
        else:
            cut_ts = float(timestamps[stable_len - 1]) + 0.35
        buffer_sec = len(self._buffer) / SAMPLE_RATE
        if cut_ts < EVICT_COMMITTED_SEC and buffer_sec <= MAX_BUFFER_SEC:
            return
        cut_samples = int(cut_ts * SAMPLE_RATE)
        if cut_samples <= 0 or cut_samples >= len(self._buffer):
            return
        self._buffer = self._buffer[cut_samples:]
        self._buffer_start_wall += cut_ts
        # Shift, don't reset: uncommitted tokens are unchanged by the cut,
        # their audio just moved to the front of the buffer.
        self._prev_tokens = list(tokens[stable_len:])
        self._prev_timestamps = [t - cut_ts for t in timestamps[stable_len:]]
        self._prev_ends = [t - cut_ts for t in ends[stable_len:]] if ends else []
        self._committed_count = 0

    def _maybe_hard_drop(self) -> None:
        """Last-resort drop when LocalAgreement keeps disagreeing.

        With continuous eviction in _maybe_trim this only fires when nothing
        has stabilized for a long stretch (applause, music, noise). Committed
        audio has already been evicted, so what we drop was never emitted —
        the loss is explicit and logged instead of silently mangled.
        """
        buffer_sec = len(self._buffer) / SAMPLE_RATE
        if buffer_sec <= HARD_DROP_BUFFER_SEC:
            return
        drop = buffer_sec - TRIM_KEEP_SEC
        keep_samples = int(TRIM_KEEP_SEC * SAMPLE_RATE)
        self._buffer = self._buffer[-keep_samples:]
        self._buffer_start_wall += drop
        self._prev_tokens = []
        self._prev_timestamps = []
        self._prev_ends = []
        self._committed_count = 0
        logger.warning(
            "parakeet: dropped {:.1f}s of audio that never stabilized "
            "(buffer hit {:.1f}s; kept last {:.1f}s) — that speech is lost",
            drop, buffer_sec, TRIM_KEEP_SEC,
        )
