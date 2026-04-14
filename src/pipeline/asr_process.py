"""
ASR Subprocess Wrapper

Runs Whisper inference in a dedicated child process so it has its own Python
interpreter and GIL.  The audio callback thread can always run immediately
regardless of how long Whisper takes, which eliminates input overflow errors.

Flow (batch mode only — streaming still uses direct ASR):

    Main process                        ASR child process
    ────────────────                    ─────────────────
    _audio_callback  ──► input_q  ──►  _asr_worker loads model,
    (fast, no block)                    loops: transcribe → output_q
                                                  │
    _asr_result_loop ◄── output_q ◄──────────────┘
    dispatches to translation pipelines
"""

import multiprocessing
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Metadata that travels with each audio chunk through the queue
# ---------------------------------------------------------------------------

@dataclass
class ASRChunkMeta:
    """Audio chunk context preserved while the chunk waits in the queue."""
    chunk_start_time: float
    chunk_duration: float
    emit_reason: str
    peak_rms: float
    sample_rate: int
    asr_time: float = 0.0   # Filled in by worker after transcription completes


# ---------------------------------------------------------------------------
# Worker — runs inside the child process
# ---------------------------------------------------------------------------

def _asr_worker(
    model_size: str,
    device: str,
    language: str,
    download_root: Optional[str],
    input_q: "multiprocessing.Queue",
    output_q: "multiprocessing.Queue",
) -> None:
    """
    Entry point for the ASR subprocess.

    Loads the Whisper model, sends a 'ready' signal, then loops: read audio
    from input_q → transcribe → put result on output_q.
    """
    # Ensure src/ is importable in the spawned process
    src_dir = str(Path(__file__).parent.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Load ROCm environment variables for AMD GPU (written by install.sh)
    _env_file = Path(__file__).parent.parent.parent / ".env.rocm"
    if _env_file.exists():
        import os
        with open(_env_file) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line.startswith("export "):
                    _line = _line[7:]
                if "=" in _line and not _line.startswith("#"):
                    _key, _, _val = _line.partition("=")
                    os.environ.setdefault(_key.strip(), _val.strip())

    # Load model
    try:
        if device == "cuda":
            from pipeline.asr import WhisperTransformersService
            asr = WhisperTransformersService(
                model_size=model_size,
                language=language,
                device=device,
                download_root=download_root,
            )
        else:
            from pipeline.asr import ASRService
            asr = ASRService(
                model_size=model_size,
                language=language,
                device=device,
                download_root=download_root,
            )
        asr.load()
        output_q.put(("ready", None, None))
    except Exception as exc:
        output_q.put(("load_error", str(exc), None))
        return

    # Processing loop
    while True:
        try:
            item = input_q.get()
        except (EOFError, OSError):
            break

        if item is None:        # shutdown sentinel
            break

        audio, meta = item
        try:
            t0 = time.time()
            result = asr.transcribe(audio, meta.sample_rate)
            meta.asr_time = time.time() - t0
            output_q.put(("result", result, meta))
        except Exception as exc:
            output_q.put(("error", str(exc), meta))

    output_q.put(("done", None, None))


# ---------------------------------------------------------------------------
# ASRProcess — used by TranslationCoordinator
# ---------------------------------------------------------------------------

class ASRProcess:
    """
    Wraps Whisper ASR in a dedicated subprocess (separate GIL).

    Usage::

        proc = ASRProcess(model_size="large-v3", device="cuda", language="en")
        proc.start()                      # blocks until model is loaded
        proc.submit(audio_array, meta)    # non-blocking
        result_item = proc.get_result()   # (TranscriptionResult, ASRChunkMeta) or None
        proc.stop()
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        language: str = "en",
        download_root: Optional[str] = None,
    ):
        self._model_size = model_size
        self._device = device
        self._language = language
        self._download_root = download_root

        # Use spawn so the child gets a clean interpreter (required for ROCm/CUDA)
        ctx = multiprocessing.get_context("spawn")
        self._input_q = ctx.Queue(maxsize=4)   # bounded — drop if falling behind
        self._output_q = ctx.Queue()
        self._process = ctx.Process(
            target=_asr_worker,
            args=(model_size, device, language, download_root,
                  self._input_q, self._output_q),
            daemon=True,
            name="ASRProcess",
        )

    def start(self, load_timeout: float = 300.0) -> None:
        """
        Start the subprocess and block until the model finishes loading.

        Args:
            load_timeout: Max seconds to wait (large-v3 can take ~30s on first run).
        """
        print(f"Starting ASR subprocess (model={self._model_size}, device={self._device})...")
        self._process.start()

        try:
            msg_type, data, _ = self._output_q.get(timeout=load_timeout)
        except Exception:
            self._process.terminate()
            raise RuntimeError("ASR subprocess timed out waiting for model load")

        if msg_type == "load_error":
            self._process.terminate()
            raise RuntimeError(f"ASR subprocess failed to load: {data}")

        print(f"ASR subprocess ready (pid={self._process.pid})")

    def submit(self, audio: np.ndarray, meta: ASRChunkMeta) -> bool:
        """
        Submit an audio chunk for transcription. Non-blocking.

        Returns False if the queue is full (4 chunks backed up) — caller
        should log the drop and continue; don't block.
        """
        try:
            self._input_q.put_nowait((audio, meta))
            return True
        except Exception:
            return False

    def get_result(self, timeout: float = 0.1) -> Optional[Tuple]:
        """
        Try to get a completed result from the subprocess.

        Returns (TranscriptionResult, ASRChunkMeta), or None if not ready.
        """
        try:
            msg_type, result, meta = self._output_q.get(timeout=timeout)
        except Exception:
            return None

        if msg_type == "result":
            return result, meta
        if msg_type == "error":
            from loguru import logger
            logger.error("ASR subprocess error: {}", result)
            return None
        return None     # "done" or unexpected

    def stop(self) -> None:
        """Gracefully shut down the subprocess."""
        try:
            self._input_q.put_nowait(None)  # sentinel
        except Exception:
            pass
        if self._process.is_alive():
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)
