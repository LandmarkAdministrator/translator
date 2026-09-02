#!/bin/bash
#
# Production stack — the live translator pipeline configured the way we'll
# ship it. Sets every environment variable so this is fully reproducible.
#
# Stack:
#   ASR             nemo-parakeet-tdt-0.6b-v3   CPU (onnx-asr default)
#   Translation     facebook/nllb-200-distilled-1.3B   CUDA fp16
#   Spanish TTS     hexgrad/Kokoro-82M voice em_alex   CUDA
#   Haitian TTS     facebook/mms-tts-hat   CUDA
#
# Sentence buffer between Parakeet and translation: 2 s silence timeout,
# 10 s hard timeout, min 3 words, max 800 chars.
#
# Usage:
#   ./scripts/run_production.sh                                      # live mic input
#   ./scripts/run_production.sh --input-file path/to/audio.wav       # file input (reproducible test)
#
# First run downloads ~5 GiB (NLLB-1.3B) + ~330 MiB (Kokoro) + ~150 MiB
# (MMS-TTS-hat) + ~600 MiB (Parakeet) into models/.

set -e
cd "$(dirname "$0")/.."

# ----- Translation backend ---------------------------------------------------
export NLLB_MODEL="facebook/nllb-200-distilled-1.3B"
export NLLB_DEVICE="cuda"
export NLLB_DTYPE="fp16"

# ----- TTS backends ----------------------------------------------------------
export ES_TTS="kokoro"
export KOKORO_DEVICE="cuda"
export HT_TTS="mms"
export MMS_DEVICE="cuda"

# ----- Sentence buffer (defaults match the comparison report) ----------------
# Overridable from the environment (the timing sweep and head-to-head runs
# rely on this; before 2026-09-01 these were hard-coded and silently ignored
# any override).
export SENTENCE_SILENCE_TIMEOUT="${SENTENCE_SILENCE_TIMEOUT:-2.0}"
export SENTENCE_HARD_TIMEOUT="${SENTENCE_HARD_TIMEOUT:-10.0}"
export SENTENCE_MIN_WORDS="${SENTENCE_MIN_WORDS:-3}"
export SENTENCE_MAX_CHARS="${SENTENCE_MAX_CHARS:-800}"

# ----- ASR -------------------------------------------------------------------
# Parakeet 0.6B v3 multilingual TDT — current best streaming-compatible ASR
# in the onnx-asr library. Default in coordinator unless PARAKEET_MODEL is set.
# export PARAKEET_MODEL="nemo-parakeet-tdt-0.6b-v3"

# ----- Environment setup -----------------------------------------------------
# onnxruntime-gpu needs PyTorch's bundled CUDA shared libs on its loader path.
VENV_NVIDIA="$(pwd)/venv/lib/python3.13/site-packages/nvidia"
if [ -d "$VENV_NVIDIA" ]; then
    LD_PATHS="$(find "$VENV_NVIDIA" -name 'lib' -type d | tr '\n' ':')"
    export LD_LIBRARY_PATH="${LD_PATHS}${LD_LIBRARY_PATH:-}"
fi

# Kokoro's phonemizer-fork needs the bundled espeak-ng library and data dir.
ESPEAK_LIB="$(./venv/bin/python -c 'import espeakng_loader; print(espeakng_loader.get_library_path())' 2>/dev/null || true)"
ESPEAK_DATA="$(./venv/bin/python -c 'import espeakng_loader; print(espeakng_loader.get_data_path())' 2>/dev/null || true)"
[ -n "$ESPEAK_LIB" ] && export PHONEMIZER_ESPEAK_LIBRARY="$ESPEAK_LIB"
[ -n "$ESPEAK_DATA" ] && export ESPEAK_DATA_PATH="$ESPEAK_DATA"

echo "============================================================"
echo "Production pipeline"
echo "  ASR:         Parakeet TDT 0.6B v3 (CPU)"
echo "  Translation: $NLLB_MODEL ($NLLB_DEVICE, $NLLB_DTYPE)"
echo "  Spanish TTS: Kokoro 82M em_alex ($KOKORO_DEVICE)"
echo "  Haitian TTS: $HT_TTS-tts-hat ($MMS_DEVICE)"
echo "  Sentence buffer: silence=${SENTENCE_SILENCE_TIMEOUT}s hard=${SENTENCE_HARD_TIMEOUT}s min_words=${SENTENCE_MIN_WORDS}"
echo "============================================================"

exec ./venv/bin/python run.py --parakeet "$@"
