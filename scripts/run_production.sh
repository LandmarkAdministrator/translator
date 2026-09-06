#!/bin/bash
#
# Production stack — the live translator pipeline configured the way we'll
# ship it. Sets every environment variable so this is fully reproducible.
#
# Stack:
#   ASR             nvidia/parakeet-unified-en-0.6b via the NeMo subprocess
#                   (PARAKEET_MODEL=unified-remote, set by the service launcher
#                   ~/bin/start-translate-unified); onnx-asr TDT otherwise
#   Translation     facebook/nllb-200-distilled-1.3B   CUDA fp16
#   Spanish TTS     hexgrad/Kokoro-82M voice em_alex   CUDA
#   Haitian TTS     facebook/mms-tts-hat   CUDA
#   Russian TTS     facebook/mms-tts-rus   CUDA
#
# Sentence buffer between the ASR and translation: sentences close on the
# ASR's own punctuation boundary; the 60-word cap and 15 s hard timeout are
# safety valves (docs/CHANGES-2026-09-06.md §2). The defaults below ARE the
# production values, so running this script directly behaves like the service.
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
# Russian uses MMS too. Without this it falls through to the Piper backend,
# which has no Russian voice, and the whole service dies in a restart loop:
#   ValueError: No voice found for language 'ru'. Available: ['ht','es','fr','en']
# Any language added here needs its backend named the same way.
export RU_TTS="mms"
export MMS_DEVICE="cuda"

# ----- Sentence buffer -------------------------------------------------------
# Overridable from the environment (the timing sweep and head-to-head runs
# rely on this; before 2026-09-01 these were hard-coded and silently ignored
# any override). Until 2026-09-06 the defaults here were the OLD policy
# (2 s / 10 s, no cap, no punctuation boundary) while the launcher overrode
# them, so a direct run of this script did not behave like the service.
export SENTENCE_SILENCE_TIMEOUT="${SENTENCE_SILENCE_TIMEOUT:-30.0}"
export SENTENCE_HARD_TIMEOUT="${SENTENCE_HARD_TIMEOUT:-15.0}"
export SENTENCE_MIN_WORDS="${SENTENCE_MIN_WORDS:-3}"
export SENTENCE_MAX_CHARS="${SENTENCE_MAX_CHARS:-800}"
export SENTENCE_MAX_WORDS="${SENTENCE_MAX_WORDS:-60}"
export SENTENCE_PUNCT_BOUNDARY="${SENTENCE_PUNCT_BOUNDARY:-1}"

# ----- ASR -------------------------------------------------------------------
# Production: PARAKEET_MODEL=unified-remote (+ UNIFIED_PYTHON, UNIFIED_BIAS_*),
# exported by the launcher. Unset, the coordinator loads the onnx-asr TDT
# model, which needs no second venv — useful on a machine without NeMo.
# export PARAKEET_MODEL="unified-remote"

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
echo "  ASR:         ${PARAKEET_MODEL:-nemo-parakeet-tdt-0.6b-v3 (onnx-asr)}"
echo "  Translation: $NLLB_MODEL ($NLLB_DEVICE, $NLLB_DTYPE)"
echo "  Spanish TTS: Kokoro 82M em_alex ($KOKORO_DEVICE)"
echo "  Haitian TTS: $HT_TTS-tts-hat ($MMS_DEVICE)"
echo "  Russian TTS: $RU_TTS-tts-rus ($MMS_DEVICE)"
echo "  Sentence buffer: punct_boundary=${SENTENCE_PUNCT_BOUNDARY} max_words=${SENTENCE_MAX_WORDS} hard=${SENTENCE_HARD_TIMEOUT}s silence=${SENTENCE_SILENCE_TIMEOUT}s min_words=${SENTENCE_MIN_WORDS}"
echo "============================================================"

exec ./venv/bin/python run.py "$@"
