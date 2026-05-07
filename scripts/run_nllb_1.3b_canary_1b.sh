#!/bin/bash
#
# ASR-upgrade test rig.
#
# Same NLLB-200-distilled-1.3B + Kokoro (es) + MMS-TTS-hat (ht) translation
# stack as run_nllb_1.3b.sh, but swaps Parakeet TDT 0.6B v3 → NVIDIA Canary
# 1B v2. Canary is the officially-onnx-asr-supported "1B class" English ASR
# from NVIDIA — Parakeet 1.1B isn't in onnx-asr's blessed list.
#
# Note: PARAKEET_MODEL env var is the existing knob in the coordinator;
# the name predates Canary support. It's just the onnx-asr model id passed
# through to load_model().
#
# First run downloads ~1 GB extra into models/asr/parakeet/.
#
# Usage:
#   ./scripts/run_nllb_1.3b_canary_1b.sh --input-file path/to/audio.wav
#   ./scripts/run_nllb_1.3b_canary_1b.sh                 # mic input

set -e
cd "$(dirname "$0")/.."

export NLLB_MODEL="facebook/nllb-200-distilled-1.3B"
export HT_TTS="mms"
export ES_TTS="kokoro"
export PARAKEET_MODEL="nemo-canary-1b-v2"

echo "============================================================"
echo "ASR: NeMo Canary 1B v2 + NLLB-200-distilled-1.3B translation"
echo "  PARAKEET_MODEL=$PARAKEET_MODEL  (passed to onnx-asr.load_model)"
echo "  ES TTS: Kokoro (em_alex)"
echo "  HT TTS: facebook/mms-tts-hat"
echo "  NLLB_DEVICE=${NLLB_DEVICE:-auto}"
echo "============================================================"

exec ./venv/bin/python run.py --parakeet "$@"
