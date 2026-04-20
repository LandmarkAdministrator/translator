#!/bin/bash
#
# Launch Parakeet streaming + NLLB-200 3.3B (non-distilled, heaviest) + MMS-TTS-hat.
#
# This is the quality ceiling for translation: 3.3B params, non-distilled, likely
# the best NLLB quality we can get on this hardware. First run will download
# ~17 GB into models/translation/. GPU (ROCm) is used automatically if available.
#
# If this keeps up in real time (queue_depth stays near 0, no DROP lines),
# stick with it — you don't need to try the lighter variants.

set -e
cd "$(dirname "$0")/.."

export NLLB_MODEL="facebook/nllb-200-3.3B"
export HT_TTS="mms"

echo "============================================================"
echo "Parakeet ASR + NLLB-200 3.3B translation + MMS-TTS (ht)"
echo "NLLB_MODEL=$NLLB_MODEL"
echo "NLLB_DEVICE=${NLLB_DEVICE:-auto}  HT_TTS=$HT_TTS"
echo "============================================================"

exec ./venv/bin/python run.py --parakeet "$@"
