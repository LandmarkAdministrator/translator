#!/bin/bash
#
# Launch Parakeet streaming + NLLB-200-distilled-600M + MMS-TTS-hat.
#
# Lightweight fallback: distilled 600M — smallest of the three, still much
# better than Opus-MT for Haitian. First run downloads ~2.4 GB into
# models/translation/. Fast enough to run on CPU if GPU isn't available.

set -e
cd "$(dirname "$0")/.."

export NLLB_MODEL="facebook/nllb-200-distilled-600M"
export HT_TTS="mms"

echo "============================================================"
echo "Parakeet ASR + NLLB-200-distilled-600M translation + MMS-TTS (ht)"
echo "NLLB_MODEL=$NLLB_MODEL"
echo "NLLB_DEVICE=${NLLB_DEVICE:-auto}  HT_TTS=$HT_TTS"
echo "============================================================"

exec ./venv/bin/python run.py --parakeet "$@"
