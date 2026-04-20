#!/bin/bash
#
# Launch Parakeet streaming + NLLB-200-distilled-1.3B + MMS-TTS-hat.
#
# Mid-tier: distilled 1.3B. Quality meaningfully better than Opus-MT, usually
# on par with the 3.3B model for short sentences, much faster. First run
# downloads ~5 GB into models/translation/.

set -e
cd "$(dirname "$0")/.."

export NLLB_MODEL="facebook/nllb-200-distilled-1.3B"
export HT_TTS="mms"

echo "============================================================"
echo "Parakeet ASR + NLLB-200-distilled-1.3B translation + MMS-TTS (ht)"
echo "NLLB_MODEL=$NLLB_MODEL"
echo "NLLB_DEVICE=${NLLB_DEVICE:-auto}  HT_TTS=$HT_TTS"
echo "============================================================"

exec ./venv/bin/python run.py --parakeet "$@"
