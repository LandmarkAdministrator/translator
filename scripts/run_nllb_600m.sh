#!/bin/bash
#
# Launch Parakeet streaming + NLLB-200-distilled-600M + Kokoro (es) + MMS-TTS-hat (ht).
#
# Lightweight + fast: distilled 600M is the smallest NLLB variant. First
# run downloads ~2.4 GB (NLLB) + ~330 MB (Kokoro) + ~150 MB (MMS-tts-hat).
#
# Pass `--input-file path/to/audio.wav` to run from a file instead of the mic
# (reproducible offline tests; pipeline shuts down at EOF).

set -e
cd "$(dirname "$0")/.."

export NLLB_MODEL="facebook/nllb-200-distilled-600M"
export HT_TTS="mms"
export ES_TTS="kokoro"

echo "============================================================"
echo "Parakeet ASR + NLLB-200-distilled-600M translation"
echo "  ES TTS: Kokoro (em_alex)"
echo "  HT TTS: facebook/mms-tts-hat"
echo "  NLLB_DEVICE=${NLLB_DEVICE:-auto}"
echo "============================================================"

exec ./venv/bin/python run.py --parakeet "$@"
