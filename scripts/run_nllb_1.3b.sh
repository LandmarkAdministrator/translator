#!/bin/bash
#
# Launch Parakeet streaming + NLLB-200-distilled-1.3B + Kokoro (es) + MMS-TTS-hat (ht).
#
# Mid-tier: distilled 1.3B. Quality on par with the 3.3B for short sentences,
# but ~3x faster, and stable on this hardware where 3.3B in bf16 hallucinates.
# First run downloads ~5 GB (NLLB) + ~330 MB (Kokoro) + ~150 MB (MMS-tts-hat).
#
# Pass `--input-file path/to/audio.wav` to run from a file instead of the mic.

set -e
cd "$(dirname "$0")/.."

export NLLB_MODEL="facebook/nllb-200-distilled-1.3B"
export HT_TTS="mms"
export ES_TTS="kokoro"

echo "============================================================"
echo "Parakeet ASR + NLLB-200-distilled-1.3B translation"
echo "  ES TTS: Kokoro (em_alex)"
echo "  HT TTS: facebook/mms-tts-hat"
echo "  NLLB_DEVICE=${NLLB_DEVICE:-auto}"
echo "============================================================"

exec ./venv/bin/python run.py --parakeet "$@"
