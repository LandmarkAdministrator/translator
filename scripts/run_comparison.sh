#!/bin/bash
#
# Run the 600M and 1.3B NLLB launchers back-to-back against the same
# audio file and capture every stream we'll need for analysis.
#
# Each test runs the full duration of the audio file (60 min real-time on
# the default stitched test), then drains the pipeline cleanly via
# FileInputStream's EOF auto-stop. Total wall-clock for the default file
# is ~2 hours plus ~10 minutes of model downloads and load on first run.
#
# Outputs (under tests/comparison/<RUN_ID>/):
#   600m.stdout.log         full pipeline stdout (incl. [EN]/[ES]/[HT] lines)
#   600m.session_log_path   path to translator's gzipped per-session log
#   1.3b.stdout.log         "
#   1.3b.session_log_path   "
#   summary.txt             durations, logs paths, exit codes
#
# Usage:
#   ./scripts/run_comparison.sh                     # default stitched audio
#   ./scripts/run_comparison.sh path/to/audio.wav   # custom audio
#
# Notes:
#   - First run downloads ~7.5 GB of weights total (NLLB-600M, NLLB-1.3B,
#     Kokoro 82M, MMS-TTS-hat) into models/. Subsequent runs skip downloads.
#   - Ctrl+C interrupts the *current* test only and skips ahead. Use
#     Ctrl+C twice quickly to abort the whole script.
#   - We don't run them in parallel — they would fight for the GPU.

set -u
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

INPUT_FILE="${1:-tests/ab_test/audio/stitched_test.wav}"
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: input file not found: $INPUT_FILE"
    exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="tests/comparison/$RUN_ID"
mkdir -p "$OUT_DIR"

ABS_INPUT="$(readlink -f "$INPUT_FILE")"
INPUT_DURATION_SEC=$(./venv/bin/python -c "
import soundfile as sf
info = sf.info('$ABS_INPUT')
print(int(info.duration))
" 2>/dev/null || echo "unknown")

cat <<HEADER | tee "$OUT_DIR/summary.txt"
============================================================
NLLB 600M vs 1.3B — back-to-back comparison
  Run ID:    $RUN_ID
  Input:     $ABS_INPUT
  Duration:  ${INPUT_DURATION_SEC}s
  Output:    $OUT_DIR/
  Started:   $(date -Iseconds)
============================================================
HEADER

run_one () {
    local LABEL="$1"
    local LAUNCHER="$2"
    local STDOUT_LOG="$OUT_DIR/${LABEL}.stdout.log"
    local SESSION_PATH_FILE="$OUT_DIR/${LABEL}.session_log_path"

    echo
    echo "------------------------------------------------------------"
    echo "[${LABEL}] starting at $(date +%H:%M:%S)"
    echo "  launcher:    $LAUNCHER --input-file $ABS_INPUT"
    echo "  stdout log:  $STDOUT_LOG"
    echo "------------------------------------------------------------"

    # Snapshot existing logs so we can identify which one this run created.
    local PRE_LOGS
    PRE_LOGS="$(ls -1 logs/translator_*.log.gz 2>/dev/null | sort -u)"

    local START_EPOCH
    START_EPOCH=$(date +%s)

    # tee so the user sees progress AND we capture for analysis.
    bash "$LAUNCHER" --input-file "$ABS_INPUT" 2>&1 | tee "$STDOUT_LOG"
    local EXIT_CODE=${PIPESTATUS[0]}

    local END_EPOCH
    END_EPOCH=$(date +%s)
    local ELAPSED=$(( END_EPOCH - START_EPOCH ))

    # Find the new translator log file (the one that wasn't there before).
    local POST_LOGS NEW_LOG
    POST_LOGS="$(ls -1 logs/translator_*.log.gz 2>/dev/null | sort -u)"
    NEW_LOG="$(comm -13 <(echo "$PRE_LOGS") <(echo "$POST_LOGS") | tail -1)"
    echo "$NEW_LOG" > "$SESSION_PATH_FILE"

    cat <<FOOTER | tee -a "$OUT_DIR/summary.txt"
------------------------------------------------------------
[${LABEL}] finished at $(date +%H:%M:%S)
  elapsed:     ${ELAPSED}s
  exit code:   ${EXIT_CODE}
  session log: ${NEW_LOG:-<none-detected>}
  stdout log:  $STDOUT_LOG
------------------------------------------------------------
FOOTER
}

run_one "600m"  "scripts/run_nllb_600m.sh"
run_one "1.3b"  "scripts/run_nllb_1.3b.sh"

echo | tee -a "$OUT_DIR/summary.txt"
echo "============================================================" | tee -a "$OUT_DIR/summary.txt"
echo "Comparison complete." | tee -a "$OUT_DIR/summary.txt"
echo "  Finished:  $(date -Iseconds)" | tee -a "$OUT_DIR/summary.txt"
echo "  Outputs:   $OUT_DIR/" | tee -a "$OUT_DIR/summary.txt"
echo "============================================================" | tee -a "$OUT_DIR/summary.txt"
