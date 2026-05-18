#!/bin/bash
#
# Run the production pipeline three times with different silence_timeout
# settings (0.75, 1.0, 2.0 seconds) against the same audio file. Used to
# pick the best sentence-flush trigger for the 1.1B Parakeet which emits
# unpunctuated output.
#
# Assumes the 1.1B model is in models/asr/parakeet-1.1b/ with config.json.
# Each run takes ~62 minutes (60-min audio at realtime + load/drain), so
# the whole sweep is ~3 hours. Runs sequentially — would compete for GPU
# memory otherwise.
#
# Output: tests/benchmarks/sweep_<run_id>/{0.75s,1.0s,2.0s}.stdout.log
# plus the per-session translator log paths recorded as side files.

set -u
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

INPUT_FILE="${1:-tests/ab_test/audio/stitched_test.wav}"
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: input file not found: $INPUT_FILE"
    exit 1
fi

# Verify the 1.1B model is on disk before kicking off three hours of work
if [ ! -f models/asr/parakeet-1.1b/encoder-model.onnx ] \
   || [ ! -f models/asr/parakeet-1.1b/config.json ]; then
    echo "ERROR: 1.1B model not found at models/asr/parakeet-1.1b/"
    echo "       expected encoder-model.onnx + config.json + ..."
    exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="tests/benchmarks/sweep_$RUN_ID"
mkdir -p "$OUT_DIR"
ABS_INPUT="$(readlink -f "$INPUT_FILE")"

cat <<HEADER | tee "$OUT_DIR/summary.txt"
============================================================
Silence-timeout sweep — Parakeet 1.1B + NLLB-1.3B + Kokoro + MMS-TTS
  Run ID:       $RUN_ID
  Input:        $ABS_INPUT
  Output:       $OUT_DIR/
  Started:      $(date -Iseconds)
  Configs:      silence_timeout = 0.75s, 1.0s, 2.0s
HEADER

run_one () {
    local TIMEOUT="$1"
    local LABEL="${TIMEOUT}s"
    local STDOUT_LOG="$OUT_DIR/$LABEL.stdout.log"
    local SESSION_PATH_FILE="$OUT_DIR/$LABEL.session_log_path"

    echo
    echo "------------------------------------------------------------"
    echo "[silence_timeout=${TIMEOUT}s] starting at $(date +%H:%M:%S)"
    echo "  stdout log: $STDOUT_LOG"
    echo "------------------------------------------------------------"

    # Snapshot existing translator logs so we can identify the new one
    local PRE_LOGS
    PRE_LOGS="$(ls -1 logs/translator_*.log.gz 2>/dev/null | sort -u)"

    local START_EPOCH; START_EPOCH=$(date +%s)

    SENTENCE_SILENCE_TIMEOUT="$TIMEOUT" \
    PARAKEET_MODEL="nemo-conformer-tdt" \
    PARAKEET_PATH="$PROJECT_ROOT/models/asr/parakeet-1.1b" \
        bash scripts/run_production.sh --input-file "$ABS_INPUT" 2>&1 \
        | tee "$STDOUT_LOG"
    local EXIT_CODE=${PIPESTATUS[0]}

    local END_EPOCH; END_EPOCH=$(date +%s)
    local ELAPSED=$(( END_EPOCH - START_EPOCH ))

    local POST_LOGS NEW_LOG
    POST_LOGS="$(ls -1 logs/translator_*.log.gz 2>/dev/null | sort -u)"
    NEW_LOG="$(comm -13 <(echo "$PRE_LOGS") <(echo "$POST_LOGS") | tail -1)"
    echo "$NEW_LOG" > "$SESSION_PATH_FILE"

    cat <<FOOTER | tee -a "$OUT_DIR/summary.txt"
------------------------------------------------------------
[silence_timeout=${TIMEOUT}s] finished at $(date +%H:%M:%S)
  elapsed:     ${ELAPSED}s
  exit code:   ${EXIT_CODE}
  session log: ${NEW_LOG:-<none-detected>}
  stdout log:  $STDOUT_LOG
------------------------------------------------------------
FOOTER
}

run_one "0.75"
run_one "1.0"
run_one "2.0"

echo | tee -a "$OUT_DIR/summary.txt"
echo "============================================================" | tee -a "$OUT_DIR/summary.txt"
echo "Sweep complete." | tee -a "$OUT_DIR/summary.txt"
echo "  Finished:  $(date -Iseconds)" | tee -a "$OUT_DIR/summary.txt"
echo "  Outputs:   $OUT_DIR/" | tee -a "$OUT_DIR/summary.txt"
echo "============================================================" | tee -a "$OUT_DIR/summary.txt"
