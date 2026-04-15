#!/bin/bash
# ==============================================================================
# ASR A/B Test Orchestrator — 8 runs (JFK × 4, Jobs × 4), batch vs streaming
#
# Runs on the LOCAL machine (where audio plays through speakers).
# SSHs to the TEST machine to launch/stop the translator.
#
# Per run:
#   1. Kill any leftover translator on remote
#   2. Launch translator on remote in chosen mode
#   3. Wait WARMUP_SEC for model load
#   4. Play audio locally via ffplay
#   5. Wait TAIL_SEC for last chunk to process
#   6. Send SIGINT to translator → clean shutdown writes SESSION_END
#   7. Archive remote log to tests/ab_test/results/runN_src_mode.log
#   8. Pause INTERRUN_SEC
#
# After all runs: run analyzer on each log + build SUMMARY.txt + SCP back.
#
# Total time: ~2h15m (4 × JFK 14.6min + 4 × Jobs 15.1min + overhead).
# ==============================================================================

set -uo pipefail  # NOT -e: we want to survive a single-run failure

# ---- Config ----
REMOTE_USER="administrator"
REMOTE_HOST="10.1.170.184"
SSH_KEY="/home/administrator/.ssh/mykey"
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ConnectTimeout=10)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIO_DIR="$SCRIPT_DIR/audio"

WARMUP_SEC=40
TAIL_SEC=20
INTERRUN_SEC=30             # Give ALSA/PipeWire time to fully release the device
SHUTDOWN_TIMEOUT=45
PLAYBACK_MAX_SEC=1200       # Hard cap on ffplay (longest audio is ~15.1 min)

# Test plan: run_number | source_key | audio_file | mode
TESTS=(
  "1|jfk|JFK_Inaugural_Address.mp3|batch"
  "2|jfk|JFK_Inaugural_Address.mp3|batch"
  "3|jfk|JFK_Inaugural_Address.mp3|streaming"
  "4|jfk|JFK_Inaugural_Address.mp3|streaming"
  "5|jobs|Steve_Jobs_Stanford.mp3|batch"
  "6|jobs|Steve_Jobs_Stanford.mp3|batch"
  "7|jobs|Steve_Jobs_Stanford.mp3|streaming"
  "8|jobs|Steve_Jobs_Stanford.mp3|streaming"
)

ssh_r() { ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$REMOTE_HOST" "$@"; }
ssh_rf() { ssh -f "${SSH_OPTS[@]}" "$REMOTE_USER@$REMOTE_HOST" "$@"; }

# Kill any running translator on remote. We can't use `pkill -f run.py` because
# the ssh-invoked remote shell's own cmdline contains "run.py" → self-kill (exit 255).
# Filter by comm=python so only actual python processes match.
remote_kill_translator() {
    ssh_r '
        pids=$(ps -eo pid,comm,args --no-headers | awk "\$2==\"python\" && /run\.py/ {print \$1}")
        if [ -n "$pids" ]; then
            kill -INT $pids 2>/dev/null || true
            sleep 3
            kill -9 $pids 2>/dev/null || true
        fi
    '
}

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
hms() { printf '%02d:%02d:%02d' $(( $1 / 3600 )) $(( ($1 % 3600) / 60 )) $(( $1 % 60 )); }

log() { echo "[$(timestamp)] $*"; }

# ---- Pre-flight ----
log "=== PRE-FLIGHT ==="
for f in JFK_Inaugural_Address.mp3 Steve_Jobs_Stanford.mp3; do
    [ -f "$AUDIO_DIR/$f" ] || { log "ERROR: local audio missing: $AUDIO_DIR/$f"; exit 1; }
done
log "Local audio files OK"

command -v ffplay >/dev/null || { log "ERROR: ffplay not installed locally"; exit 1; }
log "ffplay OK"

ssh_r true || { log "ERROR: cannot SSH to $REMOTE_USER@$REMOTE_HOST"; exit 1; }
log "SSH to remote OK"

# Reference files and GPU on remote
ssh_r 'test -f ~/translator/tests/ab_test/reference_jfk_inaugural.txt' || { log "ERROR: remote reference_jfk_inaugural.txt missing"; exit 1; }
ssh_r '
    ref_jobs=~/translator/tests/ab_test/reference_jobs.txt
    test -f "$ref_jobs" || exit 1
    # Count words after the BEGIN marker — fail if empty
    python3 -c "
import sys
from pathlib import Path
p = Path(\"$ref_jobs\")
text = p.read_text()
if \"---REFERENCE BEGINS---\" in text:
    body = text.split(\"---REFERENCE BEGINS---\", 1)[1]
else:
    body = text
body = \"\\n\".join(l for l in body.splitlines() if not l.lstrip().startswith(\"#\")).strip()
words = len(body.split())
if words < 500:
    sys.exit(f\"reference_jobs.txt has only {words} words — paste full transcript\")
print(f\"reference_jobs.txt: {words} words\")
"
' || { log "ERROR: remote reference_jobs.txt is empty or too short — paste the Jobs transcript first"; exit 1; }

ssh_r '~/translator/venv/bin/python -c "import torch; assert torch.cuda.is_available(), \"no GPU\"; print(\"GPU:\", torch.cuda.get_device_name(0))"' \
    || { log "ERROR: remote GPU check failed"; exit 1; }

ssh_r 'mkdir -p ~/translator/tests/ab_test/results'
log "Pre-flight complete"
echo

# ---- Run loop ----
START_TS=$(date +%s)

for test in "${TESTS[@]}"; do
    IFS='|' read -r run src audio mode <<< "$test"
    flags=""
    [ "$mode" = "streaming" ] && flags="--streaming"

    echo
    echo "######################################################################"
    echo "# Run $run/8 | source=$src | mode=$mode | audio=$audio"
    echo "# $(timestamp)"
    echo "######################################################################"

    # 1. Kill leftovers on remote (anything calling run.py)
    remote_kill_translator

    # 2. Launch translator on remote.
    # Use `ssh -f` so SSH returns immediately after auth; plain `ssh "... & disown"`
    # hangs because SSH holds the channel open until remote stdio fds close.
    log "Launching translator on remote (mode=$mode)..."
    ssh_rf "cd ~/translator && nohup venv/bin/python run.py $flags > /tmp/translator_run${run}.out 2>&1 < /dev/null &"
    sleep 5

    # Discover actual python PID. Filtering by comm=python avoids matching the
    # bash wrapper whose cmdline literally contains "python run.py".
    REMOTE_PID=$(ssh_r 'ps -eo pid,comm,args --no-headers | awk "\$2==\"python\" && /run\.py/ {print \$1; exit}"')
    log "  Remote PID: ${REMOTE_PID:-UNKNOWN}"
    if [ -z "$REMOTE_PID" ]; then
        log "  ERROR: translator failed to start. Last output:"
        ssh_r "tail -30 /tmp/translator_run${run}.out"
        log "  Skipping run $run"
        sleep "$INTERRUN_SEC"
        continue
    fi

    # 3. Warmup
    log "Warmup ${WARMUP_SEC}s..."
    sleep "$WARMUP_SEC"

    # Sanity-check translator is alive
    if ! ssh_r "kill -0 $REMOTE_PID 2>/dev/null"; then
        log "  ERROR: translator not running after warmup. Last output:"
        ssh_r "tail -30 /tmp/translator_run${run}.out"
        log "  Skipping run $run"
        sleep "$INTERRUN_SEC"
        continue
    fi

    # 4. Play audio locally. `timeout --kill-after` hard-stops ffplay if the
    # local audio stack wedges (Run 2 of the prior attempt hung for 18h here).
    AUDIO_START=$(date +%s)
    log "Playing $audio locally via ffplay (cap ${PLAYBACK_MAX_SEC}s)..."
    timeout --kill-after=10 "$PLAYBACK_MAX_SEC" \
        ffplay -nodisp -autoexit -loglevel error "$AUDIO_DIR/$audio" < /dev/null
    ff_rc=$?
    AUDIO_END=$(date +%s)
    elapsed=$(( AUDIO_END - AUDIO_START ))
    if [ "$ff_rc" -eq 124 ] || [ "$ff_rc" -eq 137 ]; then
        log "  WARN: ffplay killed by timeout after ${elapsed}s (rc=$ff_rc)"
    else
        log "  Playback finished (${elapsed}s, rc=$ff_rc)"
    fi

    # 5. Tail wait
    log "Tail wait ${TAIL_SEC}s for final chunk processing..."
    sleep "$TAIL_SEC"

    # 6. SIGINT for clean shutdown
    log "Sending SIGINT to remote PID $REMOTE_PID ..."
    ssh_r "kill -INT $REMOTE_PID 2>/dev/null" || true

    # Wait for clean exit
    for i in $(seq 1 "$SHUTDOWN_TIMEOUT"); do
        if ! ssh_r "kill -0 $REMOTE_PID 2>/dev/null"; then
            log "  Parent exited cleanly after ${i}s"
            break
        fi
        sleep 1
    done
    # Force-kill parent if still hung
    ssh_r "kill -9 $REMOTE_PID 2>/dev/null" || true

    # CRITICAL: kill any surviving child python processes. The Whisper ASR
    # subprocess sometimes outlives the parent SIGINT path and keeps the USB
    # audio device open, causing the next run's InputStream to fail with -9998.
    # Filter by comm=python so we don't self-kill the ssh shell.
    ssh_r '
        pids=$(ps -eo pid,comm,args --no-headers | awk "\$2==\"python\" && /run\.py|whisper|multiprocessing/ {print \$1}")
        if [ -n "$pids" ]; then
            kill -9 $pids 2>/dev/null || true
        fi
    '
    sleep 3

    # Wait for the USB audio device to actually be free before the next run.
    # PipeWire can hold it for a few seconds after the owner exits. Probe the
    # device up to 60s; only proceed when InputStream open succeeds.
    log "Waiting for USB audio device to be available..."
    dev_ready=0
    for i in $(seq 1 30); do
        if ssh_r '~/translator/venv/bin/python -c "
import sounddevice as sd, numpy as np, sys
try:
    s = sd.InputStream(device=0, samplerate=48000, channels=2, dtype=np.float32, blocksize=4800)
    s.start(); s.stop(); s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null'; then
            log "  Device ready after ${i}s"
            dev_ready=1
            break
        fi
        sleep 2
    done
    if [ "$dev_ready" = "0" ]; then
        log "  WARN: device still busy after 60s — launching anyway (retry-in-start may recover)"
    fi

    # 7. Archive log on remote.
    # The translator gzips its log on clean shutdown, so the newest file is
    # often "*.log.gz", not "*.log". Ranking both patterns together (and
    # gunzipping if needed) ensures we archive the run we just completed,
    # not some stale non-rotated log from an earlier aborted session.
    LOG_NAME="run${run}_${src}_${mode}.log"
    ssh_r "
        cd ~/translator && \
        LATEST=\$(ls -t logs/translator_*.log logs/translator_*.log.gz 2>/dev/null | head -1); \
        if [ -n \"\$LATEST\" ]; then \
            case \"\$LATEST\" in \
                *.gz) zcat \"\$LATEST\" > tests/ab_test/results/$LOG_NAME ;; \
                *)    cp    \"\$LATEST\" tests/ab_test/results/$LOG_NAME ;; \
            esac && \
            echo 'Archived: $LOG_NAME (source: '\$(basename \$LATEST)', size: '\$(stat -c %s tests/ab_test/results/$LOG_NAME)' bytes)'; \
        else \
            echo 'ERROR: no log to archive'; \
        fi
    "

    # 8. Between-run pause
    log "Between-run pause ${INTERRUN_SEC}s..."
    sleep "$INTERRUN_SEC"
done

END_TS=$(date +%s)
TOTAL=$(( END_TS - START_TS ))
echo
echo "######################################################################"
echo "# ALL 8 RUNS COMPLETE — total $(hms "$TOTAL")"
echo "######################################################################"
echo

# ---- Analysis phase ----
log "Running analyzer on each run log..."
ssh_r '
    set -u
    cd ~/translator
    mkdir -p tests/ab_test/results/analysis
    for f in tests/ab_test/results/run*.log; do
        [ -f "$f" ] || continue
        name=$(basename "$f" .log)
        case "$name" in
            *jfk*)  ref=reference_jfk_inaugural.txt ;;
            *jobs*) ref=reference_jobs.txt ;;
            *)      ref=reference_jfk_inaugural.txt ;;
        esac
        venv/bin/python tests/ab_test/analyze.py "$f" "tests/ab_test/$ref" \
            > "tests/ab_test/results/analysis/${name}.txt" 2>&1
        echo "  Analyzed: $name  (ref=$ref)"
    done
'

log "Building SUMMARY.txt..."
ssh_r '
    cd ~/translator
    {
        echo "=== A/B TEST SUMMARY  $(date) ==="
        echo
        for f in tests/ab_test/results/analysis/run*.txt; do
            [ -f "$f" ] || continue
            name=$(basename "$f" .txt)
            echo "### $name ###"
            grep -E "ASR mode|^  WER |^  substitutions|^  deletions|^  insertions|^  ref words|^  hyp words|^  asr_time|^  chunk_dur|^  RTF |^  e2e |^  translate |^  tts |^  queue_was" "$f" || true
            echo
        done
    } > tests/ab_test/results/SUMMARY.txt
    echo "  Wrote tests/ab_test/results/SUMMARY.txt"
'

# SCP results back to local
LOCAL_RESULTS="$SCRIPT_DIR/results"
mkdir -p "$LOCAL_RESULTS"
log "Copying results to local $LOCAL_RESULTS ..."
scp "${SSH_OPTS[@]}" -r "$REMOTE_USER@$REMOTE_HOST:~/translator/tests/ab_test/results/." "$LOCAL_RESULTS/" 2>&1 | tail -5 || log "  WARN: scp of results had issues — check remote directly"

echo
log "=== DONE ==="
echo "View SUMMARY.txt:"
echo "  cat $LOCAL_RESULTS/SUMMARY.txt"
echo "Per-run detail:"
echo "  ls $LOCAL_RESULTS/analysis/"
