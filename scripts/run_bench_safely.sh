#!/usr/bin/env bash
# Run the translation benchmark with the sermon-archive worker stopped.
#
# The archive worker and the benchmark both want the GPU. A benchmark run
# beside an encode job measures contention, not hardware, so this stops
# the worker first, verifies it actually stopped, runs everything, and
# puts it back.
#
#   ./scripts/run_bench_safely.sh              # stop worker, bench, restart
#   ./scripts/run_bench_safely.sh --quick      # 30 sentences instead of 120
#   ./scripts/run_bench_safely.sh --no-restart # leave the worker stopped
#   ./scripts/run_bench_safely.sh --no-stop    # I already stopped it myself
#
# Graceful stop means the worker finishes its CURRENT sermon first. An
# encode job runs about 75 minutes, so the wait can be long — that is the
# worker behaving correctly, not a hang. Use --wait to change the limit.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
TRANSLATOR_DIR="$PWD"
SERMONS_DIR="${SERMONS_DIR:-$HOME/Multi-Bitrate-Sermons}"
[ -d "$SERMONS_DIR" ] || SERMONS_DIR="$HOME/Projects/Multi-Bitrate-Sermons"

DO_STOP=1
DO_RESTART=1
WAIT_LIMIT=5400          # 90 min: an encode job plus margin
BENCH_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --no-stop)     DO_STOP=0 ;;
    --no-restart)  DO_RESTART=0 ;;
    --wait=*)      WAIT_LIMIT="${arg#*=}" ;;
    --quick|--with-madlad|--nllb-only) BENCH_ARGS+=("$arg") ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done

say() { printf '\n=== %s ===\n' "$*"; }

worker_running() {
  pgrep -f 'unified_worker\.py|asr_worker\.py|dub_worker\.py|encode_worker_v2\.py' \
    >/dev/null 2>&1
}

show_workers() {
  pgrep -af 'unified_worker\.py|asr_worker\.py|dub_worker\.py|encode_worker_v2\.py' \
    2>/dev/null | sed 's/^/    /'
}

# --------------------------------------------------------------------------
say "pre-flight"

echo "  translator : $TRANSLATOR_DIR"
echo "  sermons    : $SERMONS_DIR $([ -d "$SERMONS_DIR" ] || echo '(not found)')"
echo "  host       : $(hostname)"

# The live translator must not be running either — it owns the GPU during
# service windows and would contend just as badly as the archive worker.
if systemctl --user is-active --quiet translate.service 2>/dev/null; then
  echo
  echo "  REFUSING TO START: translate.service is active."
  echo "  A service window is open, or it was started by hand. Benchmarking"
  echo "  now would measure contention and could disrupt a live service."
  echo "  Stop it, or wait for the window to close, then re-run."
  exit 1
fi
echo "  translate.service: not active — good"

# --------------------------------------------------------------------------
STOPPED_BY_US=0
if [ "$DO_STOP" = "1" ]; then
  say "stopping the sermon archive worker"
  if ! worker_running; then
    echo "  not running — nothing to stop"
  else
    echo "  currently running:"
    show_workers
    STOP_SCRIPT="$SERMONS_DIR/scripts/lbc-stop-unified-worker.sh"
    if [ -x "$STOP_SCRIPT" ]; then
      echo "  calling lbc-stop-unified-worker.sh (graceful)"
      "$STOP_SCRIPT"
    elif [ -d "$SERMONS_DIR" ]; then
      # The documented fallback signal: stop.flag is honored between jobs.
      echo "  stop script not executable; touching stop.flag instead"
      touch "$SERMONS_DIR/stop.flag"
    else
      echo "  WARNING: sermons repo not found; cannot stop the worker."
      echo "  Set SERMONS_DIR, or stop it yourself and re-run with --no-stop."
      exit 1
    fi
    STOPPED_BY_US=1

    echo "  waiting up to $((WAIT_LIMIT / 60)) min for the current job to finish"
    echo "  (graceful: the worker finishes its sermon before exiting)"
    waited=0
    while worker_running && [ "$waited" -lt "$WAIT_LIMIT" ]; do
      sleep 15
      waited=$((waited + 15))
      if [ $((waited % 300)) -eq 0 ]; then
        echo "    still finishing... ${waited}s elapsed"
      fi
    done

    if worker_running; then
      echo
      echo "  REFUSING TO START: the worker is still running after"
      echo "  $((WAIT_LIMIT / 60)) minutes. It is probably mid-encode."
      show_workers
      echo "  Re-run later, raise the limit with --wait=SECONDS, or stop it"
      echo "  yourself and use --no-stop."
      exit 1
    fi
    echo "  worker stopped after ${waited}s"
  fi
else
  say "skipping the stop step (--no-stop)"
  if worker_running; then
    echo "  WARNING: the archive worker IS running. These numbers will"
    echo "  measure contention, not hardware."
    show_workers
  fi
fi

restore_worker() {
  if [ "$STOPPED_BY_US" = "1" ] && [ "$DO_RESTART" = "1" ]; then
    say "restarting the sermon archive worker"
    rm -f "$SERMONS_DIR/stop.flag" 2>/dev/null
    START_SCRIPT="$SERMONS_DIR/scripts/lbc-start-unified-worker.sh"
    if [ -x "$START_SCRIPT" ]; then
      "$START_SCRIPT" && echo "  restarted" || echo "  FAILED — start it by hand"
    else
      echo "  start script not found; start it by hand:"
      echo "    $START_SCRIPT"
    fi
  elif [ "$STOPPED_BY_US" = "1" ]; then
    say "worker left stopped (--no-restart)"
    echo "  restart it with: $SERMONS_DIR/scripts/lbc-start-unified-worker.sh"
    echo "  and remove:      $SERMONS_DIR/stop.flag"
  fi
}
# Restore on any exit, including Ctrl-C — leaving a church's archive
# worker stopped overnight because a benchmark was interrupted would be
# a bad trade.
trap restore_worker EXIT INT TERM

# --------------------------------------------------------------------------
say "benchmark"
./scripts/overnight_translate_bench.sh "${BENCH_ARGS[@]}"
RC=$?

say "done (exit $RC)"
echo "  report: benchmarks/translate/REPORT.txt"
echo "  raw   : benchmarks/translate/*.json"
exit $RC
