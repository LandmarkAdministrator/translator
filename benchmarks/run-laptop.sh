#!/bin/bash
# Laptop side of docs/RUN-BENCHMARKS.md: performance power profile for the
# run, the same environment run.py gives the stack, then the safe wrapper.
set -u
cd /home/administrator/Projects/translator || exit 1
LOG=benchmarks/translate/laptop-run.log
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
prev=$(powerprofilesctl get 2>/dev/null || echo "")
restore() { [ -n "$prev" ] && powerprofilesctl set "$prev" 2>/dev/null; say "power profile restored to ${prev:-unknown}; runner exiting"; }
trap restore EXIT INT TERM
powerprofilesctl set performance 2>/dev/null && say "power profile: performance (was $prev)"
[ -f .env.rocm ] && { set +u; set -a; . ./.env.rocm; set +a; set -u; say "sourced .env.rocm (HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-unset})"; }
say "AC online: $(cat /sys/class/power_supply/*/online 2>/dev/null | head -1)"
MATCH="${MATCH:-}" ./scripts/bench_queue.sh >> "$LOG" 2>&1
say "bench_queue.sh exited rc=$?"
