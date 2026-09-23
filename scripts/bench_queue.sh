#!/usr/bin/env bash
# Round two: the permissively licensed candidates, same queue on both machines
# so the numbers compare directly.
#
#   ./scripts/bench_queue.sh              # the whole queue
#   ./scripts/bench_queue.sh opus madlad  # only entries whose model matches
#
# Each configuration writes its own JSON and is skipped if that file exists,
# so the queue is safe to re-run and a crash costs only its own entry.
# Big models run in 4-bit, serial only: three instances of a 13B model do not
# fit a 12 GB card, and round one showed per-language instances buy nothing
# on a GPU anyway.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="./venv/bin/python"
[ -x "$PY" ] || PY="python3"
OUT="benchmarks/translate"
LANGS=(es ht ru)
# SOURCE=suite runs the head-to-head over all five tracks of the 56-minute
# test suite (396 sentences, 7,144 words) instead of the sermon alone.
SOURCE="${SOURCE:-sermon}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/queue.log") 2>&1

#       model             device dtype threads sentences
QUEUE=(
  "opus               cpu    fp32  1  120"
  "opus               cpu    fp32  3  120"
  "opus               cuda   fp16  1  120"
  "opus               cuda   fp16  3  120"
  "madlad-3b          cuda   bf16  1  120"
  "aya-101            cuda   int4  1   60"
  "qwen3-8b           cuda   int4  1   60"
  "mistral-nemo-12b   cuda   int4  1   60"
  # controls: is 4-bit what hurt Qwen's Creole, and does MADLAD's bigger
  # sibling do better? Both fit a 12 GB card at 8-bit.
  # a 256k-token embedding stays unquantized (~2 GB), so 7B at 8-bit
  # does not fit a 12 GB card; 4-bit does.
  "madlad-7b          cuda   int4  1   60"
)

if [ "$SOURCE" = "suite" ]; then
  QUEUE=(
    "nllb-1.3b          cuda   fp16  1   0"
    "opus               cuda   fp16  1   0"
    "madlad-3b          cuda   bf16  1   0"
  )
fi
# GUARD=1 repeats the queue with the number guard on, under its own labels.
GUARD_ARG=""
if [ "${GUARD:-0}" = "1" ]; then GUARD_ARG="--number-guard"; fi

echo "=============================================================="
echo "benchmark queue — $(date)"
echo "source: $SOURCE"
echo "host: $(hostname)   filter: ${*:-none}"
echo "=============================================================="
$PY tests/device_report.py --out "$OUT/device_report.json" | grep -E "^\[|GPU|torch see" || true

for entry in "${QUEUE[@]}"; do
  # shellcheck disable=SC2086
  set -- $entry
  model="$1"; device="$2"; dtype="$3"; threads="$4"; sentences="$5"
  if [ "$#" -ge 5 ] && [ -n "${FILTER:-}" ]; then :; fi
  keep=1
  if [ -n "${MATCH:-}" ]; then
    keep=0
    for m in $MATCH; do case "$model" in *"$m"*) keep=1 ;; esac; done
  fi
  [ "$keep" = 1 ] || continue
  label="$model-$device-$dtype"
  [ "$SOURCE" != "sermon" ] && label="$label-$SOURCE"
  [ -n "$GUARD_ARG" ] && label="$label-guarded"
  extra=()
  if [ "$threads" != "1" ]; then label="$label-x${#LANGS[@]}"; extra+=(--threads); fi
  if [ -f "$OUT/$label.json" ]; then echo ">>> SKIP $label (already done)"; continue; fi
  echo ">>> $label   $(date +%H:%M:%S)"
  timeout 21600 $PY tests/translate_bench.py \
      --model "$model" --device "$device" --dtype "$dtype" \
      --langs "${LANGS[@]}" --sentences "$sentences" --source "$SOURCE" \
      --out "$OUT" --label "$label" $GUARD_ARG "${extra[@]}"
  rc=$?
  [ "$rc" -ne 0 ] && echo "!!! $label exited $rc — continuing"
  echo
done

echo "queue finished $(date)"
$PY tests/translate_bench.py --compare "$OUT" --samples 10 > "$OUT/REPORT.txt" 2>&1
echo "report: $OUT/REPORT.txt"
