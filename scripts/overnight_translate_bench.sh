#!/usr/bin/env bash
# Overnight translation benchmark: every model on every device, then the
# per-language-instance comparison, then one report.
#
# Written for the laptop, where "cuda" is ROCm via HIP. Real CUDA numbers
# need the same script on the RTX 3060 box; one machine cannot answer
# CUDA vs ROCm.
#
#   ./scripts/overnight_translate_bench.sh
#   ./scripts/overnight_translate_bench.sh --quick     # fewer sentences
#   ./scripts/overnight_translate_bench.sh --with-madlad
#
# Safe to re-run: each configuration writes its own JSON and the report is
# rebuilt from whatever is on disk, so a crashed run costs only that run.
set -uo pipefail          # deliberately NOT -e: one model failing to
                          # download must not abandon the whole night

cd "$(dirname "$0")/.." || exit 1
PY="./venv/bin/python"
[ -x "$PY" ] || PY="python3"

OUT="benchmarks/translate"
LOG="$OUT/overnight.log"
SENTENCES=120
MODELS=("nllb-1.3b" "m2m100-418m" "m2m100-1.2b")
LANGS=(es ht ru)

for arg in "$@"; do
  case "$arg" in
    --quick)        SENTENCES=30 ;;
    --with-madlad)  MODELS+=("madlad-3b") ;;
    --nllb-only)    MODELS=("nllb-1.3b") ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

echo "=============================================================="
echo "overnight translation benchmark — $(date)"
echo "host      : $(hostname)"
echo "sentences : $SENTENCES"
echo "models    : ${MODELS[*]}"
echo "languages : ${LANGS[*]}"
echo "=============================================================="

# Which devices are real on this box. On AMD, cuda == ROCm.
DEVICES=(cpu)
if $PY - <<'EOF' 2>/dev/null
import sys, torch
sys.exit(0 if torch.cuda.is_available() else 1)
EOF
then
  DEVICES+=(cuda)
  echo "GPU detected — will measure cpu and cuda (ROCm on AMD)"
else
  echo "No GPU visible to torch — CPU only"
fi
echo

echo "--- device report ---"
$PY tests/device_report.py --out "$OUT/device_report.json"
echo

run() {   # run <model> <device> <extra-args...>
  local model="$1" device="$2"; shift 2
  local label="$model-$device"
  [ "${1:-}" = "--threads" ] && label="$label-x${#LANGS[@]}"
  if [ -f "$OUT/$label.json" ]; then
    echo ">>> SKIP $label (already done — delete the JSON to redo)"
    return 0
  fi
  echo ">>> $label   $(date +%H:%M:%S)"
  timeout 7200 $PY tests/translate_bench.py \
      --model "$model" --device "$device" \
      --langs "${LANGS[@]}" --sentences "$SENTENCES" \
      --out "$OUT" --label "$label" "$@"
  local rc=$?
  [ $rc -ne 0 ] && echo "!!! $label exited $rc — continuing"
  echo
}

# Pass 1: one shared model, each device. Answers "which model, and where".
for model in "${MODELS[@]}"; do
  for device in "${DEVICES[@]}"; do
    run "$model" "$device"
  done
done

# Pass 2: per-language instances. Answers the question that decides how
# many languages a box can carry — whether dropping the shared lock buys
# real parallelism, and what it costs in memory.
for model in "${MODELS[@]}"; do
  for device in "${DEVICES[@]}"; do
    run "$model" "$device" --threads
  done
done

echo "=============================================================="
echo "finished $(date)"
echo "=============================================================="
$PY tests/translate_bench.py --compare "$OUT" --samples 10 \
    | tee "$OUT/REPORT.txt"

echo
echo "Report: $OUT/REPORT.txt"
echo "Raw   : $OUT/*.json"
echo "Copy the whole $OUT directory to compare against the other machine."
