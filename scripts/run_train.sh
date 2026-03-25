#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <scene_path> <run_name> [quant_mode] [iters]"
  echo "quant_mode: none|dc|rest|all (default: none)"
  echo "iters: default 30000"
  exit 1
fi

SCENE_PATH="$1"
RUN_NAME="$2"
QUANT_MODE="${3:-none}"
ITERS="${4:-30000}"

if ! command -v micromamba >/dev/null 2>&1; then
  echo "micromamba not found in PATH"
  exit 1
fi

# Initialize shell hook to allow `micromamba activate` in non-interactive shells.
eval "$(micromamba shell hook --shell bash)"
micromamba activate gs

mkdir -p logs artifacts checkpoints output
MODEL_PATH="output/${RUN_NAME}"
LOG_PATH="logs/train_${RUN_NAME}.log"

echo "[INFO] scene=${SCENE_PATH} model=${MODEL_PATH} quant=${QUANT_MODE} iters=${ITERS}" | tee "${LOG_PATH}"
python train.py \
  -s "${SCENE_PATH}" \
  -m "${MODEL_PATH}" \
  --eval \
  --iterations "${ITERS}" \
  --disable_viewer \
  --sh_int8_quantization "${QUANT_MODE}" 2>&1 | tee -a "${LOG_PATH}"

echo "[DONE] Training finished: ${MODEL_PATH}" | tee -a "${LOG_PATH}"
