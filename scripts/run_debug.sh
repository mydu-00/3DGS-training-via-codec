#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <scene_path> [iters]"
  exit 1
fi

SCENE_PATH="$1"
ITERS="${2:-200}"
RUN_NAME="debug_$(date +%Y%m%d_%H%M%S)"

if ! command -v micromamba >/dev/null 2>&1; then
  echo "micromamba not found in PATH"
  exit 1
fi

# Initialize shell hook to allow `micromamba activate` in non-interactive shells.
eval "$(micromamba shell hook --shell bash)"
micromamba activate gs

mkdir -p logs
LOG_PATH="logs/${RUN_NAME}.log"

echo "[INFO] Short debug run: scene=${SCENE_PATH} iters=${ITERS}" | tee "${LOG_PATH}"
python train.py \
  -s "${SCENE_PATH}" \
  -m "output/${RUN_NAME}" \
  --eval \
  --iterations "${ITERS}" \
  --disable_viewer \
  --test_iterations 100 "${ITERS}" \
  --save_iterations "${ITERS}" 2>&1 | tee -a "${LOG_PATH}"

echo "[DONE] Debug run complete: output/${RUN_NAME}" | tee -a "${LOG_PATH}"
