#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <run_name>"
  exit 1
fi

RUN_NAME="$1"
MODEL_PATH="output/${RUN_NAME}"
LOG_PATH="logs/eval_${RUN_NAME}.log"

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Model path not found: ${MODEL_PATH}"
  exit 1
fi

if ! command -v micromamba >/dev/null 2>&1; then
  echo "micromamba not found in PATH"
  exit 1
fi

# Initialize shell hook to allow `micromamba activate` in non-interactive shells.
eval "$(micromamba shell hook --shell bash)"
micromamba activate gs

mkdir -p logs artifacts

echo "[INFO] Rendering and metric evaluation: ${MODEL_PATH}" | tee "${LOG_PATH}"
python render.py -m "${MODEL_PATH}" --skip_train 2>&1 | tee -a "${LOG_PATH}"
python metrics.py -m "${MODEL_PATH}" 2>&1 | tee -a "${LOG_PATH}"

if [ -f "${MODEL_PATH}/results.json" ]; then
  cp "${MODEL_PATH}/results.json" "artifacts/results_${RUN_NAME}.json"
  echo "[DONE] metrics copied to artifacts/results_${RUN_NAME}.json" | tee -a "${LOG_PATH}"
else
  echo "[WARN] results.json not found under ${MODEL_PATH}" | tee -a "${LOG_PATH}"
fi
