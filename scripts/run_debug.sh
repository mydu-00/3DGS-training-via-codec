#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <scene_path> [iters] [quant_mode] [quant_granularity] [quant_group_size] [quant_gaussian_group_size]"
  echo "  iters:                     training iterations (default: 200)"
  echo "  quant_mode:                none|dc|rest|all   (default: none)"
  echo "  quant_granularity:         tensor|channel|group|gaussian_group (default: tensor)"
  echo "  quant_group_size:          group size for 'group' granularity (default: 15)"
  echo "  quant_gaussian_group_size: group size for 'gaussian_group' granularity (default: 256)"
  exit 1
fi

SCENE_PATH="$1"
ITERS="${2:-200}"
QUANT_MODE="${3:-none}"
QUANT_GRANULARITY="${4:-tensor}"
QUANT_GROUP_SIZE="${5:-15}"
QUANT_GAUSSIAN_GROUP_SIZE="${6:-256}"
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

echo "[INFO] Short debug run: scene=${SCENE_PATH} iters=${ITERS} quant=${QUANT_MODE} granularity=${QUANT_GRANULARITY} group_size=${QUANT_GROUP_SIZE} gaussian_group_size=${QUANT_GAUSSIAN_GROUP_SIZE}" | tee "${LOG_PATH}"
python train.py \
  -s "${SCENE_PATH}" \
  -m "output/${RUN_NAME}" \
  --eval \
  --iterations "${ITERS}" \
  --disable_viewer \
  --test_iterations 100 "${ITERS}" \
  --save_iterations "${ITERS}" \
  --sh_int8_quantization "${QUANT_MODE}" \
  --quant_granularity "${QUANT_GRANULARITY}" \
  --quant_group_size "${QUANT_GROUP_SIZE}" \
  --quant_gaussian_group_size "${QUANT_GAUSSIAN_GROUP_SIZE}" 2>&1 | tee -a "${LOG_PATH}"

echo "[DONE] Debug run complete: output/${RUN_NAME}" | tee -a "${LOG_PATH}"
