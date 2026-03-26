#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <scene_path> <run_name> [quant_mode] [iters] [quant_granularity] [quant_group_size] [quant_gaussian_group_size]"
  echo "  quant_mode:             none|dc|rest|all            (default: none)"
  echo "  iters:                  training iterations         (default: 30000)"
  echo "  quant_granularity:      tensor|channel|group|gaussian_group|gaussian_group_channel|gaussian_group_group"
  echo "                          (default: tensor)"
  echo "                            tensor                  – one global scale per tensor"
  echo "                            channel                 – one scale per (SH-order, colour) position"
  echo "                            group                   – one scale per block of quant_group_size channels (global)"
  echo "                            gaussian_group          – one scale per block of gaussians (all channels)"
  echo "                            gaussian_group_channel  – gaussian blocks with per-channel quantization"
  echo "                            gaussian_group_group    – gaussian blocks with channel-group quantization"
  echo "  quant_group_size:       group size for 'group' and 'gaussian_group_group' (default: 15)"
  echo "                            divisors of 45 (SH-rest): 1 3 5 9 15 45"
  echo "  quant_gaussian_group_size: gaussian group size (default: 256)"
  echo "                            common values: 128, 256, 512"
  echo ""
  echo "Examples:"
  echo "  # Basic baseline (no quantization)"
  echo "  $0 ~/data/truck truck_base none 30000"
  echo ""
  echo "  # Channel-wise and group quantization (now with global scales)"
  echo "  $0 ~/data/truck truck_qrest_ch rest 30000 channel"
  echo "  $0 ~/data/truck truck_qrest_g9 rest 30000 group 9"
  echo ""
  echo "  # Gaussian grouping variants"
  echo "  $0 ~/data/truck truck_qall_gg256 all 30000 gaussian_group 15 256"
  echo "  $0 ~/data/truck truck_qall_ggch256 all 30000 gaussian_group_channel 15 256"
  echo "  $0 ~/data/truck truck_qall_ggg256 all 30000 gaussian_group_group 15 256"
  echo ""
  echo "  # Different gaussian group sizes"
  echo "  $0 ~/data/truck truck_qall_gg128 all 30000 gaussian_group 15 128"
  echo "  $0 ~/data/truck truck_qall_gg512 all 30000 gaussian_group 15 512"
  exit 1
fi

SCENE_PATH="$1"
RUN_NAME="$2"
QUANT_MODE="${3:-none}"
ITERS="${4:-30000}"
QUANT_GRANULARITY="${5:-tensor}"
QUANT_GROUP_SIZE="${6:-15}"
QUANT_GAUSSIAN_GROUP_SIZE="${7:-256}"

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

echo "[INFO] scene=${SCENE_PATH} model=${MODEL_PATH} quant=${QUANT_MODE} granularity=${QUANT_GRANULARITY} group_size=${QUANT_GROUP_SIZE} gaussian_group_size=${QUANT_GAUSSIAN_GROUP_SIZE} iters=${ITERS}" | tee "${LOG_PATH}"
python train.py \
  -s "${SCENE_PATH}" \
  -m "${MODEL_PATH}" \
  --eval \
  --iterations "${ITERS}" \
  --disable_viewer \
  --sh_int8_quantization "${QUANT_MODE}" \
  --quant_granularity "${QUANT_GRANULARITY}" \
  --quant_group_size "${QUANT_GROUP_SIZE}" \
  --quant_gaussian_group_size "${QUANT_GAUSSIAN_GROUP_SIZE}" 2>&1 | tee -a "${LOG_PATH}"

echo "[DONE] Training finished: ${MODEL_PATH}" | tee -a "${LOG_PATH}"
