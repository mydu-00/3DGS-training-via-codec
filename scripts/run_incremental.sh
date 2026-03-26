#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <scene_path> [run_tag] [iters]"
  echo "Example: $0 /data/tandt/truck truck 30000"
  echo ""
  echo "This script runs incremental experiments with corrected/new quantization methods:"
  echo "  1. Re-run group quantization experiments (now with global scales, fixed)"
  echo "  2. New gaussian_group variants with finer granularity"
  echo "  3. Different gaussian group sizes (128, 256, 512)"
  exit 1
fi

SCENE_PATH="$1"
RUN_TAG="${2:-truck}"
ITERS="${3:-30000}"

mkdir -p artifacts logs

STEP=1
TOTAL=16

echo "=========================================="
echo "Incremental Experiments (Fixed & New)"
echo "=========================================="

# Re-run group quantization (now fixed to use global scales)
echo ""
echo "[${STEP}/${TOTAL}] Re-train high-frequency SH with group granularity (size=3, FIXED)"
bash scripts/run_train.sh "${SCENE_PATH}" "${RUN_TAG}_qrest_g3_v2" rest "${ITERS}" group 3
STEP=$((STEP + 1))

echo "[${STEP}/${TOTAL}] Eval high-frequency SH group 3 v2"
bash scripts/run_eval.sh "${RUN_TAG}_qrest_g3_v2"
STEP=$((STEP + 1))

echo ""
echo "[${STEP}/${TOTAL}] Re-train high-frequency SH with group granularity (size=9, FIXED)"
bash scripts/run_train.sh "${SCENE_PATH}" "${RUN_TAG}_qrest_g9_v2" rest "${ITERS}" group 9
STEP=$((STEP + 1))

echo "[${STEP}/${TOTAL}] Eval high-frequency SH group 9 v2"
bash scripts/run_eval.sh "${RUN_TAG}_qrest_g9_v2"
STEP=$((STEP + 1))

# New: gaussian_group_channel variants
echo ""
echo "[${STEP}/${TOTAL}] Train all SH with gaussian_group_channel (256 gaussians, per-channel within)"
bash scripts/run_train.sh "${SCENE_PATH}" "${RUN_TAG}_qall_ggch256" all "${ITERS}" gaussian_group_channel 15 256
STEP=$((STEP + 1))

echo "[${STEP}/${TOTAL}] Eval all SH gaussian_group_channel 256"
bash scripts/run_eval.sh "${RUN_TAG}_qall_ggch256"
STEP=$((STEP + 1))

# New: gaussian_group_group variants
echo ""
echo "[${STEP}/${TOTAL}] Train all SH with gaussian_group_group (256 gaussians, group=15 within)"
bash scripts/run_train.sh "${SCENE_PATH}" "${RUN_TAG}_qall_ggg256_g15" all "${ITERS}" gaussian_group_group 15 256
STEP=$((STEP + 1))

echo "[${STEP}/${TOTAL}] Eval all SH gaussian_group_group 256_g15"
bash scripts/run_eval.sh "${RUN_TAG}_qall_ggg256_g15"
STEP=$((STEP + 1))

# Different gaussian group sizes
echo ""
echo "[${STEP}/${TOTAL}] Train all SH with gaussian_group size=128"
bash scripts/run_train.sh "${SCENE_PATH}" "${RUN_TAG}_qall_gg128" all "${ITERS}" gaussian_group 15 128
STEP=$((STEP + 1))

echo "[${STEP}/${TOTAL}] Eval all SH gaussian_group 128"
bash scripts/run_eval.sh "${RUN_TAG}_qall_gg128"
STEP=$((STEP + 1))

echo ""
echo "[${STEP}/${TOTAL}] Train all SH with gaussian_group size=512"
bash scripts/run_train.sh "${SCENE_PATH}" "${RUN_TAG}_qall_gg512" all "${ITERS}" gaussian_group 15 512
STEP=$((STEP + 1))

echo "[${STEP}/${TOTAL}] Eval all SH gaussian_group 512"
bash scripts/run_eval.sh "${RUN_TAG}_qall_gg512"
STEP=$((STEP + 1))

# Bonus: gaussian_group_channel with different sizes
echo ""
echo "[${STEP}/${TOTAL}] Train all SH with gaussian_group_channel size=128"
bash scripts/run_train.sh "${SCENE_PATH}" "${RUN_TAG}_qall_ggch128" all "${ITERS}" gaussian_group_channel 15 128
STEP=$((STEP + 1))

echo "[${STEP}/${TOTAL}] Eval all SH gaussian_group_channel 128"
bash scripts/run_eval.sh "${RUN_TAG}_qall_ggch128"
STEP=$((STEP + 1))

echo ""
echo "[${STEP}/${TOTAL}] Train all SH with gaussian_group_channel size=512"
bash scripts/run_train.sh "${SCENE_PATH}" "${RUN_TAG}_qall_ggch512" all "${ITERS}" gaussian_group_channel 15 512
STEP=$((STEP + 1))

echo "[${STEP}/${TOTAL}] Eval all SH gaussian_group_channel 512"
bash scripts/run_eval.sh "${RUN_TAG}_qall_ggch512"
STEP=$((STEP + 1))

# Generate summary
SUMMARY_PATH="artifacts/summary_incremental_${RUN_TAG}.csv"
export RUN_TAG
python - <<'PY'
import csv
import json
import os

run_tag = os.environ.get("RUN_TAG")
runs = [
    (f"{run_tag}_qrest_g3_v2", "rest_group_3_v2_fixed", "rest", "group", "3"),
    (f"{run_tag}_qrest_g9_v2", "rest_group_9_v2_fixed", "rest", "group", "9"),
    (f"{run_tag}_qall_ggch256", "all_gg_channel_256", "all", "gaussian_group_channel", "256"),
    (f"{run_tag}_qall_ggg256_g15", "all_gg_group_256_g15", "all", "gaussian_group_group", "256/15"),
    (f"{run_tag}_qall_gg128", "all_gg_128", "all", "gaussian_group", "128"),
    (f"{run_tag}_qall_gg512", "all_gg_512", "all", "gaussian_group", "512"),
    (f"{run_tag}_qall_ggch128", "all_gg_channel_128", "all", "gaussian_group_channel", "128"),
    (f"{run_tag}_qall_ggch512", "all_gg_channel_512", "all", "gaussian_group_channel", "512"),
]
rows = []
for run_name, label, quant_mode, granularity, group_param in runs:
    path = f"artifacts/results_{run_name}.json"
    if not os.path.exists(path):
        rows.append([run_name, label, quant_mode, granularity, group_param, "", "", "", f"missing {path}"])
        continue
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not payload:
        rows.append([run_name, label, quant_mode, granularity, group_param, "", "", "", "empty results json"])
        continue
    method_key = next(iter(payload.keys()))
    metrics = payload[method_key]
    rows.append([
        run_name,
        label,
        quant_mode,
        granularity,
        group_param,
        metrics.get("SSIM", ""),
        metrics.get("PSNR", ""),
        metrics.get("LPIPS", ""),
        "",
    ])

summary_path = f"artifacts/summary_incremental_{run_tag}.csv"
with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["run_name", "variant", "quant_mode", "granularity", "group_param", "SSIM", "PSNR", "LPIPS", "note"])
    writer.writerows(rows)

print(f"[DONE] Wrote {summary_path}")
print("\nIncremental Results Summary:")
print("-" * 120)
print(f"{'Variant':<30} {'Mode':<8} {'Granularity':<25} {'Param':<10} {'SSIM':<10} {'PSNR':<10} {'LPIPS':<10}")
print("-" * 120)
for r in rows:
    print(f"{r[1]:<30} {r[2]:<8} {r[3]:<25} {r[4]:<10} {str(r[5]):<10} {str(r[6]):<10} {str(r[7]):<10}")
print("-" * 120)
PY

echo "[DONE] Incremental experiments completed. Summary: ${SUMMARY_PATH}"
