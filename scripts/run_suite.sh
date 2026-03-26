#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <scene_path> [run_tag] [iters]"
  echo "Example: $0 /data/tandt/truck truck 30000"
  echo ""
  echo "This script runs a comprehensive suite of quantization experiments:"
  echo "  1. Baseline (no quantization)"
  echo "  2. High-frequency SH with channel granularity"
  echo "  3. High-frequency SH with group granularity (sizes: 3, 9, 15)"
  echo "  4. All SH with channel granularity"
  echo "  5. All SH with group granularity (sizes: 3, 9, 15)"
  echo "  6. All SH with gaussian_group granularity (256 gaussians per group)"
  exit 1
fi

SCENE_PATH="$1"
RUN_TAG="${2:-truck}"
ITERS="${3:-30000}"

mkdir -p artifacts logs

STEP=1
TOTAL=20

# 1. Baseline
echo "[${STEP}/${TOTAL}] Train baseline (no quantization)"
bash scripts/run_train.sh "${SCENE_PATH}" "${RUN_TAG}_base" none "${ITERS}"
STEP=$((STEP + 1))

echo "[${STEP}/${TOTAL}] Eval baseline"
bash scripts/run_eval.sh "${RUN_TAG}_base"
STEP=$((STEP + 1))

# 2. High-frequency SH quantization - channel granularity
echo "[${STEP}/${TOTAL}] Train high-frequency SH with channel granularity"
bash scripts/run_train.sh "${SCENE_PATH}" "${RUN_TAG}_qrest_ch" rest "${ITERS}" channel
STEP=$((STEP + 1))

echo "[${STEP}/${TOTAL}] Eval high-frequency SH channel"
bash scripts/run_eval.sh "${RUN_TAG}_qrest_ch"
STEP=$((STEP + 1))

# 3. High-frequency SH quantization - group granularity (multiple group sizes)
echo "[${STEP}/${TOTAL}] Train high-frequency SH with group granularity (size=3)"
bash scripts/run_train.sh "${SCENE_PATH}" "${RUN_TAG}_qrest_g3" rest "${ITERS}" group 3
STEP=$((STEP + 1))

echo "[${STEP}/${TOTAL}] Eval high-frequency SH group 3"
bash scripts/run_eval.sh "${RUN_TAG}_qrest_g3"
STEP=$((STEP + 1))

echo "[${STEP}/${TOTAL}] Train high-frequency SH with group granularity (size=9)"
bash scripts/run_train.sh "${SCENE_PATH}" "${RUN_TAG}_qrest_g9" rest "${ITERS}" group 9
STEP=$((STEP + 1))

echo "[${STEP}/${TOTAL}] Eval high-frequency SH group 9"
bash scripts/run_eval.sh "${RUN_TAG}_qrest_g9"
STEP=$((STEP + 1))

echo "[${STEP}/${TOTAL}] Train high-frequency SH with group granularity (size=15)"
bash scripts/run_train.sh "${SCENE_PATH}" "${RUN_TAG}_qrest_g15" rest "${ITERS}" group 15
STEP=$((STEP + 1))

echo "[${STEP}/${TOTAL}] Eval high-frequency SH group 15"
bash scripts/run_eval.sh "${RUN_TAG}_qrest_g15"
STEP=$((STEP + 1))

# 4. All SH quantization - channel granularity
echo "[${STEP}/${TOTAL}] Train all SH with channel granularity"
bash scripts/run_train.sh "${SCENE_PATH}" "${RUN_TAG}_qall_ch" all "${ITERS}" channel
STEP=$((STEP + 1))

echo "[${STEP}/${TOTAL}] Eval all SH channel"
bash scripts/run_eval.sh "${RUN_TAG}_qall_ch"
STEP=$((STEP + 1))

# 5. All SH quantization - group granularity (multiple group sizes)
echo "[${STEP}/${TOTAL}] Train all SH with group granularity (size=3)"
bash scripts/run_train.sh "${SCENE_PATH}" "${RUN_TAG}_qall_g3" all "${ITERS}" group 3
STEP=$((STEP + 1))

echo "[${STEP}/${TOTAL}] Eval all SH group 3"
bash scripts/run_eval.sh "${RUN_TAG}_qall_g3"
STEP=$((STEP + 1))

echo "[${STEP}/${TOTAL}] Train all SH with group granularity (size=9)"
bash scripts/run_train.sh "${SCENE_PATH}" "${RUN_TAG}_qall_g9" all "${ITERS}" group 9
STEP=$((STEP + 1))

echo "[${STEP}/${TOTAL}] Eval all SH group 9"
bash scripts/run_eval.sh "${RUN_TAG}_qall_g9"
STEP=$((STEP + 1))

echo "[${STEP}/${TOTAL}] Train all SH with group granularity (size=15)"
bash scripts/run_train.sh "${SCENE_PATH}" "${RUN_TAG}_qall_g15" all "${ITERS}" group 15
STEP=$((STEP + 1))

echo "[${STEP}/${TOTAL}] Eval all SH group 15"
bash scripts/run_eval.sh "${RUN_TAG}_qall_g15"
STEP=$((STEP + 1))

# 6. All SH quantization - gaussian_group granularity
echo "[${STEP}/${TOTAL}] Train all SH with gaussian_group granularity (256 gaussians per group)"
bash scripts/run_train.sh "${SCENE_PATH}" "${RUN_TAG}_qall_gg256" all "${ITERS}" gaussian_group 15 256
STEP=$((STEP + 1))

echo "[${STEP}/${TOTAL}] Eval all SH gaussian_group 256"
bash scripts/run_eval.sh "${RUN_TAG}_qall_gg256"
STEP=$((STEP + 1))

SUMMARY_PATH="artifacts/summary_${RUN_TAG}.csv"
export RUN_TAG
python - <<'PY'
import csv
import json
import os

run_tag = os.environ.get("RUN_TAG")
runs = [
    (f"{run_tag}_base", "baseline", "none", "tensor", "-"),
    (f"{run_tag}_qrest_ch", "rest_channel", "rest", "channel", "-"),
    (f"{run_tag}_qrest_g3", "rest_group_3", "rest", "group", "3"),
    (f"{run_tag}_qrest_g9", "rest_group_9", "rest", "group", "9"),
    (f"{run_tag}_qrest_g15", "rest_group_15", "rest", "group", "15"),
    (f"{run_tag}_qall_ch", "all_channel", "all", "channel", "-"),
    (f"{run_tag}_qall_g3", "all_group_3", "all", "group", "3"),
    (f"{run_tag}_qall_g9", "all_group_9", "all", "group", "9"),
    (f"{run_tag}_qall_g15", "all_group_15", "all", "group", "15"),
    (f"{run_tag}_qall_gg256", "all_gaussian_group_256", "all", "gaussian_group", "256"),
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

summary_path = f"artifacts/summary_{run_tag}.csv"
with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["run_name", "variant", "quant_mode", "granularity", "group_param", "SSIM", "PSNR", "LPIPS", "note"])
    writer.writerows(rows)

print(f"[DONE] Wrote {summary_path}")
print("\nResults Summary:")
print("-" * 120)
print(f"{'Variant':<25} {'Mode':<8} {'Granularity':<15} {'Param':<8} {'SSIM':<10} {'PSNR':<10} {'LPIPS':<10}")
print("-" * 120)
for r in rows:
    print(f"{r[1]:<25} {r[2]:<8} {r[3]:<15} {r[4]:<8} {str(r[5]):<10} {str(r[6]):<10} {str(r[7]):<10}")
print("-" * 120)
PY

echo "[DONE] Full suite completed. Summary: ${SUMMARY_PATH}"
