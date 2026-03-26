#!/usr/bin/env bash
# Incremental experiment: per-gaussian channel QDQ for 'rest' and 'all' modes.
#
# This script runs only the two new variants that use the bug-fixed per-gaussian
# channel quantization (--quant_granularity channel).  It uses distinct run names
# (_qrest_ch_pg, _qall_ch_pg) so that existing artifacts are never overwritten.
#
# Usage:
#   bash scripts/run_incremental_channel_pg.sh <scene_path> [run_tag] [iters]
#
# Examples:
#   bash scripts/run_incremental_channel_pg.sh ~/datasets/tandt/truck truck 30000
#   bash scripts/run_incremental_channel_pg.sh ~/datasets/tandt/truck truck_v2 10000
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <scene_path> [run_tag] [iters]"
  echo "  run_tag: prefix for output directories (default: truck)"
  echo "  iters:   training iterations (default: 30000)"
  echo ""
  echo "Runs two new variants with per-gaussian channel QDQ:"
  echo "  <run_tag>_qrest_ch_pg  -- SH rest, granularity=channel (per-gaussian)"
  echo "  <run_tag>_qall_ch_pg   -- SH all,  granularity=channel (per-gaussian)"
  echo ""
  echo "Artifacts are written to artifacts/results_<run_tag>_q{rest,all}_ch_pg.json"
  echo "and appended to artifacts/summary_<run_tag>_ch_pg.csv.  Existing files"
  echo "from prior runs are not touched."
  exit 1
fi

SCENE_PATH="$1"
RUN_TAG="${2:-truck}"
ITERS="${3:-30000}"

REST_RUN="${RUN_TAG}_qrest_ch_pg"
ALL_RUN="${RUN_TAG}_qall_ch_pg"

mkdir -p artifacts logs

echo "[1/4] Train SH-rest, per-gaussian channel QDQ"
bash scripts/run_train.sh "${SCENE_PATH}" "${REST_RUN}" rest "${ITERS}" channel

echo "[2/4] Eval SH-rest, per-gaussian channel QDQ"
bash scripts/run_eval.sh "${REST_RUN}"

echo "[3/4] Train SH-all, per-gaussian channel QDQ"
bash scripts/run_train.sh "${SCENE_PATH}" "${ALL_RUN}" all "${ITERS}" channel

echo "[4/4] Eval SH-all, per-gaussian channel QDQ"
bash scripts/run_eval.sh "${ALL_RUN}"

SUMMARY_PATH="artifacts/summary_${RUN_TAG}_ch_pg.csv"
export RUN_TAG REST_RUN ALL_RUN
python - <<'PY'
import csv
import json
import os

run_tag  = os.environ["RUN_TAG"]
rest_run = os.environ["REST_RUN"]
all_run  = os.environ["ALL_RUN"]

runs = [
    (rest_run, "quant_sh_rest_ch_pg", "rest",  "channel"),
    (all_run,  "quant_sh_all_ch_pg",  "all",   "channel"),
]
rows = []
for run_name, label, quant_mode, granularity in runs:
    path = f"artifacts/results_{run_name}.json"
    if not os.path.exists(path):
        rows.append([run_name, label, quant_mode, granularity, "", "", "", f"missing {path}"])
        continue
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not payload:
        rows.append([run_name, label, quant_mode, granularity, "", "", "", "empty results json"])
        continue
    method_key = next(iter(payload.keys()))
    metrics = payload[method_key]
    rows.append([
        run_name, label, quant_mode, granularity,
        metrics.get("SSIM", ""),
        metrics.get("PSNR", ""),
        metrics.get("LPIPS", ""),
        "",
    ])

summary_path = f"artifacts/summary_{run_tag}_ch_pg.csv"
with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["run_name", "variant", "quant_mode", "granularity", "SSIM", "PSNR", "LPIPS", "note"])
    writer.writerows(rows)

print(f"[DONE] Wrote {summary_path}")
for r in rows:
    print(r)
PY

echo "[DONE] Incremental channel-pg experiment completed. Summary: ${SUMMARY_PATH}"
