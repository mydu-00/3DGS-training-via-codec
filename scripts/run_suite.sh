#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <scene_path> [run_tag] [iters]"
  echo "Example: $0 /data/tandt/truck truck 30000"
  exit 1
fi

SCENE_PATH="$1"
RUN_TAG="${2:-truck}"
ITERS="${3:-30000}"

BASE_RUN="${RUN_TAG}_base"
REST_RUN="${RUN_TAG}_qrest"
ALL_RUN="${RUN_TAG}_qall"

mkdir -p artifacts logs

echo "[1/6] Train base"
bash scripts/run_train.sh "${SCENE_PATH}" "${BASE_RUN}" none "${ITERS}"

echo "[2/6] Eval base"
bash scripts/run_eval.sh "${BASE_RUN}"

echo "[3/6] Train SH-rest int8"
bash scripts/run_train.sh "${SCENE_PATH}" "${REST_RUN}" rest "${ITERS}"

echo "[4/6] Eval SH-rest int8"
bash scripts/run_eval.sh "${REST_RUN}"

echo "[5/6] Train SH-all int8"
bash scripts/run_train.sh "${SCENE_PATH}" "${ALL_RUN}" all "${ITERS}"

echo "[6/6] Eval SH-all int8"
bash scripts/run_eval.sh "${ALL_RUN}"

SUMMARY_PATH="artifacts/summary_${RUN_TAG}.csv"
export RUN_TAG
python - <<'PY'
import csv
import json
import os

run_tag = os.environ.get("RUN_TAG")
runs = [
    (f"{run_tag}_base", "base", "none"),
    (f"{run_tag}_qrest", "quant_sh_rest", "rest"),
    (f"{run_tag}_qall", "quant_sh_all", "all"),
]
rows = []
for run_name, label, quant_mode in runs:
    path = f"artifacts/results_{run_name}.json"
    if not os.path.exists(path):
        rows.append([run_name, label, quant_mode, "", "", "", f"missing {path}"])
        continue
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not payload:
        rows.append([run_name, label, quant_mode, "", "", "", "empty results json"])
        continue
    method_key = next(iter(payload.keys()))
    metrics = payload[method_key]
    rows.append([
        run_name,
        label,
        quant_mode,
        metrics.get("SSIM", ""),
        metrics.get("PSNR", ""),
        metrics.get("LPIPS", ""),
        "",
    ])

summary_path = f"artifacts/summary_{run_tag}.csv"
with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["run_name", "variant", "quant_mode", "SSIM", "PSNR", "LPIPS", "note"])
    writer.writerows(rows)

print(f"[DONE] Wrote {summary_path}")
for r in rows:
    print(r)
PY

echo "[DONE] Full suite completed. Summary: ${SUMMARY_PATH}"
