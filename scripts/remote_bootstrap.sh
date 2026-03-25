#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <repo_url> <target_dir>"
  echo "Example: $0 git@github.com:mydu-00/3DGS-training-via-codec.git 3dgs-work"
  exit 1
fi

REPO_URL="$1"
TARGET_DIR="$2"

if [ -e "${TARGET_DIR}" ]; then
  echo "Refusing to overwrite existing path: ${TARGET_DIR}"
  exit 1
fi

git clone --recursive "${REPO_URL}" "${TARGET_DIR}"
cd "${TARGET_DIR}"

git submodule sync --recursive
git submodule update --init --recursive

echo "[OK] Repository ready at ${PWD}"
echo "[NEXT] Activate env and run:"
echo "  eval \"\$(micromamba shell hook --shell bash)\""
echo "  micromamba activate gs"
echo "  bash scripts/run_debug.sh <scene_path> 200"
