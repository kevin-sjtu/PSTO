#!/usr/bin/env bash
set -euo pipefail
CASE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$CASE_DIR"
PYTHON=${PYTHON:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
start_args=()
if [[ -n "${TEST_DATA_IDX:-}" ]]; then
  start_args=(--data_idx "$TEST_DATA_IDX")
fi
exec "$PYTHON" -B src/test.py \
  --checkpoint_root "${TRAIN_OUTPUT_ROOT:-checkpoints/models}" \
  --models PSTO FNO Unet UFNO \
  --steps 100 \
  "${start_args[@]}" \
  --device cuda \
  --save_dir "${TEST_SAVE_DIR:-analysis/figures}" \
  "$@"
