#!/usr/bin/env bash
set -euo pipefail
CASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CASE_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
read -r -a models <<< "${TEST_MODELS:-PSTO FNO Unet UFNO}"
exec "${PYTHON:-python}" -B src/test.py \
  --checkpoint_root "${TRAIN_OUTPUT_ROOT:-checkpoints/models}" \
  --checkpoint checkpoint_best.pt --models "${models[@]}" \
  --steps 100 --device "${DEVICE:-cuda}" --source_backend gpu_sdirk2 \
  --splitting strang --data_idx "${TEST_DATA_IDX:-1001}" \
  --save_dir "${TEST_SAVE_DIR:-analysis/figures}" "$@"
