#!/usr/bin/env bash
set -euo pipefail
CASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CASE_DIR"
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON:-python}" -B analysis/plot.py \
  --directory "${PLOT_DIRECTORY:-${TEST_SAVE_DIR:-analysis/figures}}" \
  --step "${PLOT_STEP:-100}" "$@"
