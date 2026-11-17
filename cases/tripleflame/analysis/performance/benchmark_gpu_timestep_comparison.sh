#!/usr/bin/env bash
set -euo pipefail
CASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CASE_DIR"
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg OMP_NUM_THREADS=2
exec "${PYTHON:-python}" -B analysis/performance/benchmark_gpu_timestep_comparison.py "$@"
