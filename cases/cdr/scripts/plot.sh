#!/usr/bin/env bash
set -euo pipefail
CASE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$CASE_DIR"
PYTHON=${PYTHON:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/psto_cdr_mpl}
mkdir -p "$MPLCONFIGDIR"
directory=${PLOT_DIRECTORY:-analysis/figures}
step=${PLOT_STEP:-100}
while (( $# )); do
  case "$1" in
    --directory) directory=${2:?--directory requires a path}; shift 2 ;;
    --step) step=${2:?--step requires a positive integer}; shift 2 ;;
    -h|--help)
      echo "Usage: bash scripts/plot.sh [--directory RESULT_DIRECTORY] [--step STEP]"
      echo "Reads locally evaluated rollout.npz and comparison.csv; writes PNG figures and residual diagnostics."
      exit 0 ;;
    *) echo "Unknown plotting argument: $1" >&2; exit 2 ;;
  esac
done
"$PYTHON" -B analysis/plot.py --directory "$directory" --step "$step"
"$PYTHON" -B analysis/analyze_residual.py --directory "$directory"
