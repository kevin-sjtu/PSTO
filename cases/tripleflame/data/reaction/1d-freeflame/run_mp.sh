#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
exec "${PYTHON:-python}" -B "$SCRIPT_DIR/mp-1d-random-sampling.py" "$@"
