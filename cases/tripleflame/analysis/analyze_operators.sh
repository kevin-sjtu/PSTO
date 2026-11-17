#!/usr/bin/env bash
set -euo pipefail
CASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CASE_DIR"
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
exec "${PYTHON:-python}" -B analysis/analyze_operators.py "$@"
