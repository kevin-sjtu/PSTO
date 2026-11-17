#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" -B "$DATA_DIR/data_extract.py" "$@"
