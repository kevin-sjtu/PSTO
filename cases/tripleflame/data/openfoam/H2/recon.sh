#!/usr/bin/env bash
set -euo pipefail
CASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec reconstructPar -case "$CASE_DIR" "$@"
