#!/usr/bin/env bash
set -euo pipefail
CASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$CASE_DIR"
if [[ "${1:-}" == --help ]]; then
  echo "Activate the compatible OpenFOAM/Cantera build, then run this case; NPROCS defaults to 1."
  exit 0
fi
# Activate the required OpenFOAM/Cantera build before running this script.
command -v reactingFoam >/dev/null
if [[ ! -f constant/polyMesh/points && ! -f constant/polyMesh/points.gz ]]; then
  blockMesh
fi
NPROCS=${NPROCS:-1}
if [[ "$NPROCS" == 1 ]]; then
  reactingFoam -case "$CASE_DIR"
else
  # The template has 128 subdomains; change the local dictionary explicitly
  # before selecting another MPI size.
  configured=$(foamDictionary system/decomposeParDict -entry numberOfSubdomains -value)
  if [[ "$NPROCS" != "$configured" ]]; then
    echo "Set numberOfSubdomains=$NPROCS in system/decomposeParDict before MPI execution." >&2
    exit 2
  fi
  decomposePar -case "$CASE_DIR"
  "${MPIEXEC:-mpirun}" -np "$NPROCS" reactingFoam -parallel -case "$CASE_DIR"
  reconstructPar -case "$CASE_DIR"
fi
