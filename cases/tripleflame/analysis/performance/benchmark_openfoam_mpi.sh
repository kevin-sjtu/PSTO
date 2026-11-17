#!/usr/bin/env bash
# Externally time the same decomposition/solver/reconstruction phases as the paper.
set -euo pipefail
CASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PERFORMANCE_DIR="$CASE_ROOT/analysis/performance"
if [[ "${1:-}" == --help ]]; then
  echo 'Set START_SNAPSHOT, CANTERA_XML and ELEMENTS_XML to locally generated inputs.'
  echo 'Activate the compatible OpenFOAM/Cantera/MPI build. Optional: NPROCS=128, MPIEXEC, FOAM_TEMPLATE, BENCHMARK_OUTPUT, KEEP_WORK=1.'
  exit 0
fi
: "${START_SNAPSHOT:?Set START_SNAPSHOT to the locally generated fields at t=0.0001 s}"
: "${CANTERA_XML:?Set CANTERA_XML to the compatible h2o2 XML mechanism}"
: "${ELEMENTS_XML:?Set ELEMENTS_XML to the compatible Cantera elements.xml}"
FOAM_TEMPLATE=${FOAM_TEMPLATE:-$CASE_ROOT/data/openfoam/H2}
NPROCS=${NPROCS:-128}
PYTHON=${PYTHON:-python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
work_dir=$(mktemp -d "${TMPDIR:-/tmp}/psto_foam_benchmark.XXXXXXXX")
cleanup() {
  if [[ "${KEEP_WORK:-0}" == 1 ]]; then
    echo "Benchmark working directory retained: $work_dir"
  else
    rm -rf -- "$work_dir"
  fi
}
trap cleanup EXIT
cp -a "$FOAM_TEMPLATE/0" "$FOAM_TEMPLATE/system" "$FOAM_TEMPLATE/constant" "$work_dir/"
cp -a "$START_SNAPSHOT" "$work_dir/0.0001"
cp "$CANTERA_XML" "$work_dir/openfoam_h2o2.xml"
cp "$ELEMENTS_XML" "$work_dir/elements.xml"
"$PYTHON" -B "$PERFORMANCE_DIR/benchmark_openfoam_mpi.py" prepare "$work_dir"
if [[ ! -f "$work_dir/constant/polyMesh/points" && ! -f "$work_dir/constant/polyMesh/points.gz" ]]; then
  blockMesh -case "$work_dir" > "$work_dir/mesh.log" 2>&1
fi
foamDictionary "$work_dir/system/decomposeParDict" -entry numberOfSubdomains -set "$NPROCS"
/usr/bin/time -f '%e' -o "$work_dir/decomposePar.time" decomposePar -case "$work_dir" -force -time 0.0001 > "$work_dir/decompose.log" 2>&1
export CANTERA_DATA="$work_dir"
set +e
/usr/bin/time -q -f '%e' -o "$work_dir/mpirun.time" "${MPIEXEC:-mpirun}" -np "$NPROCS" reactingFoam -parallel -case "$work_dir" > "$work_dir/solver.log" 2>&1
mpi_status=$?
set -e
/usr/bin/time -f '%e' -o "$work_dir/reconstructPar.time" reconstructPar -case "$work_dir" -time 0.0002 > "$work_dir/reconstruct.log" 2>&1
cleanup_args=()
if [[ "$mpi_status" != 0 ]]; then cleanup_args=(--postrun-cleanup-error); fi
"$PYTHON" -B "$PERFORMANCE_DIR/benchmark_openfoam_mpi.py" parse "$work_dir/solver.log"   --host "${HOSTNAME:-unknown}" --case-dir "$work_dir" --mpi-exit-code "$mpi_status"   --source-case "$FOAM_TEMPLATE" --n-procs "$NPROCS"   --output "${BENCHMARK_OUTPUT:-$PERFORMANCE_DIR/openfoam_timing_100.json}" "${cleanup_args[@]}"
