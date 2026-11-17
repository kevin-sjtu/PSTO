# Generating Robertson data

The OpenFOAM templates retain their original `0/` initial fields and
`system/constant` configuration. Generated data and logs are not distributed.

| Template | Chemistry | Local HDF5 output |
|---|---|---|
| `openfoam/2Dtranscounterflow` | off | `Data_trans.h5`: 5,001 frames over 0.5 s |
| `openfoam/2Dtransrobertcounterflow` | on | `Data_transrobert.h5`: 4,001 frames over 0.4 s |

Both use a 200×100×1 mesh and save frames every `1e-4 s`.

## Build and run

Load an OpenFOAM 6 environment and build the supplied solver:

```bash
# From cases/robertson:
(cd data/openfoam/solver/pimpletransrobertFoam && wmake)
python -B data/run_openfoam.py lf
python -B data/run_openfoam.py hf
```

The Make configuration retains the original Cantera include variables, so
`WM_CANTERA_DIR` must be defined. The target is `FOAM_USER_APPBIN`. OpenFOAM
copyright/license notices remain in the source. A clean solver build and CFD
run on another installation have not yet been validated.

The runner creates fresh cases under `work/openfoam/{lf,hf}`, then executes
`blockMesh`, `checkMesh`, the solver and export. It rejects existing work
folders or output datasets. Add `--processes N` to decompose, run MPI and
reconstruct with an OpenFOAM-compatible MPI installation.

For a three-frame integration check on the original mesh:

```bash
python -B data/run_openfoam.py lf --end-time 0.0002 \
  --work-dir work/openfoam/lf-smoke --output work/lf-smoke.h5
```

## Export an existing local run

```bash
python -B data/data_extr.py --case-dir work/openfoam/lf
python -B data/data_extract.py --case-dir work/openfoam/hf
```

Defaults write the distinct LF/HF filenames above beside these programs,
independently of the shell's working directory. `--output` chooses a new file;
`--start/--end/--dt` or `--times` selects frames. See `--help` for all options.

HDF5 key `data` stores float32 arrays of shape `(time, 6, 100, 200)` with channels
`A, B, C, D, u, v`; key `time` records frame times. The exporter reads uniform
or nonuniform ASCII fields, including gzip files, using x-fast cell ordering.
It retains the original convention of copying adjacent interior rows to the
top/bottom rows for all six channels. Binary fields are unsupported.

The auxiliary historical `real-T` workflow is not part of this main LF/HF
pipeline. Meshes, processor cases, later time fields and HDF5 files are generated
locally and excluded from publication.
