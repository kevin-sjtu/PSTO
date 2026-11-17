# Generating Robertson data

The two OpenFOAM templates contain their original `0/` fields, `system/`
dictionaries and physical/chemical configuration under `constant/`:

| Template | Chemistry | Export |
|---|---|---|
| `openfoam/2Dtranscounterflow` | off (LF transport) | `Data_trans.h5`, 5,001 frames |
| `openfoam/2Dtransrobertcounterflow` | on (HF reaction/transport) | `Data_transrobert.h5`, 4,001 frames |

Both use the supplied `pimpletransrobertFoam` source, a 200×100×1 mesh and a
saved-frame interval of `1e-4 s`. The dictionary end times are 0.5 s for LF and
0.4 s for HF. Initial fields, dictionary numerical settings and reaction
configuration are retained. Generated meshes, later time directories,
decomposed processor cases, logs and HDF5 files are not included.

## Build the solver

Use an OpenFOAM 6 environment with its compilation tools and required libraries.
The supplied Make options retain the original Cantera include variables; make
sure `WM_CANTERA_DIR` is defined for that environment. With OpenFOAM's environment
already sourced, build from this case directory:

```bash
cd data/openfoam/solver/pimpletransrobertFoam
wmake
```

The target is `FOAM_USER_APPBIN`, not the global OpenFOAM application directory.
The source retains its OpenFOAM copyright and license notices. The release
preparation checked source files and CLI syntax, but did not compile this solver
or establish that another OpenFOAM installation has identical local library
modifications. A clean build and short CFD run remain required validation.

## Run and export

Return to `cases/robertson`, then run:

```bash
python -B data/run_openfoam.py lf
python -B data/run_openfoam.py hf
```

The runner creates a fresh copy of `0/`, `constant/` and `system/` beneath
`work/openfoam/lf` or `work/openfoam/hf`. It runs `blockMesh`, `checkMesh`, the
solver, and then the exporter. It never runs in the checked-in template or an
external original experiment directory. Existing work directories and datasets
are not overwritten. Use a new `--work-dir` below this case's `work/` to rerun.

For parallel CFD, append `--processes 4` (or another desired process count).
The runner updates only the copied decomposition dictionary, then invokes
`decomposePar`, `mpirun` and `reconstructPar`. MPI must be compatible with the
OpenFOAM installation. Processor output stays in the ignored work directory;
the runner does not perform automatic deletion.

A short integration test can use:

```bash
python -B data/run_openfoam.py lf --end-time 0.0002 \
  --work-dir work/openfoam/lf-smoke --output work/lf-smoke.h5
```

This generates three frames on the original mesh and is not a reproduction of
the full training trajectory. `--no-export` leaves the reconstructed CFD output
for separate inspection.

## Export an existing local run

```bash
python -B data/data_extr.py --case-dir work/openfoam/lf
python -B data/data_extract.py --case-dir work/openfoam/hf
```

The LF and HF wrappers select distinct defaults in this directory:
`Data_trans.h5` and `Data_transrobert.h5`. Output paths are independent of the
shell's working directory. Use `--output` for a different new destination;
existing files are rejected. `--start`, `--end`, `--dt`, or explicit `--times`
select frames. `--nx/--ny` are available for deliberately generated test grids;
the paper configuration is 200×100.

The HDF5 `data` array has shape `(time, 6, 100, 200)` and channel order
`A, B, C, D, u, v`, stored as float32. A `time` dataset records the selected
times. The exporter reads uniform or nonuniform ASCII internal fields,
including gzip-compressed field files. It retains the original x-fast cell
ordering and copies the adjacent interior row to each top/bottom row for all
six channels. It reads the actual `0/` initial fields instead of inferring a
missing time frame. Binary OpenFOAM fields are deliberately rejected.

The previous LF exporter selected only 4,000 intervals and all exporters used
the same HF output filename. This release uses the 5,000 LF intervals supported
by the original LF dictionary and existing LF dataset dimensions, separates LF
and HF output names, and parses field declarations rather than fixed text-line
numbers. It does not alter the CFD numerical scheme or reaction model.

The released files do not include the historical auxiliary `real-T` workflow
or generated Cantera YAML. Those are not required for the main LF/HF pipeline.
