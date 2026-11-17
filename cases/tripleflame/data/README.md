# Flame data preparation

This directory provides OpenFOAM configurations, mechanism definitions,
initial `0/` fields and reaction utilities. Datasets, later-time CFD fields,
meshes, decomposed cases and logs must be generated locally.

## Two-dimensional cases

`openfoam/H2/` is reacting (`chemistry on`); `openfoam/H2_flow/` is reaction-free
(`chemistry off`). Both retain the original mesh, physical and discretization
settings. They require a Cantera-enabled OpenFOAM build providing
`canteraPsiThermo` and `canteraChemistryReader`; the full extension/build dependency
closure has not yet been verified or bundled.

After loading a compatible solver environment, run from `cases/tripleflame`:

```bash
bash data/openfoam/H2/run.sh
bash data/openfoam/H2_flow/run.sh
```

The wrappers use `reactingFoam`, build the mesh if absent and run serially by
default. For MPI, `NPROCS` must match `system/decomposeParDict`; `MPIEXEC` selects
the launcher. Existing initial fields are retained. The `funkySet*Dict` utilities
are not automatically applied; different initial states require a confirmed
initialization sequence and compatible tools.

## Reaction utilities

`reaction/1d-freeflame/` contains auxiliary source utilities and a one-dimensional
reference case. Its original `dfLowMachFoam` entry is retained, but the actual
auxiliary solver workflow remains unverified and is not run by the main wrappers.

| Program | Input and purpose |
|---|---|
| `equilibrium_change_replace.py` | Construct premixed/equilibrium initial fields in a selected local directory; inspect with `--dry-run`. |
| `0D_data_extraction.py` | Extract existing 1D CFD fields into `(samples, variables, points)` arrays. |
| `compute_formation_enthalpies.py` | Obtain formation-enthalpy coefficients from the YAML mechanism. |
| `mp-1d-random-sampling.py` | Perturb CFD states and apply reaction filtering; requires source data and enthalpies, using `dt=1e-5 s`. |
| `single_step.py` | Integrate a supplied 2D temperature/species matrix using `dt=5e-7 s`. |

Use each program's `--help` for explicit input/output paths. The extraction and
single-step utilities expect different layouts and are not a complete pipeline.
Original sampling seeds and the exact assembly of the final training files
have not been recovered; the Linux multiprocessing sampling scripts have not
been validated through a full generation run.

## Required links for full regeneration

1. A validated 2D CFD-to-HDF5 exporter, including spatial/channel ordering, for
   `H2_tripleflow.h5` and `H2_tripleflame.h5`.
2. The coarse-pair construction and temporal-split procedure for
   `H2_tripleflow_dt1e-5_pairs.h5`; `src/paired_data.py` only loads and caches it.
3. Final assembly and provenance of `0d_data.npy` and `0d_data_1e-5.npy`.
4. Generation of the constitutive source matrix, such as `0d_dataset_cfd.npy`;
   the constitutive trainer builds labels from an existing source.
5. Production of the additional transport-only and exact Lie-reaction reference
   trajectories required by the operator diagnostics.

These missing steps remain necessary even though pretrained models are supplied.
Mechanism attribution and retained third-party terms are recorded in the
[repository notices](../../../THIRD_PARTY_NOTICES.md).
