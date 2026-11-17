# Data preparation

This directory contains source programs, OpenFOAM dictionaries, mechanism definitions and the explicitly retained initial `0/` fields. It does not contain generated datasets, nonzero-time fields, decomposed cases, generated meshes, logs or model weights.

## Two-dimensional OpenFOAM cases

`openfoam/H2/` contains the reacting case (`chemistry on`); `openfoam/H2_flow/` contains its reaction-free counterpart (`chemistry off`). Both retain their original mesh dictionary, discretization, physical properties and initial fields. The `application` label is set to `reactingFoam`, matching the actual supplied run scripts and benchmark. Alternative CanteraTorch configuration is not included.

The cases require the compatible OpenFOAM build with `canteraPsiThermo` and `canteraChemistryReader`, plus its Cantera dependencies. This is not a stock-OpenFOAM-only setup. The full extension/build dependency closure has not yet been verified or bundled. Activate the appropriate environment yourself before using:

```bash
bash data/openfoam/H2/run.sh --help
bash data/openfoam/H2_flow/run.sh --help
# After the solver build is available:
bash data/openfoam/H2/run.sh
bash data/openfoam/H2_flow/run.sh
```

The wrappers generate the mesh using `blockMesh` if absent, then run the case serially by default. For MPI, set `NPROCS` and ensure `system/decomposeParDict` has the same number of subdomains; the wrapper validates the match. `MPIEXEC` can select another MPI launcher. Existing initial fields are retained, so the `funkySet*Dict` utilities are not run automatically; recreating alternative initial states requires the compatible funky tools and a confirmed initialization sequence.

Generated `constant/polyMesh`, `processor*`, nonzero-time directories and solver output are local artifacts. The H2/H2_flow files alone do not supply the missing two-dimensional CFD-to-HDF5 exporter.

## Reaction utilities

`reaction/1d-freeflame/` retains source utilities and an auxiliary one-dimensional reference configuration. Its original `dfLowMachFoam` application entry is preserved because the actual solver used for that auxiliary case has not been confirmed. It is **not an independently validated data-generation workflow** and is not called by the main automatic OpenFOAM run wrappers. The nested `0/0` backup and all multi-condition/time-result archives are excluded.

| Program | What it does and required inputs |
|---|---|
| `equilibrium_change_replace.py` | Patches a selected local initial-state directory with the original premixed/equilibrium construction. Use `--dry-run` to inspect the intended operation. |
| `0D_data_extraction.py` | Extracts existing local one-dimensional CFD fields into a three-dimensional `(samples, variables, points)` array. `--base-dir`, `--folders` and `--output` select local paths. |
| `compute_formation_enthalpies.py` | Computes formation-enthalpy coefficients from the supplied YAML mechanism. |
| `mp-1d-random-sampling.py` | Applies the original CFD-state perturbation and reaction filtering procedure; requires `--hf`, `--formation-enthalpies` and `--output`. Its reaction interval remains `1e-5 s`. |
| `single_step.py` | Integrates an existing two-dimensional temperature/species matrix; requires `--input` and `--output`. Its original interval remains `5e-7 s`. |

The single-step utility's input layout differs from the extraction program's output; they must not be presented as a ready-to-run pipeline. No conversion or timestep change is invented here. Likewise, the exact relationship between these utilities' outputs and the current fine/coarse reaction-training files has not been verified. The original sampling seeds are not available. Path and output arguments were made explicit; worker upper-bound checks and exception propagation were repaired without changing valid samples' chemistry or selection formulas.

Use `--help` on these programs before preparing inputs. The sampling utilities remain Linux multiprocessing scripts and have not been executed as part of repository preparation. Do not infer full reproducibility from their presence.

## Missing links before full reproduction

1. The two-dimensional OpenFOAM-field-to-HDF5 export and spatial/channel ordering procedure for `H2_tripleflow.h5` and `H2_tripleflame.h5` must be supplied and validated.
2. The generator for `H2_tripleflow_dt1e-5_pairs.h5`, including its temporal split and metadata, is not yet present. `src/paired_data.py` only validates/loads paired files and builds a local cache.
3. The final assembly and provenance of `0d_data.npy` and `0d_data_1e-5.npy` must be confirmed. Existing reaction utilities cannot yet be claimed to generate those exact inputs.
4. Constitutive training expects a source matrix such as `0d_dataset_cfd.npy`; its complete generation path remains unverified. `src/train_constitutive.py` builds labels from a supplied source, not from an invented substitute.
5. Operator diagnostics require additional transport-only and exact Lie-reaction reference trajectories. Their production procedures are not yet supplied as complete verified commands.

All listed datasets and intermediate outputs are generated locally and excluded from publication. Mechanism source comments are preserved; third-party attribution and redistribution terms still require repository-level confirmation.
