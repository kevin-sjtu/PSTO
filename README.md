# PDE-Structured Transport Operator (PSTO)

Research code for **PSTO: PDE-structured transfer learning for Multiscale convection–diffusion–reaction systems with shared implicit updates**, by Kaiwen Li, Dezhi Zhou and Xingcai Lu.

PSTO learns transport from reaction-free convection–diffusion data and couples it with a separate reaction solver or neural reaction surrogate. A transport block shared across physical variables is reused in relaxed fixed-point iterations. The comparison models are FNO, U-Net and U-FNO.

## Cases

| Case | Predicted variables | Time step | Coupled rollout | Reaction treatment |
|---|---|---|---|---|
| [Canonical CDR](cases/cdr/README.md) | `u, v, phi` | `2.5e-4 s` | 100 steps | Polynomial source terms |
| [Robertson counterflow](cases/robertson/README.md) | Four species mass fractions | `1e-4 s` | 100 steps | Adaptive SDIRK2, Strang splitting |
| [Hydrogen planar jet flame](cases/tripleflame/README.md) | Temperature, nine species mass fractions, `u, v` | `1e-6 s` | 100 steps | Species MLP and enthalpy-based temperature recovery, Lie–Trotter splitting |

The flame case also includes a separately trained `1e-5 s` configuration with ten coupled steps over the same `1e-4 s` physical horizon. Use matching transport and reaction models for each time step.

PSTO uses 197,841 trainable transport parameters, eight latent channels per variable, 16 Fourier modes per direction and four shared iterations. The manuscript's main training schedules are 100 epochs for CDR and 40 for the other cases. Checkpoints are selected by reaction-free validation loss.

## Repository layout

```text
cases/
  cdr/          # Numerical data preparation and transport–reaction experiment
  robertson/    # OpenFOAM configurations, exporter, solver and neural experiment
  tripleflame/  # OpenFOAM configurations, reaction scripts and two time steps
```

Each case keeps its own `src/`, `utils/`, `scripts/`, `analysis/` and data-preparation instructions. The implementation and dependencies differ between cases, so run their commands from the indicated case directory.

Only source code, configuration files and OpenFOAM initial fields in `0/` are included. Datasets, trained weights, training/test logs, generated meshes, time-dependent CFD output and saved prediction results remain local.

## Installation

The local checks use Linux and Python 3.12. Create a separate environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, choose a PyTorch build appropriate for the machine using the [official installation instructions](https://pytorch.org/get-started/locally/). `requirements.txt` records the direct package versions used for local checks; it is not a claim that all hardware configurations have been tested.

OpenFOAM data generation requires a separate OpenFOAM 6 environment and the case-specific solver dependencies described in each data README. The Robertson neural reaction solver defaults to `gpu_sdirk2`; Assimulo is an optional backend and is not needed for the default path. The flame case uses Cantera in Python and additional Cantera support in its CFD solver.

## Prepare data

- **CDR:** follow [data/README.md](cases/cdr/data/README.md) to obtain compatible Burgers velocity input from the [PeRCNN resources](https://github.com/Raocp/PeRCNN), then run `python data/numer.py` from `cases/cdr/`. The MAT file is not redistributed here.
- **Robertson:** use the two OpenFOAM cases and exporter described in [data/README.md](cases/robertson/data/README.md). The reaction-free and reacting configurations produce distinct local HDF5 inputs.
- **Flame:** follow [data/README.md](cases/tripleflame/data/README.md) for CFD configurations, reaction-data preparation and required data layouts. Remaining gaps in the historical export pipeline are documented there; the presence of case dictionaries alone does not establish complete dataset regeneration.

Run simulations and generators in a working copy. Generated datasets and logs are excluded by `.gitignore`.

## Train, evaluate and plot

The case READMEs provide the commands and required inputs for:

1. Preparing reaction-free training data and reacting reference trajectories.
2. Training PSTO or a baseline with the case's configured loss and schedule.
3. Coupling the learned transport map with its reaction solver.
4. Computing metrics and producing PNG figures from local evaluation output.

Training scripts expose process counts and accept command-line overrides. When changing the number of processes or batch size, preserve the intended number of epochs and optimizer warmup updates. The two flame time steps have separate training and evaluation entrypoints.

The main accuracy metric computes relative L2 error separately for each predicted variable in physical units, then takes their arithmetic mean. The main rollouts report steps 1, 5, 10, 20, 50 and 100; the larger flame time step reports steps 1, 2, 5 and 10. The flame sensitivity diagnostic uses a separate whole-state relative norm, as described in the manuscript.

Existing experimental weights and results are not included. In particular, historical weights predate some later PDE-loss implementation corrections. No new training run has established that the packaged training code reproduces the historical manuscript numbers exactly. The case READMEs distinguish runnable entrypoints from analyses with missing historical preprocessing.

## Lightweight checks

These tests generate temporary inputs and do not need the full datasets:

```bash
(cd cases/cdr && python -B -m unittest discover -s tests -p test_physics_loss.py -v)
(cd cases/tripleflame && python -B -m unittest discover -s tests -p test_paired_data.py -v)
(cd cases/tripleflame && python -B -m unittest discover -s tests -p test_pde_velocity.py -v)
```

The flame Lie-splitting tests additionally require locally prepared reaction data. Full CFD runs, model training and 100-step evaluations are separate reproduction tasks, not part of the lightweight checks.

## Attribution and licensing

The manuscript title and authors are listed above; no publication DOI or journal status is asserted here. See the [third-party notices](THIRD_PARTY_NOTICES.md) for retained upstream material and data references. A project-wide open-source license has not been selected. Existing third-party notices and license texts remain attached to their respective files.
