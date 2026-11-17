# Canonical convection–diffusion–reaction case

This case learns reaction-free transport for `u`, `v`, and `phi`, then couples the learned transport with the polynomial reaction terms. The reference configuration uses a periodic `100 × 100` grid, `dt = 2.5e-4 s`, and 100 autoregressive evaluation steps.

Run the commands below from this directory after installing the repository dependencies. No datasets, checkpoints, training logs, or evaluation results are distributed with the code.

## 1. Prepare data

Follow [data/README.md](data/README.md) to obtain compatible Burgers velocity data from the PeRCNN resources. Place the local MAT beside `data/numer.py`, then generate the source-free and reacting trajectories:

```bash
python data/numer.py
```

This creates `data/CD.h5` and `data/CDR.h5`. The input MAT and generated HDF5 files remain local. An upstream file with a different grid or stored-frame interval must not be substituted by renaming it.

## 2. Train transport models

The local launchers use the formal experiment parameters and the Python environment that is already active; they do not load cluster modules or activate a machine-specific environment.

```bash
bash scripts/train.sh PSTO
bash scripts/train.sh FNO
bash scripts/train.sh Unet
bash scripts/train.sh UFNO
```

PSTO defaults to four local GPU processes and 2,500 training microsteps; each baseline defaults to one GPU and 10,000 microsteps. All use batch size 16 per process, gradient accumulation 2, and the original loss weights. The launchers retain the formal architecture defaults, including 16 Fourier modes where applicable. Checkpoints and training records are generated under `checkpoints/models/model_<name>/`.

Pass additional Python CLI options after the model name to override a default. Set `PYTHON` to select an interpreter and `NPROC_PER_NODE` to change the process count. For example, this single-GPU PSTO command retains the total number of presented training samples by using 10,000 microsteps:

```bash
NPROC_PER_NODE=1 bash scripts/train.sh PSTO --n_iters 10000
bash scripts/train.sh PSTO --help
```

Changing parallelism changes the optimization trajectory; this command is not a claim of identical trained weights. Existing historical weights are not included, and no results from retraining this release are claimed.

## 3. Evaluate coupled rollouts

After training all four models:

```bash
bash scripts/test.sh
```

The default evaluates the reacting dataset for 100 steps, beginning with its initial state, and writes `analysis/figures/rollout.npz` and `comparison.csv`. The reported main error is the arithmetic mean of the relative L2 errors of the three physical variables. Override the device, model subset, start index, duration, or output path through the CLI:

```bash
bash scripts/test.sh --device cpu --models PSTO --steps 1 --save_dir /tmp/psto_cdr_check
bash scripts/test.sh --dataset lf --save_dir analysis/figures/lf
```

The source-free evaluation defaults to `data_idx=200`; the reacting evaluation defaults to `data_idx=1`. Set `--data_idx` explicitly to select another reference window. Short checks still require locally generated data and trained checkpoints.

## 4. Plot local evaluation results

The standard figure scripts expect an evaluation containing all four models:

```bash
bash scripts/plot.sh
bash scripts/plot.sh --directory analysis/figures --step 100
```

Figures are saved only as PNG in the selected result directory. The residual analysis also writes local JSON/NPZ diagnostics. Nothing is copied to a manuscript directory. Matplotlib uses Times New Roman if installed, otherwise DejaVu Serif; no font files are distributed.

The residual diagnostic uses its original periodic centered advection and five-point diffusion definition. The training loss retains its own existing centered spatial implementation; the release preparation does not change either numerical method.

## 5. Lightweight check

The two existing physics-loss tests use analytic inputs and require no experiment data:

```bash
python -B -m unittest discover -s tests -p test_physics_loss.py -v
```

Generated datasets, checkpoints, logs, diagnostic arrays, and figures should remain outside the committed source tree.
