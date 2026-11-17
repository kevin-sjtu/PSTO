# Robertson counterflow

The transport surrogate predicts four species mass fractions, conditioned on
the two velocity components. The coupled evaluation uses a time step of
`1e-4 s`, Strang splitting and the adaptive SDIRK2 reaction backend. Models are
PSTO, FNO, U-Net (`Unet`) and U-FNO (`UFNO`).

Run the commands below from this directory after installing the repository's
Python dependencies. No datasets, trained weights, logs or saved results are
included. Generate the LF/HF data first using [data/README.md](data/README.md).

## Training

```bash
bash scripts/train.sh PSTO
bash scripts/train.sh FNO
bash scripts/train.sh Unet
bash scripts/train.sh UFNO
```

The default script uses four GPU processes, batch size 16 per process, two
gradient accumulation steps and 2,500 iterations. With 4,000 training pairs,
this presents 40 epochs of examples. PSTO uses eight latent features per
variable, 16 Fourier modes and four stages. The baseline width is 20.
Learning rate, weight decay and loss coefficients are set explicitly in the
script, following the existing experiment configuration.

`NPROC_PER_NODE` changes the process count; the default iteration count becomes
`10000 / NPROC_PER_NODE` to retain the number of presented examples. A changed
process count changes the effective batch size and does not promise identical
optimization or results. `N_ITERS` overrides the iteration count. Additional
arguments are forwarded to the training CLI:

```bash
NPROC_PER_NODE=1 N_ITERS=2 bash scripts/train.sh PSTO --device cpu --num_workers 0
```

This is a short execution test after data generation, not a paper reproduction.
Checkpoint and training-history files are generated locally beneath
`checkpoints/models/model_<MODEL>/`. `TRAIN_OUTPUT_ROOT` overrides that root.

## Coupled evaluation and plotting

```bash
bash scripts/test.sh
bash scripts/plot.sh
```

Evaluation requires the four locally trained `checkpoint_best.pt` files. The
default evaluates 100 steps from HF frame index 1001, using the `gpu_sdirk2`
reaction backend and Strang splitting. `DEVICE=cpu` selects CPU execution;
`TEST_MODELS=PSTO` selects a subset for evaluation. The standard comparison
plots require all four model outputs.

`TEST_SAVE_DIR` controls the evaluation output directory and the default plot
input directory. `PLOT_DIRECTORY` and `PLOT_STEP` override plotting inputs.
The scripts generate CSV/NPZ evaluation outputs locally and PNG figures only;
they do not write to a separate paper directory. These generated files are not
repository inputs. The score is the arithmetic mean of the four species'
relative L2 errors, computed in physical variables.

## Additional analysis

```bash
python -B analysis/analyze_anchored_stages.py --device cpu
```

This computes anchored-stage diagnostics from the local PSTO checkpoint and
LF validation data. Use `--checkpoint`, `--samples`, `--figure-dir` and
`--result-dir` to select inputs and outputs.

`analysis/analyze_distribution_shift.py` only plots an existing JSON containing
`embedding`, `indices.lf`, and `mmd`. The procedure that computes the embedding
and MMD is not provided, and the historical JSON is not included. It is therefore
an optional plotting utility, not a complete distribution-shift reproduction:

```bash
python -B analysis/analyze_distribution_shift.py --input /path/to/local.json \
  --output analysis/figures/inference_transfer.png
```

## Small tests

```bash
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

These tests generate tiny temporary ASCII fields and verify the exporter,
channel/cell ordering, boundary convention and output safety. They require no
experimental data or OpenFOAM executable. Full CFD generation, training and
100-step evaluation are separate, substantially longer procedures.
