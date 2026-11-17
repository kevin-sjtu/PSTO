# Planar hydrogen jet flame

This case learns reaction-free transport and couples the transport surrogate to a species MLP with enthalpy-based temperature recovery. PSTO, FNO, U-Net and U-FNO predict temperature, nine species mass fractions and two velocity components. The main rollout uses 100 updates of `1e-6 s`; the larger-step experiment uses 10 updates of `1e-5 s` over the same physical horizon.

Install the Python dependencies described in the [repository README](../../README.md). Python training and evaluation require locally generated datasets and locally trained auxiliary models. They are not included in this repository. See [data preparation](data/README.md) for the available generators and the remaining gaps.

## Local commands

Run from this directory. Each command also supports `--dry-run`, which prints the actual command without opening datasets or launching training. Additional options are forwarded to the underlying Python program.

```bash
# Auxiliary constitutive model (requires its local source dataset).
python -B scripts/run.py constitutive --dt 1e-6

# Species reaction models must match the prediction interval.
python -B scripts/run.py source --dt 1e-6
python -B scripts/run.py source --dt 1e-5

# Train each of PSTO, FNO, Unet, UFNO separately for each interval.
bash scripts/train.sh --dt 1e-6 --model PSTO
bash scripts/train.sh --dt 1e-5 --model PSTO

# Evaluate all four locally trained transport models with the matching source MLP.
bash scripts/test.sh --dt 1e-6 --device cuda
bash scripts/test.sh --dt 1e-5 --device cuda

# Plot the saved local rollouts. Figure files use PNG only.
bash scripts/plot.sh --dt 1e-6
bash scripts/plot.sh --dt 1e-5
```

`train_1e-5.sh`, `test_1e-5.sh` and `train_source_1e-5.sh` are aliases with the interval already set. CPU evaluation is available through `--device cpu`; `--device cpu` on the training launcher requests CPU execution through Accelerate.

The public training recipe retains the transport architecture and loss coefficients of the supplied experiment scripts: latent width 8 for PSTO, baseline width 20, 16 Fourier modes where applicable, four layers/iterations, learning rate `3e-4`, weight decay `1e-5`, ROI/PDE weights `0.1`, wavelet weight `0.02`, and mass/nonnegativity weights `1.0`. The default effective batch is 32: one process, batch 4 and eight accumulation microsteps. The default 40,000 microsteps process 160,000 examples, corresponding to 40 epochs for 4,000 fine-step training pairs; the coarse pairing excludes split-crossing pairs and gives approximately 40 epochs. Warmup spans 100 optimizer updates.

`TRAIN_NPROC=4` scales the default accumulation to 2 and microsteps to 10,000, preserving the effective batch and example count. This is a documented public recipe, not a recovered command for every historical checkpoint. Validation uses the supplied script's fixed temporal subset of at most 256 pairs; set `TRAIN_MAX_VAL_PAIRS=0` to evaluate the full validation split.

| Setting | Default |
|---|---|
| `TRAIN_DATA_FILE` | `data/H2_tripleflow.h5` or `data/H2_tripleflow_dt1e-5_pairs.h5` |
| `HF_DATA_FILE` | `data/H2_tripleflame.h5` |
| `TRAIN_OUTPUT_ROOT` | `checkpoints/models` or `checkpoints/models_1e-5` |
| `TRAIN_CACHE_DIR` | `<TRAIN_OUTPUT_ROOT>/data_cache` |
| `SOURCE_DATA_FILE` | `data/0d_data.npy` or `data/0d_data_1e-5.npy` |
| `SOURCE_CHECKPOINT` | `<TRAIN_OUTPUT_ROOT>/model_DNN/dnn_model_Y.pt` |
| `COEFFICIENT_CHECKPOINT` | `checkpoints/models/model_coeff/constitutive_DNN_BCT.pt` |
| `TEST_SAVE_DIR` | `analysis/figures` or `analysis/results_1e-5/lie_10` |
| `DATA_IDX` | 2001, using the evaluator's one-based indexing |

New training runs refuse to overwrite an existing model directory. Coarse evaluation similarly requires a new output directory. Locally generated training records and normalization metadata are required for coarse checkpoint verification; the launcher runs the reload verification after coarse training. These generated files must remain local even though they are necessary at runtime.

Training, evaluation and plotting outputs remain inside this case or the paths explicitly supplied by the user. Plotting does not copy files into a paper directory. Fine training caches and statistics are written under the selected cache directory, not beside the source HDF5 file.

## Checks and additional analyses

The seven tests below use temporary synthetic inputs and need no experimental dataset:

```bash
python -B -m unittest discover -s tests -p test_paired_data.py -v
python -B -m unittest discover -s tests -p test_pde_velocity.py -v
```

`test_lie_1e5.py` is an integration test that requires local reaction data and the mechanism; it is not part of the data-free checks. A training smoke test is not a reproduction of the paper's reported errors.

`analysis/analyze_operators.py` evaluates one-step errors, local transport sensitivity and coupled rollout accumulation. It additionally requires the locally generated transport-only and exact reaction reference trajectories. Its scaling uses the full reference trajectory's channel extrema, recomputed locally instead of loading an unpublished statistics file. The diagnostic definitions are unchanged.

Performance measurement and plotting commands are documented in [analysis/performance/README.md](analysis/performance/README.md). GPU and OpenFOAM timings must be measured on the actual machine; no timing results are bundled or assumed.

The scientific model, loss and rollout implementations are retained. Repository preparation changes paths, launchers, output handling and selected utility error checks. Historical weights are not supplied; current training has not been rerun to claim identical historical values.
