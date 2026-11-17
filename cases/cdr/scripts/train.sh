#!/usr/bin/env bash
# Official defaults: PSTO uses four processes; baselines use one process.
set -euo pipefail
CASE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$CASE_DIR"
PYTHON=${PYTHON:-python}
MODEL=${MODEL:-PSTO}
if (( $# )) && [[ "$1" != -* ]]; then
  MODEL=$1
  shift
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

case "$MODEL" in
  PSTO)
    architecture=(--latent_per_var 8 --modes 16)
    iterations=2500
    processes=${NPROC_PER_NODE:-4}
    ;;
  FNO|UFNO)
    architecture=(--hidden_channels 20 --modes 16)
    iterations=10000
    processes=${NPROC_PER_NODE:-1}
    ;;
  Unet)
    architecture=(--hidden_channels 20)
    iterations=10000
    processes=${NPROC_PER_NODE:-1}
    ;;
  *) echo "Unknown training target: $MODEL (choose PSTO, FNO, Unet, UFNO)" >&2; exit 2 ;;
esac
if [[ ! "$processes" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE must be a positive integer" >&2
  exit 2
fi
common=(
  --model_type "$MODEL"
  --n_layers 4
  --batch_size 16
  --grad_accum_steps 2
  --num_workers 2
  --n_iters "$iterations"
  --learning_rate 3e-4
  --weight_decay 1e-5
  --warmup_iters 200
  --scheduler_units updates
  --max_val_pairs 256
  --print_every 50
  --save_every 100
  --start_save_step 50
  --physics_weight 0.1
  --roi_weight 0.1
  --weighted_power 1.0
  --wavelet_weight 0.02
  --wavelet_level 2
  --ckpt_dir "${TRAIN_OUTPUT_ROOT:-checkpoints/models}/model_${MODEL}"
)
launcher=("$PYTHON" -B)
for argument in "$@"; do
  if [[ "$argument" == --help || "$argument" == -h ]]; then
    exec "${launcher[@]}" src/CDR.py "${common[@]}" "${architecture[@]}" "$@"
  fi
done
if (( processes > 1 )); then
  launcher+=( -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$processes" )
fi
exec "${launcher[@]}" src/CDR.py "${common[@]}" "${architecture[@]}" "$@"
