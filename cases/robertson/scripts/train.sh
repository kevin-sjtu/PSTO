#!/usr/bin/env bash
set -euo pipefail
CASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CASE_DIR"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo 'Usage: bash scripts/train.sh [PSTO|FNO|Unet|UFNO] [training CLI options]'
  echo 'NPROC_PER_NODE=4 reproduces the training process count; PYTHON selects Python.'
  echo 'N_ITERS defaults to 10000/NPROC_PER_NODE to preserve the number of training examples.'
  exit 0
fi
MODEL="${1:-PSTO}"
if (( $# )); then shift; fi
PROCESSES="${NPROC_PER_NODE:-4}"
if ! [[ "$PROCESSES" =~ ^[1-9][0-9]*$ ]] || (( 10000 % PROCESSES != 0 )); then
  echo 'NPROC_PER_NODE must be a positive divisor of 10000.' >&2; exit 2
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
case "$MODEL" in
  PSTO) architecture=(--latent_per_var 8 --modes 16) ;;
  FNO|UFNO) architecture=(--hidden_channels 20 --modes 16) ;;
  Unet) architecture=(--hidden_channels 20) ;;
  *) echo "Unknown model: $MODEL" >&2; exit 2 ;;
esac
common=(--model_type "$MODEL" --n_layers 4 --batch_size 16
  --grad_accum_steps 2 --num_workers 2 --n_iters "${N_ITERS:-$((10000 / PROCESSES))}"
  --learning_rate 3e-4 --weight_decay 1e-5 --warmup_iters 200
  --max_val_pairs 1000 --print_every 50 --save_every 100 --start_save_step 50
  --physics_weight 0.1 --roi_weight 0.1 --weighted_power 1.0
  --wavelet_weight 0.02 --wavelet_level 2 --mass_fraction_weight 1.0
  --nonneg_weight 1.0 --ckpt_dir "${TRAIN_OUTPUT_ROOT:-checkpoints/models}/model_${MODEL}")
launcher=("${PYTHON:-python}" -B)
for argument in "$@"; do
  if [[ "$argument" == --help || "$argument" == -h ]]; then
    exec "${launcher[@]}" src/robertson.py "${common[@]}" "${architecture[@]}" "$@"
  fi
done
if (( PROCESSES > 1 )); then
  launcher+=(-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$PROCESSES")
fi
exec "${launcher[@]}" src/robertson.py "${common[@]}" "${architecture[@]}" "$@"
