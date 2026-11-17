"""Portable entry points. All generated files stay in the selected case directory."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

CASE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('train', 'test', 'plot', 'source', 'constitutive'))
    parser.add_argument('--dt', choices=('1e-6', '1e-5'), required=True)
    parser.add_argument('--model', choices=('PSTO', 'FNO', 'Unet', 'UFNO'), default='PSTO')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--dry-run', action='store_true', help='Print the command without loading data or launching work')
    args, extra = parser.parse_known_args()
    coarse = args.dt == '1e-5'
    root = os.environ.get('TRAIN_OUTPUT_ROOT', 'checkpoints/models_1e-5' if coarse else 'checkpoints/models')
    data = os.environ.get('TRAIN_DATA_FILE', 'data/H2_tripleflow_dt1e-5_pairs.h5' if coarse else 'data/H2_tripleflow.h5')
    output = os.environ.get('TEST_SAVE_DIR', 'analysis/results_1e-5/lie_10' if coarse else 'analysis/figures')
    coefficient = os.environ.get('COEFFICIENT_CHECKPOINT', 'checkpoints/models/model_coeff/constitutive_DNN_BCT.pt')
    source = os.environ.get('SOURCE_CHECKPOINT', root + '/model_DNN/dnn_model_Y.pt')
    mechanism = os.environ.get('MECHANISM', 'mech/h2o2.yaml')
    hf = os.environ.get('HF_DATA_FILE', 'data/H2_tripleflame.h5')
    start = os.environ.get('DATA_IDX', '2001')
    python = sys.executable
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
    if args.device == 'cpu':
        env['ACCELERATE_USE_CPU'] = 'true'
    if args.action == 'train':
        nproc = int(os.environ.get('TRAIN_NPROC', '1'))
        batch = int(os.environ.get('TRAIN_BATCH_SIZE', '4'))
        if nproc <= 0 or batch <= 0 or 32 % (nproc * batch):
            parser.error('TRAIN_NPROC * TRAIN_BATCH_SIZE must divide effective batch 32')
        accum = int(os.environ.get('TRAIN_GRAD_ACCUM', str(32 // (nproc * batch))))
        iterations = os.environ.get('TRAIN_ITERS', str(160000 // (nproc * batch)))
        architecture = (['--latent_per_var', '8', '--modes', '16'] if args.model == 'PSTO'
                        else ['--hidden_channels', '20'] + ([] if args.model == 'Unet' else ['--modes', '16']))
        command = [python, '-B', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                   f'--nproc_per_node={nproc}', 'src/tripleflame.py',
                   '--model_type', args.model, '--n_layers', '4',
                   '--batch_size', str(batch), '--grad_accum_steps', str(accum),
                   '--n_iters', iterations, '--num_workers', os.environ.get('TRAIN_WORKERS', '2'),
                   '--learning_rate', '3e-4', '--weight_decay', '1e-5',
                   '--warmup_iters', os.environ.get('TRAIN_WARMUP', str(100 * accum)),
                   '--scheduler_units', 'updates',
                   '--max_pairs', os.environ.get('TRAIN_MAX_PAIRS', '0'),
                   '--max_val_pairs', os.environ.get('TRAIN_MAX_VAL_PAIRS', '256'),
                   '--print_every', os.environ.get('TRAIN_PRINT_EVERY', str(25 * accum)),
                   '--save_every', os.environ.get('TRAIN_SAVE_EVERY', str(50 * accum)),
                   '--start_save_step', os.environ.get('TRAIN_START_SAVE', str(25 * accum)),
                   '--physics_weight', '.1', '--roi_weight', '.1', '--weighted_power', '1',
                   '--wavelet_weight', '.02', '--wavelet_level', '2',
                   '--mass_fraction_weight', '1', '--nonneg_weight', '1',
                   '--coefficient_backend', 'dnn', '--coefficient_checkpoint', coefficient,
                   '--mechanism', mechanism, '--pressure', '101325',
                   '--lf_file', data, '--dt', args.dt,
                   '--cache_dir', os.environ.get('TRAIN_CACHE_DIR', root + '/data_cache'),
                   '--ckpt_dir', root + '/model_' + args.model] + architecture
    elif args.action == 'test' and coarse:
        command = [python, '-B', 'src/test_1e-5.py', '--models', 'PSTO', 'FNO', 'Unet', 'UFNO',
                   '--steps', os.environ.get('TEST_STEPS', '10'), '--data-idx', start,
                   '--dt', '1e-5', '--reference-dt', '1e-6', '--hf', hf, '--ckpt-dir', root,
                   '--source-checkpoint', source, '--coefficient-checkpoint', coefficient,
                   '--mechanism', mechanism, '--save-dir', output, '--device', args.device]
    elif args.action == 'test':
        command = [python, '-B', 'src/test.py', '--models', 'PSTO', 'FNO', 'Unet', 'UFNO',
                   '--steps', os.environ.get('TEST_STEPS', '100'), '--print_steps',
                   *[str(step) for step in (1, 5, 10, 20, 50, 100)
                     if step <= int(os.environ.get('TEST_STEPS', '100'))],
                   '--coefficient_backend', 'dnn', '--coefficient_checkpoint', coefficient,
                   '--mechanism', mechanism, '--pressure', '101325', '--modes', '16',
                   '--n_layers', '4', '--hidden_channels', '20', '--data_idx', start, '--hf', hf,
                   '--ckpt_dir', root, '--source_checkpoint', source, '--save_dir', output, '--save_name', f'dnn_{start}_all',
                   '--device', args.device]
    elif args.action == 'plot':
        if coarse:
            command = [python, '-B', 'analysis/plot_coarse.py', '--directory', output,
                       '--steps', os.environ.get('TEST_STEPS', '10'),
                       '--step', os.environ.get('PLOT_STEP', '10')]
        else:
            command = [python, '-B', 'analysis/plot.py', '--directory', output,
                       '--data-idx', start, '--steps', os.environ.get('TEST_STEPS', '100'),
                       '--step', os.environ.get('PLOT_STEP', '100')]
    elif args.action == 'source':
        command = [python, '-B', 'src/train_source_List.py', '--data-file',
                   os.environ.get('SOURCE_DATA_FILE', 'data/0d_data_1e-5.npy' if coarse else 'data/0d_data.npy'),
                   '--model-dir', root + '/model_DNN', '--dt', args.dt,
                   '--device', 'cpu' if args.device == 'cpu' else 'cuda:0',
                   '--max-epochs', os.environ.get('SOURCE_MAX_EPOCHS', '2500'),
                   '--batch-size', os.environ.get('SOURCE_BATCH_SIZE', '2000'),
                   '--patience', os.environ.get('SOURCE_PATIENCE', '200')]
    else:
        command = [python, '-B', 'src/train_constitutive.py',
                   '--source', os.environ.get('CONSTITUTIVE_SOURCE', 'data/0d_dataset_cfd.npy'),
                   '--dataset_h5', os.environ.get('CONSTITUTIVE_DATASET', 'data/constitutive_data.h5'),
                   '--output', coefficient, '--device', 'cpu' if args.device == 'cpu' else 'cuda',
                   '--epochs', '2500', '--batch_size', '2000']
    command.extend(extra)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=CASE, env=env, check=True)
        if args.action == 'train' and coarse:
            subprocess.run([python, '-B', 'src/verify_training_run.py',
                            root + '/model_' + args.model], cwd=CASE, env=env, check=True)


if __name__ == '__main__':
    main()
