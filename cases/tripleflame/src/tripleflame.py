import os
import json
from pathlib import Path
import sys
import h5py
import torch
import time
import argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from PSTO import PSTO
from Unet import Unet
from UFNO import UFNO
from FNO import FNO
from train import trainer
from accelerate import Accelerator
from paired_data import pair_metadata, paired_datasets
from utils.functions import get_parameter_net, EfficientDataset, compute_lewis_numbers, channel_bounds

def configuration():
    return {
        'dt': 1e-6,
        'dx': 1e-4,
        'lf': 'data/H2_tripleflow.h5',
        'hf': 'data/H2_tripleflame.h5',
        'load_channels': [9, 0, 4, 5, 1, 3, 2, 6, 7, 8, 11, 12],
        'channels': ['T', 'H2', 'H', 'O', 'O2', 'OH', 'H2O', 'HO2', 'H2O2', 'N2', 'u', 'v'],
    }


def train_main():
    parser = argparse.ArgumentParser(description='Neural Operator Training')
    parser.add_argument('--model_type', type=str, default='PSTO',
                        choices=['PSTO', 'FNO', 'Unet', 'UFNO'])
    parser.add_argument('--latent_per_var', type=int, default=8)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--modes', type=int, default=16)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--max_pairs', type=int, default=0,
                        help='0: all training data; paired datasets subsample N training pairs for debugging.')
    parser.add_argument('--max_val_pairs', type=int, default=0,
                        help='0: full validation; >0: fixed temporal subsample for frequent checks.')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--warmup_iters', type=int, default=200)
    parser.add_argument('--scheduler_units', choices=('updates', 'microsteps'), default='updates',
                        help='microsteps reproduces the schedule of the existing baseline checkpoints.')
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--start_save_step', type=int, default=1000)
    parser.add_argument('--grad_accum_steps', type=int, default=2)
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--reset_optimizer', action='store_true')
    parser.add_argument('--roi_weight', type=float, default=0.1)
    parser.add_argument('--weighted_power', type=float, default=1.0)
    parser.add_argument('--mass_fraction_weight', type=float, default=1.0)
    parser.add_argument('--nonneg_weight', type=float, default=0.1)
    parser.add_argument('--wavelet_weight', type=float, default=0.02)
    parser.add_argument('--wavelet_level', type=int, default=2)
    parser.add_argument('--physics_weight', type=float, default=0.0)
    parser.add_argument('--ckpt_dir', type=str, default=None, help='output dir for this run; defaults to checkpoints/models/model_<type>. Set a new dir to avoid touching existing models.')
    parser.add_argument('--lf_file', default=None, help='Original frames or explicit input-target HDF5 file')
    parser.add_argument('--dt', type=float, default=None, help='Physical prediction interval; must match paired metadata')
    parser.add_argument('--cache_dir', default=None, help='Writable cache directory for either frames or explicit pairs')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader worker processes (0=main process only)')
    parser.add_argument('--coefficient_checkpoint', type=str,
                        default='checkpoints/models/model_coeff/constitutive_DNN_BCT.pt')
    parser.add_argument('--coefficient_backend', choices=('cantera', 'dnn'), default='dnn')
    parser.add_argument('--mechanism', default='mech/h2o2.yaml')
    parser.add_argument('--pressure', type=float, default=101325.0)
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir or f'checkpoints/models/model_{args.model_type}'
    torch.manual_seed(88)

    lewis_dict = compute_lewis_numbers(args.mechanism)
    scalar_name = ['H2', 'H', 'O', 'O2', 'OH', 'H2O', 'HO2', 'H2O2', 'N2']
    lewis_vals = [lewis_dict[name] for name in scalar_name]


    def build_model(model_type, in_ch, data_max=None, data_min=None):
        common = dict(dt=dt, dx=dx)
        if model_type == 'PSTO':
            model = PSTO(
                in_ch,
                latent_per_var=args.latent_per_var,
                fno_modes=args.modes,
                n_layers=args.n_layers,
                data_max=data_max,
                data_min=data_min,
                lewis=lewis_vals,
                coefficient_checkpoint=args.coefficient_checkpoint,
                coefficient_backend=args.coefficient_backend,
                mechanism=args.mechanism,
                pressure=args.pressure,
                **common,
            )
            return model
        elif model_type == 'FNO':
            return FNO(
                in_ch,
                hidden_channels=args.hidden_channels,
                modes=args.modes,
                n_layers=args.n_layers,
                dt=dt,
            )
        elif model_type == 'Unet':
            return Unet(in_ch, hidden_channels=args.hidden_channels, n_layers=args.n_layers, dt=dt)
        elif model_type == 'UFNO':
            return UFNO(in_ch, hidden_channels=args.hidden_channels, modes=args.modes,
                        n_layers=args.n_layers, dt=dt)
        raise ValueError(f"Unknown model_type: {model_type}")


    config = configuration()
    lf_file = args.lf_file or config['lf']
    with h5py.File(lf_file, 'r') as data:
        paired = 'inputs' in data and 'targets' in data
    expected_dt = pair_metadata(lf_file)['dt'] if paired else config['dt']
    dt, dx = (expected_dt if args.dt is None else args.dt), config['dx']
    if not np.isfinite(dt) or dt <= 0 or (paired and not np.isclose(dt, expected_dt, rtol=1e-12, atol=0)):
        raise ValueError('Training dt must match the paired dataset prediction interval')
    load_channels = config['load_channels']
    input_channels = len(load_channels)

    accelerator = Accelerator()
    rank = accelerator.process_index
    # A fresh run owns a new directory. Existing model files are never reused implicitly.
    error = [None]
    if accelerator.is_main_process:
        try:
            if args.restart:
                if not (Path(ckpt_dir) / 'checkpoint_best.pt').is_file():
                    raise FileNotFoundError('Restart requires an existing checkpoint')
            else:
                Path(ckpt_dir).mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            error[0] = f'Refusing to overwrite an existing run: {ckpt_dir}: {exc}'
    if torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])
    accelerator.wait_for_everyone()

    if paired:
        cache_dir = args.cache_dir or str(Path(ckpt_dir).parent / 'data_cache')
        train_dataset, val_dataset, lf_data_min, lf_data_max, run_metadata = paired_datasets(
            lf_file, cache_dir, load_channels, args, accelerator)
        accelerator.print(f'PAIRED_DATA dt={dt:g} lag={run_metadata["lag_frames"]} '
                          f'train={len(train_dataset)} validation={len(val_dataset)} '
                          f'checkpoint_dir={Path(ckpt_dir).resolve()}')
    else:
        cache_dir = args.cache_dir or str(Path(ckpt_dir).parent / 'data_cache')
        os.makedirs(cache_dir, exist_ok=True)

        with h5py.File(lf_file, "r") as f:
            lf_total_frames = int(f["data"].shape[0])
        n_frames_use = lf_total_frames if args.max_pairs <= 0 else min(lf_total_frames, args.max_pairs + 1)
        n_pairs = n_frames_use - 1
        stats_path = os.path.join(cache_dir, "maxmin_stats.npz")
        cache_file = os.path.join(cache_dir, f"data_all_{n_frames_use}.pt")


        if accelerator.is_main_process:
            rebuild_cache = not os.path.exists(cache_file)
            if not rebuild_cache:
                try:
                    cached = torch.load(cache_file, map_location="cpu", weights_only=True, mmap=True)
                    rebuild_cache = cached["lf"].shape[0] != n_frames_use
                except (OSError, KeyError):
                    rebuild_cache = True
            if rebuild_cache:
                print(f"[Rank {rank}] Loading {n_frames_use}/{lf_total_frames} frames from HDF5 "
                      f"({n_pairs} pairs)...")
                with h5py.File(lf_file, "r") as f:
                    lf_data = torch.from_numpy(f["data"][:n_frames_use]).float()
                torch.save({"lf": lf_data}, cache_file)
                print(f"[Rank {rank}] Saved cache to {cache_file}")
            else:
                print(f"[Rank {rank}] Cache exists: {cache_file}")
        accelerator.wait_for_everyone()

        print(f"[Rank {rank}] Loading from cache...")
        cached = torch.load(cache_file, map_location="cpu", weights_only=True, mmap=True)
        lf_data = cached["lf"]
        if lf_data.shape[0] != n_frames_use:
            raise ValueError(f"Cache has {lf_data.shape[0]} frames, expected {n_frames_use}")
        print(f"[Rank {rank}] Cache loaded: {lf_data.shape[0]} frames, {lf_data.shape[0] - 1} pairs.")

        if accelerator.is_main_process:
            stats_dict = {}
            if os.path.exists(stats_path):
                with np.load(stats_path) as saved:
                    stats_dict = {key: saved[key] for key in saved.files}
            stamp = os.stat(lf_file).st_mtime_ns
            valid = (stats_dict.get('lf_bounds_scope', '') == 'full_dataset'
                     and int(stats_dict.get('lf_bounds_frame_count', 0)) == lf_total_frames
                     and int(stats_dict.get('lf_bounds_source_mtime_ns', -1)) == stamp
                     and np.array_equal(stats_dict.get('lf_bounds_channels', []), load_channels))
            if not valid:
                print(f'[stats] Scanning all {lf_total_frames} LF frames for channel bounds', flush=True)
                with h5py.File(lf_file, 'r') as f:
                    lower, upper = channel_bounds(f['data'], load_channels)
                stats_dict.update(lf_data_min=lower.numpy(), lf_data_max=upper.numpy(),
                                  lf_bounds_scope=np.array('full_dataset'),
                                  lf_bounds_frame_count=np.array(lf_total_frames),
                                  lf_bounds_source_mtime_ns=np.array(stamp),
                                  lf_bounds_channels=np.array(load_channels))
                temporary = f'{stats_path}.{os.getpid()}.tmp'
                with open(temporary, 'wb') as stream:
                    np.savez(stream, **stats_dict)
                os.replace(temporary, stats_path)
        accelerator.wait_for_everyone()
        with np.load(stats_path) as saved:
            lf_data_min = torch.from_numpy(saved['lf_data_min']).float()
            lf_data_max = torch.from_numpy(saved['lf_data_max']).float()
        args.clipping_stats_scope = 'full_lf_dataset'
        args.clipping_stats_frames = lf_total_frames

        n_pairs = lf_data.shape[0] - 1
        train_size = max(1, int(0.8 * n_pairs))
        val_size = max(1, n_pairs - train_size)

        train_dataset = EfficientDataset(
            lf_data=lf_data[:train_size + 1], hfx_data=None, hfy_data=None,
            n_lf=train_size, n_hf=0, chs=load_channels,
            stats=(lf_data_max, lf_data_min) if args.coefficient_backend == 'dnn'
                  else (lf_data_max, lf_data_min, None, None),
            pressure_reference=args.pressure if args.model_type == 'PSTO' else None,
            normalize_state=args.model_type != 'PSTO',
        )
        val_dataset = EfficientDataset(
            lf_data=lf_data[train_size:], hfx_data=None, hfy_data=None,
            n_lf=val_size, n_hf=0, chs=load_channels,
            stats=(lf_data_max, lf_data_min) if args.coefficient_backend == 'dnn'
                  else (lf_data_max, lf_data_min, None, None),
            pressure_reference=args.pressure if args.model_type == 'PSTO' else None,
            normalize_state=args.model_type != 'PSTO',
        )
        run_metadata = dict(data_file=str(Path(lf_file).resolve()), prediction_dt=dt,
                            lag_frames=1, train_pairs=len(train_dataset),
                            validation_pairs=len(val_dataset))

    if args.max_val_pairs > 0 and len(val_dataset) > args.max_val_pairs:
        indices = torch.linspace(0, len(val_dataset) - 1,
                                 args.max_val_pairs).round().long().tolist()
        val_dataset = torch.utils.data.Subset(val_dataset, indices)
    run_metadata.update(model_type=args.model_type, dt=dt, dx=dx,
                        auxiliary_velocity_time_level='input',
                        checkpoint_dir=str(Path(ckpt_dir).resolve()),
                        validation_pairs_used=len(val_dataset),
                        world_size=accelerator.num_processes,
                        arguments=vars(args))
    if accelerator.is_main_process and not args.restart:
        with (Path(ckpt_dir) / 'run_config.json').open('x') as stream:
            json.dump(run_metadata, stream, indent=2)
            stream.write('\n')
    accelerator.print(f'Train {len(train_dataset)} | Validation {len(val_dataset)} | dt={dt:g}')

    dl_kw = dict(num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        dl_kw['persistent_workers'] = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(88), **dl_kw)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **dl_kw)

    model = build_model(
        args.model_type, input_channels, data_max=lf_data_max, data_min=lf_data_min
    ).to(accelerator.device)
    if accelerator.is_main_process:
        get_parameter_net(model)

    start_time = time.time()

    _data_max, _data_min = lf_data_max, lf_data_min
    _val_data_max, _val_data_min = lf_data_max, lf_data_min

    train_obj = trainer(
        model, dt=dt, dx=dx,
        data_tuple=(train_loader, val_loader),
        learning_rate=args.learning_rate,
        n_iters=args.n_iters,
        print_every=args.print_every,
        save_every=args.save_every,
        start_save_step=args.start_save_step,
        grad_accum_steps=args.grad_accum_steps,
        warmup_iters=args.warmup_iters,
        weight_decay=args.weight_decay,
        accelerator=accelerator,
        data_max=_data_max,
        data_min=_data_min,
        val_data_max=_val_data_max,
        val_data_min=_val_data_min,
        roi_weight=args.roi_weight,
        weighted_power=args.weighted_power,
        wavelet_weight=args.wavelet_weight,
        wavelet_level=args.wavelet_level,
        mass_fraction_weight=args.mass_fraction_weight,
        nonneg_weight=args.nonneg_weight,
        physics_weight=args.physics_weight,
        scheduler_units=args.scheduler_units,
        lewis=lewis_vals,
        restart=args.restart,
        reset_optimizer=args.reset_optimizer,
        ckpt_dir=ckpt_dir,
        coefficient_checkpoint=args.coefficient_checkpoint,
        mechanism=args.mechanism, pressure=args.pressure, run_metadata=run_metadata,
    )

    train_obj.train()
    peak = torch.tensor(torch.cuda.max_memory_allocated(accelerator.device)/2**30
                        if accelerator.device.type == 'cuda' else 0., device=accelerator.device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
    accelerator.print(f'TRAIN_COMPLETE model={args.model_type} elapsed={time.time()-start_time:.2f}s peak_GPU={peak.item():.2f}GiB')
    accelerator.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train_main()
