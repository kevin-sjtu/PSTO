from robertson import configuration

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from model import AssimuloSource, RobertsonSource
from train import MODELS, ROOT, Transport, build_model, channel_rel_l2, error_fields, load_frames
from utils.functions import enforce_boundaries


def dataset_bounds(path):
    with h5py.File(path, 'r') as f:
        data = f['data']
        lo = np.full(data.shape[1], np.inf)
        hi = -lo
        stride = data.chunks[0] if data.chunks else 32
        for start in range(0, len(data), stride):
            block = data[start:start + stride]
            lo = np.minimum(lo, block.min((0, 2, 3)))
            hi = np.maximum(hi, block.max((0, 2, 3)))
    return (
        torch.tensor(lo, dtype=torch.float32).reshape(-1, 1, 1),
        torch.tensor(hi, dtype=torch.float32).reshape(-1, 1, 1),
    )


def checkpoint_architecture(name, state):

    spectral = next((value for key, value in state.items() if key.endswith('weights1')), None)
    architecture = {'n_layers': 4}
    if spectral is not None:
        architecture['modes'] = int(spectral.shape[-1])
    if name == 'PSTO':
        architecture['latent_per_var'] = int(state['transport_branch.lift.weight'].shape[0])
    else:
        architecture['hidden_channels'] = int(state['lift.weight'].shape[0])
        prefix = 'core.encoders.' if name == 'Unet' else 'blocks.'
        indices = [int(key[len(prefix):].split('.')[0]) for key in state
                   if key.startswith(prefix) and key[len(prefix):].split('.')[0].isdigit()]
        if indices:
            architecture['n_layers'] = max(indices) + 1
    return architecture


@torch.no_grad()
def test_main(arguments=None):
    config = configuration()
    parser = argparse.ArgumentParser(description='HF Strang/Lie rollout with reaction splitting')
    parser.add_argument('--models', nargs='+', default=list(MODELS))
    parser.add_argument('--checkpoint_root', default=str(ROOT / 'checkpoints/models'))
    parser.add_argument(
        '--baseline_checkpoint_root',
        default=None,
        help='Optional root for baseline models; defaults to --checkpoint_root',
    )
    parser.add_argument('--checkpoint', default='checkpoint_best.pt', choices=['checkpoint_best.pt'],
                        help='Checkpoint filename under each model_* directory')
    parser.add_argument('--data_idx', type=int, default=None)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--dataset', choices=('lf', 'hf'), default='hf',
                        help='LF pure transport or HF transport coupled with chemistry')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--splitting', choices=('lie', 'strang'), default='strang')
    parser.add_argument('--source_backend', choices=('assimulo', 'gpu_sdirk2'), default='gpu_sdirk2')
    parser.add_argument('--save_dir', default=str(ROOT / 'analysis/figures'))
    args = parser.parse_args(arguments)

    torch.set_num_threads(1)
    device = torch.device(args.device)
    if args.data_idx is None:
        args.data_idx = 1001 if args.dataset == 'hf' else 501
    frames = load_frames(ROOT / config[args.dataset], args.data_idx - 1, args.steps + 1).to(device)
    training_frames = load_frames(ROOT / config['lf'], config['start'],
                                  config['train_pairs'] + 1)
    train_lo = training_frames.amin((0, 2, 3)).reshape(-1, 1, 1).to(device)
    train_hi = training_frames.amax((0, 2, 3)).reshape(-1, 1, 1).to(device)
    del training_frames
    report_lo, report_hi = dataset_bounds(ROOT / config[args.dataset])
    report_lo, report_hi = report_lo.to(device), report_hi.to(device)

    source_class = AssimuloSource if args.source_backend == 'assimulo' else RobertsonSource
    source_dt = config['dt'] / 2 if args.splitting == 'strang' else config['dt']
    chemistry = source_class(
        source_dt,
        density=config['density'],
        molecular_weight=config['molecular_weight'],
    ).to(device) if args.dataset == 'hf' else torch.nn.Identity().to(device)
    if args.dataset == 'lf':
        args.source_backend = 'none'
    predictions = {}
    for name in args.models:
        root = args.checkpoint_root
        if name != 'PSTO' and args.baseline_checkpoint_root:
            root = args.baseline_checkpoint_root
        path = Path(root) / f'model_{name}' / args.checkpoint
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        architecture = checkpoint_architecture(name, checkpoint['model'])
        model = build_model(name, config, train_lo, train_hi, **architecture).to(device)
        model.load_state_dict(checkpoint['model'])
        if args.dataset == 'lf':
            report_lo, report_hi = train_lo, train_hi
        operator = Transport(model.eval(), report_lo, report_hi).to(device)

        current = frames[:1].clone()
        outputs = []
        for step in range(1, args.steps + 1):
            boundary = current
            stage = 'reaction_pre' if args.splitting == 'strang' else 'transport'
            try:
                if args.splitting == 'strang':
                    current = chemistry(current)
                    if not torch.isfinite(current).all():
                        raise ValueError('nonfinite reaction output')
                    current = enforce_boundaries(current, boundary)
                stage = 'transport'
                species = operator(current)
                current = torch.cat((species, current[:, -2:]), dim=1)
                if not torch.isfinite(current).all():
                    raise ValueError('nonfinite transport output before chemistry')
                stage = 'reaction_post'
                current = chemistry(current)
                if not torch.isfinite(current).all():
                    raise ValueError('nonfinite reaction output')
                current = enforce_boundaries(current, boundary)
            except ValueError as exc:
                raise ValueError(f'{name} failed at step {step}, stage={stage}: {exc}') from exc
            if not torch.isfinite(current).all():
                raise ValueError(f'{name} produced nonfinite prediction at step {step}')
            outputs.append(current[:, :4].clone())
        predictions[name] = torch.cat(outputs, 0)
        del model, operator, outputs, current

    
    rows = []
    report_steps = sorted({step for step in (1, 5, 10, 20, 50, 100, 500, 1000, args.steps)
                           if step <= args.steps})
    for name in [*args.models, 'Initial']:
        for step in report_steps:
            if name != 'Initial' and (name not in predictions or step > len(predictions[name])):
                continue
            pred = frames[:1, :4] if name == 'Initial' else predictions[name][step - 1:step]
            target_frame = frames[step:step + 1, :4]
            physical_num = (pred.double()-target_frame.double()).square().sum((0,2,3)).sqrt()
            physical_den = target_frame.double().square().sum((0,2,3)).sqrt()
            physical_errors = physical_num/physical_den
            physical_errors = torch.where((physical_den==0)&(physical_num==0),0.,physical_errors)
            errors = channel_rel_l2(error_fields(pred, report_lo[:4], report_hi[:4]),
                                    error_fields(target_frame, report_lo[:4], report_hi[:4]))
            row = dict(
                model=name,
                step=step,
                rel_L2=physical_errors.mean().item(),
                normalized_rel_L2=errors.mean().item(),
                **{channel: physical_errors[i].item() for i, channel in enumerate(config['channels'][:4])},
                **{'normalized_'+channel: errors[i].item() for i, channel in enumerate(config['channels'][:4])},
            )
            rows.append(row)
            print(json.dumps(row), flush=True)

    target = Path(args.save_dir)
    target.mkdir(parents=True, exist_ok=True)
    np.savez(
        target / 'rollout.npz',
        input=frames[0].cpu().numpy(),
        ground_truth=frames[1:, :4].cpu().numpy(),
        data_min=report_lo[:4].cpu().numpy(),
        data_max=report_hi[:4].cpu().numpy(),
        channels=np.array(config['channels'][:4]),
        data_idx=args.data_idx,
        physics_version=config['physics_version'],
        source_backend=args.source_backend,
        splitting=args.splitting,
        metric='physical_channel_relative_L2',
        dataset=args.dataset,
        **{name: value.cpu().numpy() for name, value in predictions.items()},
    )
    with (target / 'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    test_main()
