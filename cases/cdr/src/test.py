from CDR import configuration

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from model import CDRSource
from train import MODELS, ROOT, Transport, build_model, channel_rel_l2, error_fields, load_frames


def dataset_bounds(path):
    with h5py.File(path, 'r') as stream:
        data = stream['data']
        lo = np.full(data.shape[1], np.inf)
        hi = -lo
        stride = data.chunks[0] if data.chunks else 32
        for start in range(0, len(data), stride):
            block = data[start:start + stride]
            lo = np.minimum(lo, block.min((0, 2, 3)))
            hi = np.maximum(hi, block.max((0, 2, 3)))
    return (torch.tensor(lo, dtype=torch.float32).reshape(-1, 1, 1),
            torch.tensor(hi, dtype=torch.float32).reshape(-1, 1, 1))


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('hf', 'lf'), default='hf')
    parser.add_argument('--models', nargs='+', default=list(MODELS))
    parser.add_argument('--data_idx', type=int, default=None)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--checkpoint_root', default=str(ROOT / 'checkpoints/models'))
    parser.add_argument('--save_dir', default=str(ROOT / 'analysis/figures'))
    args = parser.parse_args(arguments)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if args.data_idx is None:
        args.data_idx = 1 if args.dataset == 'hf' else 200
    frames = load_frames(ROOT / config[args.dataset], args.data_idx - 1, args.steps + 1).to(device)
    training_frames = load_frames(ROOT / config['lf'], config['start'], config['train_pairs'] + 1)
    train_lo = training_frames.amin((0, 2, 3)).reshape(-1, 1, 1).to(device)
    train_hi = training_frames.amax((0, 2, 3)).reshape(-1, 1, 1).to(device)
    del training_frames
    report_lo, report_hi = dataset_bounds(ROOT / config[args.dataset])
    report_lo, report_hi = report_lo.to(device), report_hi.to(device)
    chemistry = CDRSource(config['dt'] / 2).to(device) if args.dataset == 'hf' else torch.nn.Identity().to(device)

    predictions = {}
    for name in args.models:
        path = Path(args.checkpoint_root) / f'model_{name}' / 'checkpoint_best.pt'
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        architecture = checkpoint_architecture(name, checkpoint['model'])
        model = build_model(name, config, train_lo, train_hi, **architecture).to(device)
        model.load_state_dict(checkpoint['model'])
        if args.dataset == 'lf':
            report_lo, report_hi = train_lo, train_hi
        operator = Transport(model.eval(), config, report_lo, report_hi).to(device)
        current = frames[:1].clone()
        outputs = []
        for step in range(1, args.steps + 1):
            current = chemistry(current)
            current = operator(current)
            current = chemistry(current)
            if not torch.isfinite(current).all():
                raise ValueError(f'{name} produced nonfinite output at step {step}')
            outputs.append(current.clone())
        predictions[name] = torch.cat(outputs, 0)
        del model, operator, outputs, current

    rows = []
    report_steps = [step for step in (1, 5, 10, 20, 50, 100, args.steps) if step <= args.steps]
    for name in [*args.models, 'Initial']:
        for step in sorted(set(report_steps)):
            pred = frames[:1] if name == 'Initial' else predictions[name][step - 1:step]
            target = frames[step:step + 1]
            normalized = channel_rel_l2(error_fields(pred, report_lo, report_hi),
                                        error_fields(target, report_lo, report_hi))
            numerator = (pred.double() - target.double()).square().sum((0, 2, 3)).sqrt()
            denominator = target.double().square().sum((0, 2, 3)).sqrt()
            errors = torch.where((denominator == 0) & (numerator == 0), 0., numerator / denominator)
            row = dict(model=name, step=step, rel_L2=errors.mean().item(),
                       normalized_rel_L2=normalized.mean().item(),
                       **{channel: errors[i].item() for i, channel in enumerate(config['channels'])})
            rows.append(row)
            print(json.dumps(row), flush=True)

    target = Path(args.save_dir)
    target.mkdir(parents=True, exist_ok=True)
    np.savez(target / 'rollout.npz', input=frames[0].cpu().numpy(),
             ground_truth=frames[1:].cpu().numpy(), data_min=report_lo.cpu().numpy(),
             data_max=report_hi.cpu().numpy(), channels=np.array(config['channels']),
             data_idx=args.data_idx, dataset=args.dataset,
             **{name: value.cpu().numpy() for name, value in predictions.items()})
    with (target / 'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    test_main()
