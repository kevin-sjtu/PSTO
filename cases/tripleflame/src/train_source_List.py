"""Train the species-only reaction model; recover temperature from enthalpy.

The existing architecture is retained: eight scalar MLPs predict increments in
Box--Cox transformed active-species mass fractions. N2 is unchanged.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from DNN import NN_MLP

SPECIES = ('H2', 'H', 'O', 'O2', 'OH', 'H2O', 'HO2', 'H2O2', 'N2')
LAYERS = [11, 400, 400, 400, 400, 1]


def BCT(x, lam=0.1):
    return np.log(x) if lam == 0 else (np.power(x, lam) - 1) / lam


def load_reaction_data(path, max_samples=0):
    data = np.load(path)
    if data.ndim != 2 or data.shape[1] not in (40, 44):
        raise ValueError(f'Expected 40 or 44 columns, got {data.shape}')
    if not np.isfinite(data).all():
        raise ValueError('Nonfinite reaction training data')
    if data.shape[1] == 44:
        # Cantera order contains AR before N2; AR must be absent in this case.
        if np.any(data[:, [10, 32]] != 0):
            raise ValueError('The nine-species model requires zero AR mass fraction')
        data = np.delete(data, [10, 20, 32, 42], axis=1)
    if max_samples > 0 and len(data) > max_samples:
        data = data[np.linspace(0, len(data)-1, max_samples).round().astype(int)]
    if len(data) < 10:
        raise ValueError('At least ten reaction samples are required')
    if (data[:, :11] < 0).any() or (data[:, 22:31] < 0).any():
        raise ValueError('Negative temperature, pressure, or mass fraction')
    if not np.allclose(data[:, 1], 101325.0) or not np.allclose(data[:, 21], 101325.0):
        raise ValueError('The source model is configured for 101325 Pa')
    if not np.allclose(data[:, 10], data[:, 30], rtol=0, atol=1e-12):
        raise ValueError('N2 is not conserved in the reaction labels')
    return data


def transformed_pairs(data):
    inputs = data[:, :11].copy()
    inputs[:, 2:] = BCT(inputs[:, 2:])
    targets = BCT(data[:, 22:30]) - inputs[:, 2:-1]
    return inputs, targets


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-file', default='data/0d_data.npy')
    ap.add_argument('--model-dir', default='checkpoints/models/model_DNN')
    ap.add_argument('--dt', type=float, default=1e-6, help='Time interval of the supplied reaction labels')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=2048)
    ap.add_argument('--batch-size', type=int, default=2000)
    ap.add_argument('--max-epochs', type=int, default=2500)
    ap.add_argument('--patience', type=int, default=200)
    ap.add_argument('--validate-every', type=int, default=2)
    ap.add_argument('--learning-rate', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-5)
    ap.add_argument('--max-samples', type=int, default=0, help='0 uses all data; positive values are for debugging')
    return ap


def train(args):
    if (not np.isfinite(args.dt) or args.dt <= 0 or args.batch_size <= 0
            or args.max_epochs <= 0 or args.validate_every <= 0 or args.patience < 0):
        raise ValueError('Invalid time interval or training configuration')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    directory = Path(args.model_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f'Refusing to overwrite a nonempty model directory: {directory}')
    path = Path(args.data_file).resolve()
    data = load_reaction_data(path, args.max_samples)
    split = int(0.8 * len(data))
    config = dict(arguments=vars(args), data_file=str(path), dt=args.dt,
                  data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  species=list(SPECIES), layers=LAYERS, n_train=split,
                  n_validation=len(data)-split, split='first_80_percent_then_last_20_percent',
                  temperature_model=False, temperature_recovery='constant_enthalpy_YtoT',
                  pressure=101325.0, architecture='eight_scalar_MLPs',
                  preprocessing='BCT_0.1_increment_then_train_standardization')
    # Exclusive creation also protects against simultaneous submissions.
    with (directory / 'run_config.json').open('x') as stream:
        json.dump(config, stream, indent=2)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    x, y = transformed_pairs(data)
    stats = dict(data_in_mean=x[:split].mean(0), data_in_std=x[:split].std(0, ddof=1),
                 data_target_mean=y[:split].mean(0), data_target_std=y[:split].std(0, ddof=1))
    for key in ('data_in_std', 'data_target_std'):
        stats[key][stats[key] == 0] = 1
    x = torch.as_tensor((x-stats['data_in_mean'])/stats['data_in_std'], dtype=torch.float32)
    y = torch.as_tensor((y-stats['data_target_mean'])/stats['data_target_std'], dtype=torch.float32)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Nonfinite standardized reaction data')
    models = nn.ModuleList([NN_MLP(LAYERS) for _ in range(8)]).to(device)
    optimizer = torch.optim.Adam(models.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 2000], gamma=0.4)

    def predict(inputs):
        return torch.cat([model(inputs) for model in models], dim=1)

    def validate():
        models.eval()
        totals = np.zeros(2)
        with torch.no_grad():
            for first in range(split, len(x), args.batch_size):
                diff = predict(x[first:first+args.batch_size].to(device)) - y[first:first+args.batch_size].to(device)
                totals += [diff.abs().sum().item(), diff.square().sum().item()]
        return totals / ((len(x)-split)*8)

    best_loss, best_epoch = float('inf'), 0
    start = time.time()
    print(f'Y_ONLY dt={args.dt:g} train={split} validation={len(x)-split} output={directory}', flush=True)
    with (directory / 'train_val_loss.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch', 'train_L1', 'validation_L1', 'validation_MSE', 'learning_rate'])
        for epoch in range(1, args.max_epochs+1):
            models.train()
            permutation = np.random.permutation(split)
            total = 0.0
            for first in range(0, split, args.batch_size):
                indices = permutation[first:first+args.batch_size]
                prediction = predict(x[indices].to(device))
                loss = nn.functional.l1_loss(prediction, y[indices].to(device))
                if not torch.isfinite(loss):
                    raise ValueError(f'Nonfinite training loss at epoch {epoch}')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item()*len(indices)
            if epoch == 1 or epoch % args.validate_every == 0 or epoch == args.max_epochs:
                val_l1, val_mse = validate()
                if not np.isfinite([val_l1, val_mse]).all():
                    raise ValueError(f'Nonfinite validation loss at epoch {epoch}')
                writer.writerow([epoch, total/split, val_l1, val_mse, optimizer.param_groups[0]['lr']])
                stream.flush()
                print(f'Epoch {epoch}: train_L1={total/split:.6e} val_L1={val_l1:.6e} val_MSE={val_mse:.6e}', flush=True)
                if val_l1 < best_loss:
                    best_loss, best_epoch = float(val_l1), epoch
                    saved = dict(stats, dt=args.dt, epoch=epoch, validation_L1=best_loss,
                                 run_metadata=config, layers=LAYERS, species=list(SPECIES))
                    saved.update({f'net{i}': model.state_dict() for i, model in enumerate(models)})
                    temporary = directory / 'dnn_model_Y.pt.tmp'
                    torch.save(saved, temporary)
                    os.replace(temporary, directory / 'dnn_model_Y.pt')
                    print(f'SAVED dnn_model_Y.pt epoch={epoch}', flush=True)
                if args.patience and epoch-best_epoch >= args.patience:
                    print(f'EARLY_STOP epoch={epoch} best_epoch={best_epoch}', flush=True)
                    break
            scheduler.step()
    saved = torch.load(directory / 'dnn_model_Y.pt', map_location=device, weights_only=False)
    for i, model in enumerate(models):
        model.load_state_dict(saved[f'net{i}'], strict=True)
    loaded_l1, _ = validate()
    if not np.isclose(loaded_l1, best_loss, rtol=1e-5, atol=1e-8):
        raise ValueError('Reloaded model validation differs from the saved best result')
    report = dict(training_complete=True, best_epoch=best_epoch, last_epoch=epoch,
                  validation_L1=best_loss, reload_verified=True, dt=args.dt,
                  elapsed_seconds=time.time()-start)
    with (directory / 'training_complete.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print('TRAINING_COMPLETE ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    train(parser().parse_args())
