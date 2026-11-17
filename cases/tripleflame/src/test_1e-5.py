"""Ten-microsecond Lie rollout with a matching Y-only reaction checkpoint.

Published checkpoints use verified inference metadata; independent local runs
retain their training-completion checks. Targets use every ten original frames. As in test.py, each sample's normalization/clipping bounds include its
initial frame and every original reference frame in the same physical window.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from test import CHANNEL_NAMES, _safe_torch_load, resolve_path
from src.DNN import NN_MLP
from src.model import source_Y, transport, build_backbone
from utils.functions import STATE_SPECIES, channel_bounds, load_transport_state_dict
from paired_data import CHANNELS
from inference_bundle import (find_entry, validate_physics, validate_constitutive,
                              channel_bounds as bundle_channel_bounds, require_close)


def frame_indices(start_1based, steps, dt, reference_dt, total):
    if (steps <= 0 or start_1based <= 0 or not np.isfinite([dt, reference_dt]).all()
            or dt <= 0 or reference_dt <= 0):
        raise ValueError('Positive finite intervals, steps, and 1-based initial indices are required')
    ratio = dt/reference_dt
    stride = int(round(ratio))
    if stride < 1 or not np.isclose(ratio, stride, rtol=1e-10, atol=0):
        raise ValueError('Prediction dt must be an integer multiple of reference dt')
    indices = start_1based-1 + np.arange(steps+1)*stride
    if indices[-1] >= total:
        raise ValueError(f'Need zero-based frame {indices[-1]}, but only {total} frames exist')
    return indices


def load_reference_window(data, indices):
    """Match test.py bounds, including frames between the coarse targets.

    References are used for evaluation and fixed channel bounds, as in the
    original retrospective test; they never replace autoregressive states.
    Chunked extrema are equivalent to rollout_channel_bounds on the full window.
    """
    block = torch.from_numpy(data[indices][:, CHANNELS]).float()
    lower, upper = channel_bounds(data, CHANNELS, int(indices[0]), int(indices[-1])+1)
    if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
        raise ValueError('Nonfinite reference-window bounds')
    return block[0], block[1:], lower, upper


def _load_training_source(path, dt, pressure, mechanism, device):
    saved = _safe_torch_load(str(path), device)
    if not np.isclose(saved.get('dt', np.nan), dt, rtol=1e-6, atol=0):
        raise ValueError('Reaction checkpoint dt does not match rollout dt')
    completion_path = path.parent/'training_complete.json'
    if not completion_path.is_file():
        raise ValueError('Reaction training is not complete; a checkpoint alone is insufficient')
    completion = json.loads(completion_path.read_text())
    if (not completion.get('training_complete') or not completion.get('reload_verified')
            or completion.get('best_epoch') != saved.get('epoch')
            or not np.isclose(completion.get('dt', np.nan), dt, rtol=1e-6, atol=0)):
        raise ValueError('Reaction completion record does not match the verified checkpoint')
    meta = saved['run_metadata']
    if meta.get('temperature_model') is not False or meta.get('species') != list(STATE_SPECIES):
        raise ValueError('Expected the species-only checkpoint in the physical channel order')
    if not np.isclose(meta['pressure'], pressure):
        raise ValueError('Reaction training pressure differs from rollout pressure')
    models = [NN_MLP(saved['layers']) for _ in range(8)]
    for i, model in enumerate(models):
        model.load_state_dict(saved[f'net{i}'], strict=True)
    source = source_Y(models, p=pressure, mech_file=mechanism)
    source.load_stats({key: torch.as_tensor(saved[key], dtype=torch.float32).view(1, -1, 1, 1)
                       for key in ('data_in_mean', 'data_in_std', 'data_target_mean', 'data_target_std')})
    return source.requires_grad_(False).eval().to(device)


def _load_training_transport(directory, name, args, device):
    """Load trained weights and stats; evaluate applies the original test bounds."""
    config = json.loads((directory/'run_config.json').read_text())
    completion_path = directory/'reload_verification.json'
    history_path = directory/'train_val_loss.csv'
    if not completion_path.is_file() or not history_path.is_file():
        raise ValueError(f'{name}: transport training has not completed and passed reload verification')
    completion = json.loads(completion_path.read_text())
    history = np.loadtxt(history_path, delimiter=',', skiprows=1, ndmin=2)
    if (not completion.get('reload_and_forward_passed') or not history.size
            or int(history[-1, 0]) != config['arguments']['n_iters']):
        raise ValueError(f'{name}: transport training is incomplete')
    path = directory/'checkpoint_best.pt'
    saved = _safe_torch_load(str(path), device)
    if completion.get('checkpoint_step') != saved.get('step'):
        raise ValueError(f'{name}: completion record and checkpoint differ')
    if config['model_type'] != name:
        raise ValueError('Transport model name and configuration differ')
    for value in (config['dt'], saved.get('dt', np.nan), saved['model']['time_step'].item()):
        if not np.isclose(value, args.dt, rtol=1e-6, atol=0):
            raise ValueError(f'{name}: transport checkpoint dt does not match rollout dt')
    cache = json.loads(Path(config['frame_cache']).with_name('metadata.json').read_text())
    if (cache['signature']['channels'] != CHANNELS
            or not np.isclose(cache['dt'], args.dt, rtol=1e-6, atol=0)):
        raise ValueError('Transport normalization metadata does not match the model')
    lower = torch.tensor(cache['lower'], device=device).view(-1, 1, 1)
    upper = torch.tensor(cache['upper'], device=device).view(-1, 1, 1)
    options = dict(config['arguments'])
    options.update(dt=args.dt, dx=config['dx'], pressure=args.pressure,
                   mechanism=args.mechanism, coefficient_checkpoint=args.coefficient_checkpoint)
    backbone = build_backbone(name, len(CHANNELS), SimpleNamespace(**options),
                              data_min=lower, data_max=upper, state=saved['model'])
    if name == 'PSTO':
        load_transport_state_dict(backbone, saved['model'], strict=True)
    else:
        backbone.load_state_dict(saved['model'], strict=True)
    model = transport(backbone).requires_grad_(False).eval().to(device)
    model.load_stats({'data_min': lower, 'data_max': upper})
    report = dict(path=str(path), checkpoint_step=int(saved['step']),
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  training_bounds=str(Path(config['frame_cache']).with_name('metadata.json')))
    return model, report



def load_source(path, dt, pressure, mechanism, device):
    """Load either a hashed published source model or a verified local run."""
    bundle = find_entry(path, 'reaction')
    if bundle is None:
        return _load_training_source(path, dt, pressure, mechanism, device)
    validate_physics(bundle, dt, pressure, mechanism, CHANNEL_NAMES, CHANNELS)
    entry = bundle.entry
    if (entry.get('species') != list(STATE_SPECIES)
            or entry.get('temperature_model') is not False
            or entry.get('temperature_recovery') != 'constant_enthalpy_YtoT'):
        raise ValueError('Inference bundle reaction convention mismatch')
    require_close(entry['pressure'], pressure, 'reaction pressure')
    saved = _safe_torch_load(str(path), device)
    require_close(saved.get('dt', np.nan), dt, 'reaction checkpoint interval')
    meta = saved['run_metadata']
    if (meta.get('species') != entry['species'] or meta.get('temperature_model') is not False
            or saved['layers'] != entry['layers']):
        raise ValueError('Reaction checkpoint and inference bundle metadata differ')
    require_close(meta['pressure'], pressure, 'checkpoint reaction pressure')
    models = [NN_MLP(saved['layers']) for _ in range(8)]
    for i, model in enumerate(models):
        model.load_state_dict(saved[f'net{i}'], strict=True)
    source = source_Y(models, p=pressure, mech_file=mechanism)
    source.load_stats({key: torch.as_tensor(saved[key], dtype=torch.float32).view(1, -1, 1, 1)
                       for key in ('data_in_mean', 'data_in_std', 'data_target_mean', 'data_target_std')})
    return source.requires_grad_(False).eval().to(device)


def load_transport(directory, name, args, device):
    """Published bundles need weights/configuration, not training logs or caches."""
    path = Path(directory) / 'checkpoint_best.pt'
    bundle = find_entry(path, 'transport')
    if bundle is None:
        return _load_training_transport(directory, name, args, device)
    entry = bundle.entry
    if entry.get('model_type') != name:
        raise ValueError('Transport model name and inference bundle differ')
    dx = getattr(args, 'dx', 1e-4)
    validate_physics(bundle, args.dt, args.pressure, args.mechanism,
                     CHANNEL_NAMES, CHANNELS, dx=dx)
    validate_constitutive(bundle, args.coefficient_checkpoint)
    saved = _safe_torch_load(str(path), device)
    for value in (saved.get('dt', np.nan), saved['model']['time_step'].item()):
        require_close(value, args.dt, 'transport checkpoint interval')
    if 'grid_step' in saved['model']:
        require_close(saved['model']['grid_step'].item(), dx, 'checkpoint grid spacing')
    if saved.get('step') != entry['checkpoint_step']:
        raise ValueError('Checkpoint step and inference bundle differ')
    lower, upper = bundle_channel_bounds(bundle, len(CHANNELS))
    lower = torch.from_numpy(lower).to(device)
    upper = torch.from_numpy(upper).to(device)
    architecture = entry['architecture']
    required = ('hidden_channels', 'latent_per_var', 'n_layers', 'modes')
    if any(not isinstance(architecture.get(key), int) or architecture[key] <= 0 for key in required):
        raise ValueError('Inference bundle has invalid model architecture')
    options = {key: architecture[key] for key in required}
    options.update(dt=args.dt, dx=dx, pressure=args.pressure, mechanism=args.mechanism,
                   coefficient_backend='dnn', coefficient_checkpoint=args.coefficient_checkpoint)
    backbone = build_backbone(name, len(CHANNELS), SimpleNamespace(**options),
                              data_min=lower, data_max=upper)
    if name == 'PSTO':
        load_transport_state_dict(backbone, saved['model'], strict=True)
    else:
        backbone.load_state_dict(saved['model'], strict=True)
    model = transport(backbone).requires_grad_(False).eval().to(device)
    model.load_stats({'data_min': lower, 'data_max': upper})
    report = dict(path=str(path), checkpoint_step=int(saved['step']),
                  sha256=entry['sha256'], inference_bundle=str(bundle.manifest_path),
                  training_bounds=f'{bundle.manifest_path}#transport/{bundle.key}/normalization')
    return model, report

def assess_input_range(initial, lower, upper):
    """Expose an inference distribution mismatch without changing trained weights."""
    lower, upper = lower.to(initial), upper.to(initial)
    outside = (initial < lower) | (initial > upper)
    normalized = 2*(initial-lower)/(upper-lower).clamp_min(1e-30)-1
    return dict(any_outside_training_range=bool(outside.any()),
                fraction_outside_per_channel=outside.float().mean((0, 2, 3)).cpu().tolist(),
                max_abs_normalized_per_channel=normalized.abs().amax((0, 2, 3)).cpu().tolist())


def relative_errors(prediction, reference):
    # One norm per physical channel, over samples and spatial dimensions.
    axes = (0, 2, 3)
    numerator = (prediction.double()-reference.double()).square().sum(axes).sqrt()
    denominator = reference.double().square().sum(axes).sqrt()
    return torch.where((numerator == 0) & (denominator == 0), 0.0, numerator/denominator)


def evaluate(args):
    torch.manual_seed(args.seed)
    if not np.isclose(args.dt, 1e-5, rtol=1e-12, atol=0):
        raise ValueError('This entry point is for dt=1e-5 s')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device('cuda' if args.device == 'cuda' or (
        args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    output = Path(args.save_dir)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing rollout directory: {output}')
    references, inputs, reference_indices, window_bounds = [], [], [], []
    with h5py.File(args.hf, 'r') as stream:
        data = stream['data']
        for start in args.data_idx:
            indices = frame_indices(start, args.steps, args.dt, args.reference_dt, len(data))
            sample, targets, lower, upper = load_reference_window(data, indices)
            inputs.append(sample)
            references.append(targets)
            reference_indices.append(indices.tolist())
            window_bounds.append((lower, upper))
    initial, truth = torch.stack(inputs), torch.stack(references)
    if not torch.isfinite(initial).all() or not torch.isfinite(truth).all():
        raise ValueError('Nonfinite reference data')
    source = load_source(Path(args.source_checkpoint), args.dt, args.pressure, args.mechanism, device)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(dt=args.dt, reference_dt=args.reference_dt, steps=args.steps,
                  horizon_seconds=args.dt*args.steps, data_idx_1based=args.data_idx,
                  reference_indices_0based=reference_indices, reference_file=args.hf,
                  source_checkpoint=args.source_checkpoint,
                  source_sha256=hashlib.sha256(Path(args.source_checkpoint).read_bytes()).hexdigest(),
                  splitting='transport_then_reaction', temperature_recovery='constant_enthalpy_YtoT',
                  normalization='original_test_full_reference_window',
                  evaluation_bounds=dict(
                      includes_future_reference=True, channels=CHANNEL_NAMES,
                      frame_intervals_0based_inclusive=[[indices[0], indices[-1]] for indices in reference_indices],
                      frame_stride=1,
                      lower_per_sample=[lower.flatten().tolist() for lower, _ in window_bounds],
                      upper_per_sample=[upper.flatten().tolist() for _, upper in window_bounds]),
                  species_normalization='shared_normalize_species',
                  device=str(device), checkpoints={})
    with (output/'run_config.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    predictions, diagnostics = {}, {}
    persistence_errors = [relative_errors(initial, truth[:, i]).mean().item() for i in range(args.steps)]
    report['persistence_mean_channel_relative_L2'] = persistence_errors
    with (output/'channel_errors.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'step', 'elapsed_physical_seconds', 'mean_channel_relative_L2']+CHANNEL_NAMES)
        for name in args.models:
            model, checkpoint_report = load_transport(Path(args.ckpt_dir)/f'model_{name}', name, args, device)
            checkpoint_report['input_range_check'] = assess_input_range(initial.to(device), model.data_min, model.data_max)
            report['checkpoints'][name] = checkpoint_report
            if checkpoint_report['input_range_check']['any_outside_training_range']:
                print(f'{name}: initial state exceeds the training range (diagnostic only); '
                      'rollout uses the original full-reference-window bounds', flush=True)
            trajectories = []
            max_mass_error = max_n2_change = max_enthalpy_scaled_error = max_velocity_change = 0.0
            started = time.perf_counter()
            with torch.inference_mode():
                for sample, (lower, upper) in zip(initial, window_bounds):
                    model.load_stats({'data_min': lower.to(device), 'data_max': upper.to(device)})
                    current = sample.unsqueeze(0).clone().to(device)
                    trajectory = []
                    for step in range(1, args.steps+1):
                        intermediate = model(current)
                        current = source(intermediate)
                        if not torch.isfinite(current).all() or (current[:, 0] <= 0).any():
                            raise ValueError(f'{name}: invalid state at step {step}')
                        if (current[:, 1:10] < 0).any():
                            raise ValueError(f'{name}: negative mass fraction at step {step}')
                        mass_error = (current[:, 1:10].sum(1)-1).abs().max().item()
                        n2_change = (current[:, 9]-intermediate[:, 9]).abs().max().item()
                        velocity_change = (current[:, -2:]-intermediate[:, -2:]).abs().max().item()
                        old_h, _ = source.Y_to_T.enthalpy_cp(intermediate[:, 0].reshape(-1),
                            intermediate[:, 1:10].movedim(1, -1).reshape(-1, 9))
                        new_h, _ = source.Y_to_T.enthalpy_cp(current[:, 0].reshape(-1),
                            current[:, 1:10].movedim(1, -1).reshape(-1, 9))
                        # Absolute enthalpy may cross zero; scale by max(|h|, 1 MJ/kg).
                        enthalpy_error = ((new_h-old_h).abs()/old_h.abs().clamp_min(1e6)).max().item()
                        if mass_error > 2e-6 or n2_change > 2e-6 or velocity_change != 0 or enthalpy_error > 1e-5:
                            raise ValueError(f'{name}: reaction conservation check failed at step {step}')
                        max_mass_error = max(max_mass_error, mass_error)
                        max_n2_change = max(max_n2_change, n2_change)
                        max_velocity_change = max(max_velocity_change, velocity_change)
                        max_enthalpy_scaled_error = max(max_enthalpy_scaled_error, enthalpy_error)
                        trajectory.append(current[0].cpu())
                    trajectories.append(torch.stack(trajectory))
            predictions[name] = torch.stack(trajectories)
            diagnostics[name] = dict(rollout_seconds=time.perf_counter()-started,
                max_mass_sum_error=max_mass_error, max_reaction_N2_change=max_n2_change,
                max_reaction_velocity_change=max_velocity_change,
                max_reaction_enthalpy_error_scaled=max_enthalpy_scaled_error)
            for step in range(args.steps):
                errors = relative_errors(predictions[name][:, step], truth[:, step])
                writer.writerow([name, step+1, (step+1)*args.dt, errors.mean().item()]+errors.tolist())
                print(f'{name} step {step+1}/{args.steps}: mean_physical_channel_rel_L2={errors.mean().item():.6e}', flush=True)
            diagnostics[name]['final_mean_channel_relative_L2'] = errors.mean().item()
            diagnostics[name]['final_mean_error_below_persistence'] = errors.mean().item() < persistence_errors[-1]
            stream.flush()
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    arrays = {name: value.numpy() for name, value in predictions.items()}
    arrays.update(input=initial.numpy(), ground_truth=truth.numpy(),
                  reference_indices=np.array(reference_indices), dt=np.array(args.dt))
    with (output/f'lie_1e-5_{args.steps}.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    report.update(complete=True, diagnostics=diagnostics)
    with (output/'summary.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(f'ROLLOUT_COMPLETE {output}', flush=True)
    return output


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models', nargs='+', choices=['PSTO', 'FNO', 'Unet', 'UFNO'], default=['PSTO', 'FNO', 'Unet', 'UFNO'])
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--data-idx', type=int, nargs='+', default=[2001], help='1-based original reference frame indices')
    ap.add_argument('--dt', type=float, default=1e-5)
    ap.add_argument('--reference-dt', type=float, default=1e-6)
    ap.add_argument('--hf', default='data/H2_tripleflame.h5')
    ap.add_argument('--ckpt-dir', default='checkpoints/models_1e-5')
    ap.add_argument('--source-checkpoint', default='checkpoints/models_1e-5/model_DNN/dnn_model_Y.pt')
    ap.add_argument('--save-dir', default='analysis/results_1e-5/lie_10')
    ap.add_argument('--mechanism', default='mech/h2o2.yaml')
    ap.add_argument('--coefficient-checkpoint', default='checkpoints/models/model_coeff/constitutive_DNN_BCT.pt')
    ap.add_argument('--pressure', type=float, default=101325.0)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--seed', type=int, default=88)
    return ap


if __name__ == '__main__':
    options = parser().parse_args()
    for key in ('hf', 'ckpt_dir', 'source_checkpoint', 'save_dir', 'mechanism', 'coefficient_checkpoint'):
        setattr(options, key, resolve_path(getattr(options, key)))
    evaluate(options)
