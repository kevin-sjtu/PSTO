import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


CASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE / 'src'))
sys.path.insert(1, str(CASE))

from robertson import configuration
from test import checkpoint_architecture
from train import build_model
from utils.functions import enforce_boundaries, normalize_species


def load_checkpoint(device, path):
    config = configuration()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint['model']
    lower, upper = state['data_min'].to(device), state['data_max'].to(device)
    model = build_model('PSTO', config, lower, upper,
                        **checkpoint_architecture('PSTO', state)).to(device)
    model.load_state_dict(state)
    return model.eval(), config


def validation_frames(config, samples):
    path = CASE / config['lf']
    indices = np.linspace(config['train_pairs'],
                          config['train_pairs'] + config['val_pairs'] - 1,
                          samples, dtype=int)
    with h5py.File(path, 'r') as file:
        data = file['data']
        inputs = np.stack([data[index] for index in indices])
        targets = np.stack([data[index + 1] for index in indices])
    return torch.from_numpy(inputs).float(), torch.from_numpy(targets).float()


@torch.no_grad()
def stage_statistics(model, inputs, targets, batch_size=4):
    branch = model.transport_branch
    stage_residuals = []
    errors = [[] for _ in range(model.n_layers)]
    for first in range(0, len(inputs), batch_size):
        physical = inputs[first:first + batch_size].to(model.data_min.device)
        truth = targets[first:first + batch_size, :4].to(physical.device)
        batch, _, height, width = physical.shape
        species0, velocity0 = physical[:, :4], physical[:, -2:]
        flat0 = species0.reshape(batch * 4, 1, height, width)
        z0 = branch.lift(flat0)
        z = z0
        coefficient = model.species_diffusivity.expand_as(flat0)
        density = torch.ones_like(flat0)
        residual = []
        for layer in range(model.n_layers + 1):
            z_next = branch(z, z0, velocity0, coefficient,
                            model.time_step, model.grid_step, density)
            update = (z_next - z).reshape(batch, 4, -1)
            residual.append(torch.linalg.vector_norm(update, dim=(1, 2)))
            if layer < model.n_layers:
                z = z_next
                species = (flat0 + branch.project(z - z0)).reshape_as(species0)
                species = enforce_boundaries(
                    species.clamp(min=model.data_min[:4], max=model.data_max[:4]),
                    species0)
                species = normalize_species(species)
                if layer == model.n_layers - 1:
                    torch.testing.assert_close(species, model(physical), rtol=1e-6, atol=1e-7)
                numerator = torch.linalg.vector_norm(
                    (species - truth).reshape(batch, 4, -1), dim=2)
                denominator = torch.linalg.vector_norm(
                    truth.reshape(batch, 4, -1), dim=2)
                errors[layer].append((numerator / denominator).mean(1))
        residual = torch.stack(residual, 1)
        stage_residuals.append(residual / residual[:, :1])
    residuals = torch.cat(stage_residuals).cpu().numpy()
    errors = np.stack([torch.cat(values).cpu().numpy() for values in errors], 1)
    return residuals, errors


@torch.no_grad()
def sharing_audit(model, physical):
    branch = model.transport_branch
    physical = physical[:1].to(model.data_min.device)
    species, velocity = physical[:, :4], physical[:, -2:]
    permutation = torch.tensor([2, 0, 3, 1], device=physical.device)
    coefficient = model.species_diffusivity * torch.tensor(
        [.5, 1., 1.5, 2.], device=physical.device).reshape(4, 1, 1, 1)
    coefficient = coefficient.expand(4, 1, *physical.shape[-2:])

    def raw_transport(fields, coefficients):
        flat = fields.reshape(4, 1, *physical.shape[-2:])
        z0 = branch.lift(flat)
        z = z0
        for _ in range(model.n_layers):
            z = branch(z, z0, velocity, coefficients, model.time_step,
                       model.grid_step, torch.ones_like(flat))
        return (flat + branch.project(z - z0)).reshape_as(fields)

    expected = raw_transport(species, coefficient)[:, permutation]
    observed = raw_transport(species[:, permutation], coefficient[permutation])
    torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-7)
    return (torch.linalg.vector_norm(observed - expected)
            / torch.linalg.vector_norm(expected)).item()


def configure_style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 14.5,
        'axes.titlesize': 15.5, 'axes.linewidth': .8,
        'mathtext.fontset': 'stix', 'xtick.direction': 'in',
        'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    })


def plot_evidence(residuals, errors, relaxation, diffusion_number, output):
    configure_style()
    stages = np.arange(residuals.shape[1])
    layers = np.arange(1, errors.shape[1] + 1)
    figure, axes = plt.subplots(1, 3, figsize=(11.6, 3.25), layout='constrained')

    median = np.median(residuals, axis=0)
    low, high = np.percentile(residuals, [25, 75], axis=0)
    axes[0].fill_between(stages, low, high, color='#3274AA', alpha=.18,
                         linewidth=0, label='25th–75th percentiles')
    axes[0].plot(stages, median, 'o-', color='#3274AA', linewidth=1.7,
                 markersize=4.5, label='Median')
    axes[0].axhline(1, color='.45', linewidth=.8, linestyle=':')
    axes[0].set(xlabel=r'Stage index $\ell$',
                title=r'(a) Normalized residual $r_\ell/r_0$')
    axes[0].set_yscale('log')
    axes[0].set_xticks(stages)
    axes[0].legend(frameon=False, fontsize=11)

    error_median = 100 * np.median(errors, axis=0)
    error_low, error_high = 100 * np.percentile(errors, [25, 75], axis=0)
    axes[1].fill_between(layers, error_low, error_high, color='#BF3D35',
                         alpha=.18, linewidth=0)
    axes[1].plot(layers, error_median, 'o-', color='#BF3D35', linewidth=1.7,
                 markersize=4.5)
    axes[1].set(xlabel='Number of anchored stages',
                title=r'(b) One-step mean relative $L_2$ (%)')
    axes[1].set_xticks(layers)

    xi_max = 8 * diffusion_number
    xi = np.linspace(0, xi_max, 500)
    implicit = 1 / (1 + xi)
    explicit = 1 - xi
    iteration = 1 - relaxation * (1 + xi)
    finite = implicit + xi / (1 + xi) * iteration ** errors.shape[1]
    axes[2].plot(xi, explicit, color='#9464B2', linewidth=1.35,
                 linestyle='--', label='Explicit Euler')
    axes[2].plot(xi, implicit, color='#40947D', linewidth=1.7,
                 label='Implicit Euler')
    axes[2].plot(xi, finite, color='#BF3D35', linewidth=1.7,
                 label=f'{errors.shape[1]} anchored stages')
    axes[2].axhline(0, color='.45', linewidth=.8)
    axes[2].set(xlabel=r'Scaled diffusion eigenvalue $\xi$',
                title='(c) Amplification factor')
    axes[2].set_ylim(-1.15, 1.05)
    axes[2].legend(frameon=False, fontsize=10.5, loc='lower right')

    for axis in axes:
        axis.grid(alpha=.16, linewidth=.55)
    figure.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='Compute anchored-stage diagnostics from a locally trained checkpoint.')
    parser.add_argument('--checkpoint', type=Path, default=CASE / 'checkpoints/models/model_PSTO/checkpoint_best.pt')
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--figure-dir', type=Path, default=CASE / 'analysis/figures')
    parser.add_argument('--result-dir', type=Path, default=CASE / 'analysis/results')
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    device = torch.device(args.device)
    torch.set_num_threads(1)
    model, config = load_checkpoint(device, args.checkpoint)
    inputs, targets = validation_frames(config, samples=args.samples)
    residuals, errors = stage_statistics(model, inputs, targets)
    permutation_error = sharing_audit(model, inputs)
    relaxation = (.5 * torch.sigmoid(
        model.transport_branch.block.update_logit)).item()
    diffusion_number = (config['species_diffusivity'] * config['dt']
                        / config['dx'] ** 2)
    figure_dir = args.figure_dir
    result_dir = args.result_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    output = figure_dir / 'anchored_stage_evidence.png'
    plot_evidence(residuals, errors, relaxation, diffusion_number, output)
    summary = {
        'validation_samples': len(inputs),
        'permutation_relative_error': permutation_error,
        'stage_projection_matches_forward': True,
        'relaxation': relaxation,
        'diffusion_number': diffusion_number,
        'xi_max': 8 * diffusion_number,
        'convergence_limit': 2 / (1 + 8 * diffusion_number),
        'nonoscillatory_limit': 1 / (1 + 8 * diffusion_number),
        'median_normalized_residual': np.median(residuals, axis=0).tolist(),
        'median_one_step_error_percent': (100 * np.median(errors, axis=0)).tolist(),
        'mean_one_step_error_percent': (100 * np.mean(errors, axis=0)).tolist(),
    }
    (result_dir / 'anchored_stage_evidence.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    np.savez(result_dir / 'anchored_stage_evidence.npz',
             residuals=residuals, errors=errors)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
