import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from src.model import build_backbone, build_dnn_source_model, transport
from src.tripleflame import configuration
from utils.functions import load_transport_state_dict


MODELS = ('PSTO', 'FNO', 'Unet', 'UFNO')
DISPLAY = {'PSTO': 'PSTO', 'FNO': 'FNO', 'Unet': 'U-Net', 'UFNO': 'U-FNO'}
COLORS = {'PSTO': '#BF3D35', 'FNO': '#3274AA', 'Unet': '#40947D',
          'UFNO': '#9464B2'}


def configure_style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 13.5,
        'axes.titlesize': 14.5, 'axes.linewidth': .85,
        'mathtext.fontset': 'stix', 'xtick.direction': 'in',
        'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
        'savefig.dpi': 300})


def channel_window(dataset, channels, first, last):
    lower = np.full((len(channels), 1, 1), np.inf, dtype=np.float32)
    upper = np.full_like(lower, -np.inf)
    for start in range(first, last, 8):
        block = dataset[start:min(start + 8, last)][:, channels]
        lower = np.minimum(lower, block.min(axis=(0, 2, 3))[:, None, None])
        upper = np.maximum(upper, block.max(axis=(0, 2, 3))[:, None, None])
    return lower, upper


def state(dataset, index, channels, device):
    return torch.from_numpy(dataset[index][channels]).unsqueeze(0).to(
        device=device, dtype=torch.float32)


def relative_norm(difference, reference):
    numerator = torch.sqrt(torch.sum(difference.double().square(), dim=(1, 2, 3)))
    denominator = torch.sqrt(torch.sum(reference.double().square(), dim=(1, 2, 3)))
    return numerator / denominator.clamp_min(1e-30)


def load_model(name, args, data_min, data_max, device):
    checkpoint = torch.load(ROOT / f'checkpoints/models/model_{name}/checkpoint_best.pt',
                            map_location=device, weights_only=False)
    weights = checkpoint['model']
    backbone = build_backbone(name, len(configuration()['load_channels']), args,
                              data_min=data_min, data_max=data_max, state=weights)
    if name == 'PSTO':
        load_transport_state_dict(backbone, weights, strict=True)
    else:
        backbone.load_state_dict(weights, strict=True)
    model = transport(backbone.eval().to(device)).eval()
    model.load_stats({'data_min': data_min, 'data_max': data_max})
    return model


def coupled(model, reaction, values):
    return reaction(model(values))


def perturb_states(inputs, lower, upper, scale, epsilon, generator):
    noise = torch.randn(inputs.shape, device=inputs.device, dtype=inputs.dtype,
                        generator=generator)
    noise = F.avg_pool2d(noise, 9, stride=1, padding=4)
    noise = noise / noise.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-12)
    perturbed = (inputs + epsilon * scale.to(inputs.dtype) * noise).clamp(
        min=lower, max=upper)
    active = inputs[:, :1] > 800.
    perturbed[:, :1] = torch.where(
        active, perturbed[:, :1].clamp_min(800. + 1e-4),
        perturbed[:, :1].clamp_max(800.))
    species = perturbed[:, 1:-2].clamp_min(0)
    species = species / species.sum(1, keepdim=True).clamp_min(1e-12)
    perturbed[:, 1:-2] = species
    perturbed[..., 0, :] = inputs[..., 0, :]
    perturbed[..., -1, :] = inputs[..., -1, :]
    perturbed[..., :, 0] = inputs[..., :, 0]
    perturbed[..., :, -1] = inputs[..., :, -1]
    return perturbed


def fresh_start_diagnostics(model, reaction, data, channels, first, count, scale,
                            lower, upper, epsilon, device, batch_size, seed):
    """Measure coupled one-step errors and local transport sensitivity.

    The sensitivity ratio compares transport outputs before either state is
    passed to the reaction surrogate. It is evaluated at reference states;
    rollout_diagnostics measures amplification by the complete coupled map.
    """
    defects = np.empty(count)
    local_lipschitz = np.empty(count)
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        for offset in range(0, count, batch_size):
            size = min(batch_size, count - offset)
            inputs = torch.from_numpy(
                data[first + offset:first + offset + size][:, channels]).to(
                    device=device, dtype=torch.float32)
            targets = torch.from_numpy(
                data[first + offset + 1:first + offset + size + 1][:, channels]).to(
                    device=device, dtype=torch.float32)
            perturbed = perturb_states(
                inputs, lower, upper, scale, epsilon, generator)
            transported, transported_perturbed = model(
                torch.cat((inputs, perturbed))).chunk(2)
            predictions = reaction(transported)
            defects[offset:offset + size] = relative_norm(
                predictions - targets, targets).cpu().numpy()
            numerator = relative_norm(transported_perturbed - transported, inputs)
            denominator = relative_norm(perturbed - inputs, inputs)
            local_lipschitz[offset:offset + size] = (
                numerator / denominator).cpu().numpy()
    return defects, local_lipschitz


def rollout_diagnostics(model, reaction, data, channels, first, steps, scale, device):
    current = state(data, first, channels, device)
    reference = current.clone()
    defect = np.empty(steps)
    feedback = np.empty(steps)
    amplification = np.full(steps, np.nan)
    error = np.empty(steps)
    envelope = np.empty(steps)
    with torch.inference_mode():
        for step in range(steps):
            exact_current = state(data, first + step, channels, device)
            exact_next = state(data, first + step + 1, channels, device)
            previous_error = relative_norm(current - exact_current, reference).item()
            outputs = coupled(model, reaction, torch.cat((current, exact_current)))
            predicted_next, exact_input_next = outputs[:1], outputs[1:]
            defect[step] = relative_norm(exact_input_next - exact_next, reference).item()
            feedback[step] = relative_norm(
                predicted_next - exact_input_next, reference).item()
            error[step] = relative_norm(predicted_next - exact_next, reference).item()
            envelope[step] = defect[step] + feedback[step]
            if previous_error > 0:
                amplification[step] = feedback[step] / previous_error
            current = predicted_next
    bound = np.empty(steps)
    bound[0] = defect[0]
    for step in range(1, steps):
        bound[step] = defect[step] + amplification[step] * bound[step - 1]
    return defect, feedback, amplification, error, envelope, bound


def transport_diagnostics(model, full, exact_transport, indices, channels,
                          device, batch_size):
    values = np.empty(len(indices))
    with torch.inference_mode():
        for offset in range(0, len(values), batch_size):
            size = min(batch_size, len(values) - offset)
            frame = int(indices[offset])
            inputs = torch.from_numpy(
                full[frame:frame + size][:, channels]).to(
                    device=device, dtype=torch.float32)
            targets = torch.from_numpy(
                exact_transport[frame:frame + size][:, channels]).to(
                    device=device, dtype=torch.float32)
            predictions = model(inputs)
            values[offset:offset + size] = relative_norm(
                predictions - targets, targets).cpu().numpy()
    return values


def reaction_diagnostics(reaction, exact_transport, exact_reaction, indices,
                         channels, scale, device, batch_size):
    values = np.empty(len(indices))
    with torch.inference_mode():
        for offset in range(0, len(values), batch_size):
            size = min(batch_size, len(values) - offset)
            frame = int(indices[offset])
            inputs = torch.from_numpy(
                exact_transport[frame:frame + size][:, channels]).to(
                    device=device, dtype=torch.float32)
            targets = torch.from_numpy(
                exact_reaction[offset:offset + size][:, channels]).to(
                    device=device, dtype=torch.float32)
            predictions = reaction(inputs)
            values[offset:offset + size] = relative_norm(
                predictions - targets, targets).cpu().numpy()
    return values


def composition_diagnostics(model, reaction, full, exact_transport, exact_reaction,
                            indices, channels, device, batch_size):
    values = {key: np.empty(len(indices)) for key in (
        'transport', 'propagated_transport', 'reaction', 'splitting',
        'complete', 'envelope')}
    with torch.inference_mode():
        for offset in range(0, len(indices), batch_size):
            size = min(batch_size, len(indices) - offset)
            frame = int(indices[offset])
            initial = torch.from_numpy(
                full[frame:frame + size][:, channels]).to(
                device=device, dtype=torch.float32)
            exact_next = torch.from_numpy(
                full[frame + 1:frame + size + 1][:, channels]).to(
                device=device, dtype=torch.float32)
            transported = torch.from_numpy(
                exact_transport[frame:frame + size][:, channels]).to(
                device=device, dtype=torch.float32)
            split_exact = torch.from_numpy(
                exact_reaction[offset:offset + size][:, channels]).to(
                device=device, dtype=torch.float32)
            learned_transport = model(initial)
            learned_full, learned_reaction = reaction(
                torch.cat((learned_transport, transported))).chunk(2)
            batch = {
                'transport': relative_norm(
                    learned_transport - transported, exact_next),
                'propagated_transport': relative_norm(
                    learned_full - learned_reaction, exact_next),
                'reaction': relative_norm(
                    learned_reaction - split_exact, exact_next),
                'splitting': relative_norm(split_exact - exact_next, exact_next),
                'complete': relative_norm(learned_full - exact_next, exact_next)}
            batch['envelope'] = (batch['propagated_transport'] + batch['reaction']
                                 + batch['splitting'])
            for key, value in batch.items():
                values[key][offset:offset + size] = value.cpu().numpy()
    return values


def save_figure(fig, name, output):
    path = output / name
    fig.savefig(path, bbox_inches='tight', pad_inches=.06, facecolor='white')
    plt.close(fig)


def plot_results(results, output):
    configure_style()
    steps = np.arange(1, results['steps'].item() + 1)
    sample = np.arange(results['fresh_count'].item())
    markers = ('o', 's', '^', 'D')
    fig = plt.figure(figsize=(10.2, 7.0))
    outer = fig.add_gridspec(2, 1, left=.09, right=.985, bottom=.09, top=.965,
                             height_ratios=(.78, 1.0), hspace=.30)
    bound_grid = outer[0].subgridspec(1, 4, wspace=.16)
    bound_axes = []
    for column in range(4):
        axis = fig.add_subplot(
            bound_grid[0, column],
            sharex=bound_axes[0] if bound_axes else None,
            sharey=bound_axes[0] if bound_axes else None)
        bound_axes.append(axis)
    x = np.arange(results['composition_envelope'].shape[1])
    upper = results['composition_envelope'].max() * 1.06
    transport_color = '#D98B2B'
    reaction_color = '#40947D'
    splitting_color = '#9464B2'
    for index, (axis, name) in enumerate(zip(bound_axes, MODELS)):
        transport = results['composition_propagated_transport'][index]
        reaction = results['composition_reaction'][index]
        splitting = results['composition_splitting'][index]
        after_reaction = transport + reaction
        bound = after_reaction + splitting
        axis.fill_between(x, 0, transport, color=transport_color, alpha=.72,
                          linewidth=0)
        axis.fill_between(x, transport, after_reaction, color=reaction_color,
                          alpha=.78, linewidth=0)
        axis.fill_between(x, after_reaction, bound, color=splitting_color,
                          alpha=.76, linewidth=0)
        axis.plot(x, bound, color=COLORS[name], lw=2.15, ls='--', zorder=5)
        axis.plot(sample, results['fresh_defect'][index],
                  color=COLORS[name], lw=2.15, zorder=5)
        axis.text(.025, .965, DISPLAY[name], transform=axis.transAxes,
                  color=COLORS[name], fontsize=12.5, ha='left', va='top',
                  bbox=dict(facecolor='white', edgecolor='none', alpha=.82,
                            pad=.15), zorder=6)
        axis.set(xlim=(0, len(x)), ylim=(0, upper), xticks=(0, 50, 100))
        axis.tick_params(labelleft=index == 0)
        axis.grid(alpha=.16, linewidth=.45)
    bound_axes[0].legend(handles=(
        Patch(facecolor=transport_color, edgecolor='none',
              label='Transport operator'),
        Patch(facecolor=reaction_color, edgecolor='none',
              label='Reaction surrogate'),
        Patch(facecolor=splitting_color, edgecolor='none', label='Lie splitting'),
        Line2D([], [], color='#222222', lw=2.15, label='One-step defect'),
        Line2D([], [], color='#222222', lw=2.15, ls='--',
               label='One-step bound')),
        frameon=False, fontsize=8.8, ncol=1, loc='upper right',
        borderaxespad=.25, handlelength=1.6)
    fig.text(.5375, .992, '(a) One-step error decomposition', ha='center', va='top',
             fontsize=14.5)
    fig.text(.5375, bound_axes[0].get_position().y0 - .057,
             'Step index', ha='center', va='center', fontsize=13)

    lower_grid = outer[1].subgridspec(1, 2, wspace=.40)
    axes = (fig.add_subplot(lower_grid[0, 0]),
            fig.add_subplot(lower_grid[0, 1]))
    for index, name in enumerate(MODELS):
        axes[0].plot(sample, results['local_lipschitz'][index],
                     color=COLORS[name], lw=1.45, label=DISPLAY[name])
        axes[1].plot(steps, results['rollout_error'][index],
                     color=COLORS[name], lw=1.45, label=DISPLAY[name])
        axes[1].scatter(steps[9::10], results['rollout_bound'][index, 9::10],
                        marker=markers[index], facecolor='white',
                        edgecolor=COLORS[name], linewidth=.9, s=22)
    axes[0].set(title='(b) Local transport sensitivity',
                xlabel='Step index',
                ylabel=r'$L_n^{\mathrm{loc}}$')
    local_span = np.ptp(results['local_lipschitz'])
    axes[0].set_ylim(results['local_lipschitz'].min() - .08 * local_span,
                     results['local_lipschitz'].max() + .18 * local_span)
    axes[0].axhline(1.0, color='#333333', lw=1.0, ls='--', zorder=0)
    axes[1].set(title='(c) Error and recursive bound',
                xlabel='Autoregressive step',
                ylabel=r'$e_n,\ B_n$', yscale='log',
                xlim=(1, steps[-1]),
                ylim=(min(results['rollout_error'].min(),
                          results['rollout_bound'].min()) / 1.25,
                      max(results['rollout_error'].max(),
                          results['rollout_bound'].max()) * 1.45))
    for ax in axes:
        ax.grid(alpha=.18, linewidth=.5)
    axes[0].legend(frameon=False, ncol=2, fontsize=11.5, loc='upper center')
    axes[1].legend(handles=(
        Line2D([], [], color='#333333', lw=1.45, label=r'Actual $e_n$'),
        Line2D([], [], color='#333333', marker='o', markerfacecolor='white',
               linestyle='None', label='Recursive bound')),
        frameon=False, fontsize=11.5, loc='upper left')
    save_figure(fig, 'coupled_sensitivity_error.png', output)


def analyze(args):
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for the full diagnostic')
    device = torch.device('cuda', torch.cuda.current_device())
    channels = configuration()['load_channels']
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    # Diagnostic scaling uses full-HF bounds, independently of the training LF cache.
    # Compute the same channel-wise extrema locally when the original statistics
    # file is not supplied; no historical experiment data is bundled.
    with h5py.File(ROOT / 'data/H2_tripleflame.h5', 'r') as hf_file:
        lower, upper = channel_window(hf_file['data'], channels, 0,
                                      len(hf_file['data']))
    stats = {'hf_data_min': lower, 'hf_data_max': upper}
    channel_scale = torch.from_numpy(stats['hf_data_max'] - stats['hf_data_min']).to(
        device=device, dtype=torch.float32).unsqueeze(0)
    source = build_dnn_source_model(str(ROOT), 9, device,
                                    pressure=args.pressure,
                                    mechanism=str(ROOT / 'mech/h2o2.yaml'))

    results = {'models': np.asarray(MODELS), 'steps': np.asarray(args.steps),
               'rollout_start': np.asarray(args.rollout_start),
               'fresh_start': np.asarray(args.fresh_start),
               'fresh_count': np.asarray(args.fresh_count),
               'perturbation_epsilon': np.asarray(args.perturbation_epsilon)}
    fresh_defect = []
    local_lipschitz = []
    rollout = {key: [] for key in (
        'defect', 'feedback', 'amplification', 'error', 'envelope', 'bound')}
    transport_errors = []
    compositions = {key: [] for key in ('transport', 'propagated_transport', 'reaction',
                                         'splitting', 'complete', 'envelope')}
    with h5py.File(ROOT / 'data/H2_tripleflame.h5', 'r') as full_file, \
            h5py.File(ROOT / 'data/H2_tripleflame_T.h5', 'r') as transport_file, \
            h5py.File(ROOT / 'data/H2_tripleflame_R_Lie.h5', 'r') as reaction_file:
        full = full_file['data']
        exact_transport = transport_file['data']
        exact_reaction = reaction_file['data']
        indices = reaction_file['frame_indices'][:]
        if not np.array_equal(indices, np.arange(3000, 3100)):
            raise ValueError('Expected exact Lie samples for frames 3000--3099')
        reaction_errors = reaction_diagnostics(
            source, exact_transport, exact_reaction, indices, channels,
            channel_scale, device, args.fresh_batch_size)
        fresh_min, fresh_max = channel_window(
            full, channels, args.fresh_start, args.fresh_start + args.fresh_count + 1)
        rollout_min, rollout_max = channel_window(
            full, channels, args.rollout_start, args.rollout_start + args.steps + 1)
        for name in MODELS:
            data_min = torch.from_numpy(fresh_min).to(device)
            data_max = torch.from_numpy(fresh_max).to(device)
            model = load_model(name, args, data_min, data_max, device)
            defects, local_values = fresh_start_diagnostics(
                model, source, full, channels, args.fresh_start, args.fresh_count,
                channel_scale, torch.from_numpy(stats['hf_data_min']).to(device),
                torch.from_numpy(stats['hf_data_max']).to(device),
                args.perturbation_epsilon, device, args.fresh_batch_size, args.seed)
            fresh_defect.append(defects)
            local_lipschitz.append(local_values)
            model.load_stats({'data_min': torch.from_numpy(rollout_min).to(device),
                              'data_max': torch.from_numpy(rollout_max).to(device)})
            diagnostics = rollout_diagnostics(
                model, source, full, channels, args.rollout_start, args.steps,
                channel_scale, device)
            for key, values in zip(rollout, diagnostics):
                rollout[key].append(values)
            transport_errors.append(transport_diagnostics(
                model, full, exact_transport, indices, channels, device,
                args.fresh_batch_size))
            diagnostics = composition_diagnostics(
                model, source, full, exact_transport, exact_reaction, indices,
                channels, device, args.fresh_batch_size)
            for key, value in diagnostics.items():
                compositions[key].append(value)
            del model
            torch.cuda.empty_cache()
            print(f'{name}: rollout triangle max ratio '
                  f'{np.max(rollout["error"][-1] / rollout["envelope"][-1]):.9f}; '
                  f'Lie triangle max ratio '
                  f'{np.max(diagnostics["complete"] / diagnostics["envelope"]):.9f}',
                  flush=True)
    for key, values in rollout.items():
        results[f'rollout_{key}'] = np.asarray(values)
    results['fresh_defect'] = np.asarray(fresh_defect)
    results['local_lipschitz'] = np.asarray(local_lipschitz)
    results['sub_transport'] = np.asarray(transport_errors)
    results['sub_reaction'] = reaction_errors
    for key, values in compositions.items():
        results[f'composition_{key}'] = np.asarray(values)
    results['suboperator_frames'] = indices
    result_path = output / 'operator_diagnostics.npz'
    np.savez_compressed(result_path, **results)
    plot_results(results, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'analysis/figures')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--rollout-start', type=int, default=3000)
    parser.add_argument('--fresh-start', type=int, default=3000)
    parser.add_argument('--fresh-count', type=int, default=100)
    parser.add_argument('--fresh-batch-size', type=int, default=4)
    parser.add_argument('--perturbation-epsilon', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=88)
    parser.add_argument('--latent-per-var', type=int, default=8)
    parser.add_argument('--hidden-channels', type=int, default=20)
    parser.add_argument('--modes', type=int, default=16)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--dt', type=float, default=1e-6)
    parser.add_argument('--dx', type=float, default=1e-4)
    parser.add_argument('--coefficient-backend', default='dnn')
    parser.add_argument('--coefficient-checkpoint',
                        default='checkpoints/models/model_coeff/constitutive_DNN_BCT.pt')
    parser.add_argument('--mechanism', default='mech/h2o2.yaml')
    parser.add_argument('--pressure', type=float, default=101325.)
    analyze(parser.parse_args())
