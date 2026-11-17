"""Plot the discrete CDR-equation residual for the evaluated rollouts."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


CASE = Path(__file__).resolve().parents[1]
MODELS = ('PSTO', 'FNO', 'Unet', 'UFNO')
DISPLAY = {'PSTO': 'PSTO', 'FNO': 'FNO', 'Unet': 'U-Net', 'UFNO': 'U-FNO'}
COLORS = {'PSTO': '#BF3D35', 'FNO': '#3274AA', 'U-Net': '#40947D',
          'U-FNO': '#9464B2', 'Ground Truth': '#202020'}


def configure_style():
    for font_file in (CASE / 'fonts').glob('*.ttf'):
        font_manager.fontManager.addfont(font_file)
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_family = 'Times New Roman' if 'Times New Roman' in available_fonts else 'DejaVu Serif'
    plt.rcParams.update({
        'font.family': font_family, 'font.size': 14.5,
        'axes.titlesize': 16, 'axes.titlelocation': 'center',
        'axes.linewidth': .85, 'mathtext.fontset': 'stix',
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True, 'savefig.dpi': 300})


def analyze(directory):
    with np.load(directory / 'rollout.npz') as data:
        initial = data['input'].copy()
        truth = data['ground_truth'].copy()
        trajectories = {'Ground Truth': truth,
                        **{DISPLAY[name]: data[name].copy() for name in MODELS}}
        scale = (data['data_max'] - data['data_min']).reshape(3, 1, 1)
    dt, dx = .00025, .01

    def laplacian(field):
        return sum((np.roll(field, -1, axis) - 2 * field
                    + np.roll(field, 1, axis)) / dx**2
                   for axis in (-2, -1))

    def centered_advection(field, velocity, axis):
        return velocity * (np.roll(field, -1, axis)
                           - np.roll(field, 1, axis)) / (2 * dx)

    def source_terms(state):
        u, v, phi = state
        speed2 = u * u + v * v
        return np.stack(((1 - speed2) * u + speed2 * v,
                         (1 - speed2) * v - speed2 * u,
                         phi * (1 - phi) * (phi - .5)))

    def residual(old, current):
        # Periodic centered stencils with the step-initial velocity, as in
        # eq:canonical_rollout_residual. The canonical grid stores x on axis -2.
        advection = (centered_advection(current, old[0], -2)
                     + centered_advection(current, old[1], -1))
        return ((current - old) / dt + advection
                - .005 * laplacian(current) - source_terms(current)) / scale

    curves, maps = {}, {}
    for name, trajectory in trajectories.items():
        old, values = initial.astype(np.float64), []
        for current in trajectory:
            current = current.astype(np.float64)
            field = np.sqrt(np.mean(residual(old, current) ** 2, axis=0))
            values.append(np.percentile(field, 95))
            old = current
        curves[name], maps[name] = np.asarray(values), field
    report = {
        'definition': 'First-order time difference; centered advection and five-point diffusion with periodic boundaries; step-initial velocity; reaction at predicted next state',
        'units': 's^-1',
        'dt': dt, 'dx': dx,
        'initial_spatial_95th_percentile': {name: float(values[0])
                                           for name, values in curves.items()},
        'endpoint_spatial_95th_percentile': {name: float(values[-1])
                                            for name, values in curves.items()},
        'endpoint_to_initial_ratio': {name: float(values[-1] / values[0])
                                      for name, values in curves.items()}}
    (directory / 'pde_residual_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez(directory / 'pde_residual_diagnostics.npz',
             **{f'curve_{name}': value for name, value in curves.items()},
             **{f'map_{name}': value for name, value in maps.items()})
    print(json.dumps(report, indent=2))

    fig = plt.figure(figsize=(9.0, 5.8))
    left, right, gap = .14, .86, .012
    map_width = (right - left - 3 * gap) / 4
    map_height = map_width * 9.0 / 5.8
    map_bottom = .62
    vmax = max(float(field.max()) for field in maps.values())
    image = None
    for column, name in enumerate(DISPLAY.values()):
        axis = fig.add_axes([left + column * (map_width + gap), map_bottom,
                             map_width, map_height])
        image = axis.imshow(maps[name], origin='lower', cmap='magma', vmin=0,
                            vmax=vmax, interpolation='nearest')
        axis.set_title(name, pad=6)
        axis.set(xticks=[], yticks=[])
    bar_axis = fig.add_axes([left, .565, right - left, .025])
    fig.colorbar(image, cax=bar_axis, orientation='horizontal')
    bar_axis.tick_params(axis='x', labelsize=10)
    axis = fig.add_axes([left, .13, right - left, .32])
    for name in trajectories:
        axis.plot(np.arange(1, len(curves[name]) + 1), curves[name], color=COLORS[name],
                  lw=1.7, label=name)
    axis.set(title='(b) Residual evolution', xlabel='Autoregressive step',
             xlim=(1, len(truth)), ylabel=r'95th percentile (s$^{-1}$)')
    axis.grid(alpha=.18, linewidth=.55)
    axis.legend(frameon=False, ncol=3)
    fig.suptitle('(a) Normalized PDE residual', y=.982, fontsize=15)

    destination = directory / 'pde_residual_analysis.png'
    fig.savefig(destination, bbox_inches='tight', pad_inches=.06, facecolor='white',
                format='png')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=CASE / 'analysis/figures',
                        help='Directory containing rollout.npz and receiving local diagnostics.')
    args = parser.parse_args()
    configure_style()
    analyze(args.directory)
