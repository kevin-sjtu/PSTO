import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np


MODELS = ('PSTO', 'FNO', 'Unet', 'UFNO')
DISPLAY = {'PSTO': 'PSTO', 'FNO': 'FNO', 'Unet': 'U-Net', 'UFNO': 'U-FNO'}
CHANNEL_TITLES = (r'$Y_A$', r'$Y_B$', r'$Y_C$', r'$Y_D$')
COLORS = ('#BF3D35', '#3274AA', '#40947D', '#9464B2')


def configure_style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 14,
        'axes.titlesize': 17, 'axes.titlelocation': 'center',
        'axes.linewidth': .85, 'mathtext.fontset': 'stix',
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True, 'savefig.dpi': 300,
    })


def add_colorbar(fig, image, rect):
    cax = fig.add_axes(rect)
    bar = fig.colorbar(image, cax=cax, orientation='horizontal')
    bar.locator = MaxNLocator(nbins=4)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))
    bar.formatter = formatter
    bar.update_ticks()
    bar.ax.tick_params(direction='out', top=False, bottom=True, length=2.8, pad=2)
    bar.outline.set_linewidth(.75)


def plot_error_history(directory):
    with (directory / 'comparison.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
    for name, color in zip(MODELS, COLORS):
        values = [row for row in rows if row['model'] == name]
        ax.plot([int(row['step']) for row in values],
                [float(row['rel_L2']) for row in values],
                marker='o', color=color, label=DISPLAY[name])
    ax.set(xlabel='Autoregressive step', ylabel=r'Mean variable-wise relative $L_2$ error')
    ax.set_yscale('log')
    ax.grid(alpha=.2, linewidth=.6)
    ax.legend(frameon=False)
    fig.savefig(directory / 'relative_l2.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_fields(directory, requested_step):
    with np.load(directory / 'rollout.npz') as data:
        step = min(requested_step, len(data['ground_truth']))
        truth = data['ground_truth'][step - 1]
        predictions = {name: data[name][step - 1] for name in MODELS}

    row_names = ('Ground Truth', *(DISPLAY[name] for name in MODELS))
    fields = (truth, *(predictions[name] for name in MODELS))
    image_width = 2.75
    image_height = image_width * truth.shape[-2] / truth.shape[-1]
    left, right, top, bottom = .52, .13, .50, .66
    column_gap, row_gap = .20, .18
    figure_width = left + 4 * image_width + 3 * column_gap + right
    figure_height = top + 5 * image_height + 4 * row_gap + bottom
    fig = plt.figure(figsize=(figure_width, figure_height))

    for column, title in enumerate(CHANNEL_TITLES):
        x = left + column * (image_width + column_gap)
        vmin, vmax = float(truth[column].min()), float(truth[column].max())
        image = None
        for row, (name, field) in enumerate(zip(row_names, fields)):
            y = bottom + (4 - row) * (image_height + row_gap)
            ax = fig.add_axes([x / figure_width, y / figure_height,
                               image_width / figure_width, image_height / figure_height])
            image = ax.imshow(field[column], origin='lower', cmap='turbo',
                              vmin=vmin, vmax=vmax, aspect='equal', interpolation='bilinear')
            ax.set(xticks=[], yticks=[])
            if row == 0:
                ax.set_title(title, pad=8)
            if column == 0:
                ax.set_ylabel(name, rotation=90, ha='center', va='bottom',
                              labelpad=10, fontsize=16)
        add_colorbar(fig, image,
                     [x / figure_width, .36 / figure_height,
                      image_width / figure_width, .095 / figure_height])

    output = directory / f'fields_step{step}.png'
    fig.savefig(output, bbox_inches='tight', pad_inches=.07, facecolor='white')
    plt.close(fig)


def plot_spatial_error(directory, requested_step):
    with np.load(directory / 'rollout.npz') as data:
        step = min(requested_step, len(data['ground_truth']))
        truth = data['ground_truth'][step - 1, 2]
        fields = {name: np.abs(data[name][step - 1, 2] - truth)
                  for name in MODELS}
    scale = max(float(truth.max() - truth.min()), np.finfo(np.float32).eps)
    fields = {name: field / scale for name, field in fields.items()}
    vmax = max(float(field.max()) for field in fields.values())
    fig, axes = plt.subplots(2, 2, figsize=(8.7, 4.9), layout='compressed')
    image = None
    for ax, name in zip(axes.flat, MODELS):
        image = ax.imshow(fields[name], origin='lower', cmap='magma', vmin=0,
                          vmax=vmax, aspect='equal', interpolation='nearest')
        ax.set(xticks=[], yticks=[])
        ax.set_title(DISPLAY[name], pad=5)
    fig.colorbar(image, ax=axes, orientation='horizontal', shrink=.9, pad=.055)
    output = directory / 'spatial_errors_Robertson.png'
    fig.savefig(output, bbox_inches='tight', pad_inches=.07, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parents[1] / 'analysis/figures')
    parser.add_argument('--step', type=int, default=100)
    args = parser.parse_args()
    if args.step < 1:
        raise ValueError('step must be positive')
    configure_style()
    plot_error_history(args.directory)
    plot_fields(args.directory, args.step)
    plot_spatial_error(args.directory, args.step)


if __name__ == '__main__':
    main()
