import argparse
import csv
import gc
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np


MODELS = ('PSTO', 'FNO', 'Unet', 'UFNO')
DISPLAY = {'PSTO': 'PSTO', 'FNO': 'FNO', 'Unet': 'U-Net', 'UFNO': 'U-FNO'}
CHANNELS = ('T', 'H2', 'H', 'O', 'O2', 'OH', 'H2O', 'HO2', 'H2O2', 'N2', 'u', 'v')
FIELD_CHANNELS = (0, 6, 10, 11)
FIELD_TITLES = (r'$T$ (K)', r'$Y_{\mathrm{H}_2\mathrm{O}}$', r'$u$ (m/s)', r'$v$ (m/s)')
COLORS = ('#BF3D35', '#3274AA', '#40947D', '#9464B2')


def configure_style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 14,
        'axes.titlesize': 17, 'axes.titlelocation': 'center',
        'axes.linewidth': .85, 'mathtext.fontset': 'stix',
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True, 'savefig.dpi': 300,
    })


def channel_errors(prediction, truth):
    difference = prediction.astype(np.float64) - truth.astype(np.float64)
    numerator = np.sqrt(np.sum(difference * difference, axis=(0, 2, 3)))
    denominator = np.sqrt(np.sum(truth.astype(np.float64) ** 2, axis=(0, 2, 3)))
    errors = np.divide(numerator, denominator, out=np.full_like(numerator, np.inf),
                       where=denominator > 0)
    errors[(numerator == 0) & (denominator == 0)] = 0
    return errors


def load_results(path, requested_step, directory):
    archive = np.load(path)
    truth = archive['ground_truth'][0]
    step = min(requested_step, truth.shape[0])
    report_steps = [value for value in (1, 5, 10, 20, 50, 100)
                    if value <= truth.shape[0]]
    reference = truth[step - 1].copy()
    predictions, histories, endpoint_errors = {}, {}, {}
    for name in MODELS:
        prediction = archive[name][0]
        predictions[name] = prediction[step - 1].copy()
        histories[name] = np.array([
            channel_errors(prediction[value:value + 1], truth[value:value + 1])
            for value in range(truth.shape[0])])
        endpoint_errors[name] = histories[name][step - 1]
        del prediction
        gc.collect()
    with (directory / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('model', 'step', 'rel_L2', *CHANNELS))
        for name in MODELS:
            for value in report_steps:
                errors = histories[name][value - 1]
                writer.writerow((name, value, errors.mean(), *errors))
    np.savez(directory / 'relative_l2_history.npz',
             **{name: values.mean(axis=1) for name, values in histories.items()})
    archive.close()
    return step, reference, predictions, histories, endpoint_errors


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


def plot_error_history(directory, histories):
    fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
    for name, color in zip(MODELS, COLORS):
        curve = histories[name].mean(axis=1)
        ax.plot(np.arange(1, len(curve) + 1), curve, color=color, label=DISPLAY[name])
    ax.set(xlabel='Autoregressive step', ylabel=r'Mean variable-wise relative $L_2$ error')
    ax.set_yscale('log')
    ax.grid(alpha=.2, linewidth=.6)
    ax.legend(frameon=False)
    fig.savefig(directory / 'relative_l2.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_channel_errors(directory, errors, step):
    x = np.arange(len(CHANNELS))
    width = .19
    fig, ax = plt.subplots(figsize=(11, 4.5), layout='constrained')
    for index, (name, color) in enumerate(zip(MODELS, COLORS)):
        ax.bar(x + (index - 1.5) * width, errors[name], width,
               color=color, label=DISPLAY[name])
    ax.set_xticks(x, CHANNELS)
    ax.set_yscale('log')
    ax.set_ylabel(r'Channel relative $L_2$')
    ax.set_title(f'Autoregressive step {step}', loc='center')
    ax.grid(axis='y', alpha=.2, linewidth=.6)
    ax.legend(frameon=False, ncol=4)
    fig.savefig(directory / f'channel_relative_l2_step{step}.png',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_fields(directory, step, truth, predictions):
    row_names = ('Ground Truth', *(DISPLAY[name] for name in MODELS))
    fields = (truth, *(predictions[name] for name in MODELS))
    image_width = 2.75
    image_height = image_width * truth.shape[-2] / truth.shape[-1]
    left, right, top, bottom = .52, .13, .50, .66
    column_gap, row_gap = .20, .18
    figure_width = left + 4 * image_width + 3 * column_gap + right
    figure_height = top + 5 * image_height + 4 * row_gap + bottom
    fig = plt.figure(figsize=(figure_width, figure_height))

    for column, (channel, title) in enumerate(zip(FIELD_CHANNELS, FIELD_TITLES)):
        x = left + column * (image_width + column_gap)
        vmin, vmax = float(truth[channel].min()), float(truth[channel].max())
        image = None
        for row, (name, field) in enumerate(zip(row_names, fields)):
            y = bottom + (4 - row) * (image_height + row_gap)
            ax = fig.add_axes([x / figure_width, y / figure_height,
                               image_width / figure_width, image_height / figure_height])
            interpolation = 'bilinear' if channel in (0, 6) else 'nearest'
            image = ax.imshow(field[channel], origin='lower', cmap='turbo',
                              vmin=vmin, vmax=vmax, aspect='equal',
                              interpolation=interpolation)
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


def plot_spatial_error(directory, step, truth, predictions):
    channel = 0
    reference = truth[channel]
    scale = max(float(reference.max() - reference.min()), np.finfo(np.float32).eps)
    fields = {name: np.abs(field[channel] - reference) / scale
              for name, field in predictions.items()}
    vmax = max(float(field.max()) for field in fields.values())
    fig, axes = plt.subplots(2, 2, figsize=(8.7, 5.0), layout='compressed')
    image = None
    for ax, name in zip(axes.flat, MODELS):
        image = ax.imshow(fields[name], origin='lower', cmap='magma', vmin=0,
                          vmax=vmax, aspect='equal', interpolation='nearest')
        ax.set(xticks=[], yticks=[])
        ax.set_title(DISPLAY[name], pad=5)
    fig.colorbar(image, ax=axes, orientation='horizontal', shrink=.9, pad=.055)
    output = directory / 'spatial_errors_PlanarJet.png'
    fig.savefig(output, bbox_inches='tight', pad_inches=.07, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=Path('analysis/figures'))
    parser.add_argument('--data-idx', type=int, default=2001)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--step', type=int, default=100)
    args = parser.parse_args()
    if min(args.data_idx, args.steps, args.step) < 1:
        raise ValueError('data index and steps must be positive')
    configure_style()
    path = args.directory / f'dnn_{args.data_idx}_all_{args.steps}.npz'
    result = load_results(path, args.step, args.directory)
    step, truth, predictions, histories, endpoint_errors = result
    plot_error_history(args.directory, histories)
    plot_channel_errors(args.directory, endpoint_errors, step)
    plot_fields(args.directory, step, truth, predictions)
    plot_spatial_error(args.directory, step, truth, predictions)


if __name__ == '__main__':
    main()
