"""Plot planar-jet runtime and accuracy--cost comparisons."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


CASE = Path(__file__).resolve().parents[2]
ROOT = CASE.parent
PERFORMANCE = CASE / 'analysis/performance'
LOCAL = CASE / 'analysis/figures/computational_performance.png'

MODELS = ('PSTO', 'FNO', 'Unet', 'UFNO')
DISPLAY = {'PSTO': 'PSTO', 'FNO': 'FNO', 'Unet': 'U-Net', 'UFNO': 'U-FNO'}
COLORS = {'PSTO': '#BF3D35', 'FNO': '#3274AA', 'Unet': '#40947D',
          'UFNO': '#9464B2', 'OpenFOAM': '#555555'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    font_files = []  # Use the installed serif fallback; no bundled proprietary fonts.
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
    font_family = (font_manager.FontProperties(fname=str(font_files[0])).get_name()
              if font_files else 'DejaVu Serif')
    plt.rcParams.update({
        'font.family': font_family, 'font.size': 12.5,
        'axes.titlesize': 14, 'axes.titlelocation': 'center',
        'axes.linewidth': .85, 'mathtext.fontset': 'stix',
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True, 'savefig.dpi': 300})

    timing = json.loads((PERFORMANCE / 'gpu_timing_100.json').read_text())
    accuracy = json.loads((PERFORMANCE / 'benchmark_accuracy_start2000.json').read_text())
    openfoam = json.loads((PERFORMANCE / 'openfoam_timing_100.json').read_text())
    names = (*MODELS, 'OpenFOAM')
    times = np.array([np.mean(timing['models'][name]['times_s']) for name in MODELS]
                     + [openfoam['window_clock_s']])
    errors = np.array([np.std(timing['models'][name]['times_s'], ddof=1)
                       for name in MODELS] + [0.0])
    relative_l2 = np.array([100 * accuracy['mean_relative_l2'][name]
                            for name in MODELS] + [0.0])

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.65),
                             gridspec_kw={'wspace': .26})
    positions = np.arange(len(names))
    colors = [COLORS[name] for name in names]
    axes[0].bar(positions, times, yerr=errors, color=colors, width=.72,
                error_kw={'elinewidth': .8, 'capsize': 2.5, 'capthick': .8})
    axes[0].set_yscale('log')
    axes[0].set_xticks(positions, [DISPLAY.get(name, name) for name in names])
    axes[0].tick_params(axis='x', labelsize=10.5)
    axes[0].set(title='(a) Runtime', ylabel='100-step wall time (s)')
    axes[0].grid(axis='y', alpha=.18, linewidth=.5)

    for name, runtime, error in zip(names, times, relative_l2):
        axes[1].scatter(runtime, error, s=58, color=COLORS[name],
                        edgecolor='white', linewidth=.55, label=DISPLAY.get(name, name),
                        zorder=3)
    axes[1].set_xscale('log')
    axes[1].set_ylim(bottom=0)
    axes[1].set(title='(b) Accuracy–cost', xlabel='100-step wall time (s)',
                ylabel=r'Mean variable-wise relative $L_2$ error (%)')
    axes[1].grid(alpha=.18, linewidth=.5)
    axes[1].legend(frameon=True, fontsize=10, ncol=2, loc='upper right',
                   columnspacing=.9, handletextpad=.35)

    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(LOCAL, bbox_inches='tight', pad_inches=.05, facecolor='white',
                format='png')
    plt.close(fig)


if __name__ == '__main__':
    main()
