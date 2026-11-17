"""Plot the CD-to-CDR state-distribution comparison for Robertson."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


CASE = Path(__file__).resolve().parents[1]
FIGURES = CASE / 'analysis/figures'


def main():
    parser = argparse.ArgumentParser(description='Plot an existing distribution-shift JSON; this script does not compute t-SNE or MMD.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=FIGURES / 'inference_transfer.png')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.png':
        parser.error('--output must end in .png')
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 14.5,
        'axes.linewidth': .85, 'mathtext.fontset': 'stix',
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True, 'savefig.dpi': 300})

    data = json.loads(args.input.read_text())
    embedding = np.asarray(data['embedding'])
    split = len(data['indices']['lf'])
    fig, axis = plt.subplots(figsize=(5.5, 4.3), layout='constrained')
    for values, color, label in zip(
            (embedding[:split], embedding[split:]),
            ('#3274AA', '#BF3D35'),
            ('CD dataset training', 'CDR dataset evaluation')):
        axis.scatter(values[:, 0], values[:, 1], s=30, color=color,
                     label=label, alpha=.95)
    axis.set_xlabel('t-SNE component 1', fontsize=17)
    axis.set_ylabel('t-SNE component 2', fontsize=17)
    axis.tick_params(labelsize=15.5)
    axis.text(.66, .55, rf'MMD = {data["mmd"]:.3f}', transform=axis.transAxes,
              ha='center', va='center', fontsize=17,
              bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85,
                    'pad': 1.5})
    axis.legend(frameon=False, fontsize=15, handletextpad=.6, labelspacing=.7)

    destination = args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches='tight', pad_inches=.06, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
