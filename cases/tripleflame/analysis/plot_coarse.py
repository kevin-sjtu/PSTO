"""Plot locally generated 1e-5 s rollouts using their recorded channel errors."""
import argparse
import csv
from pathlib import Path

import numpy as np
from plot import (MODELS, CHANNELS, configure_style, plot_error_history,
                  plot_channel_errors, plot_fields, plot_spatial_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('analysis/results_1e-5/lie_10'))
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--step', type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.step <= args.steps:
        parser.error('--step must be between 1 and --steps')
    with (args.directory / 'channel_errors.csv').open() as stream:
        rows = {(row['model'], int(row['step'])): row for row in csv.DictReader(stream)}
    histories = {name: np.asarray([[float(rows[name, step][channel]) for channel in CHANNELS]
                                   for step in range(1, args.steps + 1)]) for name in MODELS}
    errors = {name: values[args.step - 1] for name, values in histories.items()}
    with np.load(args.directory / f'lie_1e-5_{args.steps}.npz') as archive:
        truth = archive['ground_truth'][0, args.step - 1]
        predictions = {name: archive[name][0, args.step - 1] for name in MODELS}
    configure_style()
    plot_error_history(args.directory, histories)
    plot_channel_errors(args.directory, errors, args.step)
    plot_fields(args.directory, args.step, truth, predictions)
    plot_spatial_error(args.directory, args.step, truth, predictions)


if __name__ == '__main__':
    main()
