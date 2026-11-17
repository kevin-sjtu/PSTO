"""Plot the matched-horizon timestep comparison as PNG using archived and measured data.

Existing files are never overwritten. The 1e-6 s points and bars use exactly
the data behind computational_performance.png; the new measurements come from
benchmark_gpu_timestep_comparison.py. Only the requested local PNG and numerical summary are written.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np


CASE = Path(__file__).resolve().parents[2]
ROOT = CASE.parent
PERFORMANCE = CASE / 'analysis/performance'
MODELS = ('PSTO', 'FNO', 'Unet', 'UFNO')
DISPLAY = {'PSTO': 'PSTO', 'FNO': 'FNO', 'Unet': 'U-Net', 'UFNO': 'U-FNO'}
COLORS = {'PSTO': '#BF3D35', 'FNO': '#3274AA', 'Unet': '#40947D', 'UFNO': '#9464B2'}
COARSE_ALPHA = 0.48
TABLE_STEPS = (1, 2, 5, 10)
CHANNELS = ('T', 'H2', 'H', 'O', 'O2', 'OH', 'H2O', 'HO2', 'H2O2', 'N2', 'u', 'v')


def collect_data(errors_path):
    timing_path = PERFORMANCE / 'gpu_timing_100.json'
    accuracy_path = PERFORMANCE / 'benchmark_accuracy_start2000.json'
    measured_path = PERFORMANCE / 'gpu_timing_timestep_comparison.json'
    openfoam_path = PERFORMANCE / 'openfoam_timing_100.json'
    timing = json.loads(timing_path.read_text())
    accuracy = json.loads(accuracy_path.read_text())
    measured = json.loads(measured_path.read_text())
    openfoam = json.loads(openfoam_path.read_text())
    with errors_path.open() as stream:
        errors = {(row['model'], int(row['step'])): row for row in csv.DictReader(stream)}
    assert measured['complete'] and measured['protected_existing_files_unchanged']
    assert timing['steps'] == 100 and timing['zero_based_initial_index'] == 2000
    assert measured['physical_horizon_s'] == 1e-4
    assert set(errors) == {(model, step) for model in MODELS for step in range(1, 11)}

    data = dict(
        physical_horizon_s=1e-4, initial_frame_0based=2000, final_frame_0based=2100,
        display_names=DISPLAY, colors=COLORS, coarse_alpha=COARSE_ALPHA,
        mean_error_table_steps=list(TABLE_STEPS), channel_order=list(CHANNELS),
        runtime_statistic='mean of five synchronized full-rollout timings after two warmups',
        runtime_error_bars='sample standard deviation over the five timed repetitions',
        fine_timing_source='archived original figure data; fresh controls are not substituted',
        coarse_timing_source='matched-protocol measurement on the recorded device',
        openfoam_archived_workflow_wall_s=openfoam['window_clock_s'],
        models={}, source_sha256={str(path.relative_to(ROOT)):
                                  hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (timing_path, accuracy_path, measured_path,
                                               errors_path, openfoam_path)})
    for name in MODELS:
        fine_times = timing['models'][name]['times_s']
        coarse = measured['measurements']['1e-5']['models'][name]
        row = errors[(name, 10)]
        channels = [100 * float(row[key]) for key in CHANNELS]
        mean_error = 100 * float(row['mean_channel_relative_L2'])
        assert np.isclose(np.mean(channels), mean_error, rtol=1e-12)
        assert coarse['endpoint_verified_against_saved_results']
        fine_mean = float(np.mean(fine_times))
        data['models'][name] = dict(
            fine=dict(dt_s=1e-6, steps=100, times_s=fine_times, mean_s=fine_mean,
                      std_s=float(np.std(fine_times, ddof=1)),
                      mean_error_percent=100 * accuracy['mean_relative_l2'][name]),
            coarse=dict(dt_s=1e-5, steps=10, times_s=coarse['times_s'],
                        mean_s=coarse['mean_s'], std_s=coarse['std_s'],
                        mean_error_percent=mean_error,
                        mean_errors_percent_by_step=[100 * float(errors[(name, step)][
                            'mean_channel_relative_L2']) for step in range(1, 11)],
                        channel_errors_percent=channels),
            fresh_fine_timing_ratio=measured['measurements']['1e-6_control'][
                'models'][name]['fresh_to_archived_runtime_ratio'],
            runtime_reduction_factor=fine_mean / coarse['mean_s'],
            openfoam_workflow_speedup=openfoam['window_clock_s'] / coarse['mean_s'])
    return data


def draw(data):
    font_files = []  # Use the installed serif fallback; no bundled proprietary fonts.
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
    family = (font_manager.FontProperties(fname=str(font_files[0])).get_name()
              if font_files else 'DejaVu Serif')
    plt.rcParams.update({
        'font.family': family, 'font.size': 12.5,
        'axes.titlesize': 14, 'axes.titlelocation': 'center',
        'axes.linewidth': .85, 'mathtext.fontset': 'stix',
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True, 'savefig.dpi': 300})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.65), gridspec_kw={'wspace': .28})
    positions = np.arange(len(MODELS))
    width = .34
    all_times, all_errors = [], []
    for key, offset, alpha in (('fine', -.19, 1.), ('coarse', .19, COARSE_ALPHA)):
        times = [data['models'][name][key]['mean_s'] for name in MODELS]
        deviations = [data['models'][name][key]['std_s'] for name in MODELS]
        colors = [to_rgba(COLORS[name], alpha) for name in MODELS]
        axes[0].bar(positions + offset, times, yerr=deviations, width=width, color=colors,
                    error_kw={'elinewidth': .8, 'capsize': 2.5, 'capthick': .8}, zorder=3)
        for name, runtime, color in zip(MODELS, times, colors):
            error = data['models'][name][key]['mean_error_percent']
            axes[1].scatter(runtime, error, s=58, facecolor=color,
                            edgecolor='white', linewidth=.55, zorder=4)
            all_times.append(runtime)
            all_errors.append(error)

    axes[0].set_yscale('log')
    axes[0].set_ylim(.1, 10)
    axes[0].set_xticks(positions, [DISPLAY[name] for name in MODELS])
    axes[0].tick_params(axis='x', labelsize=10.5)
    axes[0].set(title='(a) Runtime', ylabel='Wall time (s)')
    axes[0].grid(axis='y', alpha=.18, linewidth=.5, zorder=0)
    step_legend = [Patch(facecolor=to_rgba('#555555', alpha), edgecolor='none',
                         label=rf'$\Delta t=10^{{{exponent}}}\,\mathrm{{s}}$')
                   for exponent, alpha in ((-6, 1.), (-5, COARSE_ALPHA))]
    axes[0].legend(handles=step_legend, frameon=True, fontsize=10.5, loc='upper right',
                   handlelength=1.25, handletextpad=.55)

    axes[1].set_xscale('log')
    log_times = np.log(all_times)
    xpad = .055 * np.ptp(log_times)
    axes[1].set_xlim(np.exp(log_times.min() - xpad), np.exp(log_times.max() + xpad))
    ypad = .07 * np.ptp(all_errors)
    axes[1].set_ylim(min(all_errors) - ypad, max(all_errors) + ypad)
    axes[1].xaxis.set_major_locator(FixedLocator([.2, .5, 1., 2., 5.]))
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set(title='(b) Accuracy–cost', xlabel='Wall time (s)',
                ylabel=r'Mean variable-wise relative $L_2$ error (%)')
    axes[1].grid(alpha=.18, linewidth=.5)
    model_legend = [Line2D([], [], marker='o', linestyle='none', markersize=6.5,
                           markerfacecolor=COLORS[name], markeredgecolor='white',
                           markeredgewidth=.55, label=DISPLAY[name]) for name in MODELS]
    axes[1].legend(handles=model_legend, frameon=True, fontsize=10, ncol=2,
                   loc='upper right',
                   columnspacing=.9, handletextpad=.35)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=CASE / 'analysis/figures')
    parser.add_argument('--data-output', type=Path,
                        default=PERFORMANCE / 'timestep_comparison_data.json')
    parser.add_argument('--coarse-results', type=Path, default=CASE / 'analysis/results_1e-5/lie_10')
    args = parser.parse_args()
    image_path = args.output_dir / 'computational_performance_timestep.png'
    for path in (image_path, args.data_output):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')
    data = collect_data(args.coarse_results / 'channel_errors.csv')
    fig = draw(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, bbox_inches='tight', pad_inches=.05, facecolor='white',
                format='png')
    print(image_path)
    plt.close(fig)
    with args.data_output.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
    print(args.data_output)


if __name__ == '__main__':
    main()
