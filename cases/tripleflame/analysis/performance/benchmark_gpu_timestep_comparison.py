"""Measure timestep-dependent coupled inference without changing existing results.

The previously generated local 1e-6 s timings remain the figure's fine-step reference. A fresh
fine-step run checks comparability; 1e-5 s timings use the identical workload,
two warmups, five repetitions, and synchronized wall-clock measurements.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import h5py
import numpy as np
import torch


CASE = Path(__file__).resolve().parents[2]
ROOT = CASE.parents[1]
PERFORMANCE = CASE / 'analysis/performance'
MODELS = ('PSTO', 'FNO', 'Unet', 'UFNO')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_status():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,pstate,'
         'temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,memory.used',
         '--format=csv'], capture_output=True, text=True, check=False)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=PERFORMANCE / 'gpu_timing_timestep_comparison.json')
    parser.add_argument('--coarse-results', type=Path, default=CASE / 'analysis/results_1e-5/lie_10')
    parser.add_argument('--require-device', default=None, help='Optional GPU-name constraint; measurements always record the actual device')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    os.chdir(CASE)
    sys.path[:0] = [str(CASE / 'src'), str(CASE)]
    import test as evaluation
    from src.model import transport
    from utils.functions import load_transport_state_dict, rollout_channel_bounds

    spec = importlib.util.spec_from_file_location('coarse_evaluation', CASE / 'src/test_1e-5.py')
    coarse = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(coarse)
    torch.set_num_threads(2)
    torch.manual_seed(88)
    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires an allocated CUDA GPU')
    device = torch.device('cuda')
    if args.require_device and torch.cuda.get_device_name(0) != args.require_device:
        raise RuntimeError('Allocated GPU does not match --require-device')

    archived_timing = json.loads((PERFORMANCE / 'gpu_timing_100.json').read_text())
    archived_accuracy = json.loads((PERFORMANCE / 'benchmark_accuracy_start2000.json').read_text())
    coarse_results = args.coarse_results
    with (coarse_results / 'channel_errors.csv').open() as stream:
        endpoint_rows = {row['model']: row for row in csv.DictReader(stream)
                         if int(row['step']) == 10}

    protected = [PERFORMANCE / 'gpu_timing_100.json',
                 PERFORMANCE / 'benchmark_accuracy_start2000.json',
                 PERFORMANCE / 'benchmark_gpu_100.py',
                 PERFORMANCE / 'plot_computational_performance.py']
    protected += [CASE / 'src' / name for name in
                  ('test.py', 'test_1e-5.py', 'model.py', 'PSTO.py', 'thermodynamics.py')]
    protected += [CASE / f'checkpoints/{folder}/model_{name}/checkpoint_best.pt'
                  for folder in ('models', 'models_1e-5') for name in MODELS]
    protected += [CASE / f'checkpoints/{folder}/model_DNN/dnn_model_Y.pt'
                  for folder in ('models', 'models_1e-5')]
    protected.append(CASE / 'checkpoints/models/model_coeff/constitutive_DNN_BCT.pt')
    before = {str(path.relative_to(ROOT)): digest(path) for path in protected}

    with h5py.File(CASE / 'data/H2_tripleflame.h5', 'r') as stream:
        window = torch.from_numpy(stream['data'][2000:2101]).float()
        window = window[:, coarse.CHANNELS].to(device)
    initial, truth = window[:1], window[-1:]
    lower, upper = rollout_channel_bounds(initial, window[1:].unsqueeze(0))
    lower, upper = lower[0], upper[0]
    report = dict(
        protocol='Matched-horizon coupled transport-reaction inference',
        zero_based_initial_index=2000, zero_based_final_index=2100,
        physical_horizon_s=1e-4, device=torch.cuda.get_device_name(0),
        host=os.uname().nodename, slurm_job_id=os.getenv('SLURM_JOB_ID'),
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        warmup_rollouts=2, repeats=5, cpu_threads=2,
        normalization_frame_interval_inclusive=[2000, 2100],
        normalization_frame_stride=1,
        includes=['constitutive DNN where used', 'transport operator', 'reaction DNN',
                  'species constraints', 'enthalpy-conserving temperature recovery'],
        excludes=['checkpoint and data loading', 'error and conservation diagnostics',
                  'host transfers and prediction archival'],
        gpu_status_before=gpu_status(), measurements={})

    for tag, dt, steps, folder in (
            ('1e-6_control', 1e-6, 100, 'models'),
            ('1e-5', 1e-5, 10, 'models_1e-5')):
        args = SimpleNamespace(
            hidden_channels=20, latent_per_var=8, n_layers=4, modes=16,
            dt=dt, dx=1e-4, coefficient_backend='dnn',
            coefficient_checkpoint=str(CASE / 'checkpoints/models/model_coeff/constitutive_DNN_BCT.pt'),
            mechanism=str(CASE / 'mech/h2o2.yaml'), pressure=101325.)
        source_path = CASE / f'checkpoints/{folder}/model_DNN/dnn_model_Y.pt'
        source = (evaluation.build_reaction_source(args, device) if dt == 1e-6 else
                  coarse.load_source(source_path, dt, args.pressure, args.mechanism, device))
        group = dict(dt_s=dt, steps=steps, source_checkpoint=str(source_path.relative_to(ROOT)),
                     source_sha256=digest(source_path), models={})
        for name in MODELS:
            path = CASE / f'checkpoints/{folder}/model_{name}/checkpoint_best.pt'
            if dt == 1e-5:
                operator, _ = coarse.load_transport(path.parent, name, args, device)
            else:
                saved = torch.load(path, map_location='cpu', weights_only=False)
                net = evaluation.build_backbone(
                    name, 12, args, data_min=lower, data_max=upper, state=saved['model']).to(device)
                if name == 'PSTO':
                    load_transport_state_dict(net, saved['model'], strict=True)
                else:
                    net.load_state_dict(saved['model'], strict=True)
                operator = transport(net.eval()).to(device)
                del saved, net
            operator.load_stats({'data_min': lower, 'data_max': upper})
            if any(value.device.type != 'cuda' for module in (operator, source)
                   for value in (*module.parameters(), *module.buffers())):
                raise RuntimeError('All inference parameters and buffers must stay on CUDA')

            def rollout():
                current = initial.clone()
                for _ in range(steps):
                    current = source(operator(current))
                    if current.device.type != 'cuda':
                        raise RuntimeError('Coupled inference left CUDA')
                return current

            with torch.inference_mode():
                for _ in range(2):
                    current = rollout()
                torch.cuda.synchronize()
                times = []
                for repeat in range(5):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    current = rollout()
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    times.append(elapsed)
                    print(tag, name, repeat + 1, f'{elapsed:.6f} s', flush=True)
                if not torch.isfinite(current).all():
                    raise ValueError('Nonfinite prediction')
                errors = coarse.relative_errors(current, truth).cpu().numpy()
            expected = (np.array(archived_accuracy['channel_relative_l2'][name])
                        if dt == 1e-6 else
                        np.array([float(endpoint_rows[name][key]) for key in evaluation.CHANNEL_NAMES]))
            max_error_difference = float(np.max(np.abs(errors - expected)))
            if not np.allclose(errors, expected, rtol=1e-4, atol=2e-6):
                raise AssertionError(f'{tag} {name}: saved-result mismatch {max_error_difference}')
            result = dict(
                times_s=times, mean_s=float(np.mean(times)), std_s=float(np.std(times, ddof=1)),
                checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=digest(path),
                channel_relative_l2=errors.tolist(), mean_relative_l2=float(errors.mean()),
                max_channel_error_difference_from_archived=max_error_difference,
                endpoint_verified_against_saved_results=True)
            if dt == 1e-6:
                archived_mean = float(np.mean(archived_timing['models'][name]['times_s']))
                result.update(archived_mean_s=archived_mean,
                              fresh_to_archived_runtime_ratio=result['mean_s'] / archived_mean)
            group['models'][name] = result
            del operator, current
            torch.cuda.empty_cache()
        report['measurements'][tag] = group
        del source
        torch.cuda.empty_cache()

    after = {str(path.relative_to(ROOT)): digest(path) for path in protected}
    if before != after:
        raise AssertionError('A protected existing file changed during benchmarking')
    report.update(gpu_status_after=gpu_status(), protected_existing_files_sha256=after,
                  protected_existing_files_unchanged=True, complete=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(f'TIMESTEP_BENCHMARK_COMPLETE {output}', flush=True)


if __name__ == '__main__':
    main()
