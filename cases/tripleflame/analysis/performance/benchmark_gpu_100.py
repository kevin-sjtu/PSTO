"""Benchmark the current coupled models over planar-jet frames 2000--2100."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import torch


CASE = Path(__file__).resolve().parents[2]
ROOT = CASE.parents[1]
MODELS = ('PSTO', 'FNO', 'Unet', 'UFNO')
OUTPUT = CASE / 'analysis/performance/gpu_timing_100.json'
ACCURACY = CASE / 'analysis/performance/benchmark_accuracy_start2000.json'
START = 2000
STEPS = 100


def channel_errors(prediction, truth):
    numerator = torch.linalg.vector_norm((prediction - truth).double(), dim=(-2, -1))
    denominator = torch.linalg.vector_norm(truth.double(), dim=(-2, -1))
    return torch.where((numerator == 0) & (denominator == 0), 0.,
                       numerator / denominator)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    os.chdir(CASE)
    sys.path[:0] = [str(CASE / 'src'), str(CASE)]
    import test as evaluation
    from model import transport
    from tripleflame import configuration
    from utils.functions import load_transport_state_dict, rollout_channel_bounds

    torch.set_num_threads(2)
    device = torch.device('cuda')
    args = SimpleNamespace(
        hidden_channels=20, latent_per_var=8, n_layers=4, modes=16,
        dt=1e-6, dx=1e-4, coefficient_backend='dnn',
        coefficient_checkpoint=str(CASE / 'checkpoints/models/model_coeff/constitutive_DNN_BCT.pt'),
        mechanism=str(CASE / 'mech/h2o2.yaml'), pressure=101325.)
    channels = configuration()['load_channels']
    with h5py.File(CASE / 'data/H2_tripleflame.h5', 'r') as data:
        window = torch.from_numpy(data['data'][START:START + STEPS + 1]).float()
        window = window[:, channels].to(device)
    initial, truth = window[:1], window[-1:]
    window_min, window_max = rollout_channel_bounds(initial, window[1:].unsqueeze(0))
    lower, upper = window_min[0], window_max[0]
    source = evaluation.build_reaction_source(args, device)

    report = {
        'protocol': '100-step coupled transport-reaction inference',
        'zero_based_initial_index': START, 'steps': STEPS,
        'neural_time_step_s': 1e-6, 'device': torch.cuda.get_device_name(0),
        'warmup_rollouts': 2, 'repeats': 5,
        'timing_excludes': ['checkpoint loading', 'initial state loading'],
        'tensor_computation_device': 'cuda',
        'includes': ['constitutive DNN', 'transport operator', 'reaction DNN',
                     'species constraints', 'enthalpy-conserving temperature recovery'],
        'models': {}}
    accuracy = {
        'start_zero_based': START, 'end_zero_based': START + STEPS, 'steps': STEPS,
        'metric': 'mean_physical_variable_wise_relative_l2', 'mean_relative_l2': {},
        'channel_relative_l2': {}}

    for name in MODELS:
        path = CASE / f'checkpoints/models/model_{name}/checkpoint_best.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state = checkpoint['model']
        net = evaluation.build_backbone(
            name, 12, args, data_min=lower, data_max=upper, state=state).to(device)
        if name == 'PSTO':
            load_transport_state_dict(net, state, strict=True)
        else:
            net.load_state_dict(state, strict=True)
        operator = transport(net.eval()).to(device)
        operator.load_stats({'data_min': lower, 'data_max': upper})
        modules = (operator, source)
        if any(value.device.type != 'cuda' for module in modules
               for value in (*module.parameters(), *module.buffers())):
            raise RuntimeError('Inference parameters and buffers must remain on CUDA')

        def rollout():
            current = initial.clone()
            for _ in range(STEPS):
                current = source(operator(current))
                if current.device.type != 'cuda':
                    raise RuntimeError('Coupled inference left CUDA')
            return current

        with torch.inference_mode():
            for _ in range(2):
                current = rollout()
            torch.cuda.synchronize()
            times = []
            for run in range(5):
                torch.cuda.synchronize()
                start = time.perf_counter()
                current = rollout()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if not torch.isfinite(current).all():
                    raise ValueError(f'{name} produced a nonfinite field')
                times.append(elapsed)
                print(name, run + 1, f'{elapsed:.6f} s', flush=True)
            errors = channel_errors(current, truth)[0].cpu().numpy()
        report['models'][name] = {
            'times_s': times, 'checkpoint': str(path.relative_to(ROOT))}
        accuracy['mean_relative_l2'][name] = float(errors.mean())
        accuracy['channel_relative_l2'][name] = errors.tolist()
        OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
        ACCURACY.write_text(json.dumps(accuracy, indent=2) + '\n')
        del checkpoint, net, operator
        torch.cuda.empty_cache()
    print('GPU_BENCHMARK_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
