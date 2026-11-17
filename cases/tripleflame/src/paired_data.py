"""Explicit temporal pairs, with a shared read-only frame cache for training."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch


CHANNELS = [9, 0, 4, 5, 1, 3, 2, 6, 7, 8, 11, 12]


def pair_metadata(path):
    path = Path(path).resolve()
    with h5py.File(path, 'r') as data:
        inputs = data['input_frame_indices'][:]
        targets = data['target_frame_indices'][:]
        train = data['splits/train'][:]
        validation = data['splits/validation'][:]
        shape = tuple(data['frames'].shape)
        lag = int(data.attrs['target_lag_frames'])
        dt = float(data.attrs['prediction_dt_seconds'])
        if data['inputs'].shape != data['targets'].shape or len(inputs) != len(targets):
            raise ValueError('Inconsistent paired dataset shapes')
        if not np.all(targets - inputs == lag) or inputs.min() < 0 or targets.max() >= shape[0]:
            raise ValueError('Invalid temporal pair indices')
        if len(train) == 0 or len(validation) == 0 or np.intersect1d(train, validation).size:
            raise ValueError('Invalid train/validation split')
        if targets[train].max() > inputs[validation].min():
            raise ValueError('Training and validation temporal intervals overlap')
        if not np.isclose(dt, lag * float(data.attrs['source_dt_seconds']), rtol=1e-12):
            raise ValueError('Inconsistent physical time interval')
    return dict(path=str(path), inputs=inputs, targets=targets, train=train,
                validation=validation, shape=shape, lag=lag, dt=dt)


def prepare_cache(path, cache_root, channels=CHANNELS):
    """Store frames once as a memmap; compute bounds only on the training interval."""
    meta = pair_metadata(path)
    stat = Path(meta['path']).stat()
    start = int(meta['inputs'][meta['train']].min())
    stop = int(meta['targets'][meta['train']].max()) + 1
    signature = dict(version=1, source=meta['path'], size=stat.st_size,
                     mtime_ns=stat.st_mtime_ns, channels=list(channels),
                     training_frame_interval=[start, stop])
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(cache_root).resolve() / key
    directory.mkdir(parents=True, exist_ok=True)
    frame_path, manifest_path = directory / 'frames.npy', directory / 'metadata.json'
    with (directory / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest_path.exists() and frame_path.exists():
            saved = json.loads(manifest_path.read_text())
            cached = np.load(frame_path, mmap_mode='r')
            if saved['signature'] == signature and tuple(cached.shape) == meta['shape']:
                return str(frame_path), saved, meta
            raise ValueError('Existing paired cache has inconsistent metadata')
        temporary = directory / f'frames.{os.getpid()}.building.npy'
        lower = np.full(len(channels), np.inf, dtype=np.float32)
        upper = np.full(len(channels), -np.inf, dtype=np.float32)
        with h5py.File(meta['path'], 'r') as data:
            source = data['frames']
            cached = np.lib.format.open_memmap(temporary, mode='w+',
                                              dtype=source.dtype, shape=source.shape)
            for first in range(0, len(source), 16):
                last = min(first + 16, len(source))
                block = source[first:last]
                if not np.isfinite(block).all():
                    raise ValueError(f'Nonfinite source frame in {first}:{last}')
                cached[first:last] = block
                a, b = max(start, first), min(stop, last)
                if a < b:
                    physical = block[a-first:b-first, channels]
                    lower = np.minimum(lower, physical.min(axis=(0, 2, 3)))
                    upper = np.maximum(upper, physical.max(axis=(0, 2, 3)))
                if first % 512 == 0:
                    print(f'PAIR_CACHE {last}/{len(source)} frames', flush=True)
            cached.flush()
            del cached
        current = Path(meta['path']).stat()
        if (current.st_size, current.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise RuntimeError('Paired source changed while caching')
        temporary.rename(frame_path)
        saved = dict(signature=signature, lower=lower.tolist(), upper=upper.tolist(),
                     dt=meta['dt'], lag=meta['lag'], train_pairs=len(meta['train']),
                     validation_pairs=len(meta['validation']))
        temporary_manifest = directory / f'metadata.{os.getpid()}.building.json'
        temporary_manifest.write_text(json.dumps(saved, indent=2) + '\n')
        temporary_manifest.rename(manifest_path)
        print(f'PAIR_CACHE_READY {frame_path}', flush=True)
    return str(frame_path), saved, meta


class PairedFlowDataset(torch.utils.data.Dataset):
    def __init__(self, frame_path, inputs, targets, channels, lower, upper,
                 normalize_state=False, return_physics=False, pressure_reference=None):
        self.frame_path = frame_path
        self.inputs, self.targets = np.asarray(inputs), np.asarray(targets)
        self.channels = list(channels)
        self.lower, self.upper = lower, upper
        self.normalize_state, self.return_physics = normalize_state, return_physics
        self.pressure_reference = pressure_reference
        self._frames = None

    def __len__(self):
        return len(self.inputs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_frames'] = None
        return state

    def __getitem__(self, index):
        if self._frames is None:
            self._frames = np.load(self.frame_path, mmap_mode='r')
        previous = self._frames[self.inputs[index]]
        following = self._frames[self.targets[index]]
        x = torch.from_numpy(previous[self.channels].copy()).float()
        y = torch.from_numpy(following[self.channels].copy()).float()
        if self.normalize_state:
            span = (self.upper - self.lower).clamp_min(1e-30)
            x = 2 * (x - self.lower) / span - 1
            y = 2 * (y - self.lower) / span - 1
        if not self.return_physics:
            return x, y
        rho, cp, conductivity, mu = torch.from_numpy(previous[-4:].copy()).float().split(1)
        if self.pressure_reference is not None:
            pressure = torch.from_numpy(previous[10:11].copy()).float()
            rho = rho * (self.pressure_reference / pressure)
        return x, y, torch.cat((rho, mu, conductivity / (cp + 1e-12)))


def paired_datasets(path, cache_root, channels, args, accelerator):
    if accelerator.is_main_process:
        prepare_cache(path, cache_root, channels)
    accelerator.wait_for_everyone()
    frame_path, saved, meta = prepare_cache(path, cache_root, channels)
    lower = torch.tensor(saved['lower'], dtype=torch.float32).view(-1, 1, 1)
    upper = torch.tensor(saved['upper'], dtype=torch.float32).view(-1, 1, 1)
    train = meta['train']
    if args.max_pairs > 0 and args.max_pairs < len(train):
        train = train[np.linspace(0, len(train)-1, args.max_pairs).round().astype(int)]

    def dataset(indices):
        return PairedFlowDataset(
            frame_path, meta['inputs'][indices], meta['targets'][indices], channels, lower, upper,
            normalize_state=args.model_type != 'PSTO', return_physics=args.coefficient_backend != 'dnn',
            pressure_reference=args.pressure if args.model_type == 'PSTO' else None)

    args.clipping_stats_scope = 'paired_training_interval'
    args.clipping_stats_frames = saved['signature']['training_frame_interval'][1]
    metadata = dict(data_file=meta['path'], prediction_dt=meta['dt'], lag_frames=meta['lag'],
                    train_pairs=len(train), validation_pairs=len(meta['validation']),
                    bounds_scope=args.clipping_stats_scope, frame_cache=frame_path)
    return dataset(train), dataset(meta['validation']), lower, upper, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', required=True)
    parser.add_argument('--cache_dir', required=True)
    options = parser.parse_args()
    frame_path, saved, _ = prepare_cache(options.data_file, options.cache_dir)
    print(json.dumps(dict(frame_cache=frame_path, metadata=saved), indent=2))
