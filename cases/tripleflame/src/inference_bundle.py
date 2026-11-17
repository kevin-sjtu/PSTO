"""Validate the portable metadata supplied with the published checkpoints.

Unlisted local training runs are deliberately handled by the original training
verification path. A listed checkpoint with a mismatched hash never falls back.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BundleEntry:
    manifest_path: Path
    case_root: Path
    manifest: dict
    entry: dict
    key: str


def read_manifest(path):
    with Path(path).open() as stream:
        manifest = json.load(stream)
    if manifest.get('format') != 'psto-inference-v1':
        raise ValueError(f'Unsupported inference bundle format: {path}')
    return manifest


def relative_asset(case_root, value):
    path = Path(value)
    if path.is_absolute():
        raise ValueError('Inference bundle asset paths must be relative')
    resolved = (case_root / path).resolve()
    try:
        resolved.relative_to(case_root.resolve())
    except ValueError as exc:
        raise ValueError('Inference bundle asset path escapes the case directory') from exc
    return resolved


def verify_digest(path, expected):
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError('Inference bundle requires a SHA256 digest')
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f'Inference bundle SHA256 mismatch: {Path(path).name}')


def find_entry(checkpoint, group):
    checkpoint = Path(checkpoint).resolve()
    for parent in checkpoint.parents:
        manifest_path = parent / 'inference_config.json'
        if not manifest_path.is_file():
            continue
        manifest = read_manifest(manifest_path)
        case_root = manifest_path.parent.parent.resolve()
        entries = manifest.get(group)
        if not isinstance(entries, dict):
            raise ValueError(f'Missing inference bundle group: {group}')
        for key, entry in entries.items():
            if relative_asset(case_root, entry['checkpoint']) == checkpoint:
                verify_digest(checkpoint, entry.get('sha256'))
                return BundleEntry(manifest_path, case_root, manifest, entry, key)
        # The nearby manifest does not claim this independently trained run.
        return None
    return None


def require_close(actual, expected, description):
    if not (math.isfinite(float(actual)) and math.isfinite(float(expected))
            and math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=0.0)):
        raise ValueError(f'Inference bundle {description} mismatch')


def validate_physics(bundle, dt, pressure, mechanism, channel_names,
                     load_channels, dx=None):
    manifest, entry = bundle.manifest, bundle.entry
    require_close(entry['dt'], dt, 'prediction interval')
    require_close(manifest['pressure'], pressure, 'pressure')
    if manifest.get('channel_names') != list(channel_names):
        raise ValueError('Inference bundle physical channel order mismatch')
    if manifest.get('load_channels') != list(load_channels):
        raise ValueError('Inference bundle source channel order mismatch')
    if dx is not None:
        require_close(manifest['grid_spacing'], dx, 'grid spacing')
        require_close(entry['dx'], dx, 'transport grid spacing')
    verify_digest(mechanism, manifest.get('mechanism_sha256'))


def validate_constitutive(bundle, checkpoint):
    constitutive = bundle.manifest['constitutive']
    if (relative_asset(bundle.case_root, bundle.entry['coefficient_checkpoint'])
            != relative_asset(bundle.case_root, constitutive['checkpoint'])):
        raise ValueError('Inference bundle constitutive configuration mismatch')
    if bundle.entry.get('coefficient_backend') != 'dnn':
        raise ValueError('Published inference bundle requires the constitutive DNN')
    verify_digest(checkpoint, constitutive.get('sha256'))


def channel_bounds(bundle, n_channels):
    normalization = bundle.entry['normalization']
    lower = np.asarray(normalization['lower'], dtype=np.float32)
    upper = np.asarray(normalization['upper'], dtype=np.float32)
    if (lower.shape != (n_channels,) or upper.shape != (n_channels,)
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or not np.all(upper > lower)):
        raise ValueError('Inference bundle has invalid channel bounds')
    return lower.reshape(-1, 1, 1), upper.reshape(-1, 1, 1)
