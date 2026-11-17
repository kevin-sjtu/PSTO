"""Reload a newly trained transport checkpoint and check one paired prediction."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import build_backbone
from paired_data import CHANNELS, PairedFlowDataset, pair_metadata


def verify(directory):
    directory = Path(directory).resolve()
    config = json.loads((directory / 'run_config.json').read_text())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(directory / 'checkpoint_best.pt', map_location=device, weights_only=False)
    assert np.isclose(checkpoint['dt'], 1e-5, rtol=1e-6, atol=0)
    assert checkpoint['run_metadata']['checkpoint_dir'] == str(directory)
    assert checkpoint['run_metadata']['lag_frames'] == 10
    assert np.isfinite(checkpoint['val_loss'])
    meta = pair_metadata(config['data_file'])
    cache = json.loads(Path(config['frame_cache']).with_name('metadata.json').read_text())
    lower = torch.tensor(cache['lower']).reshape(-1, 1, 1)
    upper = torch.tensor(cache['upper']).reshape(-1, 1, 1)
    kind = config['model_type']
    arguments = dict(config['arguments'], dt=config['dt'], dx=config['dx'])
    model = build_backbone(kind, len(CHANNELS), SimpleNamespace(**arguments),
                           data_min=lower, data_max=upper).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    assert np.isclose(model.time_step.item(), 1e-5, rtol=1e-6, atol=0)
    ids = meta['validation'][:1]
    if Path(config['frame_cache']).is_file():
        dataset = PairedFlowDataset(config['frame_cache'], meta['inputs'][ids], meta['targets'][ids],
                                   CHANNELS, lower, upper, normalize_state=kind != 'PSTO')
        x, target = dataset[0]
    else:
        # A completed run needs only one H5 pair for verification, not a rebuilt cache.
        with h5py.File(config['data_file'], 'r') as stream:
            x = torch.from_numpy(stream['frames'][meta['inputs'][ids[0]]][CHANNELS]).float()
            target = torch.from_numpy(stream['frames'][meta['targets'][ids[0]]][CHANNELS]).float()
        if kind != 'PSTO':
            span = (upper-lower).clamp_min(1e-30)
            x, target = 2*(x-lower)/span-1, 2*(target-lower)/span-1
    with torch.no_grad():
        prediction = model(x.unsqueeze(0).to(device))
    assert prediction.shape == (1,) + target.shape
    assert torch.isfinite(prediction).all()
    result = dict(model=kind, checkpoint=str(directory / 'checkpoint_best.pt'),
                  dt=model.time_step.item(), checkpoint_step=checkpoint['step'],
                  validation_loss=checkpoint['val_loss'], prediction_shape=list(prediction.shape),
                  reload_and_forward_passed=True)
    with (directory / 'reload_verification.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print('RELOAD_VERIFIED ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    verify(args.directory)
