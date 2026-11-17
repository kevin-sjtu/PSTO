from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from paired_data import CHANNELS, PairedFlowDataset, pair_metadata, prepare_cache


class PairedDataTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='psto-pair-test-')
        self.root = Path(self.directory.name)
        self.path = self.root / 'pairs.h5'
        self.frames = np.broadcast_to(
            np.arange(6, dtype=np.float32)[:, None, None, None] +
            np.arange(17, dtype=np.float32)[None, :, None, None], (6, 17, 3, 4)).copy()
        self.frames[5] += 1000
        with h5py.File(self.path, 'w') as f:
            f['frames'] = self.frames
            f['inputs'] = self.frames[:4]
            f['targets'] = self.frames[2:]
            f['input_frame_indices'] = np.arange(4)
            f['target_frame_indices'] = np.arange(2, 6)
            f['splits/train'] = [0, 1]
            f['splits/validation'] = [3]
            f.attrs['target_lag_frames'] = 2
            f.attrs['source_dt_seconds'] = 1e-6
            f.attrs['prediction_dt_seconds'] = 2e-6

    def tearDown(self):
        self.directory.cleanup()

    def test_pair_targets_and_training_only_bounds(self):
        path, saved, meta = prepare_cache(self.path, self.root / 'cache')
        lower = torch.tensor(saved['lower']).view(-1, 1, 1)
        upper = torch.tensor(saved['upper']).view(-1, 1, 1)
        np.testing.assert_array_equal(lower[:, 0, 0], CHANNELS)
        np.testing.assert_array_equal(upper[:, 0, 0], np.array(CHANNELS) + 3)
        dataset = PairedFlowDataset(path, meta['inputs'], meta['targets'], CHANNELS, lower, upper)
        for i in (0, 1, 3):
            x, y = dataset[i]
            np.testing.assert_array_equal(x, self.frames[i, CHANNELS])
            np.testing.assert_array_equal(y, self.frames[i+2, CHANNELS])
        again, _, _ = prepare_cache(self.path, self.root / 'cache')
        self.assertEqual(path, again)

    def test_normalization_round_trip(self):
        path, saved, meta = prepare_cache(self.path, self.root / 'cache')
        lower = torch.tensor(saved['lower']).view(-1, 1, 1)
        upper = torch.tensor(saved['upper']).view(-1, 1, 1)
        dataset = PairedFlowDataset(path, meta['inputs'], meta['targets'], CHANNELS,
                                   lower, upper, normalize_state=True)
        x, y = dataset[1]
        torch.testing.assert_close((x+1)/2*(upper-lower)+lower,
                                   torch.from_numpy(self.frames[1, CHANNELS]))
        torch.testing.assert_close((y+1)/2*(upper-lower)+lower,
                                   torch.from_numpy(self.frames[3, CHANNELS]))

    def test_training_cache_can_be_removed_and_rebuilt(self):
        path, saved, _ = prepare_cache(self.path, self.root/'cache')
        Path(path).unlink()
        self.assertTrue(Path(path).with_name('metadata.json').is_file())
        rebuilt, metadata, _ = prepare_cache(self.path, self.root/'cache')
        self.assertEqual(rebuilt, path)
        self.assertEqual(metadata, saved)
        np.testing.assert_array_equal(np.load(rebuilt, mmap_mode='r'), self.frames)

    def test_inconsistent_time_interval_rejected(self):
        with h5py.File(self.path, 'r+') as f:
            f.attrs['prediction_dt_seconds'] = 1e-6
        with self.assertRaisesRegex(ValueError, 'time interval'):
            pair_metadata(self.path)

    def test_crossing_split_rejected(self):
        with h5py.File(self.path, 'r+') as f:
            f['splits/train'][1] = 2
        with self.assertRaisesRegex(ValueError, 'overlap'):
            pair_metadata(self.path)


if __name__ == '__main__':
    unittest.main()
