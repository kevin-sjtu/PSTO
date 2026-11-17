"""Check time alignment, species-column mapping and reaction invariants."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from train_source_List import load_reaction_data, transformed_pairs, BCT

spec = importlib.util.spec_from_file_location('lie_1e5', ROOT/'src/test_1e-5.py')
lie = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lie)


class LieTests(unittest.TestCase):
    def test_ten_step_reference_alignment(self):
        np.testing.assert_array_equal(lie.frame_indices(2001, 10, 1e-5, 1e-6, 6001),
                                      np.arange(2000, 2101, 10))
        with self.assertRaises(ValueError):
            lie.frame_indices(2001, 10, 1e-5, 1e-6, 2100)
        with self.assertRaises(ValueError):
            lie.frame_indices(1, 10, 1e-5, 3e-6, 100)

    def test_original_bounds_include_unsampled_reference_frames(self):
        from utils.functions import rollout_channel_bounds
        raw = np.ones((24, 13, 2, 3), dtype=np.float32)
        raw[0, lie.CHANNELS[0], 0, 0] = -999.0  # Outside this sample's window.
        raw[23, lie.CHANNELS[0], 0, 0] = 999.0
        raw[2, lie.CHANNELS[0], 0, 0] = 2.0  # Initial frame.
        raw[7, lie.CHANNELS[0], 0, 0] = 11.0  # Neither initial nor coarse target.
        raw[8, lie.CHANNELS[1], 0, 0] = -3.0
        indices = lie.frame_indices(3, 2, 1e-5, 1e-6, len(raw))
        with tempfile.TemporaryDirectory() as temporary:
            with h5py.File(Path(temporary)/'reference.h5', 'w') as stream:
                data = stream.create_dataset('data', data=raw)
                initial, targets, lower, upper = lie.load_reference_window(data, indices)
        np.testing.assert_array_equal(initial.numpy(), raw[2, lie.CHANNELS])
        np.testing.assert_array_equal(targets.numpy(), raw[[12, 22]][:, lie.CHANNELS])
        full_reference = torch.from_numpy(raw[3:23][:, lie.CHANNELS])[None]
        expected_lower, expected_upper = rollout_channel_bounds(initial[None], full_reference)
        torch.testing.assert_close(lower, expected_lower[0], rtol=0, atol=0)
        torch.testing.assert_close(upper, expected_upper[0], rtol=0, atol=0)
        self.assertEqual(upper[0].item(), 11.0)
        self.assertEqual(lower[1].item(), -3.0)
        self.assertEqual(initial[0].max().item(), 2.0)
        self.assertEqual(targets[:, 0].max().item(), 1.0)

    def test_species_column_mapping(self):
        raw = np.load(ROOT/'data/0d_data_1e-5.npy', mmap_mode='r')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'data.npy'
            np.save(path, raw[:10])
            reduced = load_reaction_data(path)
        x, target = transformed_pairs(reduced)
        np.testing.assert_allclose(x[:, 2:], BCT(raw[:10, [2,3,4,5,6,7,8,9,11]]))
        np.testing.assert_allclose(target, BCT(raw[:10, 24:32])-BCT(raw[:10, 2:10]))

    def test_reaction_dt_and_enthalpy(self):
        saved = dict(dt=1e-5, epoch=1, layers=[11, 400, 400, 400, 400, 1],
                     run_metadata=dict(temperature_model=False, species=list(lie.STATE_SPECIES), pressure=101325.),
                     data_in_mean=np.zeros(11), data_in_std=np.ones(11),
                     data_target_mean=np.array([0.02]+[0.0]*7), data_target_std=np.ones(8))
        network = lie.NN_MLP(saved['layers'])
        for parameter in network.parameters():
            parameter.data.zero_()
        for i in range(8):
            saved[f'net{i}'] = network.state_dict()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)/'species.pt'
            torch.save(saved, checkpoint)
            with self.assertRaisesRegex(ValueError, 'dt'):
                lie.load_source(checkpoint, 1e-6, 101325., str(ROOT/'mech/h2o2.yaml'), 'cpu')
            with self.assertRaisesRegex(ValueError, 'not complete'):
                lie.load_source(checkpoint, 1e-5, 101325., str(ROOT/'mech/h2o2.yaml'), 'cpu')
            (Path(temporary)/'training_complete.json').write_text(json.dumps(dict(
                training_complete=True, reload_verified=True, best_epoch=1, dt=1e-5)))
            reaction = lie.load_source(checkpoint, 1e-5, 101325., str(ROOT/'mech/h2o2.yaml'), 'cpu')
        raw = np.load(ROOT/'data/0d_data_1e-5.npy', mmap_mode='r')
        physical = raw[[0, 44049, 88098]][:, [0,2,3,4,5,6,7,8,9,11]]
        physical = np.concatenate([physical, np.full((3, 2), 2.0)], axis=1)
        x = torch.tensor(physical.T, dtype=torch.float32)[None, :, None, :]
        with torch.inference_mode():
            y = reaction(x)
        self.assertTrue(torch.isfinite(y).all())
        self.assertGreater((y[:, 1:9]-x[:, 1:9]).abs().max().item(), 0)
        torch.testing.assert_close(y[:, -2:], x[:, -2:], rtol=0, atol=0)
        torch.testing.assert_close(y[:, 9], x[:, 9], rtol=0, atol=2e-6)
        torch.testing.assert_close(y[:, 1:10].sum(1), torch.ones_like(y[:, 0]), rtol=0, atol=2e-6)
        h0, _ = reaction.Y_to_T.enthalpy_cp(x[:, 0].reshape(-1), x[:, 1:10].movedim(1, -1).reshape(-1, 9))
        h1, _ = reaction.Y_to_T.enthalpy_cp(y[:, 0].reshape(-1), y[:, 1:10].movedim(1, -1).reshape(-1, 9))
        self.assertLess(((h1-h0).abs()/h0.abs().clamp_min(1e6)).max().item(), 1e-5)

    def test_transport_rejects_partial_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'run_config.json').write_text(json.dumps(dict(arguments=dict(n_iters=40000))))
            with self.assertRaisesRegex(ValueError, 'not completed'):
                lie.load_transport(directory, 'FNO', lie.parser().parse_args([]), 'cpu')
            (directory/'reload_verification.json').write_text(json.dumps(dict(reload_and_forward_passed=True)))
            (directory/'train_val_loss.csv').write_text('step,train_loss,val_loss\n400,0.1,0.1\n')
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                lie.load_transport(directory, 'FNO', lie.parser().parse_args([]), 'cpu')

    def test_input_range_assessment_does_not_change_fields(self):
        initial = torch.tensor([[[[0., 100.]], [[1., 2.]]]])
        before = initial.clone()
        checked = lie.assess_input_range(initial, torch.zeros(2, 1, 1), torch.ones(2, 1, 1))
        self.assertTrue(checked['any_outside_training_range'])
        self.assertEqual(checked['fraction_outside_per_channel'], [0.5, 0.5])
        self.assertEqual(checked['max_abs_normalized_per_channel'], [199., 3.])
        torch.testing.assert_close(initial, before, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
