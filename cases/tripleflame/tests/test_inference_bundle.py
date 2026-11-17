"""CPU integration checks for the shipped weights; no experimental data needed."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

CASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CASE / 'src'), str(CASE)]
import inference_bundle

spec = importlib.util.spec_from_file_location('bundled_coarse_evaluation', CASE / 'src/test_1e-5.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class InferenceBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.manifest = json.loads((CASE / 'checkpoints/inference_config.json').read_text())
        cls.device = torch.device('cpu')
        cls.options = dict(dt=1e-5, dx=1e-4, pressure=101325.,
                           mechanism=str(CASE / 'mech/h2o2.yaml'),
                           coefficient_checkpoint=str(CASE / cls.manifest['constitutive']['checkpoint']))

    def directory(self, name='PSTO'):
        return CASE / 'checkpoints/models_1e-5' / f'model_{name}'

    def load(self, name='PSTO', **changes):
        options = SimpleNamespace(**dict(self.options, **changes))
        return evaluation.load_transport(self.directory(name), name, options, self.device)

    @staticmethod
    def initial_state(size=32):
        values = torch.zeros(1, 12, size, size)
        values[:, 0] = 1000.
        values[:, 1] = .02
        values[:, 4] = .18
        values[:, 9] = .80
        values[:, 10] = 1.
        return values

    def test_transports_load_and_run_without_training_records(self):
        for name in ('PSTO', 'FNO', 'Unet', 'UFNO'):
            with self.subTest(model=name):
                for filename in ('run_config.json', 'train_val_loss.csv', 'reload_verification.json'):
                    self.assertFalse((self.directory(name) / filename).exists())
                model, report = self.load(name)
                self.assertIn('inference_bundle', report)
                with torch.inference_mode():
                    result = model(self.initial_state())
                self.assertEqual(tuple(result.shape), (1, 12, 32, 32))
                self.assertTrue(torch.isfinite(result).all())

    def test_reaction_loads_and_runs_without_completion_record(self):
        path = CASE / self.manifest['reaction']['models_1e-5']['checkpoint']
        self.assertFalse((path.parent / 'training_complete.json').exists())
        source = evaluation.load_source(path, 1e-5, 101325., self.options['mechanism'], self.device)
        values = self.initial_state(4)
        with torch.inference_mode():
            result = source(values)
        self.assertEqual(result.shape, values.shape)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.all(result[:, 0] > 0))
        self.assertTrue(torch.allclose(result[:, 1:10].sum(1), torch.ones_like(result[:, 0]), atol=2e-6))

    def test_checkpoint_hash_mismatch_rejected(self):
        modified = copy.deepcopy(self.manifest)
        modified['transport']['models_1e-5/PSTO']['sha256'] = '0' * 64
        with patch.object(inference_bundle, 'read_manifest', return_value=modified):
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                self.load()

    def test_physics_mismatch_rejected(self):
        for changes in ({'dt': 1e-6}, {'dx': 2e-4}, {'pressure': 202650.}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'mismatch'):
                self.load(**changes)
        modified = copy.deepcopy(self.manifest)
        modified['channel_names'][0:2] = reversed(modified['channel_names'][0:2])
        with patch.object(inference_bundle, 'read_manifest', return_value=modified):
            with self.assertRaisesRegex(ValueError, 'channel order mismatch'):
                self.load()
        modified = copy.deepcopy(self.manifest)
        modified['mechanism_sha256'] = '0' * 64
        with patch.object(inference_bundle, 'read_manifest', return_value=modified):
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                self.load()

    def test_invalid_normalization_rejected(self):
        modified = copy.deepcopy(self.manifest)
        modified['transport']['models_1e-5/PSTO']['normalization']['lower'][0] = float('nan')
        with patch.object(inference_bundle, 'read_manifest', return_value=modified):
            with self.assertRaisesRegex(ValueError, 'invalid channel bounds'):
                self.load()

    def test_constitutive_checkpoint_mismatch_rejected(self):
        wrong_checkpoint = self.directory('FNO') / 'checkpoint_best.pt'
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            self.load(coefficient_checkpoint=str(wrong_checkpoint))

    def test_architecture_mismatch_rejected(self):
        modified = copy.deepcopy(self.manifest)
        modified['transport']['models_1e-5/FNO']['architecture']['hidden_channels'] = 19
        with patch.object(inference_bundle, 'read_manifest', return_value=modified):
            with self.assertRaises(RuntimeError):
                self.load('FNO')

    def test_unlisted_local_run_uses_training_path(self):
        path = CASE / 'checkpoints/new_local_run/model_PSTO/checkpoint_best.pt'
        self.assertIsNone(inference_bundle.find_entry(path, 'transport'))


if __name__ == '__main__':
    unittest.main()
