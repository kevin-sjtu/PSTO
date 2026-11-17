"""Small generated fields test the exporter; no experimental data is used."""
from decimal import Decimal
import gzip
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data'))
from export_fields import export, read_internal_field


class ExportTests(unittest.TestCase):
    def make_frame(self, directory):
        directory.mkdir(parents=True)
        for name, value in zip(('A', 'B', 'C', 'D'), (0.1, 0.2, 0.3, 0.4)):
            (directory / name).write_text(f'FoamFile {{ format ascii; }}\ninternalField uniform {value};\n')
        entries = '\n'.join(f'({i} {-i} 0)' for i in range(8))
        (directory / 'U').write_text(f'internalField nonuniform List<vector>\n8\n(\n{entries}\n);\n')

    def test_cell_order_channels_boundaries_and_time_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_frame(root / '0')
            self.make_frame(root / '1e-4')
            output = root / 'output.h5'
            export(root, output, [Decimal('0'), Decimal('0.0001')], nx=2, ny=4)
            with h5py.File(output) as stream:
                self.assertEqual(stream['data'].shape, (2, 6, 4, 2))
                np.testing.assert_allclose(stream['data'][0, :4, 1, 0], [.1, .2, .3, .4])
                np.testing.assert_array_equal(stream['data'][0, 4], [[2, 3], [2, 3], [4, 5], [4, 5]])
                np.testing.assert_array_equal(stream['data'][0, 5], -stream['data'][0, 4])
                np.testing.assert_allclose(stream['time'][:], [0, .0001])
            with self.assertRaises(FileExistsError):
                export(root, output, [Decimal('0')], nx=2, ny=4)

    def test_compressed_uniform_vector(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'U'
            with gzip.open(str(path) + '.gz', 'wt') as stream:
                stream.write('internalField uniform (1 2 3);')
            np.testing.assert_array_equal(read_internal_field(path, 4, 3), [[1, 2, 3]] * 4)

    def test_malformed_field_does_not_leave_partial_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_frame(root / '0')
            (root / '0/U').write_text('internalField nonuniform List<vector> 7 ((0 0 0));')
            output = root / 'broken.h5'
            with self.assertRaises(ValueError):
                export(root, output, [Decimal('0')], nx=2, ny=4)
            self.assertFalse(output.exists())

    def test_missing_time_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'missing.h5'
            with self.assertRaises(FileNotFoundError):
                export(root, output, [Decimal('0')], nx=2, ny=4)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
