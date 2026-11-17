# CDR data preparation

The Burgers velocity benchmark follows the PeRCNN resources. See the authors' [PeRCNN repository](https://github.com/Raocp/PeRCNN), including its dataset instructions, and the [PeRCNN repository associated with the published work](https://github.com/isds-neu/PeRCNN).

The PSTO repository does not redistribute the velocity dataset or generated CD/CDR datasets. Obtain compatible reaction-free velocity data from the upstream resources, then place the input MAT file beside `numer.py` in this directory.

## Required input

The default filename is `Burgers_2001x2x100x100_[FWE,dt=00025].mat`.

The input must be readable by `scipy.io.loadmat` and contain the array `uv` with dimensions `[time, 2, ny, nx]`; the two channels are `u` and `v`. The configuration used in this project is:

| Property | Value |
|---|---|
| Array shape | `(2001, 2, 100, 100)` |
| Grid | Periodic unit square, `100 × 100` |
| Grid spacing | `0.01` |
| Time between stored frames | `2.5e-4 s` |
| Diffusivity | `0.005` |

All values must be finite. The original local input uses double precision. `numer.py` does not infer the frame interval from MAT metadata or resample an upstream trajectory. Check the grid and time interval before using an externally obtained file; renaming a different dataset does not make it compatible.

## Generate CD and CDR data

From the repository root:

```bash
cd cases/cdr
python data/numer.py
```

For a different local input path:

```bash
python data/numer.py --input /path/to/compatible_velocity.mat
```

The default command generates `data/CD.h5` and `data/CDR.h5`. Both store the dataset under the key `data`, with channels ordered as `[u, v, phi]`. CD uses the supplied velocity trajectory; the default CDR generation runs 1,000 steps and stores 1,001 frames. The input MAT and generated HDF5 files remain local and must not be committed.

For attribution, follow the citation instructions in the upstream PeRCNN resources. This preparation guide specifies the input expected by PSTO; it does not assert that an arbitrary upstream download is identical to the original local dataset.
