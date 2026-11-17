# CDR data preparation

Obtain compatible Burgers velocity data from the authors' [PeRCNN resources](https://github.com/Raocp/PeRCNN)
and follow their attribution instructions; see also the [repository associated
with the published work](https://github.com/isds-neu/PeRCNN). This repository does
not redistribute the MAT input or generated datasets.

Place `Burgers_2001x2x100x100_[FWE,dt=00025].mat` beside `numer.py`. It must be
readable by `scipy.io.loadmat` and contain finite `uv` values with these properties:

| Property | Required configuration |
|---|---|
| Shape and channels | `(2001, 2, 100, 100)`, ordered `u, v` |
| Grid | Periodic unit square, spacing `0.01` |
| Stored-frame interval | `2.5e-4 s` |
| Diffusivity | `0.005` |

The original input uses double precision. `numer.py` does not infer the time
interval or resample a different upstream trajectory; renaming an incompatible
file is insufficient. An arbitrary upstream download is not asserted to be
identical to the original experiment input.

From `cases/cdr`, run:

```bash
python -B data/numer.py
# Or select a different compatible local path:
python -B data/numer.py --input /path/to/compatible_velocity.mat
```

Outputs are `data/CD.h5` and `data/CDR.h5`, each with HDF5 key `data` and channels
`u, v, phi`. CD uses the supplied velocity trajectory; the default CDR simulation
runs 1,000 steps and stores 1,001 frames. These locally generated files supply
the training and reference inputs for the models.
