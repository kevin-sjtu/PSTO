# Third-party notices and references

## OpenFOAM material

The Robertson solver and OpenFOAM configuration files retain their existing source notices. The supplied solver headers identify OpenFOAM Foundation material under GNU GPL version 3 or later. The license text from the source OpenFOAM distribution is preserved in `LICENSES/OpenFOAM-COPYING`.

The OpenFOAM installation itself, its compiled binaries, generated meshes and runtime build products are not bundled. Flame CFD uses additional Cantera-enabled OpenFOAM components described in its data instructions.

## Burgers benchmark

The CDR velocity benchmark refers to the authors' [PeRCNN repository](https://github.com/Raocp/PeRCNN) and [the repository associated with the published work](https://github.com/isds-neu/PeRCNN). Follow their data and citation instructions. No upstream MAT dataset is redistributed in this repository.

## Chemical mechanisms

The hydrogen mechanism files retain the provenance comments present in the supplied files. These identify the hydrogen/oxygen mechanism and conversion sources used by the experiment. Mechanism configurations and their upstream notices are separate from the PSTO model source; no new ownership or project-wide license is assigned to them here.

## Python dependencies and baseline implementations

PyTorch, NumPy, SciPy, h5py, Matplotlib, PyWavelets, pytorch-wavelets, Accelerate and Cantera are installed as dependencies rather than vendored. Their own distributions provide their license notices.

FNO, U-Net and U-FNO implementations are the case-specific files supplied with the experiment. Existing comments are retained; the source records do not establish a complete upstream attribution chain. Additional attribution should be recorded when that provenance is confirmed.

No project-wide open-source license has been selected for the PSTO source in this release preparation.
