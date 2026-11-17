from __future__ import annotations

import argparse
import os

import numpy as np

try:
    import cantera as ct
except ModuleNotFoundError as e:
    raise SystemExit(
        "Cantera is not available in this Python environment.\n"
        "Please activate an environment containing Cantera "
        "or install cantera, then re-run."
    ) from e


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute formation enthalpies from a Cantera mechanism and save as .npy")
    ap.add_argument("--mech", default="./h2o2.yaml", help="Cantera mechanism file (default: ./h2o2.yaml)")
    ap.add_argument("--Tref", type=float, default=298.15, help="Reference temperature in K (default: 298.15)")
    ap.add_argument(
        "--out",
        default="formation_enthalpies.npy",
        help="Output .npy filename (default: formation_enthalpies.npy)",
    )
    args = ap.parse_args()

    gas = ct.Solution(args.mech)

    # J/kmol at Tref
    h0_kmol = np.array([gas.species(s).thermo.h(args.Tref) for s in gas.species_names], dtype=np.float64)
    mw = np.array(gas.molecular_weights, dtype=np.float64)  # kg/kmol

    # Convert to J/kg
    h0_mass = h0_kmol / mw

    np.save(args.out, h0_mass.astype(np.float64))

    print(f"OK: saved -> {os.path.abspath(args.out)}")
    print(f"mechanism: {os.path.abspath(args.mech)}")
    print(f"Tref: {args.Tref} K")
    print(f"n_species: {gas.n_species}")
    print("species order (must match gas.Y order):")
    for i, s in enumerate(gas.species_names):
        print(f"  {i:2d}  {s:8s}  h0={h0_mass[i]:.6e} J/kg")


if __name__ == "__main__":
    main()


