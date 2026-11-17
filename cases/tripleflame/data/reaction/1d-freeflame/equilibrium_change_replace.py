import os
import argparse
from pathlib import Path
import re
import cantera as ct
from typing import Dict, Tuple, Optional


# ===== User inputs =====
T_initial = 300.0  # K (premixed inlet temperature)
P_initial = 101325.0  # Pa (kept at 1 atm)

# Y_H2 = 0.014360
# Y_O2 = 0.229753  # ratio=0.5
# Y_H2 = 0.028522
# Y_O2 = 0.226354  # ratio=1
Y_H2 = 0.041876
Y_O2 = 0.223339  # ratio=1.5
# Y_H2 = 0.055066
# Y_O2 = 0.220264 # ratio=2
# Y_H2 = 0.080386
# Y_O2 = 0.214362 # ratio=3


foam0_dir = str(Path(__file__).resolve().parent / '0')
premix_n_cells = 500
mechanism_file = "h2o2.yaml"
dry_run = int(os.environ.get("DRY_RUN", "0")) == 1


def _fmt(x: float) -> str:
    if abs(x) < 5e-16:
        return "0"
    return f"{x:.6g}"


def _validate_premixed_three_species(y_h2: float, y_o2: float) -> Dict[str, float]:
    y_h2 = float(y_h2)
    y_o2 = float(y_o2)
    if y_h2 < 0 or y_o2 < 0:
        raise ValueError("Mass fractions must be non-negative.")
    if y_h2 + y_o2 > 1.0 + 1e-12:
        raise ValueError("Y_H2 + Y_O2 must not exceed 1.")
    y_n2 = 1.0 - y_h2 - y_o2
    if y_n2 < 0 and y_n2 > -1e-12:
        y_n2 = 0.0
    if y_n2 < 0:
        raise ValueError("Computed Y_N2 < 0; please check inputs.")
    return {"H2": y_h2, "O2": y_o2, "N2": y_n2}


def _compute_equilibrium_hp(mech: str, T: float, P: float, premix_Y: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    gas = ct.Solution(mech)
    missing = [sp for sp in premix_Y.keys() if sp not in gas.species_names]
    if missing:
        raise ValueError(f"Species not found in mechanism: {missing}")
    gas.TPY = T, P, premix_Y
    gas.equilibrate("HP")
    Yeq = {sp: float(gas.Y[gas.species_index(sp)]) for sp in gas.species_names}
    return float(gas.T), Yeq


def _find_total_cells_from_nonuniform(lines: list) -> Optional[int]:
    try:
        i_internal = next(i for i, s in enumerate(lines) if s.strip().startswith("internalField"))
    except StopIteration:
        return None

    if "nonuniform" not in lines[i_internal]:
        return None

    j = i_internal + 1
    while j < len(lines) and lines[j].strip() == "":
        j += 1
    try:
        return int(lines[j].strip())
    except Exception:
        return None


def _patch_internal_field(
    lines: list,
    premix_value: float,
    eq_value: float,
    premix_n: int,
    total_n: int,
) -> list:
    try:
        i_internal = next(i for i, s in enumerate(lines) if s.strip().startswith("internalField"))
    except StopIteration:
        raise ValueError("internalField not found")

    if premix_n <= 0 or premix_n >= total_n:
        raise ValueError(f"Invalid premix_n_cells={premix_n} (total_n={total_n})")

    if "nonuniform" in lines[i_internal]:
        j = i_internal + 1
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        try:
            n_total = int(lines[j].strip())
        except Exception as e:
            raise ValueError(f"Cannot parse internalField list length (line {j+1}: {lines[j].strip()})") from e

        if n_total != total_n:
            raise ValueError(f"List length mismatch: file n_total={n_total}, detected total_n={total_n}")

        j += 1
        while j < len(lines) and lines[j].strip() != "(":
            j += 1
        if j >= len(lines):
            raise ValueError("Cannot find '(' for internalField list")
        start = j + 1
        end = start + n_total - 1
        if end >= len(lines):
            raise ValueError("internalField list exceeds file length")

        premix_s = _fmt(premix_value) + "\n"
        eq_s = _fmt(eq_value) + "\n"
        for k in range(n_total):
            lines[start + k] = premix_s if k < premix_n else eq_s
        return lines

    # Case B: internalField is `uniform` -> rewrite as nonuniform so we can express two segments.
    try:
        i_bf = next(i for i, s in enumerate(lines) if s.strip() == "boundaryField")
    except StopIteration as e:
        raise ValueError("boundaryField not found (cannot rewrite uniform internalField)") from e

    premix_s = _fmt(premix_value)
    eq_s = _fmt(eq_value)
    list_block = []
    list_block.append("internalField   nonuniform List<scalar> \n")
    list_block.append(f"{total_n}\n")
    list_block.append("(\n")
    for k in range(total_n):
        list_block.append((premix_s if k < premix_n else eq_s) + "\n")
    list_block.append(")\n")
    list_block.append(";\n")
    list_block.append("\n")

    return lines[:i_internal] + list_block + lines[i_bf:]


def _patch_inlet_boundary(lines: list, inlet_value: float) -> list:
    try:
        i_bf = next(i for i, s in enumerate(lines) if s.strip() == "boundaryField")
    except StopIteration:
        raise ValueError("boundaryField not found")

    # Find inlet block
    i = i_bf
    while i < len(lines) and lines[i].strip() != "inlet":
        i += 1
    if i >= len(lines):
        raise ValueError("inlet patch not found")

    # Find value uniform line within inlet block
    i += 1
    depth = 0
    while i < len(lines):
        s = lines[i]
        if "{" in s:
            depth += s.count("{")
        if "}" in s:
            depth -= s.count("}")
            if depth <= 0:
                break

        if re.search(r"^\s*value\s+uniform\s+", s):
            new_line = re.sub(r"(value\s+uniform\s+)([^;]+)(;)", rf"\g<1>{_fmt(inlet_value)}\3", s)
            lines[i] = new_line
            return lines
        i += 1

    raise ValueError("value uniform line not found inside inlet patch block")


def _update_field_file(path: str, premix_value: float, eq_value: float, premix_n: int, total_n: int) -> None:
    with open(path, "r", encoding="utf-8", errors="strict") as f:
        lines = f.readlines()

    lines = _patch_internal_field(lines, premix_value, eq_value, premix_n, total_n)
    lines = _patch_inlet_boundary(lines, premix_value)

    if dry_run:
        print(f"[DRY_RUN] {os.path.basename(path)}: premix={_fmt(premix_value)}, eq={_fmt(eq_value)}")
        return

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main() -> None:
    premix_Y = _validate_premixed_three_species(Y_H2, Y_O2)

    print("=" * 70)
    print("Premixed (H2/O2/N2) 0D equilibrium (adiabatic HP) + patch OpenFOAM fields")
    print("=" * 70)
    print(f"Premixed inlet Y: H2={premix_Y['H2']:.8f}, O2={premix_Y['O2']:.8f}, N2={premix_Y['N2']:.8f}")
    print(f"Inlet: T={T_initial:.2f} K, P={P_initial/101325:.4f} atm")

    Teq, Yeq = _compute_equilibrium_hp(mechanism_file, T_initial, P_initial, premix_Y)
    print(f"\nEquilibrium temperature Teq: {Teq:.2f} K")
    print("\nEquilibrium mass fractions (sorted desc; filter < 1e-12):")
    print("-" * 70)
    for sp, y in sorted(Yeq.items(), key=lambda kv: kv[1], reverse=True):
        if y >= 1e-12:
            print(f"  {sp:8s}: {y:.8e}")

    # Detect total cells from an existing nonuniform field (e.g. 880)
    total_n = None
    for probe in ("T", "H2", "O2", "H2O", "OH", "HO2", "O", "H"):
        pth = os.path.join(foam0_dir, probe)
        if not os.path.isfile(pth):
            continue
        with open(pth, "r", encoding="utf-8", errors="ignore") as f:
            n = _find_total_cells_from_nonuniform(f.readlines())
        if n is not None:
            total_n = n
            break
    if total_n is None:
        raise ValueError("Cannot detect total cell count from nonuniform field files in foam0_dir.")
    if premix_n_cells >= total_n:
        raise ValueError(f"premix_n_cells={premix_n_cells} must be < total_n={total_n}")

    # Fields to patch (species premix values are only non-zero for H2/O2/N2)
    fields = {
        "H2": ("H2", premix_Y["H2"]),
        "O2": ("O2", premix_Y["O2"]),
        "N2": ("N2", premix_Y["N2"]),
        "H2O": ("H2O", 0.0),
        "H": ("H", 0.0),
        "O": ("O", 0.0),
        "OH": ("OH", 0.0),
        "HO2": ("HO2", 0.0),
        "H2O2": ("H2O2", 0.0),
    }

    # Temperature: premix segment = T_initial, equilibrium segment = Teq
    t_path = os.path.join(foam0_dir, "T")
    if os.path.isfile(t_path):
        _update_field_file(t_path, T_initial, Teq, premix_n_cells, total_n)
    else:
        print(f"WARN: {t_path} not found; skipping temperature patch")

    # Species: premix segment contains only H2/O2/N2 (others set to 0); equilibrium segment uses equilibrium Y.
    for fname, (sp, premix_val) in fields.items():
        fpath = os.path.join(foam0_dir, fname)
        if not os.path.isfile(fpath):
            print(f"WARN: {fpath} not found; skipping")
            continue
        eq_val = float(Yeq.get(sp, 0.0))
        _update_field_file(fpath, premix_val, eq_val, premix_n_cells, total_n)

    print("\nDone: premixed inlet and equilibrium results were written to OpenFOAM field files (p unchanged).")
    if dry_run:
        print("Note: DRY_RUN=1 (no files were modified). Set DRY_RUN=0 to write changes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply the original premixed/equilibrium initialization to a local 0 directory")
    parser.add_argument('--initial-dir', default=foam0_dir)
    parser.add_argument('--mechanism', default=str(Path(__file__).resolve().parent / 'h2o2.yaml'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    foam0_dir, mechanism_file = args.initial_dir, args.mechanism
    dry_run = args.dry_run or dry_run
    main()
