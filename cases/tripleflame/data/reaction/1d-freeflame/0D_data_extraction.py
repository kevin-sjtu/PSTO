from __future__ import annotations

import argparse
from pathlib import Path
import gzip
import os
import re
from typing import List, Tuple

import numpy as np


VARS: List[str] = ['T', 'p', 'H2', 'H', 'O', 'O2', 'OH', 'H2O', 'HO2', 'H2O2', 'AR', 'N2']


def is_time_dir(name: str) -> bool:
    # OpenFOAM time directories are usually float-like strings (decimal or scientific notation).
    try:
        float(name)
        return True
    except Exception:
        return False


def list_time_dirs(case_dir: str) -> List[str]:
    names = []
    for n in os.listdir(case_dir):
        p = os.path.join(case_dir, n)
        if os.path.isdir(p) and is_time_dir(n):
            names.append(n)
    names.sort(key=lambda s: float(s))
    return names


_float_re = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _parse_float(line: str) -> float:
    s = line.strip()
    if not s:
        raise ValueError("empty")
    # Strip possible trailing semicolons.
    s = s.rstrip(";")
    if s in ("(", ")", ");", ")", ");"):
        raise ValueError("non-numeric token")
    if not _float_re.match(s):
        raise ValueError(f"not-float: {s}")
    return float(s)


def _read_gz_text_lines(path: str) -> List[str]:
    with gzip.open(path, mode="rt", encoding="utf-8", errors="ignore", newline="") as f:
        return f.readlines()


def _parse_internal_field_values(lines: List[str], path_hint: str) -> Tuple[str, np.ndarray]:

    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("internalField"):
            # uniform
            m = re.search(r"internalField\s+uniform\s+([^;]+)\s*;", s)
            if m:
                v = _parse_float(m.group(1))
                return "uniform", np.asarray([v], dtype=np.float64)

            # nonuniform
            if "nonuniform" in s:
                # For nonuniform fields, the next non-empty line should be the list length (N).
                j = i + 1
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                try:
                    n_total = int(lines[j].strip())
                except Exception as e:
                    raise ValueError(
                        f"{path_hint}: cannot parse nonuniform list length "
                        f"(line: {j+1} -> {lines[j].strip()})"
                    ) from e

                j += 1
                while j < len(lines) and lines[j].strip() != "(":
                    j += 1
                if j >= len(lines):
                    raise ValueError(f"{path_hint}: cannot find '(' for nonuniform list")
                start = j + 1
                end = start + n_total
                if end > len(lines):
                    raise ValueError(f"{path_hint}: nonuniform list exceeds file length (N={n_total})")

                vals = []
                for k in range(n_total):
                    vals.append(_parse_float(lines[start + k]))
                return "nonuniform", np.asarray(vals, dtype=np.float64)

            if s.startswith("internalField") and "nonuniform" not in s:
                if i + 1 < len(lines) and "nonuniform" in lines[i + 1]:
                    lines2 = lines[:]
                    lines2[i] = (lines[i].rstrip("\n") + " " + lines[i + 1].lstrip())
                    del lines2[i + 1]
                    return _parse_internal_field_values(lines2, path_hint)

            raise ValueError(f"{path_hint}: unrecognized internalField format (line: {i+1} -> {s})")
        i += 1

    raise ValueError(f"{path_hint}: internalField not found")


def extract_internal_field_segment(path: str, start_idx: int, end_idx: int) -> np.ndarray:

    if start_idx < 1 or end_idx < start_idx:
        raise ValueError("invalid index range")

    lines = _read_gz_text_lines(path)
    mode, vals = _parse_internal_field_values(lines, path)
    n_points = end_idx - start_idx + 1

    if mode == "uniform":
        return np.full((n_points,), float(vals[0]), dtype=np.float64)

    # nonuniform
    if end_idx > len(vals):
        raise ValueError(f"{path}: nonuniform length {len(vals)} < end_idx {end_idx}")
    # Convert to 0-based slice
    return vals[start_idx - 1 : end_idx]


def build_samples(
    base_dir: str,
    folders: List[str],
    start_idx: int,
    end_idx: int,
) -> np.ndarray:

    samples: List[np.ndarray] = []

    n_points = end_idx - start_idx + 1

    for folder in folders:
        case_dir = os.path.join(base_dir, folder)
        if not os.path.isdir(case_dir):
            raise FileNotFoundError(f"Directory not found: {case_dir}")

        for t in list_time_dirs(case_dir):
            t_dir = os.path.join(case_dir, t)
            arr_vars = np.empty((len(VARS), n_points), dtype=np.float64)
            for vi, v in enumerate(VARS):
                fpath = os.path.join(t_dir, f"{v}.gz")
                if not os.path.isfile(fpath):
                    raise FileNotFoundError(f"Missing file: {fpath}")
                arr_vars[vi, :] = extract_internal_field_segment(fpath, start_idx, end_idx)

            samples.append(arr_vars)

    data = np.stack(samples, axis=0) if samples else np.empty((0, len(VARS), n_points), dtype=np.float64)
    return data


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract internalField segments from multiple 1d-freeflame folders/time steps and save as 0d_data.npy"
    )
    ap.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parent),
        help="Base directory of 1d-freeflame (contains subfolders like 0.5/ 1/ 2/ 3/ 4/)",
    )
    ap.add_argument(
        "--folders",
        nargs="+",
        default=["0.5", "1", "2", "3", "4"],
        help="Subfolders to process (default: 0.5 1 2 3 4)",
    )
    ap.add_argument("--start-idx", type=int, default=521, help="Start internalField point index (1-based, inclusive)")
    ap.add_argument("--end-idx", type=int, default=650, help="End internalField point index (1-based, inclusive)")
    ap.add_argument('--output', type=Path, default=None, help='Defaults to <base-dir>/0d_data.npy')
    args = ap.parse_args()

    data = build_samples(args.base_dir, args.folders, args.start_idx, args.end_idx)

    out_path = str(args.output.resolve()) if args.output else os.path.join(args.base_dir, '0d_data.npy')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, data.astype(np.float32))

    print(f"OK: data saved -> {out_path}")
    print(f"    shape = {data.shape} (samples, vars={len(VARS)}, points={args.end_idx - args.start_idx + 1})")


if __name__ == "__main__":
    main()