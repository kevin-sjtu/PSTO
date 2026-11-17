"""Export Robertson OpenFOAM ASCII fields without loading a full trajectory.

The channel order, x-fast cell ordering and top/bottom zero-gradient copy match
the original exporters. LF and HF have separate output names and durations.
"""
import argparse
from decimal import Decimal, InvalidOperation
import gzip
from pathlib import Path
import re

import h5py
import numpy as np


DATA_DIR = Path(__file__).resolve().parent
CASE_NAMES = {'lf': '2Dtranscounterflow', 'hf': '2Dtransrobertcounterflow'}
OUTPUT_NAMES = {'lf': 'Data_trans.h5', 'hf': 'Data_transrobert.h5'}
END_TIMES = {'lf': '0.5', 'hf': '0.4'}
CHANNELS = ('A', 'B', 'C', 'D', 'u', 'v')


def read_internal_field(path, cells, components=1):
    """Read uniform/nonuniform scalar or vector internal fields (ASCII only)."""
    path = Path(path)
    if not path.is_file() and path.with_name(path.name + '.gz').is_file():
        path = path.with_name(path.name + '.gz')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        text = stream.read()
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    if re.search(r'\bformat\s+binary\s*;', text):
        raise ValueError(f'{path}: binary fields are unsupported; use writeFormat ascii')
    uniform = re.search(r'\binternalField\s+uniform\s+([^;]+);', text)
    if uniform:
        values = np.fromstring(uniform.group(1).translate(str.maketrans('', '', '()')), sep=' ')
        if values.size != components:
            raise ValueError(f'{path}: expected {components} uniform components')
        result = np.broadcast_to(values, (cells, components)).copy()
    else:
        field = re.search(
            r'\binternalField\s+nonuniform\s+\S+\s+(\d+)\s*\((.*?)\)\s*;',
            text, flags=re.S)
        if field is None or int(field.group(1)) != cells:
            raise ValueError(f'{path}: missing internal field or unexpected cell count')
        values = np.fromstring(field.group(2).translate(str.maketrans('', '', '()')), sep=' ')
        if values.size != cells * components:
            raise ValueError(f'{path}: expected {cells * components} values, got {values.size}')
        result = values.reshape(cells, components)
    if not np.isfinite(result).all():
        raise ValueError(f'{path}: nonfinite field values')
    return result.astype(np.float32)


def read_frame(directory, nx, ny):
    frame = np.empty((6, ny, nx), dtype=np.float32)
    for channel, name in enumerate(('A', 'B', 'C', 'D')):
        frame[channel] = read_internal_field(directory / name, nx * ny).reshape(ny, nx)
    velocity = read_internal_field(directory / 'U', nx * ny, 3).reshape(ny, nx, 3)
    frame[4:] = velocity[..., :2].transpose(2, 0, 1)
    # Retain the original exporter convention for all six channels.
    frame[:, 0, :] = frame[:, 1, :]
    frame[:, -1, :] = frame[:, -2, :]
    return frame


def time_directories(case_dir):
    result = {}
    for path in case_dir.iterdir():
        if not path.is_dir():
            continue
        try:
            value = Decimal(path.name)
        except InvalidOperation:
            continue
        if not value.is_finite():
            continue
        if value in result:
            raise ValueError(f'Duplicate numeric time directories: {result[value]} and {path}')
        result[value] = path
    return result


def export(case_dir, output, times, nx=200, ny=100):
    if nx < 1 or ny < 3 or not times:
        raise ValueError('Require nx >= 1, ny >= 3 and at least one time')
    directories = time_directories(case_dir)
    missing = [str(t) for t in times if t not in directories]
    if missing:
        raise FileNotFoundError(f'Missing OpenFOAM times (first ten): {missing[:10]}')
    if output.exists():
        raise FileExistsError(f'Refusing to replace existing output: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids overwriting an existing LF/HF dataset.
    stream = h5py.File(output, 'x')
    try:
        with stream:
            data = stream.create_dataset('data', (len(times), 6, ny, nx), dtype='float32',
                                         chunks=(1, 6, ny, nx), compression='gzip')
            stream.attrs['channels'] = ','.join(CHANNELS)
            stream.attrs['cell_order'] = 'x-fast; reshape(ny,nx)'
            stream.create_dataset('time', data=np.array([float(t) for t in times]))
            for index, value in enumerate(times):
                data[index] = read_frame(directories[value], nx, ny)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return output


def main(default_dataset=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(CASE_NAMES), default=default_dataset or 'lf')
    parser.add_argument('--case-dir', type=Path,
                        help='Reconstructed OpenFOAM case; default: work/openfoam/<dataset>')
    parser.add_argument('--output', type=Path, help='New HDF5 path; existing files are not overwritten')
    parser.add_argument('--start', default='0')
    parser.add_argument('--end', help='Default LF 0.5, HF 0.4 seconds')
    parser.add_argument('--dt', default='0.0001', help='Saved-frame interval in seconds')
    parser.add_argument('--times', nargs='+', help='Explicit increasing times instead of start/end/dt')
    parser.add_argument('--nx', type=int, default=200)
    parser.add_argument('--ny', type=int, default=100)
    args = parser.parse_args()
    try:
        if args.times:
            times = [Decimal(t) for t in args.times]
        else:
            start, end, dt = (Decimal(args.start), Decimal(args.end or END_TIMES[args.dataset]),
                              Decimal(args.dt))
            if not all(t.is_finite() for t in (start, end, dt)) or dt <= 0 or end < start or (end - start) % dt:
                raise ValueError('Require dt > 0 and an integer number of saved-frame intervals')
            times = [start + i * dt for i in range(int((end - start) / dt) + 1)]
        if any(not t.is_finite() for t in times) or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError('Times must be finite and strictly increasing')
    except (ArithmeticError, ValueError) as error:
        parser.error(str(error))
    case_dir = (args.case_dir or DATA_DIR.parent / 'work/openfoam' / args.dataset).resolve()
    output = (args.output or DATA_DIR / OUTPUT_NAMES[args.dataset]).resolve()
    export(case_dir, output, times, args.nx, args.ny)
    print(f'Wrote {output}: {len(times)} frames, channels {CHANNELS}')


if __name__ == '__main__':
    main()
