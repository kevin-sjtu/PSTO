"""Generate LF/HF data in a fresh case-local work directory.

Source OpenFOAM 6 and build the supplied pimpletransrobertFoam solver first.
No global OpenFOAM installation or checked-in template is modified.
"""
import argparse
from decimal import Decimal
from pathlib import Path
import shutil
import subprocess

from export_fields import CASE_NAMES, DATA_DIR, END_TIMES, OUTPUT_NAMES, export


def run(command):
    print('+ ' + ' '.join(map(str, command)), flush=True)
    subprocess.run([str(item) for item in command], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=tuple(CASE_NAMES))
    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument('--end-time', help='Override end time for a shorter generation test')
    parser.add_argument('--work-dir', type=Path, help='Fresh directory below this case\'s work/')
    parser.add_argument('--output', type=Path, help='New HDF5 output below this case directory')
    parser.add_argument('--no-export', action='store_true', help='Run CFD without writing HDF5')
    args = parser.parse_args()
    case_root = DATA_DIR.parent.resolve()
    work_root = case_root / 'work'
    work = (args.work_dir or work_root / 'openfoam' / args.dataset).resolve()
    output = (args.output or DATA_DIR / OUTPUT_NAMES[args.dataset]).resolve()
    if not work.is_relative_to(work_root) or work == work_root:
        parser.error('--work-dir must be a subdirectory of this case\'s work/')
    if not output.is_relative_to(case_root):
        parser.error('--output must be below this case directory')
    if args.processes < 1:
        parser.error('--processes must be positive')
    try:
        end = Decimal(args.end_time or END_TIMES[args.dataset])
        interval = Decimal('0.0001')
        if not end.is_finite() or end <= 0 or end % interval:
            raise ValueError
    except (ValueError, ArithmeticError):
        parser.error('--end-time must be a positive multiple of 0.0001 seconds')
    if work.exists():
        parser.error(f'Work directory already exists: {work}; choose a new directory')
    if not args.no_export and output.exists():
        parser.error(f'Output already exists: {output}; choose a new output')
    commands = ['blockMesh', 'checkMesh', 'foamDictionary', 'pimpletransrobertFoam']
    if args.processes > 1:
        commands += ['decomposePar', 'reconstructPar', 'mpirun']
    missing = [command for command in commands if shutil.which(command) is None]
    if missing:
        parser.error('Missing executables: ' + ', '.join(missing))
    template = DATA_DIR / 'openfoam' / CASE_NAMES[args.dataset]
    work.mkdir(parents=True)
    for name in ('0', 'constant', 'system'):
        shutil.copytree(template / name, work / name)
    run(['foamDictionary', work / 'system/controlDict', '-entry', 'endTime', '-set', end])
    run(['blockMesh', '-case', work])
    run(['checkMesh', '-case', work])
    if args.processes == 1:
        run(['pimpletransrobertFoam', '-case', work])
    else:
        run(['foamDictionary', work / 'system/decomposeParDict', '-entry',
             'numberOfSubdomains', '-set', args.processes])
        run(['decomposePar', '-case', work])
        run(['mpirun', '-np', args.processes, 'pimpletransrobertFoam', '-case', work, '-parallel'])
        run(['reconstructPar', '-case', work])
    if not args.no_export:
        times = [index * interval for index in range(int(end / interval) + 1)]
        export(work, output, times)
        print(f'Wrote {output}', flush=True)


if __name__ == '__main__':
    main()
