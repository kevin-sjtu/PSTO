"""Run and parse the frame-100-to-200 OpenFOAM 128-rank MPI benchmark."""
import argparse
import json
import re
import gzip
import numpy as np
from pathlib import Path


PAPER = Path(__file__).resolve().parent
OUTPUT = PAPER / 'openfoam_timing_100.json'
START_TIME = 1e-4
END_TIME = 2e-4
DELTA_T = 1e-7
EXPECTED_STEPS = 1000

def validate_fields(directory):
    names = ('T', 'U', 'p', 'phi', 'H2', 'O2', 'H2O', 'OH', 'H', 'O', 'HO2', 'H2O2', 'N2')
    arrays, counts = {}, {}
    for name in names:
        path = directory / name
        if not path.exists():path = directory / (name + '.gz')
        content = gzip.open(path, 'rt').read() if path.suffix == '.gz' else path.read_text()
        match = re.search(r'internalField\s+nonuniform\s+List<[^>]+>\s+(\d+)\s*\((.*?)\)\s*;', content, re.S)
        if not match:raise ValueError(f'Missing nonuniform internal field: {name}')
        count = int(match.group(1))
        values = np.fromstring(match.group(2).replace('(', ' ').replace(')', ' '), sep=' ')
        if values.size != count * (3 if name == 'U' else 1) or not np.isfinite(values).all():
            raise ValueError(f'Incomplete or nonfinite reconstructed field: {name}')
        arrays[name], counts[name] = values, count
    if any(counts[name] != counts['T'] for name in names if name != 'phi'):
        raise ValueError('Reconstructed volume fields have inconsistent cell counts')
    species_sum = sum(arrays[name] for name in names[4:])
    deviation = float(np.abs(species_sum - 1).max())
    if deviation > 1e-4:raise ValueError(f'Invalid reconstructed species sum: {deviation}')
    return {'cell_count': counts['T'], 'surface_flux_count': counts['phi'],
            'all_fields_finite': True, 'species_sum_max_deviation': deviation}


def prepare(case):
    snapshot = case / '0.0001'
    required = ('T.gz', 'H2.gz', 'U.gz', 'p.gz', 'phi.gz')
    if any(not (snapshot / name).is_file() for name in required):
        raise ValueError(f'Incomplete frame-100 snapshot: {snapshot}')
    path = case / 'system/controlDict'
    content = path.read_text()
    edits = {
        'application': 'reactingFoam',
        'startFrom': 'startTime',
        'startTime': '0.0001',
        'endTime': '0.0002',
        'deltaT': '1e-7',
        'adjustTimeStep': 'off',
        'writeControl': 'timeStep',
        'writeInterval': '1000',
    }
    for key, value in edits.items():
        content, count = re.subn(rf'(?m)^(\s*{key}\s+)[^;]+;',
                                 lambda m: m.group(1) + value + ';', content, count=1)
        if count != 1:
            raise ValueError(f'Expected exactly one {key} entry in {path}')
    path.write_text(content)
    thermo = case / 'constant/thermophysicalProperties'
    content = thermo.read_text()
    content, count = re.subn(r'canteraChemistryFile\s+"h2o2\.cti"\s*;',
                             'canteraChemistryFile "openfoam_h2o2.xml";', content, count=1)
    if count != 1 or not (case / 'openfoam_h2o2.xml').is_file():
        raise ValueError('Preconverted Cantera XML mechanism is missing or not selected')
    thermo.write_text(content)
    print(f'FRAME100_MPI_PREPARED {case}', flush=True)


def parse(log, host, postrun_cleanup_error=False, case_dir=None, mpi_exit_code=0,
          output=OUTPUT, n_procs=128, source_case=None):
    times = []
    final_execution = final_clock = None
    current_time = None
    lines = log.read_text(errors='replace').splitlines()
    if postrun_cleanup_error:
        end_positions = [i for i, line in enumerate(lines) if line.strip() == 'End']
        errors = [i for i, line in enumerate(lines) if 'malloc_consolidate()' in line or 'corrupted size vs.' in line]
        if not end_positions or not errors or min(errors) <= max(end_positions):
            raise ValueError('Expected cleanup error only after completed solver End marker')
    for line in lines:
        match = re.match(r'^Time = ([0-9.eE+-]+)\s*$', line)
        if match:
            current_time = float(match.group(1))
            times.append(current_time)
        match = re.search(r'ExecutionTime = ([0-9.eE+-]+) s\s+ClockTime = ([0-9.eE+-]+) s', line)
        if match and current_time is not None:
            final_execution, final_clock = map(float, match.groups())
    if len(times) != EXPECTED_STEPS:
        raise ValueError(f'Expected {EXPECTED_STEPS} solver steps, found {len(times)}')
    if abs(times[0] - (START_TIME + DELTA_T)) > 1e-10 or abs(times[-1] - END_TIME) > 1e-10:
        raise ValueError(f'Wrong solver interval: {times[0]} to {times[-1]}')
    if final_clock is None or final_execution is None or min(final_clock, final_execution) <= 0:
        raise ValueError('Missing final positive MPI solver time')
    if case_dir is None:
        raise ValueError('Three-phase timing requires --case-dir and reconstructed final fields')
    phase_times = {}
    for phase in ('decomposePar', 'mpirun', 'reconstructPar'):
        value = float((case_dir / f'{phase}.time').read_text().strip())
        if value <= 0:
            raise ValueError(f'Invalid wall time for {phase}: {value}')
        phase_times[phase] = value
    final_fields = case_dir / '0.0002'
    required = ('T', 'U', 'p', 'phi', 'H2', 'O2', 'H2O', 'OH', 'H', 'O', 'HO2', 'H2O2', 'N2')
    for field in required:
        candidates = [final_fields / field, final_fields / (field + '.gz')]
        if not any(path.is_file() and path.stat().st_size > 0 for path in candidates):
            raise ValueError(f'Missing reconstructed final field: {field}')
    if not any(line.strip() == 'End' for line in (case_dir / 'reconstruct.log').read_text().splitlines()):
        raise ValueError('reconstructPar did not finish normally')
    result = {
        'solver': 'OpenFOAM reactingFoam',
        'host': host,
        'n_procs': n_procs,
        'parallel_mode': 'MPI',
        'source_case': str(source_case) if source_case else None,
        'start_snapshot': str(case_dir / '0.0001'),
        'start_physical_time_s': START_TIME,
        'end_physical_time_s': END_TIME,
        'internal_time_step_s': DELTA_T,
        'internal_steps_in_window': EXPECTED_STEPS,
        'first_solver_time_s': times[0],
        'last_solver_time_s': times[-1],
        'window_execution_s': final_execution,
        'solver_reported_clock_s': final_clock,
        'phase_wall_times_s': phase_times,
        'window_clock_s': sum(phase_times.values()),
        'timing_scope': 'sum of external wall times for decomposePar, mpirun reactingFoam, and reconstructPar -time 0.0002; excludes Python, input preparation, copies and archival',
        'mpi_exit_code': mpi_exit_code,
        'final_snapshot': str(final_fields),
        'reconstructed_fields': list(required),
        'field_validation': validate_fields(final_fields),
        'postrun_cleanup_error': postrun_cleanup_error,
        'timing_record_precedes_cleanup_error': postrun_cleanup_error,
        'solver_log': str(log),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('prepare', 'parse'))
    parser.add_argument('path', type=Path)
    parser.add_argument('--host', default='unknown')
    parser.add_argument('--postrun-cleanup-error', action='store_true')
    parser.add_argument('--case-dir', type=Path)
    parser.add_argument('--mpi-exit-code', type=int, default=0)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--n-procs', type=int, default=128)
    parser.add_argument('--source-case', type=Path)
    args = parser.parse_args()
    if args.operation == 'prepare':
        prepare(args.path)
    else:
        parse(args.path, args.host, args.postrun_cleanup_error, args.case_dir, args.mpi_exit_code,
              args.output, args.n_procs, args.source_case)
