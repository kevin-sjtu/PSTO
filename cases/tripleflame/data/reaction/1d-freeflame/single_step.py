import argparse
from pathlib import Path
import cantera as ct
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory

parser = argparse.ArgumentParser(description="Original constant-pressure single-step reaction utility; input must be a 2-D T/species matrix")
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--mechanism', default=str(Path(__file__).resolve().parent / 'h2o2.yaml'))
parser.add_argument('--processes', type=int, default=64)
args = parser.parse_args()
chem = args.mechanism
gas = ct.Solution(chem)
n_species = gas.n_species
n_process = args.processes
time_step = 5e-7

def single_step(npstate):
    gas = ct.Solution(chem)
    T_old, P_old, Y_old = npstate[0], npstate[1], npstate[2:]
    gas.TPY = T_old, P_old, Y_old
    res_1st = [T_old, P_old] + list(gas.Y) + list(gas.partial_molar_enthalpies/gas.molecular_weights)
    r = ct.IdealGasConstPressureReactor(gas, name='R1')
    sim = ct.ReactorNet([r])
    sim.advance(time_step)
    new_TPY = [gas.T, gas.P] + list(gas.Y) + list(gas.partial_molar_enthalpies/gas.molecular_weights)
    res_1st += new_TPY
    return res_1st

if __name__ == "__main__":
    # Load data
    test = np.load(args.input)
    print(f"Original data shape: {test.shape}")
    print(test[12222, 1:11].sum())
    # Insert constant pressure column (101325 Pa) after the first column (T)
    
    test = np.insert(test, 1, 101325.0, axis=1)
    # Insert zero column for AR species right before the last column (N2)
    # NOTE: np.insert(..., -2, ...) inserts before the *second-to-last* entry, which is wrong here.
    test = np.insert(test, -1, 0.0, axis=1)
    print(test[12222, 2:12].sum())
    print(f"After insert P and AR: {test.shape}")
    # Sanity: AR is a species in h2o2.yaml (order: ... H2O2, AR, N2). It must be exactly 0 here.
    ar_idx = ct.Solution(chem).species_index("AR")
    ar_col = 2 + ar_idx
    print(f"[SANITY] AR index in mechanism={ar_idx}, column in test={ar_col}, max|AR|={float(np.max(np.abs(test[:, ar_col]))):.3e}")

    
    test_n_rows = test.shape[0]
    out_cols = 4 + 4 * n_species
    
    # Create shared memory for output
    shm = shared_memory.SharedMemory(create=True, size=8 * test_n_rows * out_cols)
    cantera_out = np.ndarray((test_n_rows, out_cols), dtype=np.float64, buffer=shm.buf)
    
    rows_per_process = (test_n_rows + n_process - 1) // n_process
    
    def worker(pid):
        start_row = pid * rows_per_process
        end_row = min((pid + 1) * rows_per_process, test_n_rows)
        for i in range(start_row, end_row):
            cantera_out[i] = np.array(single_step(test[i, :2 + n_species]))
    
    # Run multiprocessing
    jobs = []
    pool = mp.Pool(n_process)
    for pid in range(n_process):
        jobs.append(pool.apply_async(worker, (pid,)))
    pool.close()
    pool.join()
    for job in jobs:
        job.get()
    
    print(f"Output shape: {cantera_out.shape}")
    
    # Check AR species columns (should be ~0)
    ar_cols = [10, 32]
    for idx in ar_cols:
        count = np.sum(np.abs(cantera_out[:, idx]) > 1e-10)
        print(f"Column {idx}: {count}/{cantera_out.shape[0]} values > 1e-10")
        count = np.sum(np.abs(cantera_out[:, idx]) > 1e-5)
        print(f"Column {idx}: {count}/{cantera_out.shape[0]} values > 1e-5")
    # Save result
    result = cantera_out.copy()
    # for idx in [10, 20, 32, 42]:
    #     result[:, idx+1] += result[:, idx]
    #     result[:, idx] = 0.0
    
    print(result[1222, :])
    np.save(args.output, result)
    print(f'Saved to {args.output}')
    
    # Cleanup shared memory
    shm.close()
    shm.unlink()

