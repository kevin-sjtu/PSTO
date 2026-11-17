import random
import argparse
from pathlib import Path
import h5py
import cantera as ct
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory

parser = argparse.ArgumentParser(description="Original CFD-state perturbation and reaction sampling utility; see data/README.md for unresolved provenance")
parser.add_argument('--hf', type=Path, required=True)
parser.add_argument('--mechanism', default=str(Path(__file__).resolve().parent / 'h2o2.yaml'))
parser.add_argument('--formation-enthalpies', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--processes', type=int, default=64)
args = parser.parse_args()
chem = args.mechanism
gas = ct.Solution(chem)
n_species = gas.n_species
n_process = args.processes
time_step = 1e-5

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

def double_step(npstate):
    gas = ct.Solution(chem)
    T_old, P_old, Y_old = npstate[0], npstate[1], npstate[2:]
    gas.TPY = T_old, P_old, Y_old   
    res_1st = [T_old, P_old] + list(gas.Y) + list(gas.partial_molar_enthalpies/gas.molecular_weights)
    r = ct.IdealGasConstPressureReactor(gas, name='R1')
    sim = ct.ReactorNet([r])
    
    sim.advance(time_step)
    new_TPY = [gas.T, gas.P] + list(gas.Y) + list(gas.partial_molar_enthalpies/gas.molecular_weights)
    res_1st += new_TPY

    T_old = gas.T
    P_old = gas.P
    res_2nd = [T_old, P_old]
    Y_old = gas.Y
    res_2nd += list(Y_old) + list(gas.partial_molar_enthalpies/gas.molecular_weights)

    sim.advance(2*time_step)
    new_TPY = [gas.T, gas.P] + list(gas.Y) + list(gas.partial_molar_enthalpies/gas.molecular_weights)
    res_2nd += new_TPY
    return [res_1st, res_2nd]


LOAD_CHANNELS = [9, 0, 4, 5, 1, 3, 2, 6, 7, 8]  # T, H2, H, O, O2, OH, H2O, HO2, H2O2, N2
with h5py.File(args.hf, 'r') as f:
    test = []
    for index in range(1000, 1400, 100):
        test.append(f['data'][index:index+1])
    
test = np.concatenate(test, axis=0)
test = test[:, LOAD_CHANNELS]
# test = test.transpose(0, 2, 1).reshape(-1, test.shape[1])
# Insert constant pressure column (101325 Pa) after the first column (T)
test = np.insert(test, 1, 101325.0, axis=1)
# Insert AR=0 right before the last column (N2). Using -2 would insert before the second-to-last column (wrong).
test = np.insert(test, -1, 0.0, axis=1)
test = test.transpose(0, 2, 3, 1).reshape(-1, test.shape[1])
print(test[11111, :])
print(f"base dataset shape {test.shape}")
# exit()

# ---------------------------------
# generate random data
# ---------------------------------
maxT = np.max(test[:,0])
minT = np.min(test[:,0])
maxP = np.max(test[:,1])
minP = np.min(test[:,1])
maxN2 = np.max(test[:,-1])
minN2 = np.min(test[:,-1])
print(maxT,minT,maxP,minP,maxN2,minN2)

r_n = 1    # random number
test_tmp = np.copy(test)
alpha = 0.1
test_r =  []

for j in range(r_n):
    # progvar = (1.2-(test[:,0]-minT)/(maxT - minT))
    # test_tmp[:, 0] = test_tmp[:,0] + progvar*(maxT - minT)*(2*np.random.random((test_tmp.shape[0])) - 1)*alpha
    
    ### all random on test
    test_tmp[:, 0] = test[:,0] + (maxT - minT)*(2*np.random.random((test.shape[0])) - 1.0)*alpha
    test_tmp[:, 1] = test[:,1] + (maxP - minP)*(2*np.random.random((test.shape[0])) - 1.0)*alpha*5
    test_tmp[:, -1] = test[:,-1] + (maxN2 - minN2)*(2*np.random.random((test.shape[0])) - 1)*alpha

    for i in range(2, test.shape[1] -2):
        test_tmp[:, i] = np.abs(test[:, i])**(1 + (2*np.random.random((test.shape[0])) -1)*alpha)
    
    # ### all random on random
    # test_tmp[:, 0] = test_tmp[:,0] + (maxT - minT)*(2*np.random.random((test_tmp.shape[0])) - 1)*alpha
    # test_tmp[:, -1] = test_tmp[:,-1] + (maxN2 - minN2)*(2*np.random.random((test_tmp.shape[0])) - 1)*alpha

    # for i in range(2, test_tmp.shape[1] -1):
    #     test_tmp[:, i] = test_tmp[:, i]**(1 + (2*np.random.random((test_tmp.shape[0])) -1)*alpha)

    test_tmp[:, 2: -1] = test_tmp[:, 2:-1]/np.sum(test_tmp[:, 2:-1],axis=1)[:,np.newaxis]*(1 - test_tmp[:, -1])[:, np.newaxis]
    test_r.append(test_tmp)
test_r = np.concatenate(test_r, axis=0)
print(f"test_r dataset shape {test_r.shape}")


# randomset2 = test_r.copy()
# test_n_rows = test_r.shape[0]
# shm = shared_memory.SharedMemory(create=True, size =8*test_n_rows*(4+4*n_species))
# cantera_out = np.ndarray((test_r.shape[0],4+4*n_species), dtype=np.float64, buffer=shm.buf)
# rows_per_process = (test_n_rows + n_process - 1)//n_process

# def worker_evolution(ii):
#     for i in range(ii*rows_per_process,(ii+1)*rows_per_process):
#         cantera_out[i] = np.array((single_step(randomset2[i, :2+n_species])))
# pool = mp.Pool(n_process)
# for ii in range(n_process):
#     pool.apply_async(worker_evolution, (ii,))                                                                                                                                                                                                                                                                                 
# pool.close()
# pool.join()
# print(f"single step dataset shape {cantera_out.shape}")
# np.save('dataset_cfd', cantera_out)
# print(cantera_out[2, :])

# shm.close()
# shm.unlink()
# exit()

## -------------------------------------------
## calculate Qdot for CFD data and random data
## -------------------------------------------

formation_enthalpies = np.load(args.formation_enthalpies)

test_n_rows = test.shape[0]
shm1 = shared_memory.SharedMemory(create=True, size =8*test_n_rows)
shm2 = shared_memory.SharedMemory(create=True, size =8*test_n_rows)
qdot_noram0 = np.ndarray((test_n_rows,), dtype=np.float64, buffer=shm1.buf)
qdot_noram1 = np.ndarray((test_n_rows,), dtype=np.float64, buffer=shm2.buf)
rows_per_process = (test_n_rows + n_process - 1)//n_process
def worker_calc_qdot(ii):
    for i in range(ii*rows_per_process, (ii+1)*rows_per_process):
        if i >= test_n_rows:
            continue
        twores_noram = double_step(test[i])
        twores_noram_np0 = np.array(twores_noram[0])
        twores_noram_np1 = np.array(twores_noram[1])
        qdot_noram0[i] = (-(formation_enthalpies*(twores_noram_np0[4+2*n_species:4+3*n_species]-twores_noram_np0[2:2+n_species])/time_step).sum())
        qdot_noram1[i] = (-(formation_enthalpies*(twores_noram_np1[4+2*n_species:4+3*n_species]-twores_noram_np1[2:2+n_species])/time_step).sum())

jobs = []

pool = mp.Pool(n_process)
for ii in range(n_process):
    jobs.append(pool.apply_async(worker_calc_qdot, (ii,)))
pool.close()
pool.join()
for job in jobs:
    job.get()

    
cq = 100 * alpha
    
rows_per_process = (test_r.shape[0] + n_process - 1)//n_process
def worker_evo_sel(ii, res):
    for i in range(ii*rows_per_process, (ii+1)*rows_per_process):
        if i >= test_r.shape[0]:
            break
        twores = double_step(test_r[i])
        twores_np0 = np.array(twores[0])
        twores_np1 = np.array(twores[1])
        # print(f"twores_np0.shape {twores_np0.shape}")
        # print(f"twores_np1.shape {twores_np1.shape}")
        # print(f"formation_enthalpies.shape {formation_enthalpies.shape}")
        
        qdot0 = -(formation_enthalpies*(twores_np0[4+2*n_species:4+3*n_species]-twores_np0[2:2+n_species])/time_step).sum()
        qdot1 = -(formation_enthalpies*(twores_np1[4+2*n_species:4+3*n_species]-twores_np1[2:2+n_species])/time_step).sum()
    
        r_j = divmod(i,test.shape[0])[1]

        if (twores_np0[0] > 800 and twores_np0[0] < 2600 and twores_np0[2+2*n_species] < 2600):
            if(qdot0>1/cq*qdot_noram0[r_j] and qdot0<cq*qdot_noram0[r_j]):
                res.append(twores[0])

        if (twores_np1[0] > 800 and twores_np1[0] < 2600 and twores_np1[2+2*n_species] < 2600):
            if(qdot1>1/cq*qdot_noram1[r_j] and qdot1<cq*qdot_noram1[r_j]):
                res.append(twores[1])

jobs = []

pool = mp.Pool(n_process)
manager = mp.Manager()
res = manager.list()
for ii in range(n_process):
    jobs.append(pool.apply_async(worker_evo_sel, (ii, res)))
pool.close()
pool.join()
for job in jobs:
    job.get()

res = np.array(res)
print(f"output dataset shape {res.shape}")

# Count non-101325 values for p-like channels at axis-1 indices 1 and 23
p_idx_1 = np.count_nonzero(~np.isclose(res[:, 1], 101325.0))
print(f"axis-1 index 1: count(!=101325.0) = {p_idx_1}")
p_idx_23 = np.count_nonzero(~np.isclose(res[:, 23], 101325.0))
print(f"axis-1 index 23: count(!=101325.0) = {p_idx_23}")

np.save(args.output, res)

shm1.close()
shm1.unlink()
shm2.close()
shm2.unlink()
