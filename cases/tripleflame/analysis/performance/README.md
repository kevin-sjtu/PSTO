# Performance measurements

The bundled [runtime](../../figures/computational_performance.png) and
[time-step comparison](../../figures/computational_performance_timestep.png)
PNGs report original experiments. Raw timing records and prediction arrays are
not included. Rerun the benchmarks with the supplied models and local datasets
to measure the current machine.

From `cases/tripleflame`:

```bash
bash analysis/performance/benchmark_gpu_100.sh
# Complete coarse evaluation before this comparison:
bash analysis/performance/benchmark_gpu_timestep_comparison.sh \
  --coarse-results analysis/results_1e-5/lie_10
```

The CUDA protocol uses two warmup rollouts, five synchronized timed rollouts
and two CPU threads over reference frames 2000–2100. The comparison consumes
locally generated fine timing/accuracy and coarse evaluation records.
`--require-device` optionally enforces a particular GPU name.

The CFD benchmark requires compatible Cantera-enabled `reactingFoam`, MPI,
`foamDictionary`, `blockMesh`, GNU `/usr/bin/time`, a local snapshot at
`0.0001 s`, and compatible Cantera XML files:

```bash
START_SNAPSHOT=/path/to/local/0.0001 \
CANTERA_XML=/path/to/local/h2o2.xml \
ELEMENTS_XML=/path/to/local/elements.xml \
NPROCS=128 bash analysis/performance/benchmark_openfoam_mpi.sh
```

It measures decomposition, 1,000 solver updates of `1e-7 s` and reconstruction
at `0.0002 s` in a temporary case. Input copying, mesh preparation and Python
parsing are outside that timing interval. This CFD benchmark has not been
validated during repository preparation. `FOAM_TEMPLATE`, `MPIEXEC` and
`BENCHMARK_OUTPUT` configure paths; `KEEP_WORK=1` retains the temporary case.

After measuring locally, produce PNGs with:

```bash
python -B analysis/performance/plot_computational_performance.py
python -B analysis/performance/plot_timestep_performance.py \
  --coarse-results analysis/results_1e-5/lie_10
```
