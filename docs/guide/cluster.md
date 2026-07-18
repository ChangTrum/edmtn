# Running on a cluster (multi-GPU `hpc`)

`backend='hpc'` uses exactly the GPUs/ranks **you launch it across** —
cuTensorNet's only multi-GPU mode is one MPI rank per **distinct
physical** GPU. `edmtn` does not submit jobs, ssh, or call
`srun`/`sbatch` for you; the launch is your workflow. Your script is
unchanged — just `backend='hpc'`.

## Requirements for the distributed path

- one MPI rank per distinct physical GPU;
- a **CUDA-aware MPI runtime** — the distributed contraction runs MPI
  collectives directly on device buffers — plus the cuTensorNet
  distributed-interface library selected through `CUTENSORNET_COMM_LIB`;
- `pathfinder='cuquantum'` (the default); `'cotengra'` is not supported
  multi-rank;
- an `mpi4py` whose MPI ABI matches the launcher you use.

The exact launcher command and any ABI or workaround settings are
**site-specific**: use the recipes under `cluster/` in the repository as
records to adapt, not as portable commands.

## The `cluster/` recipes

| script | status |
|---|---|
| `test_gpu_hpc.sbatch` | the **current test recipe and status record**: three hardware-gated pytest invocations (Track-1 GPU parity, Track-2 single-GPU, multi-GPU) |
| `cutensornet_sanity.sbatch` | single-GPU cuTensorNet validation job |
| `cutensornet_mpi.sbatch`, `cutensornet_multigpu.sbatch` | **legacy/historical** — their `srun --mpi=pmi2` launch matched the site as configured in 2026-06 and is not currently working; kept for reference, and both say so in their headers |

## Current hardware-validation status

- Track-1 single-GPU parity and Track-2 single-GPU cuTensorNet are
  **verified on real hardware**.
- The 4-GPU distributed re-run is currently **blocked by a
  CUDA-aware-MPI regression in the test cluster's environment** — a site
  environment issue, not a defect in the model or the distributed
  pipeline. An older job passed the 4-GPU path historically; that is a
  historical record and does **not** constitute current-environment
  acceptance.

Details and the full diagnosis: {doc}`../design/multi-gpu-cuquantum-design`.

## Hardware acceptance testing

GPU/HPC tests skip with a specific reason when their stack is absent
(see {doc}`../getting-started/installation`). For *acceptance* runs, the
`--require-*` pytest flags turn a missing stack into a non-zero exit, so
an all-skipped run can never masquerade as hardware validation:

```
pytest -q -m "gpu and not cuquantum and not multigpu" --require-gpu
pytest -q -m "cuquantum and not multigpu"             --require-gpu --require-cuquantum
pytest -q -m multigpu --require-gpu --require-cuquantum --require-multigpu=4
```

`--require-multigpu=N` additionally requires the distributed worker's
result JSON, pointed to by the `EDMTN_MULTIGPU_RESULT` environment
variable, to exist and be readable. That variable is an anti-fake-pass
contract of the **test infrastructure** — it is not part of the solver's
public API, and no production code path reads it.
