"""Phase C0 — cuTensorNet multi-GPU (MPI) distributed-contraction de-risk probe.

Run on c1 across 4×A800 (one rank per GPU):

    srun --mpi=pmi2 --ntasks=4 python examples/cutensornet_mpi_probe.py

Confirms the distributed stack end-to-end: MPI launch + per-rank GPU pinning,
**CUDA-aware MPI** (cuTensorNet's distributed reduction needs it), the
`distributed_reset_configuration` handle binding, and that a distributed
contraction matches the serial `cupy.einsum`. Staged prints isolate the failure
point (a segfault in the CUDA-aware test ⇒ MPI isn't CUDA-aware; a
`CUTENSORNET_STATUS_DISTRIBUTED_FAILURE` ⇒ the MPICH symbol-visibility issue,
needs `LD_PRELOAD=<mpi>/libmpi.so`).

Prereqs (set by the sbatch): `CUTENSORNET_COMM_LIB` → the built MPI wrapper `.so`;
on MPICH, `LD_PRELOAD` the MPI lib so its symbols are global.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from mpi4py import MPI  # initializes MPI


def main() -> int:
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    import cupy as cp
    from cupy.cuda.runtime import getDeviceCount
    ngpu = getDeviceCount()
    device_id = rank % ngpu
    cp.cuda.Device(device_id).use()

    def log(msg):
        print(f"[rank {rank}/{size} dev {device_id}/{ngpu}] {msg}", flush=True)

    comm_lib = os.environ.get("CUTENSORNET_COMM_LIB", "UNSET")
    log(f"MPI up | CUTENSORNET_COMM_LIB={comm_lib} | "
        f"LD_PRELOAD={os.environ.get('LD_PRELOAD', '')}")
    if rank == 0:
        if comm_lib == "UNSET" or not os.path.isfile(comm_lib):
            log("FATAL: CUTENSORNET_COMM_LIB unset or missing — build the wrapper first")
            comm.Abort(1)

    # -- CUDA-aware MPI check (cuTensorNet's slice reduction needs it) ----------
    log(">>> CUDA-aware MPI test (GPU-buffer Allreduce)")
    try:
        x = cp.ones(4, dtype=cp.float64) * (rank + 1)
        comm.Allreduce(MPI.IN_PLACE, x, op=MPI.SUM)
        cp.cuda.Stream.null.synchronize()
        expect = size * (size + 1) // 2
        ok = bool((x == expect).all())
        log(f"    GPU Allreduce -> {x.get()} (expect {expect}); CUDA-aware={ok}")
    except Exception as e:  # noqa: BLE001
        log(f"    GPU Allreduce raised {type(e).__name__}: {e} (MPI may not be CUDA-aware)")

    # -- distributed contraction (NVIDIA example einsum) -----------------------
    from cuquantum.bindings import cutensornet as cutn
    from cuquantum.tensornet import contract, get_mpi_comm_pointer

    expr = "ehl,gj,edhg,bif,d,c,k,iklj,cf,a->ba"
    shapes = [(8, 2, 5), (5, 7), (8, 8, 2, 5), (8, 6, 3), (8,), (6,), (5,),
              (6, 5, 5, 7), (6, 3), (3,)]
    # build on root, broadcast via CPU (safe regardless of CUDA-awareness), then to GPU
    host = ([np.random.default_rng(0).random(s) for s in shapes] if rank == 0
            else [np.empty(s) for s in shapes])
    for h in host:
        comm.Bcast(h, root=0)
    operands = [cp.asarray(h) for h in host]

    log(">>> cutn.create + distributed_reset_configuration")
    handle = cutn.create()
    cutn.distributed_reset_configuration(handle, *get_mpi_comm_pointer(comm))
    log("    distributed_reset_configuration OK")

    log(">>> distributed contract")
    result = contract(expr, *operands, options={"device_id": device_id, "handle": handle})
    log("    contract returned")

    if rank == 0:
        ref = cp.einsum(expr, *operands, optimize=True)
        match = bool(cp.allclose(result, ref))
        err = float(cp.max(cp.abs(result - ref)))
        log(f"RESULT: distributed == cupy.einsum? {match} (max|Δ|={err:.2e})")
    comm.Barrier()
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
