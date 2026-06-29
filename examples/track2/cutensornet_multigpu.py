"""Phase C1 — multi-GPU hpc solve validation (cuTensorNet distributed).

    srun --mpi=pmi2 --ntasks=4 python examples/cutensornet_multigpu.py

Each rank calls the unified ``solve(backend='hpc')`` (Track 2 = exact 2D); the
2D EDM contraction is distributed across the 4×A800 by cuTensorNet (auto-detected
from the MPI launcher). Rank 0 checks the result against the Track-1 exact fold and
prints the backend label (should show ``/4gpu``) + the optimizer slice count.
"""

from __future__ import annotations

import numpy as np
from mpi4py import MPI  # initializes MPI


def main() -> int:
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    from edmtn.driver import solve
    from edmtn.models import GaudinModel

    K, N, order = 4, 6, 1
    model = GaudinModel(g=1.0, K=K, time_step_order=order)
    res = solve(model, T=N * 0.1, eps=0.1, channel=3, backend="hpc")

    if rank != 0:
        return 0

    # Track-1 exact (uncompressed) reference on rank 0
    from edmtn.evolution.cutensornet import _make_expander
    from edmtn.evolution.separable_bath import SeparableBathEvolution
    from edmtn.kernels.separable_mpo import SeparableKernelEngine
    ke = SeparableKernelEngine.from_model(model, N * 0.1, 0.1)
    ev = SeparableBathEvolution(_make_expander(order)).run(model, ke, 0.1, N, compress=False)
    ref = ev.mps.reduced_density_matrix()

    final = res.density_matrices[-1]
    err = float(np.max(np.abs(final - ref)))
    print(f"backend label : {res.backend}")
    print(f"size (ranks)  : {size}")
    print(f"error_metrics : {res.error_metrics}")
    print(f"<S_z(t)>      : {np.asarray(res.polarization)}")
    print(f"multi-GPU final rho vs Track-1 exact: max|Δ|={err:.2e}  "
          f"{'PASS' if err < 1e-10 else 'FAIL'}")
    return 0 if err < 1e-10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
