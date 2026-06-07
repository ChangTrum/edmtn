"""CPU vs GPU benchmark for the Gaudin (separable) EDM evolution.

Unlike the spin-boson hot path (small matrices, where the CPU wins -- see
``perf_cpu_gpu.py``), the Gaudin model grows the EDM bond dimension up to the
hard cutoff ``D_c`` (the paper uses 400).  Those are large dense SVD/QR/contract
calls, which is where the GPU should overtake the CPU -- the whole reason Phase 2
makes the GPU the primary backend for this pipeline.

This script times the end-to-end solve on CPU and GPU across a few bond-dimension
caps and reports the GPU's <S_z(t)> error against the CPU reference, then names
the faster backend at the largest cap.  The driver frees pool blocks after each
sub-bath, so VRAM stays bounded across the K-fold outer loop.

Not a pytest -- the ``perf_`` prefix keeps it out of collection; run directly:
    python tests/benchmarks/perf_gaudin.py
    python tests/benchmarks/perf_gaudin.py --K 49 --T 10 --eps 0.05 --max-bonds 100,200,400
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from edmtn.driver import EDMSolver
from edmtn.models import GaudinModel


def _cupy():
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() == 0:
            return None
    except Exception:
        return None
    return cp


def run_backend(model, backend, *, T, eps, cutoff, max_bond, order):
    t0 = time.perf_counter()
    res = EDMSolver.from_model(
        model, T=T, eps=eps, expansion_order=order,
        cutoff=cutoff, max_bond=max_bond, backend=backend,
    ).solve(channel=3)
    return res, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=30, help="number of bath spins")
    ap.add_argument("--T", type=float, default=8.0, help="total time in g^-1")
    ap.add_argument("--eps", type=float, default=0.05, help="time step in g^-1")
    ap.add_argument("--cutoff", type=float, default=1e-6)
    ap.add_argument("--order", type=int, default=2, choices=(1, 2))
    ap.add_argument("--max-bonds", default="100,200,400", help="comma list of D_c caps")
    ap.add_argument("--quick", action="store_true",
                    help="fast preview (K=12, T=3, eps=0.1, D_c=50,100)")
    args = ap.parse_args()

    cp = _cupy()
    print("CuPy/GPU:", "available" if cp else "NOT available (CPU-only run)")

    if args.quick:
        K, T, eps, caps = 12, 3.0, 0.1, [50, 100]
    else:
        K, T, eps, caps = args.K, args.T, args.eps, [int(s) for s in args.max_bonds.split(",")]
    model = GaudinModel(g=args.g, K=K)

    print(f"\nGaudin K={K}, T={T} g^-1, eps={eps} g^-1, order={args.order}, "
          f"cutoff={args.cutoff}\n")
    print(f"{'D_c':>5} {'Dmax':>6} {'CPU[s]':>9} {'GPU[s]':>9} {'speedup':>8} {'max|dSz|':>10}")

    rows = []
    for D_c in caps:
        cpu_res, cpu_t = run_backend(model, "cpu", T=T, eps=eps, cutoff=args.cutoff,
                                     max_bond=D_c, order=args.order)
        dmax = int(np.max(cpu_res.bond_dims))
        if cp is not None:
            try:
                gpu_res, gpu_t = run_backend(model, "gpu", T=T, eps=eps, cutoff=args.cutoff,
                                             max_bond=D_c, order=args.order)
                err = float(np.max(np.abs(gpu_res.polarization - cpu_res.polarization)))
                speed = cpu_t / gpu_t if gpu_t > 0 else float("nan")
                print(f"{D_c:>5} {dmax:>6} {cpu_t:>9.2f} {gpu_t:>9.2f} {speed:>7.2f}x {err:>10.1e}")
                rows.append((D_c, cpu_t, gpu_t, speed))
            except Exception as exc:  # OOM / backend failure
                print(f"{D_c:>5} {dmax:>6} {cpu_t:>9.2f} {type(exc).__name__:>9} "
                      f"{'-':>8} {'-':>10}")
        else:
            print(f"{D_c:>5} {dmax:>6} {cpu_t:>9.2f} {'(no GPU)':>9} {'-':>8} {'-':>10}")

    if rows:
        D_c, cpu_t, gpu_t, speed = rows[-1]
        winner = "GPU" if gpu_t < cpu_t else "CPU"
        print(f"\nAt D_c={D_c}: {winner} is faster ({speed:.2f}x GPU/CPU). "
              f"GPU overtakes CPU once the bond dimension makes the dense "
              f"linear algebra large enough to amortise launch overhead.")


if __name__ == "__main__":
    main()
