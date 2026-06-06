"""CPU vs GPU, fp32 vs fp64 benchmark for the EDM evolution.

Times the single-bath EDM evolution (the Phase-1 hot path) on four backends --
CPU/GPU x complex128/complex64 -- and reports the error of each against the
CPU/complex128 reference, across a few problem sizes.  Concludes which backend
is the better choice for the Phase-1 spin-boson regime.

Why this comparison matters
---------------------------
The EDM algorithm is ``O(N^2)`` in the number of steps: each step contracts a
kernel MPO over the whole history and recompresses with an SVD sweep.  The
individual matrices are *small* (bond ~ tens, system d^2 = 4), and the O(N^2)
SVD/QR calls are issued sequentially with Python orchestration in between.  That
profile -- many tiny, sequential, latency-bound linear-algebra calls -- is the
worst case for a GPU (per-call launch + synchronisation overhead dominates the
negligible compute), so we expect the CPU to win for Phase-1 sizes; the GPU only
pays off once individual matrices get large.  fp32 roughly halves the work and
memory but caps accuracy near ~1e-6 relative.

Run:
    python examples/benchmark_cpu_gpu.py
    python examples/benchmark_cpu_gpu.py --sizes 100,200,300 --order 2
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from edmtn.cumulants import GaussianCumulantEngine
from edmtn.evolution import SingleBathEvolution
from edmtn.expansion import FirstOrderExpander, SecondOrderExpander
from edmtn.kernels import GaussianKernelEngine
from edmtn.models import SpinBosonModel


def _cupy():
    try:
        import cupy as cp
    except ImportError:
        return None
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            return None
    except Exception:
        return None
    return cp


def _converters(cp):
    """(label, convert, sync) tuples for each available backend/precision."""
    out = [
        ("CPU fp64", None, lambda: None),
        ("CPU fp32", lambda a: np.asarray(a, np.complex64), lambda: None),
    ]
    if cp is not None:
        sync = lambda: cp.cuda.Device().synchronize()
        out += [
            ("GPU fp64", lambda a: cp.asarray(a, cp.complex128), sync),
            ("GPU fp32", lambda a: cp.asarray(a, cp.complex64), sync),
        ]
    return out


def _sz_history(model, res):
    out = np.empty(len(res.times))
    for i, (t, rho) in enumerate(zip(res.times, res.density_matrices)):
        r = rho.get() if hasattr(rho, "get") else np.asarray(rho)
        out[i] = np.trace(model.coupling_operators_at(t)[0] @ r).real
    return out


def _free_gpu(cp):
    """Return all pooled blocks to the driver so runs don't accumulate VRAM."""
    if cp is None:
        return
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def run_one(model, kernel, eps, N, order, cutoff, convert, sync):
    exp = SecondOrderExpander() if order == 2 else FirstOrderExpander()
    engine = SingleBathEvolution(expander=exp)
    sync()
    t0 = time.perf_counter()
    res = engine.run(model, kernel, eps, N, cutoff=cutoff, record_rho=True, convert=convert)
    sync()
    wall = time.perf_counter() - t0
    return res, wall


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="80,160,240", help="comma list of step counts N")
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--order", type=int, default=1, choices=(1, 2))
    ap.add_argument("--cutoff", type=float, default=1e-6)
    ap.add_argument("--J0", type=float, default=0.7)
    args = ap.parse_args()

    cp = _cupy()
    print("CuPy/GPU:", "available" if cp else "NOT available (CPU-only run)")
    if cp is not None:
        # cap the pool so a runaway can't take the whole card / wedge the driver,
        # and free between runs (the O(N^2) sweep allocates many size-classes).
        try:
            cp.get_default_memory_pool().set_limit(fraction=0.5)
        except Exception:
            pass
    backends = _converters(cp)
    sizes = [int(s) for s in args.sizes.split(",")]
    model = SpinBosonModel(J0=args.J0, omega_c=5.0, mu=1.0)

    print(f"\nspin-boson J0={args.J0}, eps={args.eps}, order={args.order}, "
          f"cutoff={args.cutoff}\n")
    header = f"{'N':>5} {'Dmax':>5} " + " ".join(f"{lbl:>20}" for lbl, _, _ in backends)
    print(header)
    print(f"{'':>5} {'':>5} " + " ".join(f"{'wall[s] / max|dSz|':>20}" for _ in backends))

    results = {lbl: [] for lbl, _, _ in backends}
    for N in sizes:
        kernel = GaussianKernelEngine.from_model(model, T=N * args.eps, eps=args.eps,
                                                 order=args.order)
        # warm up each backend (GPU init, kernel compile) on a tiny run
        for lbl, convert, sync in backends:
            run_one(model, kernel, args.eps, 4, args.order, args.cutoff, convert, sync)
            if lbl.startswith("GPU"):
                _free_gpu(cp)

        ref_sz = None
        dmax = None
        cells = []
        for lbl, convert, sync in backends:
            try:
                res, wall = run_one(model, kernel, args.eps, N, args.order, args.cutoff,
                                    convert, sync)
                sz = _sz_history(model, res)
                if ref_sz is None:
                    ref_sz, dmax = sz, max(res.bond_dims)
                err = float(np.max(np.abs(sz - ref_sz))) if ref_sz is not None else 0.0
                results[lbl].append((N, wall, err))
                cells.append(f"{wall:7.2f} / {err:.1e}")
                del res
            except Exception as exc:  # OOM or backend failure: record and continue
                name = type(exc).__name__
                results[lbl].append((N, float("nan"), float("nan")))
                cells.append(f"{name:>20}")
            finally:
                if lbl.startswith("GPU"):
                    _free_gpu(cp)
        print(f"{N:>5} {str(dmax):>5} " + " ".join(f"{c:>20}" for c in cells))

    # -- verdict at the largest size ---------------------------------------
    print("\nVerdict (largest size):")
    big = sizes[-1]
    rows = [(lbl, next(w for n, w, e in results[lbl] if n == big),
             next(e for n, w, e in results[lbl] if n == big)) for lbl, _, _ in backends]
    finite = [r for r in rows if np.isfinite(r[1])]
    best = min(finite, key=lambda r: r[1]) if finite else (None, float("nan"), float("nan"))
    for lbl, w, e in rows:
        if not np.isfinite(w):
            print(f"  {lbl:>10}:    failed (OOM / backend error)")
            continue
        tag = "  <-- fastest" if lbl == best[0] else ""
        print(f"  {lbl:>10}: {w:7.2f} s   max|dSz|={e:.1e}{tag}")
    if best[0] is not None:
        print(f"\nFor the Phase-1 spin-boson regime (small matrices, O(N^2) sequential "
              f"SVDs), the recommended backend is: {best[0]}.")


if __name__ == "__main__":
    main()
