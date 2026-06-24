"""Phase 3 / P5a benchmark: compression strategy x backend, end-to-end Gaudin fold.

Compares the full SVD (`StandardSVD`) against single-pass and cold randomized SVD
(`RandomizedSVD`, n_iter=0 / 2) on CPU and GPU, measuring GPU-synchronised
wall-clock, accuracy vs the CPU full-SVD reference, and the final bond dimension.
The point is to locate where the GPU + GEMM-based rSVD overtakes the CPU full-SVD
pipeline (the crossover anticipated in docs/cpu-vs-gpu-edm.md and the EDMSolver
backend docstring).

Not collected by pytest (lives under tests/benchmarks/ as perf_*).  Run directly:

    python tests/benchmarks/perf_gpu_compression.py --K 24 --cutoff 1e-6
    python tests/benchmarks/perf_gpu_compression.py --K 24 --cutoff 1e-6 \
        --combos cpu:svd,gpu:svd,gpu:rsvd0,gpu:rsvd2 --repeats 3
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from edmtn.decomposition import RandomizedSVD, StandardSVD
from edmtn.driver.solver import EDMSolver
from edmtn.models import GaudinModel


def _sync(backend):
    """Force completion of queued GPU work so the timer measures real wall-clock."""
    if backend in ("gpu", "cupy"):
        try:
            import cupy as cp  # noqa: PLC0415

            cp.cuda.Device().synchronize()
        except Exception:
            pass


def _to_host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _make_decomp(kind):
    if kind == "svd":
        return lambda: StandardSVD()
    if kind == "rsvd0":
        return lambda: RandomizedSVD(n_iter=0)
    if kind == "rsvd2":
        return lambda: RandomizedSVD(n_iter=2)
    raise ValueError(f"unknown decomposition kind {kind!r}")


def run_combo(model, *, T, eps, order, cutoff, max_bond, backend, kind, repeats):
    """One (backend, decomposition) combo; returns dict or None if backend unavailable."""
    if backend in ("gpu", "cupy"):
        try:
            import cupy as cp  # noqa: PLC0415

            if cp.cuda.runtime.getDeviceCount() == 0:
                return None
        except Exception:
            return None
    make = _make_decomp(kind)
    walls, res = [], None
    for r in range(repeats + 1):              # r=0 is an untimed warm-up
        t0 = time.perf_counter()
        res = EDMSolver.from_model(
            model, T=T, eps=eps, expansion_order=order, cutoff=cutoff,
            max_bond=max_bond, backend=backend, decomposition=make(),
        ).solve(channel=3)
        _sync(backend)
        dt = time.perf_counter() - t0
        if r > 0:
            walls.append(dt)
    return dict(wall=min(walls), pol=_to_host(res.polarization), dmax=int(res.max_bond))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--cutoff", type=float, default=1e-6)
    ap.add_argument("--max-bond", type=int, default=400)
    ap.add_argument("--order", type=int, default=2, choices=(1, 2))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--combos", default="cpu:svd,gpu:svd,gpu:rsvd0,gpu:rsvd2",
                    help="comma list of backend:kind (kind = svd|rsvd0|rsvd2)")
    args = ap.parse_args()

    model = GaudinModel(g=args.g, K=args.K)
    combos = [c.strip().split(":") for c in args.combos.split(",") if c.strip()]
    print(f"GPU compression benchmark (Gaudin): K={args.K}, T={args.T} g^-1, "
          f"eps={args.eps} g^-1, order={args.order}, xi={args.cutoff:g}, repeats={args.repeats}")

    # reference = first cpu:svd combo (or first available)
    ref_pol = None
    rows = []
    for backend, kind in combos:
        r = run_combo(model, T=args.T, eps=args.eps, order=args.order, cutoff=args.cutoff,
                      max_bond=args.max_bond, backend=backend, kind=kind, repeats=args.repeats)
        if r is None:
            print(f"  {backend:>4}:{kind:<6}  -- backend unavailable, skipped")
            continue
        if ref_pol is None and backend in ("cpu", "numpy") and kind == "svd":
            ref_pol = r["pol"]
        rows.append((backend, kind, r))

    if ref_pol is None and rows:
        ref_pol = rows[0][2]["pol"]            # fall back to first available combo

    cpu_svd_wall = next((r["wall"] for b, k, r in rows if b in ("cpu", "numpy") and k == "svd"), None)

    print(f"\n  {'combo':>12} | {'wall(s)':>8} {'vs cpu-svd':>10} | {'max|dSz|':>9} | {'Dmax':>5}")
    print("  " + "-" * 56)
    for backend, kind, r in rows:
        n = min(len(r["pol"]), len(ref_pol))
        err = float(np.max(np.abs(np.asarray(r["pol"][:n]) - np.asarray(ref_pol[:n]))))
        spd = f"{cpu_svd_wall / r['wall']:.2f}x" if cpu_svd_wall else "  --"
        print(f"  {backend + ':' + kind:>12} | {r['wall']:8.2f} {spd:>10} | {err:9.2e} | {r['dmax']:5d}")

    print("\n  (speedup vs CPU full-SVD pipeline; accuracy vs CPU full-SVD reference)")


if __name__ == "__main__":
    main()
