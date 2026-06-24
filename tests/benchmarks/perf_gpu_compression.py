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
import os
import time

import numpy as np

from edmtn.decomposition import RandomizedSVD, StandardSVD
from edmtn.driver.solver import EDMSolver
from edmtn.evolution import CholeskyQR
from edmtn.models import GaudinModel


def _blas_name():
    """Best-effort name+version of the BLAS NumPy is linked against (MKL vs OpenBLAS)."""
    try:
        cfg = np.show_config("dicts")            # NumPy >= 2.0
        b = cfg.get("Build Dependencies", {}).get("blas", {})
        return f"{b.get('name', '?')} {b.get('version', '')}".strip()
    except Exception:
        try:
            import numpy.distutils.system_info as si  # noqa: PLC0415

            return "mkl" if si.get_info("mkl") else "openblas?"
        except Exception:
            return "unknown"


def _print_provenance():
    """Record the CPU BLAS / thread / GPU environment so the comparison is auditable."""
    print("  [provenance]")
    print(f"    numpy {np.__version__}  BLAS={_blas_name()}")
    threads = {k: os.environ.get(k) for k in
               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "SLURM_CPUS_PER_TASK")}
    print(f"    threads env: {threads}")
    try:
        import cupy as cp  # noqa: PLC0415

        ndev = cp.cuda.runtime.getDeviceCount()
        name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode() if ndev else "n/a"
        cublas = getattr(cp.cuda, "cublas", None)
        cublas_ver = cublas.getVersion(cublas.create()) if cublas else "?"
        print(f"    cupy {cp.__version__}  cudaRT {cp.cuda.runtime.runtimeGetVersion()} "
              f"cuBLAS {cublas_ver}  devices={ndev} ({name})")
        print("    GPU scope: SINGLE card (sequential fold; multi-GPU + cuQuantum are future work)")
    except Exception:
        print("    cupy: not available (CPU-only run)")


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


def _make_canon(kind):
    if kind in ("householder", "hh", "qr"):
        return lambda: None                       # None -> Householder QR (default)
    if kind in ("cholqr2", "cqr2"):
        return lambda: CholeskyQR(passes=2)
    if kind in ("cholqr", "cholqr1", "cqr1"):
        return lambda: CholeskyQR(passes=1)
    raise ValueError(f"unknown canon kind {kind!r}")


def run_combo(model, *, T, eps, order, cutoff, max_bond, backend, kind, canon, repeats):
    """One (backend, decomposition, canon) combo; None if backend unavailable."""
    if backend in ("gpu", "cupy"):
        try:
            import cupy as cp  # noqa: PLC0415

            if cp.cuda.runtime.getDeviceCount() == 0:
                return None
        except Exception:
            return None
    make_d = _make_decomp(kind)
    make_c = _make_canon(canon)
    walls, res, peak_gb = [], None, 0.0
    for r in range(repeats + 1):              # r=0 is an untimed warm-up
        t0 = time.perf_counter()
        res = EDMSolver.from_model(
            model, T=T, eps=eps, expansion_order=order, cutoff=cutoff,
            max_bond=max_bond, backend=backend, decomposition=make_d(),
            canonicalization=make_c(),
        ).solve(channel=3)
        _sync(backend)
        dt = time.perf_counter() - t0
        if r > 0:
            walls.append(dt)
    if backend in ("gpu", "cupy"):
        try:
            import cupy as cp  # noqa: PLC0415

            peak_gb = cp.get_default_memory_pool().total_bytes() / 2**30
        except Exception:
            peak_gb = 0.0
    return dict(wall=min(walls), pol=_to_host(res.polarization),
                dmax=int(res.max_bond), peak_gb=peak_gb)


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
    ap.add_argument("--canon", default="householder",
                    help="comma list of canon (householder|cholqr2|cholqr1); each is "
                         "crossed with every combo")
    ap.add_argument("--label", default="", help="tag printed in the header (for scaling sweeps)")
    args = ap.parse_args()

    model = GaudinModel(g=args.g, K=args.K)
    combos = [c.strip().split(":") for c in args.combos.split(",") if c.strip()]
    canons = [c.strip() for c in args.canon.split(",") if c.strip()]
    nsites = args.order * int(round(args.T / args.eps))
    tag = f" [{args.label}]" if args.label else ""
    print(f"GPU compression benchmark (Gaudin){tag}: K={args.K}, T={args.T} g^-1, "
          f"eps={args.eps} g^-1, order={args.order}, xi={args.cutoff:g}, "
          f"n_sites={nsites}, repeats={args.repeats}")
    _print_provenance()

    # reference = first cpu:svd + householder combo (or first available)
    ref_pol = None
    rows = []
    for canon in canons:
        for backend, kind in combos:
            r = run_combo(model, T=args.T, eps=args.eps, order=args.order, cutoff=args.cutoff,
                          max_bond=args.max_bond, backend=backend, kind=kind, canon=canon,
                          repeats=args.repeats)
            if r is None:
                print(f"  {backend}:{kind}/{canon}  -- backend unavailable, skipped")
                continue
            if (ref_pol is None and backend in ("cpu", "numpy") and kind == "svd"
                    and canon == "householder"):
                ref_pol = r["pol"]
            rows.append((backend, kind, canon, r))

    if ref_pol is None and rows:
        ref_pol = rows[0][3]["pol"]            # fall back to first available combo

    cpu_svd_wall = next((r["wall"] for b, k, c, r in rows
                         if b in ("cpu", "numpy") and k == "svd"), None)

    print(f"\n  {'combo/canon':>22} | {'wall(s)':>8} {'vs cpu-svd':>10} | "
          f"{'max|dSz|':>9} | {'Dmax':>5} | {'GPU GB':>7}")
    print("  " + "-" * 76)
    for backend, kind, canon, r in rows:
        n = min(len(r["pol"]), len(ref_pol))
        err = float(np.max(np.abs(np.asarray(r["pol"][:n]) - np.asarray(ref_pol[:n]))))
        spd = f"{cpu_svd_wall / r['wall']:.2f}x" if cpu_svd_wall else "  --"
        label = f"{backend}:{kind}/{canon}"
        mem = f"{r['peak_gb']:7.2f}" if r["peak_gb"] else "      -"
        print(f"  {label:>22} | {r['wall']:8.2f} {spd:>10} | {err:9.2e} | "
              f"{r['dmax']:5d} | {mem}")

    print("\n  (speedup vs CPU full-SVD pipeline; accuracy vs CPU full-SVD reference)")


if __name__ == "__main__":
    main()
