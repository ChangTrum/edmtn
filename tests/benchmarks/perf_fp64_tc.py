"""FP64 Tensor-Core (DMMA) probe -- are FP64 / complex128 GEMMs using tensor cores?

Phase-5 item 5.0 / the recommended-config open question.  On A100/A800 the FP64
tensor-core (DMMA) peak is ~2x the FP64 CUDA-core peak (A800: ~19.5 vs ~9.7
TFLOP/s).  So if a large **DGEMM** sustains well above the CUDA-core peak, cuBLAS is
dispatching it to DMMA.  We then ask whether **ZGEMM** (complex128 -- what the rSVD
GEMMs actually are) gets the same benefit, which decides whether a real/imag split
(building the complex product from DMMA-eligible real DGEMMs) is worth it.

Method: time cupy matmul for float64 and complex128 at several N; report achieved
TFLOP/s (DGEMM = 2N^3, ZGEMM = 8N^3 real flops).  Also toggles the cuBLAS math mode
(default vs PEDANTIC = no tensor cores) when reachable, so a default-vs-pedantic gap
is direct evidence of TC use.  GPU-only (skips on CPU).

    python tests/benchmarks/perf_fp64_tc.py --sizes 2048,4096,8192 --reps 5
"""

from __future__ import annotations

import argparse

import numpy as np


def _gpu():
    try:
        import cupy as cp  # noqa: PLC0415

        if cp.cuda.runtime.getDeviceCount() == 0:
            return None
        return cp
    except Exception:
        return None


def _time_matmul(cp, dtype, N, reps):
    """Return best achieved TFLOP/s over ``reps`` timed matmuls (after a warm-up)."""
    rng = cp.random.default_rng(0)
    a = rng.standard_normal((N, N)).astype(dtype)
    b = rng.standard_normal((N, N)).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        a = a + 1j * rng.standard_normal((N, N)).astype(dtype)
        b = b + 1j * rng.standard_normal((N, N)).astype(dtype)
    flop = (8.0 if np.issubdtype(dtype, np.complexfloating) else 2.0) * N ** 3
    start, end = cp.cuda.Event(), cp.cuda.Event()
    a @ b                                         # warm-up (handle + autotune)
    cp.cuda.Device().synchronize()
    best = float("inf")
    for _ in range(reps):
        start.record()
        c = a @ b                                 # noqa: F841
        end.record()
        end.synchronize()
        best = min(best, cp.cuda.get_elapsed_time(start, end) * 1e-3)  # ms -> s
    return flop / best * 1e-12                     # TFLOP/s


def _set_math_mode(cp, pedantic):
    """Best-effort cuBLAS math-mode toggle; returns True if applied."""
    try:
        from cupy.cuda import cublas, device  # noqa: PLC0415

        h = device.get_cublas_handle()
        mode = cublas.CUBLAS_PEDANTIC_MATH if pedantic else cublas.CUBLAS_DEFAULT_MATH
        cublas.setMathMode(h, mode)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="2048,4096,8192")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    cp = _gpu()
    if cp is None:
        print("no CuPy GPU available -- FP64-TC probe is GPU-only, skipping")
        return
    name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"FP64 Tensor-Core (DMMA) probe on {name}")
    print(f"  (A800 ref peaks: FP64 CUDA-core ~9.7 TFLOP/s, FP64 DMMA ~19.5 TFLOP/s)")
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    for pedantic in (False, True):
        applied = _set_math_mode(cp, pedantic)
        tag = "PEDANTIC (no TC)" if pedantic else "default"
        if not applied and pedantic:
            print("\n  (cuBLAS math-mode toggle unavailable; reporting default only)")
            break
        print(f"\n  math mode = {tag}{'' if applied else ' [toggle N/A]'}")
        print(f"    {'N':>6} | {'DGEMM TFLOP/s':>14} | {'ZGEMM TFLOP/s':>14}")
        print("    " + "-" * 42)
        for N in sizes:
            d = _time_matmul(cp, np.float64, N, args.reps)
            z = _time_matmul(cp, np.complex128, N, args.reps)
            print(f"    {N:>6} | {d:>14.1f} | {z:>14.1f}")

    print("\n  Interpretation: DGEMM >> 9.7 TFLOP/s => DMMA engaged for FP64; "
          "a default-vs-PEDANTIC gap confirms it. If ZGEMM (per-flop) tracks DGEMM, "
          "complex GEMMs already benefit; if not, a real/imag split may help.")


if __name__ == "__main__":
    main()
