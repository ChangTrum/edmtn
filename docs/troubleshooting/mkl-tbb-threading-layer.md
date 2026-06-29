---
name: mkl-tbb-threading-layer
description: numpy BLAS/LAPACK hard-crashes in the quimb env unless MKL_THREADING_LAYER=TBB is set before numpy import
metadata:
  type: project
---

In the `quimb` conda env (`D:\Productivity\Anaconda3\envs\quimb`), any numpy BLAS/LAPACK call (`np.matmul`, `np.linalg.svd`, even real-valued/small) **hard-crashes the process** with Windows fatal exception `0xc06d007f` (exit 127). It is not a Python exception, so `try/except` cannot catch it — it kills the whole interpreter (and a pytest run).

**Cause:** the env has MKL-backed BLAS (`libblas/libcblas/liblapack` = `*_mkl`, `mkl-2026.0.0`) but ships **both** OpenMP runtimes — Intel `libiomp5md.dll` and GNU `libgomp-1.dll`. MKL's default OpenMP threading layer aborts on the OpenMP clash. `KMP_DUPLICATE_LIB_OK=TRUE` does NOT fix it.

**Fix:** set `MKL_THREADING_LAYER=TBB`. MKL reads this only at init (numpy import / first BLAS call), so it must be set **before numpy is first imported**. Setting `os.environ[...]` after `import numpy` is too late (still crashes).

- Durable fix (applied): `conda env config vars set MKL_THREADING_LAYER=TBB -n quimb`, so the
  variable is present whenever the env is activated. The test suite therefore no longer sets it.
- Caveat: `conda env config vars` only inject on `conda activate`. Invoking the env's
  `python.exe` directly (without activating) does NOT get the variable — in that case pass
  `MKL_THREADING_LAYER=TBB` explicitly, or numpy BLAS/LAPACK will crash.

GPU path (CuPy / cuSOLVER) is unaffected by this — only the numpy CPU path crashes.
