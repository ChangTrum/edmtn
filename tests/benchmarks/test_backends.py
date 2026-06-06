"""Backend-correctness checks for the benchmark paths (marked ``benchmark``).

These confirm that the alternate precisions/backends the benchmark exercises
produce the *same physics* as the CPU/complex128 reference, so that the timing
numbers compare like with like.  GPU tests skip cleanly when CuPy is absent.

Run with:  pytest -m benchmark
"""

import numpy as np
import pytest

from edmtn.cumulants import GaussianCumulantEngine
from edmtn.evolution import SingleBathEvolution
from edmtn.kernels import GaussianKernelEngine
from edmtn.models import SpinBosonModel

pytestmark = pytest.mark.benchmark


def _sz_history(convert, eps=0.05, N=20, cutoff=1e-6):
    model = SpinBosonModel(J0=0.7, omega_c=5.0, mu=1.0)
    cum = GaussianCumulantEngine().compute(model, T=N * eps, eps=eps)
    eng = GaussianKernelEngine(cum)
    res = SingleBathEvolution().run(
        model, eng, eps, N, cutoff=cutoff, record_rho=True, convert=convert
    )
    out = []
    for t, rho in zip(res.times, res.density_matrices):
        r = rho.get() if hasattr(rho, "get") else rho
        out.append(np.trace(model.coupling_operators_at(t)[0] @ r).real)
    return np.array(out)


def test_cpu_fp32_matches_fp64():
    ref = _sz_history(None)
    f32 = _sz_history(lambda a: np.asarray(a, np.complex64))
    # single precision: agreement to ~1e-5
    np.testing.assert_allclose(f32, ref, atol=1e-4)


def test_gpu_matches_cpu():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("no CUDA device")
    ref = _sz_history(None)
    gpu64 = _sz_history(lambda a: cp.asarray(a, cp.complex128))
    np.testing.assert_allclose(gpu64, ref, atol=1e-8)
    gpu32 = _sz_history(lambda a: cp.asarray(a, cp.complex64))
    np.testing.assert_allclose(gpu32, ref, atol=1e-4)
