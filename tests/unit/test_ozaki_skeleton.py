"""Unit tests for the Ozaki/ADP GEMM seam (skeleton).

The real Ozaki implementation is deferred to Phase 4; here we only check that
the seam exists, hardware detection is safe on any machine, and the stub gemm()
is numerically identical to a plain matmul.
"""

import numpy as np
import pytest

import edmtn.backend as bk
from edmtn.backend.ozaki_gemm import OzakiGEMMBackend


# The one hardware-dependent test carries @pytest.mark.gpu; tests/conftest.py skips it (with a
# specific reason) when no GPU is present.


def test_exported_from_backend_package():
    assert bk.OzakiGEMMBackend is OzakiGEMMBackend


def test_detect_hardware_returns_bool():
    assert isinstance(OzakiGEMMBackend.detect_hardware(), bool)


def test_enabled_matches_detection():
    o = OzakiGEMMBackend()
    assert o.enabled == OzakiGEMMBackend.detect_hardware()


def test_invalid_strategy_rejected():
    with pytest.raises(ValueError):
        OzakiGEMMBackend(strategy="ozaki-9000")


def test_gemm_matches_plain_matmul_numpy():
    o = OzakiGEMMBackend()
    a = np.random.rand(8, 5) + 1j * np.random.rand(8, 5)
    b = np.random.rand(5, 7) + 1j * np.random.rand(5, 7)
    np.testing.assert_allclose(o.gemm(a, b), a @ b)


def test_gemm_out_argument():
    o = OzakiGEMMBackend()
    a = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
    b = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
    out = np.empty((4, 4), dtype=np.complex128)
    ret = o.gemm(a, b, out=out)
    assert ret is out
    np.testing.assert_allclose(out, a @ b)


def test_accelerated_gemm_not_implemented():
    o = OzakiGEMMBackend()
    with pytest.raises(NotImplementedError):
        o._accelerated_gemm(np.eye(2), np.eye(2))


@pytest.mark.gpu
def test_detect_hardware_true_on_blackwell():
    # RTX 5090 is compute capability 12.0; with CUDA >= 13.0 detection is True.
    import cupy as cp

    cc = cp.cuda.Device(0).compute_capability
    major = int(cc[:-1]) if len(cc) > 1 else int(cc)
    runtime = cp.cuda.runtime.runtimeGetVersion()
    expected = major >= 10 and runtime >= 13000
    assert OzakiGEMMBackend.detect_hardware() == expected
