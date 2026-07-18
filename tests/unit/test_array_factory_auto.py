"""Unit tests for GPU-primary backend resolution (resolve_backend / auto).

The fallback logic is exercised on any machine; the assertions adapt to whether
a GPU is actually present.
"""

import numpy as np
import pytest

import edmtn.backend as bk
from edmtn.backend.array_factory import resolve_backend
from edmtn.backend.precision import PrecisionPolicy


# These tests are ADAPTIVE: they run on any machine and branch on the ``gpu_available`` fixture
# (from tests/conftest.py), asserting the CPU-fallback path when no GPU is present and the GPU
# path when one is.  They are NOT gpu-only, so they carry no @pytest.mark.gpu.


# --------------------------------------------------------------------------
# resolve_backend
# --------------------------------------------------------------------------

def test_resolve_numpy_always_numpy():
    assert resolve_backend("numpy") == ("numpy", None)


def test_resolve_invalid_prefer():
    with pytest.raises(ValueError):
        resolve_backend("jax")


def test_resolve_cupy_matches_availability(gpu_available):
    name, reason = resolve_backend("cupy")
    if gpu_available:
        assert name == "cupy" and reason is None
    else:
        assert name == "numpy" and isinstance(reason, str) and reason


# --------------------------------------------------------------------------
# ArrayFactory.auto
# --------------------------------------------------------------------------

def test_auto_prefer_numpy():
    f = bk.ArrayFactory.auto(prefer="numpy")
    assert not f.is_gpu
    assert f.fallback_reason is None


def test_auto_prefer_cupy_consistent_state(gpu_available):
    f = bk.ArrayFactory.auto(prefer="cupy")
    if gpu_available:
        assert f.is_gpu and f.fallback_reason is None
    else:
        # graceful fallback, never raises
        assert not f.is_gpu and isinstance(f.fallback_reason, str)


def test_auto_threads_precision_and_keeps_defaults():
    f = bk.ArrayFactory.auto(prefer="numpy", precision=PrecisionPolicy.mixed())
    assert f.precision.contract == "f32"
    # default-constructed factory still uses full f64
    assert bk.ArrayFactory("numpy").precision == PrecisionPolicy.full_f64()


# --------------------------------------------------------------------------
# backward-compatible positional signature (Phase 1)
# --------------------------------------------------------------------------

def test_positional_signature_unchanged():
    f = bk.ArrayFactory("numpy", np.complex128, 0)
    assert not f.is_gpu
    assert f.dtype == np.dtype(np.complex128)
    assert f.precision == PrecisionPolicy.full_f64()
