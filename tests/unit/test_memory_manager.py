"""Unit tests for the Layer-0 GPU MemoryManager.

The NumPy path is a no-op and is exercised on any machine; the CuPy path is
skipped automatically when no GPU is available.
"""

import numpy as np
import pytest

import edmtn.backend as bk
from edmtn.backend.memory import MemoryManager


# CuPy tests carry @pytest.mark.gpu; tests/conftest.py skips them (with a specific reason) when
# no GPU is present.


# --------------------------------------------------------------------------
# NumPy (no-op) path — runs everywhere
# --------------------------------------------------------------------------

def test_numpy_is_not_gpu():
    assert MemoryManager("numpy").is_gpu is False


def test_invalid_backend_rejected():
    with pytest.raises(ValueError):
        MemoryManager("jax")


def test_numpy_methods_are_noops():
    m = MemoryManager("numpy")
    m.set_limit(fraction=0.5)       # no-op, must not raise
    m.set_limit(size=1 << 20)       # no-op, must not raise
    m.free_all_blocks()             # no-op


def test_numpy_scope_noop():
    m = MemoryManager("numpy")
    with m.scope() as inner:
        assert inner is m


def test_numpy_oom_guard_passes_through():
    m = MemoryManager("numpy")
    # On CPU there is nothing to catch; an unrelated error propagates unchanged.
    with pytest.raises(ZeroDivisionError):
        with m.oom_guard():
            1 / 0


def test_numpy_stats_zeroed_with_expected_keys():
    s = MemoryManager("numpy").stats()
    assert set(s) == {"used_bytes", "total_bytes", "limit", "n_free_blocks"}
    assert all(v == 0 for v in s.values())


def test_set_limit_requires_exactly_one_arg():
    m = MemoryManager("numpy")
    with pytest.raises(ValueError):
        m.set_limit()
    with pytest.raises(ValueError):
        m.set_limit(fraction=0.5, size=1 << 20)


def test_factory_carries_memory_manager():
    f = bk.ArrayFactory("numpy")
    assert isinstance(f.memory, MemoryManager)
    assert f.memory.backend == "numpy"


# --------------------------------------------------------------------------
# CuPy path
# --------------------------------------------------------------------------

@pytest.mark.gpu
def test_cupy_stats_keys_and_types():
    m = MemoryManager("cupy")
    s = m.stats()
    assert set(s) == {"used_bytes", "total_bytes", "limit", "n_free_blocks"}
    assert all(isinstance(v, int) for v in s.values())


@pytest.mark.gpu
def test_cupy_scope_frees_blocks():
    import cupy as cp

    m = MemoryManager("cupy")
    pool = cp.get_default_memory_pool()
    with m.scope():
        a = cp.ones((256, 256), dtype=cp.complex128)
        del a
    # after the scope, cached blocks have been returned to the driver
    assert pool.n_free_blocks() == 0


@pytest.mark.gpu
def test_cupy_set_limit_roundtrip():
    import cupy as cp

    m = MemoryManager("cupy")
    pool = cp.get_default_memory_pool()
    original = pool.get_limit()
    try:
        m.set_limit(size=512 << 20)
        assert pool.get_limit() == (512 << 20)
    finally:
        pool.set_limit(size=original)


@pytest.mark.gpu
def test_cupy_oom_guard_catches_and_reraises():
    import cupy as cp

    m = MemoryManager("cupy")
    pool = cp.get_default_memory_pool()
    original = pool.get_limit()
    seen = {}

    def on_oom(exc):
        seen["exc"] = exc

    try:
        m.set_limit(size=1 << 20)  # 1 MiB — too small for the alloc below
        with pytest.raises(cp.cuda.memory.OutOfMemoryError):
            with m.oom_guard(on_oom=on_oom):
                cp.empty((4096, 4096), dtype=cp.complex128)  # 256 MiB
        assert "exc" in seen
    finally:
        pool.set_limit(size=original)
        pool.free_all_blocks()
