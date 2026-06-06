"""Unit tests for Layer 0 (backend abstraction).

Covers:
  * ArrayFactory creation / dtype / host-device transfer on both backends
  * the decomposition registry
  * CuPySVDBackend SVD / QR / eigh correctness
  * QuimbSVDBackend on both NumPy and CuPy arrays
  * the autoray/CuPy compat shim (quimb split + contraction on the GPU)

GPU tests are skipped automatically if CuPy or a device is unavailable.
"""

import numpy as np
import pytest

import edmtn.backend as bk


# --------------------------------------------------------------------------
# GPU availability
# --------------------------------------------------------------------------

def _gpu_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _gpu_available(), reason="no CuPy GPU available")


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

def test_registry_has_backends():
    names = bk.available()
    assert "cupy" in names
    assert "quimb" in names


def test_registry_unknown_raises():
    with pytest.raises(KeyError):
        bk.create("does-not-exist")


def test_registry_create_returns_backend():
    backend = bk.create("quimb")
    assert isinstance(backend, bk.DecompositionBackend)


# --------------------------------------------------------------------------
# ArrayFactory - NumPy
# --------------------------------------------------------------------------

def test_array_factory_numpy_defaults():
    f = bk.ArrayFactory("numpy")
    assert not f.is_gpu
    z = f.zeros((2, 3))
    assert z.shape == (2, 3)
    assert z.dtype == np.complex128
    assert np.all(z == 0)


def test_array_factory_dtype_override():
    f = bk.ArrayFactory("numpy", dtype=np.complex64)
    assert f.eye(3).dtype == np.complex64
    assert f.zeros((2,), dtype=np.float64).dtype == np.float64


def test_array_factory_invalid_backend():
    with pytest.raises(ValueError):
        bk.ArrayFactory("jax")


def test_array_factory_numpy_to_cpu_passthrough():
    f = bk.ArrayFactory("numpy")
    a = f.asarray([1.0, 2.0, 3.0])
    out = f.to_cpu(a)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [1, 2, 3])


def test_array_factory_numpy_to_gpu_rejected():
    f = bk.ArrayFactory("numpy")
    with pytest.raises(RuntimeError):
        f.to_gpu(np.zeros(3))


# --------------------------------------------------------------------------
# ArrayFactory - CuPy
# --------------------------------------------------------------------------

@requires_gpu
def test_array_factory_cupy_roundtrip():
    f = bk.ArrayFactory("cupy")
    assert f.is_gpu
    host = np.random.rand(4, 5) + 1j * np.random.rand(4, 5)
    dev = f.asarray(host)
    assert type(dev).__module__.startswith("cupy")
    assert dev.dtype == np.complex128
    back = f.to_cpu(dev)
    assert isinstance(back, np.ndarray)
    np.testing.assert_allclose(back, host)


@requires_gpu
def test_array_factory_cupy_creation_helpers():
    f = bk.ArrayFactory("cupy")
    assert f.zeros((3, 3)).shape == (3, 3)
    assert f.ones((2,)).sum().item() == pytest.approx(2.0)
    assert f.eye(4).shape == (4, 4)


# --------------------------------------------------------------------------
# CuPySVDBackend
# --------------------------------------------------------------------------

@requires_gpu
def test_cupy_svd_reconstruction():
    import cupy as cp

    f = bk.ArrayFactory("cupy")
    a = f.asarray(np.random.rand(64, 40) + 1j * np.random.rand(64, 40))
    backend = bk.create("cupy")
    U, s, Vh = backend.svd(a, full_matrices=False)
    assert U.shape == (64, 40)
    assert s.shape == (40,)
    assert Vh.shape == (40, 40)
    recon = (U * s) @ Vh
    rel = float(cp.linalg.norm(recon - a) / cp.linalg.norm(a))
    assert rel < 1e-10
    # singular values are non-negative and descending
    s_host = cp.asnumpy(s)
    assert np.all(s_host >= 0)
    assert np.all(np.diff(s_host) <= 1e-9)


@requires_gpu
def test_cupy_qr_reconstruction():
    import cupy as cp

    f = bk.ArrayFactory("cupy")
    a = f.asarray(np.random.rand(30, 18) + 1j * np.random.rand(30, 18))
    Q, R = bk.create("cupy").qr(a)
    assert Q.shape == (30, 18)
    assert R.shape == (18, 18)
    rel = float(cp.linalg.norm(Q @ R - a) / cp.linalg.norm(a))
    assert rel < 1e-10
    # Q has orthonormal columns
    ident = Q.conj().T @ Q
    assert float(cp.linalg.norm(ident - cp.eye(18))) < 1e-9


@requires_gpu
def test_cupy_eigh_hermitian():
    import cupy as cp

    f = bk.ArrayFactory("cupy")
    m = np.random.rand(12, 12) + 1j * np.random.rand(12, 12)
    herm = f.asarray(m + m.conj().T)
    w, V = bk.create("cupy").eigh(herm)
    assert np.all(np.diff(cp.asnumpy(w)) >= -1e-9)
    recon = (V * w) @ V.conj().T
    rel = float(cp.linalg.norm(recon - herm) / cp.linalg.norm(herm))
    assert rel < 1e-9


# --------------------------------------------------------------------------
# QuimbSVDBackend (array-agnostic)
# --------------------------------------------------------------------------

def test_quimb_svd_numpy():
    backend = bk.create("quimb")
    a = np.random.rand(20, 12) + 1j * np.random.rand(20, 12)
    U, s, Vh = backend.svd(a)
    recon = (U * s) @ Vh
    assert np.linalg.norm(recon - a) / np.linalg.norm(a) < 1e-10


@requires_gpu
def test_quimb_svd_cupy():
    import cupy as cp

    backend = bk.create("quimb")
    a = bk.ArrayFactory("cupy").asarray(np.random.rand(20, 12) + 1j * np.random.rand(20, 12))
    U, s, Vh = backend.svd(a)
    assert type(U).__module__.startswith("cupy")
    recon = (U * s) @ Vh
    assert float(cp.linalg.norm(recon - a) / cp.linalg.norm(a)) < 1e-10


# --------------------------------------------------------------------------
# autoray / CuPy compat shim: quimb tensor ops must work on the GPU
# --------------------------------------------------------------------------

@requires_gpu
def test_compat_shim_quimb_split_on_cupy():
    import cupy as cp
    import quimb.tensor as qtn

    bk.apply_quimb_cupy_compat()
    data = cp.asarray(np.random.rand(4, 6, 5).astype(np.complex128))
    t = qtn.Tensor(data, inds=["a", "b", "c"])
    left, right = t.split(["a"], method="svd", max_bond=3, cutoff=1e-12, get="tensors")
    assert type(left.data).__module__.startswith("cupy")
    assert left.shape[0] == 4


@requires_gpu
def test_compat_shim_quimb_contract_on_cupy():
    import cupy as cp
    import quimb.tensor as qtn

    bk.apply_quimb_cupy_compat()
    t1 = qtn.Tensor(cp.asarray(np.random.rand(4, 5).astype(np.complex128)), inds=["a", "c"])
    t2 = qtn.Tensor(cp.asarray(np.random.rand(5, 7).astype(np.complex128)), inds=["c", "d"])
    out = (t1 & t2).contract()
    assert type(out.data).__module__.startswith("cupy")
    assert set(out.inds) == {"a", "d"}
