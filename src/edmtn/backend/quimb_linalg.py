"""quimb integration: array-agnostic decompositions and GPU compat shim.

Two responsibilities:

1. A compatibility shim that makes quimb tensor operations work on CuPy-backed
   tensors.  quimb routes truncated SVD / contraction through autoray's
   ``get_namespace``, which stores ``array.device`` in its module-level
   namespace cache key.  In current CuPy the ``Device`` class is unhashable and
   immutable, so any quimb split / contraction on a CuPy array raises
   ``TypeError: unhashable type: 'cupy.cuda.device.Device'``.  The shim swaps
   the cache for one that coerces unhashable keys.

2. :class:`QuimbSVDBackend`, an array-agnostic decomposition backend that
   dispatches through autoray and therefore works on both NumPy and CuPy arrays.
   It also exposes a passthrough to :func:`quimb.tensor.tensor_split` for the
   higher tensor-network layers.
"""

from __future__ import annotations

from .decomposition_registry import DecompositionBackend, register

_SHIM_APPLIED = False


class _SafeNamespaceCache(dict):
    """A ``dict`` that tolerates unhashable ``device`` entries in autoray keys.

    autoray builds keys ``(cls, device, dtype, submodule)``.  When the whole key
    is unhashable, the offending ``device`` element is replaced by a stable,
    hashable surrogate that still separates distinct devices.
    """

    @staticmethod
    def _coerce(key):
        try:
            hash(key)
            return key
        except TypeError:
            pass
        try:
            cls, device, dtype, submodule = key
        except (TypeError, ValueError):
            return repr(key)
        surrogate = ("__unhashable_device__", type(device).__name__, repr(device))
        return (cls, surrogate, dtype, submodule)

    def __getitem__(self, key):
        return super().__getitem__(self._coerce(key))

    def __setitem__(self, key, value):
        super().__setitem__(self._coerce(key), value)

    def __contains__(self, key):
        return super().__contains__(self._coerce(key))


def _patch_cupy_cholesky() -> None:
    """Make ``cupy.linalg.cholesky`` accept the ``upper`` keyword.

    quimb's Cholesky-QR (``method='qr:cholesky'``, our ``compress_canon='cholqr'``)
    calls ``xp.linalg.cholesky(x, upper=False)``.  NumPy 2 accepts ``upper``; CuPy's
    signature does not, so the call raises ``TypeError`` on the GPU.  CuPy returns
    the lower factor (== ``upper=False``), so the shim just absorbs the kwarg (and
    transposes for ``upper=True``).  Best-effort and idempotent; no-op without CuPy.
    """
    try:
        import cupy  # noqa: PLC0415
    except Exception:
        return
    chol = cupy.linalg.cholesky
    if getattr(chol, "_edm_upper_shim", False):
        return

    def cholesky(a, upper=False):
        L = chol(a)  # CuPy returns the lower-triangular factor
        return L.conj().swapaxes(-1, -2) if upper else L

    cholesky._edm_upper_shim = True
    cupy.linalg.cholesky = cholesky
    try:  # also override autoray's dispatch in case it cached the bare function
        import autoray
        autoray.register_function("cupy", "linalg.cholesky", cholesky)
    except Exception:
        pass


def apply_quimb_cupy_compat() -> bool:
    """Install the autoray namespace-cache shim (and the CuPy cholesky shim).

    Idempotent and harmless for the NumPy backend.  Returns ``True`` if the shim
    is active, ``False`` if autoray is unavailable or its internals changed such
    that the shim no longer applies.
    """
    global _SHIM_APPLIED
    _patch_cupy_cholesky()
    if _SHIM_APPLIED:
        return True
    try:
        import autoray.autoray as _aa
    except Exception:
        return False

    cache = getattr(_aa, "_NAMESPACE_CACHE", None)
    if cache is None:
        _SHIM_APPLIED = True
        return False
    if not isinstance(cache, _SafeNamespaceCache):
        _aa._NAMESPACE_CACHE = _SafeNamespaceCache(cache)
    _SHIM_APPLIED = True
    return True


class QuimbSVDBackend(DecompositionBackend):
    """Array-agnostic factorisations dispatched through autoray.

    Works on any array type autoray understands (NumPy, CuPy), which makes it
    the natural CPU/debug backend and a uniform fallback on the GPU.  The GPU
    compat shim is applied on construction.
    """

    name = "quimb"

    def __init__(self):
        apply_quimb_cupy_compat()

    def svd(self, matrix, full_matrices: bool = False):
        from autoray import do

        return do("linalg.svd", matrix, full_matrices=full_matrices)

    def qr(self, matrix):
        from autoray import do

        return do("linalg.qr", matrix)

    def eigh(self, matrix):
        from autoray import do

        return do("linalg.eigh", matrix)

    @staticmethod
    def tensor_split(tensor, left_inds, **kwargs):
        """Passthrough to :func:`quimb.tensor.tensor_split`.

        Provided for the higher tensor-network layers; the compat shim is
        applied first so the call is safe on CuPy-backed tensors.
        """
        apply_quimb_cupy_compat()
        from quimb.tensor import tensor_split

        return tensor_split(tensor, left_inds, **kwargs)


register("quimb", QuimbSVDBackend)
