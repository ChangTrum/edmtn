"""Array creation and device/dtype management for Layer 0.

:class:`ArrayFactory` is the single entry point through which the higher layers
create and move arrays.  They never ``import cupy`` directly; selecting the
backend (``'numpy'`` or ``'cupy'``) and the device is done here.  This keeps the
rest of the code backend-agnostic and makes the pure-CPU path (``'numpy'``)
available for debugging.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

_VALID_BACKENDS = ("numpy", "cupy")


class ArrayFactory:
    """Create and move arrays on a chosen backend and device.

    Parameters
    ----------
    backend : {'numpy', 'cupy'}
        Array library to use.  ``'cupy'`` runs on the GPU.
    dtype : data-type
        Default dtype for created arrays (``complex128`` by default, matching
        the precision the EDM construction needs).
    device_id : int
        GPU device index (ignored for the NumPy backend).
    """

    def __init__(self, backend: str = "numpy", dtype=np.complex128, device_id: int = 0):
        if backend not in _VALID_BACKENDS:
            raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {backend!r}")
        self.backend = backend
        self.dtype = np.dtype(dtype)
        self.device_id = device_id

        if backend == "cupy":
            import cupy as cp

            # Ensure quimb/autoray tensor ops are safe on this backend.
            from .quimb_linalg import apply_quimb_cupy_compat

            apply_quimb_cupy_compat()
            self._xp = cp
        else:
            self._xp = np

    # -- introspection -----------------------------------------------------

    @property
    def xp(self):
        """The underlying array module (``numpy`` or ``cupy``)."""
        return self._xp

    @property
    def is_gpu(self) -> bool:
        return self.backend == "cupy"

    @contextmanager
    def device(self):
        """Context manager that activates the configured GPU device.

        A no-op for the NumPy backend, so call sites stay uniform.
        """
        if self.is_gpu:
            with self._xp.cuda.Device(self.device_id):
                yield
        else:
            yield

    def _resolve_dtype(self, dtype):
        return self.dtype if dtype is None else np.dtype(dtype)

    # -- creation ----------------------------------------------------------

    def zeros(self, shape, dtype=None):
        with self.device():
            return self._xp.zeros(shape, dtype=self._resolve_dtype(dtype))

    def ones(self, shape, dtype=None):
        with self.device():
            return self._xp.ones(shape, dtype=self._resolve_dtype(dtype))

    def empty(self, shape, dtype=None):
        with self.device():
            return self._xp.empty(shape, dtype=self._resolve_dtype(dtype))

    def eye(self, n, dtype=None):
        with self.device():
            return self._xp.eye(n, dtype=self._resolve_dtype(dtype))

    def asarray(self, data, dtype=None):
        """Place ``data`` on this backend/device with the given (or default) dtype."""
        with self.device():
            return self._xp.asarray(data, dtype=self._resolve_dtype(dtype))

    def astype(self, arr, dtype):
        with self.device():
            return arr.astype(np.dtype(dtype))

    # -- host/device transfer ---------------------------------------------

    def to_gpu(self, arr, dtype=None):
        """Return ``arr`` as a CuPy array on the configured device.

        Raises if this factory is not a GPU factory.
        """
        if not self.is_gpu:
            raise RuntimeError("to_gpu() requires a 'cupy' ArrayFactory")
        with self.device():
            return self._xp.asarray(arr, dtype=self._resolve_dtype(dtype))

    def to_cpu(self, arr):
        """Return ``arr`` as a NumPy array, regardless of where it lives."""
        if type(arr).__module__.startswith("cupy"):
            import cupy as cp

            return cp.asnumpy(arr)
        return np.asarray(arr)
