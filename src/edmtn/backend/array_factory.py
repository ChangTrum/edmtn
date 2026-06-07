"""Array creation and device/dtype management for Layer 0.

:class:`ArrayFactory` is the single entry point through which the higher layers
create and move arrays.  They never ``import cupy`` directly; selecting the
backend (``'numpy'`` or ``'cupy'``) and the device is done here.  This keeps the
rest of the code backend-agnostic and makes the pure-CPU path (``'numpy'``)
available for debugging.

From Phase 2 on, the GPU is the primary compute path (the Gaudin / Kondo bond
dimensions are far larger than spin-boson).  :meth:`ArrayFactory.auto` resolves
to CuPy when a GPU is usable and falls back to NumPy otherwise, so callers get
"GPU-primary, CPU-fallback" without branching.  Each factory also carries a
:class:`~edmtn.backend.precision.PrecisionPolicy` and a
:class:`~edmtn.backend.memory.MemoryManager`, so per-stage dtype casting and GPU
memory management flow through the same object instead of ad-hoc lambdas.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from .memory import MemoryManager
from .precision import PrecisionPolicy

_VALID_BACKENDS = ("numpy", "cupy")


def resolve_backend(prefer: str = "cupy") -> tuple[str, str | None]:
    """Resolve a usable backend name, preferring ``prefer``.

    Returns ``(name, reason)`` where ``name`` is ``'cupy'`` or ``'numpy'`` and
    ``reason`` is ``None`` if the preferred backend was available, otherwise a
    short string explaining the fallback to NumPy.

    ``prefer='numpy'`` always resolves to NumPy.  ``prefer='cupy'`` checks that
    CuPy imports and at least one CUDA device is present; on any failure it
    falls back to NumPy with the reason recorded rather than raising.
    """
    if prefer not in _VALID_BACKENDS:
        raise ValueError(f"prefer must be one of {_VALID_BACKENDS}, got {prefer!r}")
    if prefer == "numpy":
        return "numpy", None
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() <= 0:
            return "numpy", "no CUDA device found; falling back to NumPy"
        return "cupy", None
    except Exception as exc:  # CuPy missing, CUDA driver error, etc.
        return "numpy", f"CuPy/GPU unavailable ({type(exc).__name__}); falling back to NumPy"


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
    precision : PrecisionPolicy, optional
        Per-stage precision policy (default :meth:`PrecisionPolicy.full_f64`).
        Drives :meth:`cast` / :meth:`caster`.
    memory : MemoryManager, optional
        GPU memory manager (default a fresh one matching ``backend``/``device_id``).

    Notes
    -----
    ``precision`` and ``memory`` are keyword-only with defaults, so the
    positional Phase-1 signature ``ArrayFactory('numpy')`` is unchanged.
    """

    def __init__(
        self,
        backend: str = "numpy",
        dtype=np.complex128,
        device_id: int = 0,
        *,
        precision: PrecisionPolicy | None = None,
        memory: MemoryManager | None = None,
    ):
        if backend not in _VALID_BACKENDS:
            raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {backend!r}")
        self.backend = backend
        self.dtype = np.dtype(dtype)
        self.device_id = device_id
        self.precision = precision if precision is not None else PrecisionPolicy.full_f64()
        self.memory = memory if memory is not None else MemoryManager(backend, device_id)
        #: set by :meth:`auto` when the preferred backend was unavailable
        self.fallback_reason: str | None = None

        if backend == "cupy":
            import cupy as cp

            # Ensure quimb/autoray tensor ops are safe on this backend.
            from .quimb_linalg import apply_quimb_cupy_compat

            apply_quimb_cupy_compat()
            self._xp = cp
        else:
            self._xp = np

    @classmethod
    def auto(
        cls,
        prefer: str = "cupy",
        dtype=np.complex128,
        device_id: int = 0,
        *,
        precision: PrecisionPolicy | None = None,
        memory: MemoryManager | None = None,
    ) -> "ArrayFactory":
        """Build a factory on the preferred backend, falling back to NumPy.

        GPU-primary entry point for Phase 2+: ``prefer='cupy'`` yields a CuPy
        factory when a GPU is usable, otherwise a NumPy factory whose
        :attr:`fallback_reason` records why.  Never raises for a missing GPU.
        """
        name, reason = resolve_backend(prefer)
        factory = cls(
            name, dtype=dtype, device_id=device_id, precision=precision, memory=memory
        )
        factory.fallback_reason = reason
        return factory

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

    # -- precision casting -------------------------------------------------

    def cast(self, arr, stage: str):
        """Place ``arr`` on this backend at the precision for ``stage``.

        ``stage`` is one of ``'build'``, ``'contract'``, ``'decompose'`` (see
        :class:`~edmtn.backend.precision.PrecisionPolicy`).
        """
        with self.device():
            return self.caster(stage)(arr)

    def caster(self, stage: str):
        """Return a callable casting arrays to ``stage`` precision on this backend.

        The Layer-0 replacement for the hand-written ``convert`` lambda the
        Phase-1 evolution engine threaded through; Gaudin's evolution will pass
        ``factory.caster('contract')`` instead.
        """
        return self.precision.caster(stage, self._xp)

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
