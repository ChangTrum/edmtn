"""GPU memory management for Layer 0.

Gaudin (and later Kondo) push much larger bond dimensions than the Phase-1
spin-boson problem, so the GPU memory pool stops being an afterthought.  In
Phase 1 the pool cap / ``free_all_blocks`` / OOM handling lived ad-hoc inside
``tests/benchmarks/perf_cpu_gpu.py``; :class:`MemoryManager` lifts that into
Layer 0 as a reusable component with a uniform API.

On the NumPy (CPU) backend every method is a no-op so call sites stay uniform:
the higher layers wrap allocation-heavy regions in ``manager.scope()`` /
``manager.oom_guard()`` unconditionally and pay nothing on CPU.

CuPy is imported lazily inside the methods so this module imports (and a CPU
``MemoryManager`` works) on machines without a GPU.
"""

from __future__ import annotations

from contextlib import contextmanager

_VALID_BACKENDS = ("numpy", "cupy")

# numpy/CPU placeholder so callers can read stats uniformly.
_CPU_STATS = {"used_bytes": 0, "total_bytes": 0, "limit": 0, "n_free_blocks": 0}


class MemoryManager:
    """Manage the CuPy memory pool for one device.

    Parameters
    ----------
    backend : {'numpy', 'cupy'}
        ``'numpy'`` makes every operation a no-op.
    device_id : int
        GPU device index (ignored for the NumPy backend).
    """

    def __init__(self, backend: str = "numpy", device_id: int = 0):
        if backend not in _VALID_BACKENDS:
            raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {backend!r}")
        self.backend = backend
        self.device_id = device_id

    @property
    def is_gpu(self) -> bool:
        return self.backend == "cupy"

    def _cp(self):
        import cupy as cp

        return cp

    def _pool(self):
        return self._cp().get_default_memory_pool()

    # -- limit -------------------------------------------------------------

    def set_limit(self, *, fraction: float | None = None, size: int | None = None) -> None:
        """Cap the GPU memory pool.

        Exactly one of ``fraction`` (of total device memory) or ``size`` (bytes)
        may be given.  No-op on the NumPy backend.
        """
        if (fraction is None) == (size is None):
            raise ValueError("pass exactly one of fraction= or size=")
        if not self.is_gpu:
            return
        cp = self._cp()
        with cp.cuda.Device(self.device_id):
            pool = self._pool()
            if fraction is not None:
                pool.set_limit(fraction=fraction)
            else:
                pool.set_limit(size=size)

    # -- freeing -----------------------------------------------------------

    def free_all_blocks(self) -> None:
        """Return all cached (unused) pool blocks to the driver.  No-op on CPU."""
        if not self.is_gpu:
            return
        cp = self._cp()
        with cp.cuda.Device(self.device_id):
            self._pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    @contextmanager
    def scope(self):
        """Free pool blocks when the block exits.

        Used around a self-contained allocation phase — e.g. one Gaudin
        sub-bath in the separable-bath outer loop (technical plan §8.4), whose
        intermediate MPO can be released before the next sub-bath.  No-op on CPU.
        """
        try:
            yield self
        finally:
            self.free_all_blocks()

    @contextmanager
    def oom_guard(self, on_oom=None):
        """Free blocks and run ``on_oom`` if the block raises ``OutOfMemoryError``.

        The exception is always re-raised after cleanup: deciding how to recover
        (e.g. lowering ``D_c``) belongs to the decomposition layer.  ``on_oom``,
        if given, is called with the caught exception before the re-raise.
        No-op wrapper on CPU (nothing to catch).
        """
        if not self.is_gpu:
            yield self
            return
        cp = self._cp()
        try:
            yield self
        except cp.cuda.memory.OutOfMemoryError as exc:
            self.free_all_blocks()
            if on_oom is not None:
                on_oom(exc)
            raise

    # -- introspection -----------------------------------------------------

    def stats(self) -> dict:
        """Pool usage snapshot.

        Keys: ``used_bytes``, ``total_bytes`` (pool-reserved), ``limit`` (0 means
        unlimited), ``n_free_blocks``.  Returns zeros on the NumPy backend.
        """
        if not self.is_gpu:
            return dict(_CPU_STATS)
        cp = self._cp()
        with cp.cuda.Device(self.device_id):
            pool = self._pool()
            return {
                "used_bytes": int(pool.used_bytes()),
                "total_bytes": int(pool.total_bytes()),
                "limit": int(pool.get_limit()),
                "n_free_blocks": int(pool.n_free_blocks()),
            }
