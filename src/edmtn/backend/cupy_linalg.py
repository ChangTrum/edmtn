"""CuPy-based decomposition backend (GPU).

Routes SVD / QR / eigendecomposition straight to ``cupy.linalg`` so that the
GPU fast path never depends on quimb's array-API namespace machinery.  CuPy
dispatches internally to cuSOLVER (Jacobi ``gesvdj`` for small/batched
problems, QR-based ``gesvd`` otherwise).

CuPy is imported lazily inside the methods so that this module can be imported
(and the backend registered) on machines without a GPU.
"""

from __future__ import annotations

from .decomposition_registry import DecompositionBackend, register


class CuPySVDBackend(DecompositionBackend):
    """Dense factorisations on the GPU via ``cupy.linalg``."""

    name = "cupy"

    def __init__(self, device_id: int = 0):
        self.device_id = device_id

    def _cp(self):
        import cupy as cp

        return cp

    def svd(self, matrix, full_matrices: bool = False):
        cp = self._cp()
        with cp.cuda.Device(self.device_id):
            return cp.linalg.svd(matrix, full_matrices=full_matrices)

    def qr(self, matrix):
        cp = self._cp()
        with cp.cuda.Device(self.device_id):
            return cp.linalg.qr(matrix, mode="reduced")

    def eigh(self, matrix):
        cp = self._cp()
        with cp.cuda.Device(self.device_id):
            return cp.linalg.eigh(matrix)


register("cupy", CuPySVDBackend)
