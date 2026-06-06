"""NumPy-based decomposition backend (CPU).

The pure-CPU counterpart to :class:`CuPySVDBackend`, routing factorisations to
``numpy.linalg``.  Used for debugging and as the reference path; the
decomposition strategies select it automatically for NumPy arrays.
"""

from __future__ import annotations

import numpy as np

from .decomposition_registry import DecompositionBackend, register


class NumpySVDBackend(DecompositionBackend):
    """Dense factorisations on the CPU via ``numpy.linalg``."""

    name = "numpy"

    def svd(self, matrix, full_matrices: bool = False):
        return np.linalg.svd(matrix, full_matrices=full_matrices)

    def qr(self, matrix):
        return np.linalg.qr(matrix, mode="reduced")

    def eigh(self, matrix):
        return np.linalg.eigh(matrix)


register("numpy", NumpySVDBackend)
