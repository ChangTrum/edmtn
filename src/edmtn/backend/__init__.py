"""Layer 0: backend abstraction.

Provides array creation/device management (:class:`ArrayFactory`) and the
matrix-decomposition backends (:class:`DecompositionBackend` and its registry).
Importing this package registers the available decomposition backends
(``'numpy'``, ``'cupy'`` and ``'quimb'``).
"""

from __future__ import annotations

from .array_factory import ArrayFactory
from .cupy_linalg import CuPySVDBackend
from .decomposition_registry import (
    DecompositionBackend,
    available,
    create,
    is_registered,
    register,
)
from .numpy_linalg import NumpySVDBackend
from .quimb_linalg import QuimbSVDBackend, apply_quimb_cupy_compat

__all__ = [
    "ArrayFactory",
    "DecompositionBackend",
    "NumpySVDBackend",
    "CuPySVDBackend",
    "QuimbSVDBackend",
    "apply_quimb_cupy_compat",
    "register",
    "create",
    "available",
    "is_registered",
]
