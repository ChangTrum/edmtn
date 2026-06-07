"""Layer 0: backend abstraction.

Provides array creation/device management (:class:`ArrayFactory`), the GPU
memory manager (:class:`MemoryManager`), the mixed-precision policy
(:class:`PrecisionPolicy`), the feature-flagged Ozaki GEMM seam
(:class:`OzakiGEMMBackend`), and the matrix-decomposition backends
(:class:`DecompositionBackend` and its registry).  Importing this package
registers the available decomposition backends (``'numpy'``, ``'cupy'`` and
``'quimb'``).
"""

from __future__ import annotations

from .array_factory import ArrayFactory, resolve_backend
from .cupy_linalg import CuPySVDBackend
from .decomposition_registry import (
    DecompositionBackend,
    available,
    create,
    is_registered,
    register,
)
from .memory import MemoryManager
from .numpy_linalg import NumpySVDBackend
from .ozaki_gemm import OzakiGEMMBackend
from .precision import PrecisionPolicy
from .quimb_linalg import QuimbSVDBackend, apply_quimb_cupy_compat

__all__ = [
    "ArrayFactory",
    "resolve_backend",
    "PrecisionPolicy",
    "MemoryManager",
    "OzakiGEMMBackend",
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
