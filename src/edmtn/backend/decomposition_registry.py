"""Linear-algebra decomposition backends and their registry.

A decomposition backend exposes the raw matrix factorisations (SVD, QR,
eigendecomposition) that the higher layers build on.  Backends are intentionally
thin: they return full (reduced) factorisations and do *not* perform truncation.
Bond-dimension truncation and rank selection are the responsibility of the
decomposition strategies in a higher layer.

Backends register themselves under a string name so that callers can select an
implementation (e.g. ``'cupy'`` for GPU, ``'quimb'`` for the array-agnostic
path) without importing the concrete class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

# name -> factory returning a DecompositionBackend instance
_REGISTRY: dict[str, Callable[..., "DecompositionBackend"]] = {}


class DecompositionBackend(ABC):
    """Abstract interface for matrix factorisations.

    Implementations operate on dense 2-D arrays of whatever array type they
    support (NumPy or CuPy ``ndarray``) and return arrays of the same type.
    """

    #: short identifier used in the registry
    name: str = "abstract"

    @abstractmethod
    def svd(self, matrix, full_matrices: bool = False):
        """Singular value decomposition.

        Returns ``(U, s, Vh)`` with ``s`` a 1-D array of singular values in
        descending order.  With ``full_matrices=False`` the economy form is
        returned (``U`` is ``m x k``, ``Vh`` is ``k x n``, ``k = min(m, n)``).
        """

    @abstractmethod
    def qr(self, matrix):
        """Reduced QR decomposition, returning ``(Q, R)``."""

    @abstractmethod
    def eigh(self, matrix):
        """Eigendecomposition of a Hermitian matrix, returning ``(w, V)``."""


def register(name: str, factory: Callable[..., DecompositionBackend], *, overwrite: bool = False) -> None:
    """Register a backend ``factory`` under ``name``.

    ``factory`` is any callable that returns a :class:`DecompositionBackend`
    (typically the class itself).  Re-registering an existing name requires
    ``overwrite=True``.
    """
    if not overwrite and name in _REGISTRY:
        raise KeyError(f"decomposition backend {name!r} already registered")
    _REGISTRY[name] = factory


def create(name: str, **kwargs) -> DecompositionBackend:
    """Instantiate the backend registered under ``name``."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown decomposition backend {name!r}; "
            f"available: {sorted(_REGISTRY)}"
        ) from None
    return factory(**kwargs)


def available() -> list[str]:
    """Return the sorted list of registered backend names."""
    return sorted(_REGISTRY)


def is_registered(name: str) -> bool:
    return name in _REGISTRY
