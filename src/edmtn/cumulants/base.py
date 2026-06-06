"""Cumulant / correlation engine interface (Layer 2).

A cumulant engine turns a model's bath description into the irreducible bath
correlations (cumulants) that the kernel-tensor construction consumes.  Engines
are specialised by bath type; each returns its own result structure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class CumulantEngine(ABC):
    """Base class for bath cumulant engines.

    Subclasses declare the :attr:`bath_type` they handle and implement
    :meth:`compute`, which evaluates the cumulants on the discrete time grid
    defined by the total time ``T`` and step ``eps``.
    """

    #: bath type this engine handles ('gaussian' | 'separable' | 'chain' | ...)
    bath_type: str = "abstract"

    def _check_model(self, model) -> None:
        if model.bath_type != self.bath_type:
            raise ValueError(
                f"{type(self).__name__} handles bath_type={self.bath_type!r}, "
                f"but model has bath_type={model.bath_type!r}"
            )

    @staticmethod
    def _n_steps(T: float, eps: float) -> int:
        """Number of time steps ``N = T / eps``, validated to be near-integer."""
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        n = T / eps
        n_round = round(n)
        if abs(n - n_round) > 1e-9 * max(1.0, n):
            raise ValueError(f"T/eps = {n} is not (close to) an integer")
        return int(n_round)

    @abstractmethod
    def compute(self, model, T: float, eps: float):
        """Evaluate the bath cumulants for ``model`` on the grid ``0..T`` step ``eps``."""
