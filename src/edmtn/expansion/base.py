"""Time-step expansion interface (Layer 4b).

Expands a small step of evolution ``e^{eps H^-(t)}`` into a tensor-product form,
producing the *system superoperators* ``S^phi`` that the EDM evolution applies
to the (vectorised) system density operator.  The bath side of each step is
handled separately by the cumulant/kernel layers; the ``phi`` index convention
is shared with them:

* ``phi = 0``         null (identity superoperator), bath ``I``
* ``phi = 2a - 1``    ``S^+_a`` (anticommutator), paired with bath ``B^-``
* ``phi = 2a``        ``S^-_a`` (commutator), paired with bath ``B^+``

for coupling channels ``a = 1 .. K``, giving ``phys_dim = 2 K + 1``.

Superoperators act on the *row-major* vectorisation ``vec(rho) = rho.reshape(-1)``
(C order), for which ``A rho B -> (A kron B^T) vec(rho)``.  Hence

    S^-_a rho = -i [S_a, rho]   ->  -i (S_a kron I - I kron S_a^T)
    S^+_a rho = 1/2 {S_a, rho}  ->  1/2 (S_a kron I + I kron S_a^T)

both scaled by ``eps`` in the first-order coefficients.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


def commutator_superoperator(A: np.ndarray) -> np.ndarray:
    """Return the matrix of ``rho -> -i [A, rho]`` on row-major ``vec(rho)``."""
    d = A.shape[0]
    I = np.eye(d, dtype=np.complex128)
    return -1j * (np.kron(A, I) - np.kron(I, A.T))


def anticommutator_superoperator(A: np.ndarray) -> np.ndarray:
    """Return the matrix of ``rho -> 1/2 {A, rho}`` on row-major ``vec(rho)``."""
    d = A.shape[0]
    I = np.eye(d, dtype=np.complex128)
    return 0.5 * (np.kron(A, I) + np.kron(I, A.T))


def apply_superoperator(smat: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Apply a ``d^2 x d^2`` superoperator matrix to a ``d x d`` operator ``rho``."""
    d = rho.shape[0]
    return (smat @ rho.reshape(-1)).reshape(d, d)


def first_order_superoperators(coupling_ops: list[np.ndarray], eps: float) -> np.ndarray:
    """Build the first-order system superoperators ``S^phi`` at one time.

    Parameters
    ----------
    coupling_ops : list of (d, d) arrays
        The interaction-picture coupling operators ``S_a(t)``.
    eps : float
        Time step (absorbed into the non-identity superoperators).

    Returns
    -------
    ndarray
        Array of shape ``(phys_dim, d^2, d^2)`` with ``phys_dim = 2 K + 1``;
        index 0 is the identity superoperator.
    """
    if not coupling_ops:
        raise ValueError("need at least one coupling operator")
    d = coupling_ops[0].shape[0]
    d2 = d * d
    phys_dim = 2 * len(coupling_ops) + 1
    S = np.zeros((phys_dim, d2, d2), dtype=np.complex128)
    S[0] = np.eye(d2, dtype=np.complex128)
    for a, Sa in enumerate(coupling_ops, start=1):
        S[2 * a - 1] = eps * anticommutator_superoperator(Sa)  # S^+ , pairs with B^-
        S[2 * a] = eps * commutator_superoperator(Sa)          # S^- , pairs with B^+
    return S


@dataclass
class StepSuperoperators:
    """System superoperators for one evolution step.

    Attributes
    ----------
    phys_dim : int
        Number of ``phi`` indices (``2 K + 1``).
    d : int
        System Hilbert-space dimension.
    families : list[np.ndarray]
        One ``(phys_dim, d^2, d^2)`` array per sub-step.  First order has a
        single family; second order has two, ordered ``[S_1, S_2]`` (applied to
        the state in that order, i.e. ``S_1`` first), carrying the ``(1 - i)/2``
        and ``(1 + i)/2`` coefficients respectively.
    order : int
    """

    phys_dim: int
    d: int
    families: list[np.ndarray]
    order: int


class TimeStepExpander(ABC):
    """Base class for small-step expansions of ``e^{eps H^-(t)}``."""

    #: expansion order (1 or 2)
    order: int = 1

    @abstractmethod
    def build(self, coupling_ops: list[np.ndarray], eps: float) -> StepSuperoperators:
        """Build the system superoperators for one step given ``S_a(t)`` and ``eps``."""

    def build_at(self, model, t: float, eps: float) -> StepSuperoperators:
        """Convenience: build from a model's interaction-picture operators at time ``t``."""
        return self.build(model.coupling_operators_at(t), eps)
