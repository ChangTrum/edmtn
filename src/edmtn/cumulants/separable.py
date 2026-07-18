"""Separable-bath correlation engine (Layer 2).

For a separable bath (independent sub-baths) the cumulant expansion is
infeasible -- a spin-1/2 sub-bath, for instance, has nonzero cumulants at all
orders.  Instead each sub-bath's bath-correlation tensor has an exact
matrix-product (MPS) form (paper Eq. F1).  Because the Gaudin bath spins have no
self-Hamiltonian (infinite memory time), the per-time-slice local tensors are
*time-independent*: the whole correlation tensor of sub-bath ``k`` is a single
transfer tensor repeated, so this engine just precomputes one transfer tensor
per sub-bath.

The transfer tensor is expressed in the **superoperator index** convention
shared with the rest of the pipeline (see Layer 4b / the Gaussian kernel): the
physical index ``phi`` runs over ``0 .. 2 n_ch`` with

* ``phi = 0``        the identity bath superoperator ``B^0 = I``;
* ``phi = 2a - 1``   the bath commutator ``B^-_a = -i [B_a, .]``;
* ``phi = 2a``       the bath mean field ``B^+_a = (1/2){B_a, .}``,

for channels ``a = 1..n_ch`` (``d_phys = 2 n_ch + 1`` -- ``7`` for the Gaudin
spin-1/2 central spin, whose three channels couple to ``B_a = g_k J_{x/y/z}``).
This matches the system-side convention in
:mod:`edmtn.expansion.base` (where ``phi = 2a-1`` carries ``S^+_a`` paired with
bath ``B^-`` and ``phi = 2a`` carries ``S^-_a`` paired with bath ``B^+``), so a
kernel site contracts cleanly with the system superoperator family.

The transfer tensor (Eq. F1 read in this basis) is

    A_k[phi, a, a'] = (1/2) Tr[ sigma_a  B^phi_k(sigma_{a'}) ],
    sigma_0 = I,  sigma_{1/2/3} = 2 J_{x/y/z} = Pauli,

with lateral (Liouville) bond dimension ``D_a = 4``.  Fixing the newest site's
left index and the oldest site's right index to ``0`` (``sigma_0 = I``, encoding
``Omega_k = I/2`` on the right) makes the matrix product

    C_{k; phi_T, ..., phi_1} = A_k[phi_T]_{0, .} A_k[phi_{T-1}] ... A_k[phi_1]_{., 0}

equal to the exact time-ordered bath correlation

    Tr[ B^{phi_T}_k . ... . B^{phi_1}_k (Omega_k) ]

(each ``A`` carries a factor ``1/2``; the internal sigma-basis resolution of
identity collapses them so that only the single ``1/2`` of ``Omega_k`` survives).

The picking tensor that turns this correlation into a combined-kernel MPO (with
the open arms that route each noise to its later partners) is applied downstream
in the separable kernel engine (Layer 3).
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field

import numpy as np

from .base import CumulantEngine

_I2 = np.eye(2, dtype=np.complex128)


def _identity(X):
    return X


def _commutator(B):
    return lambda X: -1j * (B @ X - X @ B)


def _anticommutator(B):
    return lambda X: 0.5 * (B @ X + X @ B)


@dataclass(frozen=True)
class SeparableCorrelation:
    """Per-sub-bath correlation transfer tensors of a separable bath.

    Attributes
    ----------
    eps : float
        Time step.
    n_steps : int
        Number of steps ``N`` (sites in the time-ordered MPS).
    couplings : np.ndarray
        Per-sub-bath couplings ``g_k`` (length ``K``), in the model's stored order.
    transfer : np.ndarray
        Transfer tensors ``A[k, phi, a, a']`` of shape ``(K, d_phys, D_a, D_a)``;
        ``A[k]`` is the (time-independent) Eq.-F1 local tensor of sub-bath ``k``
        in the superoperator-index convention.

    Both arrays are **privately copied at construction and marked read-only**: mutating the
    arrays you passed in afterwards cannot change this correlation, and writing to either
    attribute raises.
    """

    eps: float
    n_steps: int
    couplings: np.ndarray = field(repr=False)
    transfer: np.ndarray = field(repr=False)

    def __post_init__(self):
        # own private read-only copies so this correlation's recorded parameters and its
        # transfer tensors can never be mutated out from under it via a shared array
        # (frozen dataclass -> object.__setattr__)
        couplings = np.array(self.couplings, dtype=np.float64, copy=True)
        transfer = np.array(self.transfer, dtype=np.complex128, copy=True)
        couplings.setflags(write=False)
        transfer.setflags(write=False)
        object.__setattr__(self, "couplings", couplings)
        object.__setattr__(self, "transfer", transfer)

    @property
    def K(self) -> int:
        """Number of sub-baths."""
        return int(self.transfer.shape[0])

    @property
    def d_phys(self) -> int:
        """Superoperator (physical) index dimension ``2 n_ch + 1``."""
        return int(self.transfer.shape[1])

    @property
    def bond_dim(self) -> int:
        """Lateral (Liouville) bond dimension ``D_a``."""
        return int(self.transfer.shape[2])

    def transfer_for(self, k: int) -> np.ndarray:
        """Transfer tensor ``A[k][phi, a, a']`` of sub-bath ``k``."""
        return self.transfer[k]

    def correlation(self, ops, k: int = 0) -> complex:
        """Superoperator correlation ``Tr[B^{phi_T}...B^{phi_1}(Omega_k)]`` from the MPS.

        ``ops`` is the superoperator index sequence in **time order**
        ``[phi_1, ..., phi_T]`` (oldest first); the product is formed
        newest-first with both boundary lateral indices fixed to ``0``.  Provided
        mainly for verification against a brute-force superoperator chain.
        """
        A = self.transfer[k]
        M = np.eye(self.bond_dim, dtype=np.complex128)
        for phi in reversed(list(ops)):  # A[phi_T] A[phi_{T-1}] ... A[phi_1]
            M = M @ A[phi]
        return complex(M[0, 0])


class SeparableBathCorrelation(CumulantEngine):
    """Eq.-F1 superoperator transfer tensors for a spin-1/2 separable bath (Gaudin).

    Handles models whose sub-baths are spin-1/2 at infinite temperature, exposing
    ``couplings`` (``g_k``) and ``bath_spin_operators()`` (``[J_x, J_y, J_z]``).
    """

    bath_type = "separable"

    def compute(self, model, T: float, eps: float) -> SeparableCorrelation:
        """Precompute the per-sub-bath transfer tensors for ``model``.

        ``T`` / ``eps`` only fix ``n_steps`` (the MPS is uniform in time); the
        transfer tensors depend on the couplings alone.
        """
        self._check_model(model)
        self._require_infinite_temperature(model)
        J, couplings = self._require_spin_half_bath(model)
        n = self._n_steps(T, eps)

        # sigma basis: sigma_0 = I, sigma_{1..} = 2 J  (Pauli for spin-1/2)
        sigma = [_I2] + [2.0 * j for j in J]
        d_a = len(sigma)               # Liouville bond dimension D_a (= 4)
        n_ch = len(J)                  # coupling channels (= 3)
        d_phys = 2 * n_ch + 1          # superoperator index dimension (= 7)

        transfer = np.empty((len(couplings), d_phys, d_a, d_a), dtype=np.complex128)
        for k, gk in enumerate(couplings):
            bath_ops = [gk * j for j in J]  # B_a = g_k J_a
            transfer[k] = self._transfer_tensor(bath_ops, sigma)

        return SeparableCorrelation(
            eps=eps,
            n_steps=n,
            couplings=np.asarray(couplings, dtype=np.float64),
            transfer=transfer,
        )

    # -- implementation ----------------------------------------------------

    @staticmethod
    def _superoperators(bath_ops):
        """Bath superoperators ``[B^0, B^-_1, B^+_1, B^-_2, B^+_2, ...]``.

        Index ``phi`` -> callable acting on a ``2 x 2`` operator, matching the
        ``0 / 2a-1 (B^-) / 2a (B^+)`` convention.
        """
        ops = [_identity]
        for B in bath_ops:
            ops.append(_commutator(B))      # phi = 2a - 1  ->  B^-
            ops.append(_anticommutator(B))  # phi = 2a      ->  B^+
        return ops

    @classmethod
    def _transfer_tensor(cls, bath_ops, sigma) -> np.ndarray:
        """``A[phi, a, a'] = (1/2) Tr[sigma_a B^phi(sigma_{a'})]`` (Eq. F1)."""
        superops = cls._superoperators(bath_ops)
        d_phys = len(superops)
        d_a = len(sigma)
        A = np.empty((d_phys, d_a, d_a), dtype=np.complex128)
        for phi, superop in enumerate(superops):
            for ap in range(d_a):
                Y = superop(sigma[ap])  # B^phi(sigma_{a'})
                for a in range(d_a):
                    A[phi, a, ap] = 0.5 * np.trace(sigma[a] @ Y)
        return A

    @staticmethod
    def _require_infinite_temperature(model) -> None:
        # only real POSITIVE infinity is the maximally-mixed bath this engine assumes;
        # -inf / finite / nan / bool must all fail loudly (np.isinf accepted -inf)
        temp = model.bath_params().temperature
        if isinstance(temp, bool) or not isinstance(temp, numbers.Real) or temp != math.inf:
            raise NotImplementedError(
                "SeparableBathCorrelation currently supports infinite-temperature "
                f"(maximally mixed, +inf) sub-baths only (got temperature={temp!r})"
            )

    @staticmethod
    def _require_spin_half_bath(model):
        """Return ``([J_x, J_y, J_z], couplings)``; validate the spin-1/2 interface."""
        if not (hasattr(model, "bath_spin_operators") and hasattr(model, "couplings")):
            raise NotImplementedError(
                "SeparableBathCorrelation needs a model exposing bath_spin_operators() "
                "and couplings (spin-1/2 separable bath, e.g. GaudinModel)"
            )
        J = model.bath_spin_operators()
        for op in J:
            if op.shape != (2, 2):
                raise NotImplementedError(
                    "SeparableBathCorrelation supports spin-1/2 sub-baths only "
                    f"(got bath operator shape {op.shape})"
                )
        return J, np.asarray(model.couplings, dtype=np.float64)
