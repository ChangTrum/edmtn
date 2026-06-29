"""Gaussian-bath combined-kernel engine (Layer 3).

For a single-channel Gaussian bath the combined kernel tensor has the closed
form (Appendix E of the EDM paper) with a fixed lateral bond dimension of 2.
Reading the time-ordered MPO from the newest site (step ``t``) to the oldest
(step ``1``), the raw per-site tensors built from the bath correlation are

    newest site (1 x 2 boundary):
        Qraw[0] = [1, 0],  Qraw[1] = [0, 0],  Qraw[2] = [0, 1]

    interior site j, lag m = t - j (2 x 2):
        Qraw[0] = I,
        Qraw[1] = [[0, 0], [2 Im f(m), 0]],
        Qraw[2] = [[0, 0], [Re f(m), 0]]

    oldest site 1, lag m = t - 1 (2 x 1 boundary):
        Rraw[0] = [1, 0]^T,
        Rraw[1] = [0, 2 Im f(m)]^T,
        Rraw[2] = [0, Re f(m)]^T

The physical operator tensor is obtained by contracting the picking tensor,
``T[up, down, l, r] = sum_mid P[up, mid, down] Qraw[mid, l, r]``.  The lateral
bond carries whether the newest noise is still awaiting its (single, second
order) pairing partner; an interior ``Qraw[1]/Qraw[2]`` closes that pairing,
contributing the corresponding cumulant.

Second order (``order=2``)
--------------------------
The second-order Trotter expansion (Appendix E, Eq. E5-E6) replaces each step's
``B^phi S^phi`` with ``B^phi B^psi S^phi_2 S^psi_1`` -- two bath insertions per
physical step.  The evolution then runs on a *doubled* sub-step grid of length
``2 N``: sub-step ``g`` lives at physical time ``ceil(g / 2)`` (the odd sub-step
carries ``S_1``, the even one ``S_2``).  The pairwise cumulant between two
sub-steps depends on their *physical*-time separation, not their sub-step
separation.  For the kernel built at sub-step ``t`` the site at offset ``k`` from
the newest pairs with physical lag

    lag(t, k) = (k + (t mod 2)) // 2,

which equals ``ceil(t/2) - ceil((t-k)/2)`` for every ``t, k`` (so the recursive
advance reproduces all pairwise cumulants ``f(physical lag)``, including the
within-step sibling pairing at ``lag = 0``).  The lateral structure is otherwise
identical to first order, so only the lag map changes.
"""

from __future__ import annotations

import numpy as np
import opt_einsum as oe

from ..cumulants.gaussian import GaussianCumulants
from .base import KernelMPO, KernelProvider, picking_tensor

# single-channel superoperator dimension: 0 (null), 1 (B^-), 2 (B^+)
_D_PHYS = 3


class GaussianKernelEngine(KernelProvider):
    """Closed-form combined kernel for a zero-temperature Gaussian bath.

    Parameters
    ----------
    cumulants : GaussianCumulants
        Second-order cumulants on the step grid (from the Layer-2 engine).
    """

    def __init__(self, cumulants: GaussianCumulants, order: int = 1):
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        self.cum = cumulants
        self.order = order
        self.d_phys = _D_PHYS
        self._P = picking_tensor(_D_PHYS)

    @classmethod
    def from_model(cls, model, T: float, eps: float, method: str = "auto", order: int = 1):
        """Build the engine by running the Gaussian cumulant engine on ``model``.

        The cumulants are always sampled on the *physical* grid (``eps``, ``T``);
        a second-order engine drives them on the doubled sub-step grid via the
        parity-dependent lag map.
        """
        from ..cumulants.gaussian import GaussianCumulantEngine

        cum = GaussianCumulantEngine(method=method).compute(model, T, eps)
        return cls(cum, order=order)

    # -- public API --------------------------------------------------------

    def _lag(self, t: int, offset: int) -> int:
        """Physical-time lag for the site at ``offset`` from the newest at step ``t``."""
        if self.order == 1:
            return offset
        return (offset + (t % 2)) // 2

    def get_kernel_mpo(self, t: int) -> KernelMPO:
        if t < 1:
            raise ValueError(f"t must be >= 1, got {t}")
        max_lag = self._lag(t, t - 1) if t > 1 else 0
        if max_lag > self.cum.n_steps:
            raise ValueError(
                f"kernel at step t={t} needs cumulant lag {max_lag} > "
                f"available {self.cum.n_steps}"
            )
        if t == 1:
            return KernelMPO([self._operatorize(self._raw_single())], t=1, d_phys=self.d_phys)

        sites = [self._operatorize(self._raw_newest())]
        for offset in range(1, t - 1):  # interior sites at offsets 1 .. t-2
            sites.append(self._operatorize(self._raw_interior(lag=self._lag(t, offset))))
        sites.append(self._operatorize(self._raw_oldest(lag=self._lag(t, t - 1))))
        return KernelMPO(sites, t=t, d_phys=self.d_phys)

    def memory_time(self) -> int | None:
        return None

    # -- raw tensors -------------------------------------------------------

    def _cum_pair(self, lag: int) -> tuple[complex, complex]:
        """Return ``(2 Im f, Re f)`` at integer ``lag`` (the B^-/B^+ cumulants)."""
        return complex(self.cum.im2[lag]), complex(self.cum.re[lag])

    @staticmethod
    def _raw_single() -> np.ndarray:
        # t == 1: no pairing possible; raw selector delta^0 -> identity operator
        raw = np.zeros((_D_PHYS, 1, 1), dtype=np.complex128)
        raw[0, 0, 0] = 1.0
        return raw

    @staticmethod
    def _raw_newest() -> np.ndarray:
        raw = np.zeros((_D_PHYS, 1, 2), dtype=np.complex128)
        raw[0, 0, 0] = 1.0  # null -> channel 0
        raw[2, 0, 1] = 1.0  # B^+ at newest -> pending-pairing channel 1
        # B^- (index 1) at the newest time vanishes
        return raw

    def _raw_interior(self, lag: int) -> np.ndarray:
        im2, re = self._cum_pair(lag)
        raw = np.zeros((_D_PHYS, 2, 2), dtype=np.complex128)
        raw[0] = np.eye(2)  # null passes the bond
        raw[1, 1, 0] = im2  # close pairing with newest, B^- partner -> 2 Im f
        raw[2, 1, 0] = re   # close pairing with newest, B^+ partner -> Re f
        return raw

    def _raw_oldest(self, lag: int) -> np.ndarray:
        im2, re = self._cum_pair(lag)
        raw = np.zeros((_D_PHYS, 2, 1), dtype=np.complex128)
        raw[0, 0, 0] = 1.0
        raw[1, 1, 0] = im2
        raw[2, 1, 0] = re
        return raw

    def _operatorize(self, raw: np.ndarray) -> np.ndarray:
        """Apply the picking tensor: ``T[up, down, l, r] = P[up, mid, down] raw[mid, l, r]``."""
        # P[up, mid, down], raw[mid, l, r] -> T[up, down, l, r]
        return oe.contract("amd,mlr->adlr", self._P, raw, optimize="auto")
