"""Separable-bath kernel engine (Layer 3).

For a separable bath the per-sub-bath correlation tensor is already an MPS (the
Layer-2 :class:`~edmtn.cumulants.separable.SeparableCorrelation` transfer
tensors).  Turning it into the combined-kernel MPO that the evolution contracts
only requires attaching the picking tensor at every time site, exactly as the
Gaussian engine does (no cumulant->kernel construction is needed here):

    T_k[phi_up, phi_down, a_left, a_right]
        = sum_mid  P[phi_up, mid, phi_down]  A_k[mid, a_left, a_right],

with ``A_k`` the (time-independent) superoperator transfer tensor of sub-bath
``k`` and ``P`` the picking tensor (:func:`edmtn.kernels.base.picking_tensor`).
Because ``A_k`` is the same at every time slice, the MPO is *uniform*; the
boundaries fix the newest site's left lateral index and the oldest site's right
lateral index to ``0`` (``sigma_0 = I`` / ``Omega_k = I/2``), matching Eq. F1.

The separable bath is solved by an outer recursion over sub-baths (Eq. 21,
Fig. 5c): each sub-bath ``k`` contributes one such correlation MPO, applied to
the running EDM.  This engine therefore exposes a per-sub-bath
:class:`KernelProvider` via :meth:`for_sub_bath`; the Layer-5 separable evolution
loops ``k = 1..K`` and consumes one provider per sub-bath.

Closing all open arms (``phi_up = 0``) reduces the operatorised site back to the
raw transfer tensor (``P[0, mid, down] = delta_{mid, down}``), so the kernel's
all-arms-closed contraction recovers the bare superoperator correlation -- the
hook used to validate this layer against Layer 2.
"""

from __future__ import annotations

import numpy as np

from ..cumulants.separable import SeparableCorrelation
from .base import KernelMPO, KernelProvider, picking_tensor


class _SubBathKernel(KernelProvider):
    """Combined-kernel MPO provider for a single separable sub-bath.

    Holds the operatorised, time-independent site tensor and slices the lateral
    boundaries for the newest / oldest sites on demand.
    """

    def __init__(self, op_tensor: np.ndarray, d_phys: int):
        # op_tensor[phi_up, phi_down, a_left, a_right]
        self._op = op_tensor
        self.d_phys = d_phys

    def get_kernel_mpo(self, t: int) -> KernelMPO:
        if t < 1:
            raise ValueError(f"t must be >= 1, got {t}")
        op = self._op
        if t == 1:
            site = op[:, :, 0:1, 0:1]  # both lateral boundaries -> 0
            return KernelMPO([site], t=1, d_phys=self.d_phys)
        newest = op[:, :, 0:1, :]      # left boundary -> 0
        oldest = op[:, :, :, 0:1]      # right boundary -> 0
        sites = [newest] + [op] * (t - 2) + [oldest]
        return KernelMPO(sites, t=t, d_phys=self.d_phys)

    def memory_time(self) -> int | None:
        return None  # Gaudin bath spins are time-independent: infinite memory


class SeparableKernelEngine:
    """Per-sub-bath combined-kernel MPOs for a separable bath (Gaudin).

    Parameters
    ----------
    correlation : SeparableCorrelation
        Layer-2 transfer tensors (one per sub-bath), in the superoperator-index
        convention.
    """

    def __init__(self, correlation: SeparableCorrelation):
        self.corr = correlation
        self.d_phys = correlation.d_phys
        self.K = correlation.K
        self._P = picking_tensor(self.d_phys)
        # operatorise every sub-bath: op[k, up, down, l, r] = P[up, mid, down] A[k, mid, l, r]
        self._op = np.einsum("amd,kmlr->kadlr", self._P, correlation.transfer)  # tiny, one-shot

    @classmethod
    def from_model(cls, model, T: float, eps: float) -> "SeparableKernelEngine":
        """Build the engine by running the separable correlation engine on ``model``."""
        from ..cumulants.separable import SeparableBathCorrelation

        corr = SeparableBathCorrelation().compute(model, T, eps)
        return cls(corr)

    def for_sub_bath(self, k: int) -> _SubBathKernel:
        """Return the :class:`KernelProvider` for sub-bath ``k``."""
        if not 0 <= k < self.K:
            raise IndexError(f"sub-bath index {k} out of range 0..{self.K - 1}")
        return _SubBathKernel(self._op[k], self.d_phys)

    def get_kernel_mpo(self, t: int, k: int) -> KernelMPO:
        """Combined-kernel MPO that builds sub-bath ``k``'s contribution at ``t`` sites."""
        return self.for_sub_bath(k).get_kernel_mpo(t)

    def memory_time(self) -> int | None:
        return None
