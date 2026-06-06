"""Kernel-tensor engine interface (Layer 3).

A kernel provider supplies, for each evolution step, the *combined kernel
tensor* that advances the extended density matrix.  The combined kernel is
expressed as a matrix-product operator (MPO) over the time-ordered sites; the
evolution engine (Layer 5) contracts it with the EDM-MPS and recompresses.

Index conventions for a :class:`KernelMPO` site tensor ``T``:

    T[phi_up, phi_down, a_left, a_right]

* ``phi_up`` (open arm) and ``phi_down`` (physical in) each range over the
  ``d_phys`` superoperator indices.  For a single-channel bath ``d_phys = 3``
  (``0`` = null, ``1`` = ``B^-`` commutator, ``2`` = ``B^+`` mean-field).
* ``a_left`` / ``a_right`` are the lateral bond indices.

Site tensors are ordered latest-first: ``site_tensors[0]`` is the newest site
(step ``t``), ``site_tensors[-1]`` is the oldest (step ``1``).  The newest site
carries the new physical noise index that contracts with the system
superoperator; the older sites' ``phi_down`` indices contract with the existing
EDM open arms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


def picking_tensor(d_phys: int) -> np.ndarray:
    """Return the picking tensor ``P[phi_up, phi_mid, phi_down]``.

    ``P = delta^{up}_{down} delta^{mid}_0 + delta^{mid}_{down} delta^{up}_0
          - delta^0_{down} delta^{up}_0 delta^{mid}_0``.

    It routes a noise either to its open arm (``phi_up = phi_down``, ``phi_mid =
    0``) or into a cumulant (``phi_up = 0``, ``phi_mid = phi_down``); the null
    index ``0`` maps to ``(0, 0)``.
    """
    P = np.zeros((d_phys, d_phys, d_phys), dtype=np.complex128)
    for down in range(d_phys):
        for up in range(d_phys):
            for mid in range(d_phys):
                val = (
                    (up == down) * (mid == 0)
                    + (mid == down) * (up == 0)
                    - (down == 0) * (up == 0) * (mid == 0)
                )
                P[up, mid, down] = val
    return P


@dataclass
class KernelMPO:
    """Combined kernel tensor for one evolution step, as a time-ordered MPO.

    Attributes
    ----------
    site_tensors : list[np.ndarray]
        Site tensors ``T[phi_up, phi_down, a_left, a_right]`` ordered latest
        (step ``t``) first to oldest (step ``1``) last.
    t : int
        Evolution step this kernel builds (number of sites).
    d_phys : int
        Physical (superoperator) index dimension.
    """

    site_tensors: list[np.ndarray]
    t: int
    d_phys: int

    @property
    def bond_dims(self) -> list[int]:
        """Right-bond dimension of each site tensor."""
        return [T.shape[3] for T in self.site_tensors]

    def to_dense(self) -> np.ndarray:
        """Contract the lateral bonds into a dense tensor.

        Returns an array indexed ``[phi_up(t), ..., phi_up(1),
        phi_down(t), ..., phi_down(1)]`` (``2 * t`` axes of size ``d_phys``).
        Intended for testing / small-``t`` reference contractions only.
        """
        p = self.d_phys
        first = self.site_tensors[0]
        # (phi_up_t, phi_down_t, a_right)   -- left boundary bond is dim 1
        acc = first.reshape(p, p, first.shape[3])
        for T in self.site_tensors[1:]:
            # contract running bond (last axis) with this site's left bond
            acc = np.tensordot(acc, T, axes=([acc.ndim - 1], [2]))
            # acc now ends with (..., phi_up_j, phi_down_j, a_right)
        acc = acc[..., 0]  # drop the trivial right boundary bond
        # axes are interleaved [u_t, d_t, u_{t-1}, d_{t-1}, ...]; split them
        t = self.t
        up_axes = [2 * i for i in range(t)]
        down_axes = [2 * i + 1 for i in range(t)]
        return np.transpose(acc, up_axes + down_axes)


class KernelProvider(ABC):
    """Base class for kernel-tensor engines."""

    @abstractmethod
    def get_kernel_mpo(self, t: int) -> KernelMPO:
        """Return the combined kernel MPO that builds the EDM at step ``t``."""

    def memory_time(self) -> int | None:
        """Finite bath memory time in steps, or ``None``."""
        return None
