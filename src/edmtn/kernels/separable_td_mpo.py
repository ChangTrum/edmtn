"""Time-dependent separable-bath kernel engine (Layer 3).

Same construction as :mod:`edmtn.kernels.separable_mpo` -- attach the picking tensor to
each Eq.-F1 transfer tensor -- but the transfer tensor now differs from one time site to
the next, and the initial bath state enters through the oldest site's boundary::

    T_k[g][phi_up, phi_down, a_left, a_right]
        = sum_mid  P[phi_up, mid, phi_down]  A_k[g][mid, a_left, a_right]

**Two orderings run opposite ways and must not be conflated.**  The correlation array is
stored **oldest first** (``transfer[g-1]``, ``g = 1 .. n_sites``), while
:class:`~edmtn.kernels.base.KernelMPO` stores sites **newest first**.  Hence

    site_tensors[p] = operatorised( transfer[n_sites - 1 - p] ),   p = 0 .. n_sites - 1

Boundaries: the newest site's left lateral index is fixed to ``0``, and the oldest site's
right lateral index is contracted with the bath Bloch boundary vector
``r_k = (1, r_x, r_y, r_z)``.  For a maximally mixed bath ``r_k = (1, 0, 0, 0)`` and the
contraction reduces exactly to the time-independent engine's "slice to index 0".

The engine records the **full grid signature** ``(eps, n_steps, order)`` and exposes
:meth:`SeparableTDKernelEngine.check_grid` so the evolution can reject a kernel built for
a different grid.  The site count alone is not enough: ``order=1, N=4`` and
``order=2, N=2`` share it, as do two runs that differ only in ``eps`` -- and a mismatch
there is not a crash but a silently wrong Hamiltonian sampling.
"""

from __future__ import annotations

import numbers

import numpy as np

from ..cumulants.separable_td import TimeDependentSeparableCorrelation
from .base import KernelMPO, KernelProvider, picking_tensor


class _SubBathTDKernel(KernelProvider):
    """Combined-kernel MPO provider for one time-dependent separable sub-bath.

    Holds the operatorised, **oldest-first** site tensors of a single sub-bath plus its
    boundary vector, and assembles the newest-first :class:`KernelMPO` on demand.
    """

    def __init__(self, op_tensor: np.ndarray, boundary: np.ndarray, d_phys: int):
        # op_tensor[g-1, phi_up, phi_down, a_left, a_right], oldest first
        self._op = op_tensor
        self._r = boundary
        self.d_phys = d_phys
        self.n_sites = int(op_tensor.shape[0])

    def get_kernel_mpo(self, t: int, closing=None) -> KernelMPO:
        """Sites for a ``t``-site chain; ``t`` must be the grid this kernel was built for.

        ``closing`` selects what the newest site's left lateral index is contracted with.
        The lateral index carries the Pauli-coefficient vector ``x_a = Tr[sigma_a X]`` of
        the operator the chain has accumulated, so the default ``None`` -- which keeps the
        index-``0`` slice -- takes ``Tr X``, i.e. does not measure the bath.  A length-4
        vector ``v`` instead returns ``Tr[(sum_a v_a sigma_a) X]``, which is what turns
        this fold into a bath-side measurement insertion.  What ``v`` *means* is decided by
        the model (see ``DickeModel.collective_spin_closures``); this layer knows only the
        contraction.

        The contraction is **bilinear and does not conjugate** ``v``: a complex closing is
        used as given, and conjugating it would silently measure the adjoint operator.
        """
        if isinstance(t, bool) or not isinstance(t, numbers.Integral):
            raise ValueError(f"t must be an integer, got {t!r}")
        t = int(t)
        if t != self.n_sites:
            raise ValueError(
                f"this time-dependent kernel was built for {self.n_sites} sites but was "
                f"asked for {t}; the per-site tensors are grid-specific, so a mismatch "
                f"would sample the bath at the wrong times")
        if closing is not None:
            closing = np.asarray(closing, dtype=np.complex128)
            lateral = int(self._op.shape[3])
            if closing.shape != (lateral,):
                raise ValueError(
                    f"closing must be a length-{lateral} lateral vector, got shape "
                    f"{closing.shape}")
            if not np.all(np.isfinite(closing)):
                raise ValueError("closing must be finite")

        # newest-first: site p carries sub-step g = n_sites - p, i.e. index n_sites-1-p
        sites = [self._op[self.n_sites - 1 - p] for p in range(self.n_sites)]
        # oldest site: contract the right lateral index with the bath boundary vector r_k.
        # This runs **before** the newest-site closing so that at n_sites == 1, where both
        # boundaries land on the same tensor, the two contractions still compose.
        sites[-1] = np.tensordot(sites[-1], self._r, axes=([3], [0]))[..., None]
        if closing is None:
            # newest site: fix the left lateral index to 0
            sites[0] = sites[0][:, :, 0:1, :]
        else:
            sites[0] = np.tensordot(sites[0], closing, axes=([2], [0]))[:, :, None, :]
        return KernelMPO(sites, t=t, d_phys=self.d_phys)

    def memory_time(self) -> int | None:
        # no memory-time cutoff is imposed; with non-zero rates the correlation decays on
        # its own, but the pipeline never truncates the history.
        return None


class SeparableTDKernelEngine:
    """Per-sub-bath combined-kernel MPOs for a time-dependent separable bath (Dicke).

    Sub-bath tensors are built **lazily**, one at a time in :meth:`for_sub_bath`, so the
    outer fold loop never holds ``K`` operatorised arrays at once.  One provider's array
    is ``n_sites * d_phys**2 * 16`` complex entries (about 1.8 MB at ``order = 2``,
    ``N = 400``); that is the provider's own storage and does **not** include the fold and
    compression working set, which is where the memory actually goes.

    Parameters
    ----------
    correlation : TimeDependentSeparableCorrelation
        Layer-2 description of the grid, the per-sub-bath transfer tensors and the bath
        boundary vectors.
    """

    #: this engine's sub-bath providers accept a lateral ``closing`` vector, so a
    #: bath-side measurement insertion can be folded.  The evolution checks this flag
    #: rather than the bath-type string: a kernel without it would otherwise reject the
    #: argument deep inside the fold loop, as a bare ``TypeError``.
    supports_closings = True

    def __init__(self, correlation: TimeDependentSeparableCorrelation):
        self.corr = correlation
        self.d_phys = correlation.d_phys
        self.K = correlation.K
        self.order = correlation.order
        self.n_steps = correlation.n_steps
        self.eps = correlation.eps
        self.n_sites = correlation.n_sites
        self._P = picking_tensor(self.d_phys)

    @classmethod
    def from_model(cls, model, T: float, eps: float, order: int) -> "SeparableTDKernelEngine":
        """Build the engine by running the time-dependent correlation engine on ``model``.

        ``order`` must be the **resolved** expansion order the evolution will run at; it
        is passed in rather than read from the model so the kernel and the evolution
        cannot disagree about the sub-step map.
        """
        from ..cumulants.separable_td import SeparableTDBathCorrelation  # noqa: PLC0415

        return cls(SeparableTDBathCorrelation().compute(model, T, eps, order))

    @property
    def grid_signature(self) -> tuple:
        """``(eps, n_steps, order)`` of the grid this kernel was built for."""
        return self.corr.grid_signature

    def check_grid(self, eps: float, n_steps: int, order: int) -> None:
        """Raise unless the caller's time grid is exactly the one this kernel encodes.

        Called by the evolution **before** any tensor is built, so a mismatched grid fails
        fast instead of producing a silently wrong trajectory.
        """
        got = (float(eps), int(n_steps), int(order))
        if got != self.grid_signature:
            raise ValueError(
                f"kernel/evolution time-grid mismatch: the kernel was built for "
                f"(eps, n_steps, order)={self.grid_signature} but the evolution runs "
                f"{got}; the per-site tensors are grid-specific, so continuing would "
                f"sample the bath at the wrong times")

    def for_sub_bath(self, k: int) -> _SubBathTDKernel:
        """Return the :class:`KernelProvider` for sub-bath ``k`` (built on demand)."""
        if isinstance(k, bool) or not isinstance(k, numbers.Integral):
            raise ValueError(f"sub-bath index must be an integer, got {k!r}")
        k = int(k)
        if not 0 <= k < self.K:
            raise IndexError(f"sub-bath index {k} out of range 0..{self.K - 1}")
        # op[g, up, down, l, r] = P[up, mid, down] A[g, mid, l, r]
        op = np.einsum("amd,gmlr->gadlr", self._P, self.corr.transfer_for(k))
        return _SubBathTDKernel(op, self.corr.boundary_vector(k), self.d_phys)

    def get_kernel_mpo(self, t: int, k: int, closing=None) -> KernelMPO:
        """Combined-kernel MPO that builds sub-bath ``k``'s contribution over ``t`` sites.

        ``closing`` is passed through to :meth:`_SubBathTDKernel.get_kernel_mpo`.
        """
        return self.for_sub_bath(k).get_kernel_mpo(t, closing)

    def memory_time(self) -> int | None:
        return None
