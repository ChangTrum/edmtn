"""Observable extraction from the EDM-MPS (Layer 6).

Two complementary extractions:

* **Reduced density matrix** ``rho(t) = delta^0_{Phi} rho^{Phi}`` -- close every
  open arm with the closing tensor.  Any single-time expectation follows as
  ``<O(t)> = Tr[O(t) rho(t)]`` (with ``O(t)`` the interaction-picture operator).

* **Coupling-channel polarization history** (Eq. F2).  From a *single* final EDM
  ``rho^{Phi(T)}`` the expectation of the coupling-channel operator at every
  intermediate time ``t`` is read by setting the open arm at time ``t`` to the
  ``S^+`` selector (index ``2 a - 1``) and closing all others::

      <S_a(t)> = eps^{-1} Tr[ rho^{Phi(T)} delta^{2a-1}_{phi_t} delta^0_{rest} ].

  Because the newest open-arm slice already carries ``eps S^+_a`` (Layer 5
  ``apply_step``), the ``eps^{-1}`` exactly cancels it.  A left/right
  environment sweep evaluates all times in ``O(T D^2)``.

The selector convention matches Layers 3/4b: arm index ``1`` is the ``S^+``
(``B^-``-paired) channel of coupling channel 1.
"""

from __future__ import annotations

import numpy as np

from ..evolution.mps_utils import _xp

# A coupling-channel polarization is real; any imaginary part is numerical
# (Trotter + truncation) error.  First-order evolution leaves an artifact of
# order 1e-6 relative even at strong coupling/long time, so the guard only flags
# *gross* leakage (a wrong index/convention gives an imaginary part O(1)).
_IMAG_REL_TOL = 1e-3


def _vec_identity(d, like):
    """Row-major ``vec(I_d)`` as a 1-D array on the same backend as ``like``."""
    xp = _xp(like)
    return xp.asarray(np.eye(d, dtype=np.complex128).reshape(-1))


class ObservableExtractor:
    """Extract reduced states and expectation histories from an EDM-MPS."""

    # -- single-time -------------------------------------------------------

    @staticmethod
    def density_matrix(mps):
        """Reduced density matrix ``rho(t)`` at the MPS's current (final) time."""
        return mps.reduced_density_matrix()

    @staticmethod
    def trace(mps):
        """``Tr[rho(t)]`` (should stay ``1`` up to truncation error)."""
        return complex(np.trace(_as_numpy(mps.reduced_density_matrix())))

    @staticmethod
    def trace_deviation(mps) -> float:
        """``|Tr[rho(t)] - 1|`` -- a cheap precision indicator."""
        return float(abs(ObservableExtractor.trace(mps) - 1.0))

    @staticmethod
    def expectation(mps, operator) -> complex:
        """``<O> = Tr[O rho(t)]`` for an operator at the MPS's final time."""
        rho = _as_numpy(mps.reduced_density_matrix())
        op = np.asarray(operator)
        return complex(np.trace(op @ rho))

    # -- all-times coupling-channel history (Eq. F2) -----------------------

    @staticmethod
    def coupling_polarization_history(mps, eps, *, channel: int = 1):
        """``<S_a(t)>`` for the coupling channel ``a = channel`` at every time.

        Returns ``(times, values)`` ascending in time, with ``values`` real
        (the imaginary part is asserted negligible for a Hermitian observable).
        Uses a single left/right environment sweep over the final EDM.
        """
        xp = _xp(mps.tensors[0])
        n = mps.num_sites
        sel = 2 * channel - 1  # S^+ selector of channel `a`
        if not 0 < sel < mps.d_phys:
            raise ValueError(f"channel {channel} out of range for d_phys={mps.d_phys}")

        zero_mats = [t[0] for t in mps.tensors]   # phi_up = 0 slices
        sel_mats = [t[sel] for t in mps.tensors]  # phi_up = 2a-1 slices

        # left environments: e_L[p] = vec(I)^T . prod_{q<p} M_q[0]
        left = [None] * n
        left[0] = _vec_identity(mps.d, mps.tensors[0])
        for p in range(1, n):
            left[p] = left[p - 1] @ zero_mats[p - 1]
        # right environments: e_R[p] = prod_{q>p} M_q[0] . vec(rho0)
        right = [None] * n
        right[n - 1] = mps.rho0_vec
        for p in range(n - 2, -1, -1):
            right[p] = zero_mats[p + 1] @ right[p + 1]

        times = np.empty(n, dtype=np.float64)
        values = np.empty(n, dtype=np.complex128)
        for p in range(n):
            val = left[p] @ (sel_mats[p] @ right[p])
            times[n - 1 - p] = (n - p) * eps
            values[n - 1 - p] = complex(_scalar(val)) / eps

        if np.max(np.abs(values.imag)) > _IMAG_REL_TOL * (np.max(np.abs(values.real)) + 1e-12):
            raise ValueError("coupling polarization has a non-negligible imaginary part")
        return times, values.real

    # -- history from recorded reduced states (general operator) -----------

    @staticmethod
    def expectation_history(density_matrices, times, operator_fn):
        """``<O(t)> = Tr[O(t) rho(t)]`` over recorded reduced states.

        ``operator_fn(t)`` returns the (interaction-picture) operator at time
        ``t``.  Works for any single-system operator, unlike the Eq.-F2 sweep
        which is restricted to coupling channels.
        """
        out = np.empty(len(density_matrices), dtype=np.complex128)
        for i, (rho, t) in enumerate(zip(density_matrices, times)):
            op = np.asarray(operator_fn(t))
            out[i] = np.trace(op @ _as_numpy(rho))
        return np.asarray(times, dtype=np.float64), out


def _as_numpy(a):
    if type(a).__module__.split(".")[0] == "cupy":
        import cupy as cp  # noqa: PLC0415

        return cp.asnumpy(a)
    return np.asarray(a)


def _scalar(a):
    """Coerce a 0-d / length-1 array (NumPy or CuPy) to a Python complex."""
    return complex(_as_numpy(a).reshape(()))
