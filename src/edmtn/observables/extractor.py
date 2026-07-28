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

import math

import numpy as np

from ..evolution.mps_utils import _xp

# A coupling-channel polarization is real; any imaginary part is numerical
# (Trotter + truncation) error.  First-order evolution leaves an artifact of
# order 1e-6 relative even at strong coupling/long time, so the guard only flags
# *gross* leakage (a wrong index/convention gives an imaginary part O(1)).
_IMAG_REL_TOL = 1e-3
# ... and an absolute floor, because the relative rule is vacuous on a channel whose whole
# history is zero (channels 1/2 can be, exactly, by symmetry): there `values` is pure
# roundoff, whose imaginary part is no smaller than its real part.  The absolute term
# dominates when `_IMAG_REL_TOL * real_max < _IMAG_ABS_TOL`, i.e. `real_max < 1e-9`.
_IMAG_ABS_TOL = 1e-12


def _check_imaginary_part(values) -> None:
    """Raise unless ``values`` is real to within the guard, else return.

    A module-level helper rather than an inline expression so the tests exercise the
    SHIPPED rule: a copy of the formula in the test file would keep passing even if this
    guard were deleted outright.
    """
    imag_max = float(np.max(np.abs(values.imag)))
    real_max = float(np.max(np.abs(values.real)))
    threshold = max(_IMAG_ABS_TOL, _IMAG_REL_TOL * real_max)
    if imag_max > threshold:
        raise ValueError(
            f"coupling polarization has a non-negligible imaginary part: "
            f"max|Im| = {imag_max:.3e} > threshold {threshold:.3e} "
            f"(max|Re| = {real_max:.3e}, rel_tol = {_IMAG_REL_TOL:.1e}, "
            f"abs_tol = {_IMAG_ABS_TOL:.1e})")


def finite_complex_expectation(name: str, value) -> complex:
    """Return ``value`` as a Python ``complex``, refusing a non-finite result.

    A ``nan`` or ``inf`` reaching an observable means the computation broke down on legal
    parameters -- an overflowing contraction, a degenerate decomposition -- so it raises
    ``FloatingPointError``, the project's signal for exactly that, rather than being
    returned as a plausible-looking number.  This check must come **first**: a comparison
    against ``nan`` is always ``False``, so any tolerance test placed before it silently
    passes.
    """
    v = complex(value)
    if not (math.isfinite(v.real) and math.isfinite(v.imag)):
        raise FloatingPointError(
            f"{name} is not finite ({v!r}); the computation produced a non-finite value "
            f"from legal parameters")
    return v


def real_scalar_expectation(name: str, value) -> float:
    """Return ``Re value`` as a Python ``float``, refusing a gross imaginary leak.

    The scalar counterpart of :func:`_check_imaginary_part`, for an expectation that is
    physically real (a photon-number moment, ``<J_z>``): the same absolute-or-relative
    rule, so the project has one tolerance pair rather than two that can drift.  ``value``
    must already be a Python scalar -- reduce on the array's own backend and call
    ``.item()`` *last*, so a CuPy result crosses the device boundary once, as one number.

    Not for a quantity that is complex by construction -- ``<J_+>`` has a real and an
    imaginary part that are two different observables, and guarding it would reject valid
    physics.  Use :func:`finite_complex_expectation` for those; it is applied here first,
    so a ``nan`` raises instead of sliding through the comparison below.
    """
    v = finite_complex_expectation(name, value)
    threshold = max(_IMAG_ABS_TOL, _IMAG_REL_TOL * abs(v.real))
    if abs(v.imag) > threshold:
        raise ValueError(
            f"{name} is physically real but has a non-negligible imaginary part: "
            f"|Im| = {abs(v.imag):.3e} > threshold {threshold:.3e} "
            f"(|Re| = {abs(v.real):.3e}, rel_tol = {_IMAG_REL_TOL:.1e}, "
            f"abs_tol = {_IMAG_ABS_TOL:.1e})")
    return float(v.real)


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
    def coupling_polarization_history(mps, eps, *, channel: int = 1, order: int = 1):
        """``<S_a(t)>`` for the coupling channel ``a = channel`` at every time.

        Returns ``(times, values)`` ascending in time, with ``values`` real
        (the imaginary part is asserted negligible for a Hermitian observable).
        Uses a single left/right environment sweep over the final EDM.

        * ``order = 1`` (Eq. F2): select arm ``2a-1`` at every site; the slice
          already carries ``eps S^+_a`` so a ``1/eps`` factor cancels it.
        * ``order = 2`` (Eq. F3): on the doubled sub-step grid, select arm
          ``2a-1`` at each ``S_1`` sub-step (odd sub-step index), with prefactor
          ``(1+i)/eps``; the ``S_1`` slice carries ``(1-i)/2 eps S^+_a`` and
          ``(1+i)(1-i)/2 = 1``, so this returns one value per physical step.
        """
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {order}")
        # same shared validator as the solver/HPC path; the EDM-MPS carries d_phys = 2*n_ch+1
        from ..models.base import validate_channel  # noqa: PLC0415
        channel = validate_channel(channel, (mps.d_phys - 1) // 2)
        n = mps.num_sites
        sel = 2 * channel - 1  # S^+ selector of channel `a`
        if not 0 < sel < mps.d_phys:  # internal consistency defense (type check is above)
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

        if order == 1:
            times = np.empty(n, dtype=np.float64)
            values = np.empty(n, dtype=np.complex128)
            for p in range(n):
                val = left[p] @ (sel_mats[p] @ right[p])
                times[n - 1 - p] = (n - p) * eps
                values[n - 1 - p] = complex(_scalar(val)) / eps
        else:
            # one physical step per pair of sub-steps; sub-step g = n - p, with
            # odd g the S_1 (psi) channel.  m = (g+1)//2 is the physical step.
            n_phys = n // 2
            coeff = (1.0 + 1.0j) / eps
            times = np.empty(n_phys, dtype=np.float64)
            values = np.empty(n_phys, dtype=np.complex128)
            for p in range(n):
                g = n - p
                if g % 2 == 1:  # S_1 sub-step
                    m = (g + 1) // 2  # physical step 1..n_phys
                    val = left[p] @ (sel_mats[p] @ right[p])
                    times[m - 1] = m * eps
                    values[m - 1] = coeff * complex(_scalar(val))

        # Absolute OR relative, whichever is larger.  The relative part catches a genuine
        # selector/index leak on an O(1) trajectory; the absolute floor is what makes the
        # test meaningful when the channel's whole history is zero or near-zero (which
        # channels 1/2 can be exactly, by symmetry).  There, `values` is pure roundoff with
        # no reason for its imaginary part to be smaller than its real part, so a purely
        # relative rule degenerates into comparing noise with noise: the previous
        # `rel * (real_max + 1e-12)` form left a 1e-15 floor, which a compression carrying
        # slightly more arithmetic could cross while agreeing with an uncompressed run to
        # 9e-16 on every gauge-invariant contraction (measured: `dm_tracking` 3.2e-15 vs
        # `zipup` 9.2e-17 on an identically-zero Gaudin channel).  1e-12 keeps ~300x margin
        # over that measured spread and stays far below the O(1) scale of the non-zero
        # trajectories this guard exists to police.
        _check_imaginary_part(values)
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
