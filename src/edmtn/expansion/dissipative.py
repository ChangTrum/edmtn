"""Local system-side dissipation for the time-step expansion (Layer 4b).

Wraps any :class:`~edmtn.expansion.base.TimeStepExpander` so that each physical step
also applies a **local dissipative channel to the system**, placed in Strang half-steps
at the two ends of the step::

    order 2:   Q_n = E_h  F_{2,n}  F_{1,n}  E_h
    order 1:   Q_n = E_h  F_{1,n}  E_h                 h = eps / 2

with ``E_h`` the half-step channel of the *complete* dissipator; this module supplies its
system-side tensor factor ``M(h)``, while the bath-side factors ``D_k(h)`` come from the
separable time-dependent correlation engine.  The rightmost factor acts first, so the
site tensors are

* earlier sub-step (``S_1``):  ``S^phi_1 M(h)``
* later sub-step (``S_2``):    ``M(h) S^phi_2``
* order 1 (single site):       ``M(h) S^phi M(h)``

**``h = eps/2`` at both orders**: the two halves belong to the *physical* step, not to the
algebraic sub-steps, so ``h`` does not scale with ``order``.  Consecutive steps merge
their adjacent halves into one exact full-step channel, so the dissipative time increment
consumed per physical step is exactly ``eps`` -- none double-counted, none lost.

The **identity superoperator entry** ``phi = 0`` is multiplied by the channel too: every
term of the tensor-product expansion carries the same system-side dissipation, exactly as
the bath-side ``phi = 0`` transfer tensor carries ``D_k``.

Why this placement and not one half-channel inside each sub-step: the latter leaves a
per-step error ``((1-i)/4) eps^2 [L_D, H^-]``, first order globally whenever the
dissipator and the Hamiltonian fail to commute.  Derivation, symbolic expansion and the
measured convergence orders are in ``docs/design/dicke-second-order-discretisation.md``.
"""

from __future__ import annotations

import math
import numbers

import numpy as np

from .base import StepSuperoperators, TimeStepExpander


def _nonnegative_finite(name: str, value) -> float:
    """Coerce ``value`` to a finite, non-negative float, else raise ``ValueError``.

    A public entry point, so a bool, a non-real, a NaN/Inf, a negative rate, or a Python
    ``int`` too large for float64 (``10**400`` -- a ``numbers.Real`` that overflows the
    conversion) must all surface as the project's ``ValueError``, never a leaked
    ``TypeError`` / ``OverflowError``.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be representable as a finite real number, got {value!r}") from exc
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return v


def amplitude_damping_matrix(d: int, kappa: float, dt: float) -> np.ndarray:
    """Exact ``exp(dt * kappa * D[a])`` on row-major ``vec(rho)``, for a Fock space of
    dimension ``d``.

    Returns the ``(d**2, d**2)`` matrix of the cavity amplitude-damping channel::

        rho'_{mu,nu} = x^{(mu+nu)/2}
                       sum_{m=0}^{min(d-1-mu, d-1-nu)} (1-x)^m
                       sqrt( C(mu+m, m) C(nu+m, m) ) rho_{mu+m, nu+m},
        x = exp(-dt kappa),

    which is the **full** Kraus sum ``sum_m V_m rho V_m^dag`` restricted to the truncated
    space -- not a low-order truncation of it.  Keeping every ``m`` costs nothing (the
    matrix is built once) and buys two properties the truncated version does not have:
    the channel is **exactly trace preserving on the truncated space**, and it is an exact
    one-parameter semigroup, ``M(dt/2) M(dt/2) = M(dt)``.  (A two-term Kraus truncation
    leaks trace at order ``eps^2`` on the top Fock level.)

    ``kappa == 0`` or ``dt == 0`` returns the identity **exactly**, so a closed model is
    bit-for-bit unaffected by wrapping its expander.
    """
    if isinstance(d, bool) or not isinstance(d, numbers.Integral) or int(d) < 1:
        raise ValueError(f"d must be a positive integer, got {d!r}")
    d = int(d)
    kappa = _nonnegative_finite("kappa", kappa)
    dt = _nonnegative_finite("dt", dt)

    d2 = d * d
    if kappa == 0.0 or dt == 0.0:
        return np.eye(d2, dtype=np.complex128)

    x = math.exp(-dt * kappa)
    # binomial coefficients up to d-1, exactly (integers, then to float)
    binom = np.zeros((d, d), dtype=np.float64)
    for n in range(d):
        for m in range(n + 1):
            binom[n, m] = float(math.comb(n, m))

    M = np.zeros((d2, d2), dtype=np.complex128)
    for mu in range(d):
        for nu in range(d):
            row = mu * d + nu
            pref = x ** (0.5 * (mu + nu))
            for m in range(min(d - 1 - mu, d - 1 - nu) + 1):
                M[row, (mu + m) * d + (nu + m)] = (
                    pref * (1.0 - x) ** m * math.sqrt(binom[mu + m, m] * binom[nu + m, m])
                )
    return M


class DissipativeExpander(TimeStepExpander):
    """Wrap a time-step expander with a Strang-placed system-side channel.

    The wrapper reports the base expander's :attr:`order`, so every downstream check
    (``SeparableBathEvolution``, ``validate_expansion_order``, the sub-step map) sees the
    order it actually runs at.

    Parameters
    ----------
    base : TimeStepExpander
        The undamped expansion (``FirstOrderExpander`` or ``SecondOrderExpander``); its
        superoperator families are used unchanged apart from the channel factors.
    channel : callable
        ``channel(d, dt) -> (d**2, d**2)`` array: the system-side dissipative channel
        acting on row-major ``vec(rho)`` for a system of dimension ``d`` over a time
        ``dt``.  Called with the **half-step** ``dt = eps/2`` and cached per ``(d, dt)``,
        so a constant time grid builds it exactly once.
    """

    def __init__(self, base: TimeStepExpander, channel):
        if getattr(base, "order", None) not in (1, 2):
            raise ValueError(
                f"base expander must have order 1 or 2, got {getattr(base, 'order', None)!r}")
        if not callable(channel):
            raise ValueError("channel must be callable as channel(d, dt) -> (d**2, d**2)")
        self.base = base
        self._channel = channel
        self._cache: dict[tuple[int, float], np.ndarray] = {}

    @property
    def order(self) -> int:  # type: ignore[override]
        return self.base.order

    def _half_step_channel(self, d: int, eps: float) -> np.ndarray:
        """``M(eps/2)`` -- the half-step is eps/2 at BOTH orders (see the module docstring)."""
        key = (d, 0.5 * eps)
        if key not in self._cache:
            M = np.asarray(self._channel(d, 0.5 * eps), dtype=np.complex128)
            if M.shape != (d * d, d * d):
                raise ValueError(
                    f"channel(d={d}, dt={0.5 * eps}) must return a ({d * d}, {d * d}) "
                    f"matrix, got shape {M.shape}")
            if not np.all(np.isfinite(M)):
                raise ValueError("channel returned a non-finite matrix")
            self._cache[key] = M
        return self._cache[key]

    def build(self, coupling_ops: list[np.ndarray], eps: float) -> StepSuperoperators:
        st = self.base.build(coupling_ops, eps)
        M = self._half_step_channel(st.d, eps)
        # NumPy matmul broadcasts over the leading phi axis, so `fam @ M` applies M to
        # every superoperator slice, phi = 0 included.
        if st.order == 1:
            families = [M @ (st.families[0] @ M)]
        else:
            early, late = st.families                 # families[0] acts first
            families = [early @ M, M @ late]
        return StepSuperoperators(
            phys_dim=st.phys_dim, d=st.d, families=families, order=st.order)

    def __repr__(self) -> str:
        return f"DissipativeExpander(base={self.base!r})"


def cavity_damping_channel(kappa: float):
    """Return the ``channel(d, dt)`` callable for cavity amplitude damping at rate ``kappa``."""
    return lambda d, dt: amplitude_damping_matrix(d, kappa, dt)
