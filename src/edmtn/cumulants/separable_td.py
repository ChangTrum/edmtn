"""Time-dependent separable-bath correlation engine (Layer 2).

The separable engine in :mod:`edmtn.cumulants.separable` assumes the sub-baths have no
self-Hamiltonian, so one Eq.-F1 transfer tensor per sub-bath serves every time slice.
The Dicke bath spins **do** have a self-Hamiltonian, so their interaction-picture
operators rotate,

    B_k(t) = g_k [ cos(omega_k t) sigma_x - sin(omega_k t) sigma_y ],

and each time slice needs its own transfer tensor.  They may in addition carry local
Lindblad dissipation, which enters as a fixed 4x4 channel on the same Liouville index.

Per sub-step ``g = 1 .. order * n_steps`` (oldest first), with physical step
``n = (g-1)//order + 1`` and midpoint ``t_n^* = (n - 1/2) eps``:

    A_k(t)[phi, a, a'] = (1/2) Tr[ sigma_a  B^phi_k(t)(sigma_{a'}) ],
        phi = 0 -> identity,  phi = 1 -> -i[B_k(t), .],  phi = 2 -> (1/2){B_k(t), .}
        sigma = (I, sigma_x, sigma_y, sigma_z)          (Pauli, lateral dimension 4)

    order 2, earlier sub-step:  A_k(t_n^*) D_k(h)
    order 2, later sub-step:    D_k(h) A_k(t_n^*)
    order 1, single sub-step:   D_k(h) A_k(t_n^*) D_k(h)          h = eps/2

so that one physical step contributes ``D_k(h) A_k A_k D_k(h)`` -- the bath-side factors
of the Strang map ``E_h F_2 F_1 E_h``.  **``h = eps/2`` at both orders**: the halves belong
to the physical step, not to the algebraic sub-steps.

Two conventions that are easy to get wrong and are therefore stated once here:

* the transfer tensors carry **no** ``c_1``/``c_2`` coefficients -- the second-order
  complex coefficients live only on the system-side superoperator families, and squaring
  them by also applying them here would be a silent error;
* ``B_k`` is read from ``model.bath_operator_at(k, t)`` rather than rebuilt from the
  couplings, so the model and the transfer tensors cannot drift apart.

The initial bath state enters through the **oldest site's boundary vector**
``r_k = (1, r_x, r_y, r_z)`` with ``Omega_k = (1/2)(I + r_k . sigma)``; the maximally
mixed case ``r = (1, 0, 0, 0)`` reproduces the time-independent engine's "slice the
oldest lateral index to 0" convention.

Derivation, error orders and the verification record:
``docs/design/dicke-second-order-discretisation.md``.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field

import numpy as np

from .base import CumulantEngine
from .separable import SeparableBathCorrelation

_PAULI = [
    np.eye(2, dtype=np.complex128),
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
]


def _nonnegative_finite(name: str, value) -> float:
    """Coerce ``value`` to a finite, non-negative float, else raise ``ValueError``.

    These helpers are exported, so an illegal argument must not reach the exponentials and
    produce a non-physical channel: a negative rate would give ``exp(+|gamma| dt) > 1``,
    an amplifying "relaxation".  A bool, a non-real, a NaN/Inf, or a Python ``int`` too
    large for float64 all raise here rather than leaking ``TypeError`` / ``OverflowError``.
    ``0`` is legal for both a rate and ``dt``.
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


def relaxation_factor(gamma: float, dt: float) -> float:
    """``(1 - exp(-gamma dt)) / gamma``, evaluated stably, with the limit ``dt`` at ``gamma = 0``.

    The naive expression is ``0/0`` at ``gamma = 0`` -- which is the **default**
    configuration, every rate zero -- and loses precision to cancellation for small
    ``gamma dt``.  ``-expm1(-gamma dt) / gamma`` has neither problem.
    """
    gamma = _nonnegative_finite("gamma", gamma)
    dt = _nonnegative_finite("dt", dt)
    if gamma == 0.0:
        return dt
    return -math.expm1(-gamma * dt) / gamma


def bath_channel_matrix(pump: float, emission: float, dephasing: float, dt: float) -> np.ndarray:
    """``D_k(dt) = exp(dt L_k)`` on the Pauli-coefficient vector ``(r_0, r_x, r_y, r_z)``.

    With ``Gamma_1 = pump + emission`` and ``Gamma_2 = Gamma_1/2 + dephasing``::

        D_k(dt) = [[1,     0,            0,            0          ],
                   [0,     e^{-G2 dt},   0,            0          ],
                   [0,     0,            e^{-G2 dt},   0          ],
                   [d_k,   0,            0,            e^{-G1 dt} ]]
        d_k = (pump - emission) * relaxation_factor(Gamma_1, dt)

    Pure dephasing (``Gamma_1 = 0``, ``dephasing > 0``) is a legitimate regime: the affine
    entry is then ``0`` while ``Gamma_2 > 0`` still damps the transverse components.
    """
    pump = _nonnegative_finite("pump", pump)
    emission = _nonnegative_finite("emission", emission)
    dephasing = _nonnegative_finite("dephasing", dephasing)
    dt = _nonnegative_finite("dt", dt)
    g1 = pump + emission
    g2 = 0.5 * g1 + dephasing
    D = np.zeros((4, 4), dtype=np.complex128)
    D[0, 0] = 1.0
    D[1, 1] = D[2, 2] = math.exp(-g2 * dt)
    D[3, 3] = math.exp(-g1 * dt)
    D[3, 0] = (pump - emission) * relaxation_factor(g1, dt)
    return D


@dataclass(frozen=True)
class TimeDependentSeparableCorrelation:
    """Per-sub-bath, per-sub-step Eq.-F1 transfer tensors of a time-dependent separable bath.

    Attributes
    ----------
    eps : float
        Physical time step.
    n_steps : int
        Number of **physical** steps ``N``; the site grid has ``order * N`` sub-steps.
    order : int
        Resolved expansion order (``1`` or ``2``) -- part of the grid signature, because
        the sub-step to physical-step map depends on it.
    bloch : np.ndarray
        Initial Bloch vectors ``(K, 3)``.
    rates : np.ndarray
        Per-spin ``(pump, emission, dephasing)``, shape ``(K, 3)``.
    bath_operator : callable
        ``bath_operator(k, t) -> (2, 2)`` returning ``B_k(t)`` (Pauli units).

    ``bloch`` and ``rates`` are **privately copied at construction and marked read-only**.
    Transfer tensors are **not** stored: :meth:`transfer_for` builds one sub-bath's
    ``(n_sites, 3, 4, 4)`` array on demand, so the outer fold loop holds one at a time
    instead of ``K`` of them.
    """

    eps: float
    n_steps: int
    order: int
    bloch: np.ndarray = field(repr=False)
    rates: np.ndarray = field(repr=False)
    bath_operator: object = field(repr=False)

    def __post_init__(self):
        for name in ("bloch", "rates"):
            arr = np.array(getattr(self, name), dtype=np.float64, copy=True)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        if self.bloch.ndim != 2 or self.bloch.shape[1] != 3:
            raise ValueError(f"bloch must have shape (K, 3), got {self.bloch.shape}")
        if self.rates.shape != self.bloch.shape:
            raise ValueError(
                f"rates shape {self.rates.shape} != bloch shape {self.bloch.shape}")
        if self.order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {self.order!r}")

    # -- shape / grid ------------------------------------------------------

    @property
    def K(self) -> int:
        """Number of sub-baths."""
        return int(self.bloch.shape[0])

    @property
    def d_phys(self) -> int:
        """Superoperator (physical) index dimension; ``3`` for this single-channel bath."""
        return 3

    @property
    def bond_dim(self) -> int:
        """Lateral (Liouville) bond dimension ``D_a = 4``."""
        return 4

    @property
    def n_sites(self) -> int:
        """Number of sub-step sites, ``order * n_steps``."""
        return self.order * self.n_steps

    @property
    def grid_signature(self) -> tuple:
        """``(eps, n_steps, order)`` -- the full time grid, not just its length.

        ``n_sites`` alone does not identify a grid: ``order=1, N=4`` and ``order=2, N=2``
        share it, and so do two runs with different ``eps``.  Consumers compare this.
        """
        return (float(self.eps), int(self.n_steps), int(self.order))

    def sample_time(self, g: int) -> float:
        """Midpoint ``t_n^* = (n - 1/2) eps`` of the physical step owning sub-step ``g``.

        ``g`` is 1-based and counts from the **oldest** sub-step; both algebraic sub-steps
        of a physical step share the one midpoint.  The system side must use the same map.
        """
        n = (g - 1) // self.order + 1
        return (n - 0.5) * self.eps

    # -- tensors -----------------------------------------------------------

    def transfer_for(self, k: int) -> np.ndarray:
        """Transfer tensors of sub-bath ``k``, shape ``(n_sites, 3, 4, 4)``.

        Index ``0`` is the **oldest** sub-step (``g = 1``); the kernel reverses this to get
        its newest-first site order.  Built on demand, not cached.
        """
        if isinstance(k, bool) or not isinstance(k, numbers.Integral):
            raise ValueError(f"k must be an integer in 0..{self.K - 1}, got {k!r}")
        k = int(k)
        if not 0 <= k < self.K:
            raise IndexError(f"sub-bath index {k} out of range 0..{self.K - 1}")

        pump, emission, dephasing = (float(x) for x in self.rates[k])
        D_h = bath_channel_matrix(pump, emission, dephasing, 0.5 * self.eps)

        out = np.empty((self.n_sites, self.d_phys, 4, 4), dtype=np.complex128)
        cache: dict[float, np.ndarray] = {}
        for g in range(1, self.n_sites + 1):
            t = self.sample_time(g)
            if t not in cache:
                B = np.asarray(self.bath_operator(k, t), dtype=np.complex128)
                if B.shape != (2, 2):
                    raise ValueError(
                        f"bath_operator({k}, {t}) must return a (2, 2) matrix, "
                        f"got shape {B.shape}")
                # the Eq.-F1 formula itself is shared with the time-independent engine,
                # so the two cannot disagree on conventions
                cache[t] = SeparableBathCorrelation._transfer_tensor([B], _PAULI)
            A = cache[t]
            if self.order == 1:
                out[g - 1] = D_h @ A @ D_h
            elif (g - 1) % 2 == 0:          # earlier algebraic sub-step
                out[g - 1] = A @ D_h
            else:                           # later algebraic sub-step
                out[g - 1] = D_h @ A
        return out

    def boundary_vector(self, k: int) -> np.ndarray:
        """Oldest-site boundary ``r_k = (1, r_x, r_y, r_z)`` from ``Omega_k = (1/2)(I + r.sigma)``.

        ``r_{k,0} = Tr[Omega_k] = 1`` always; the maximally mixed bath gives
        ``(1, 0, 0, 0)``, i.e. exactly the "slice the oldest lateral index to 0" boundary
        of the time-independent engine.
        """
        if isinstance(k, bool) or not isinstance(k, numbers.Integral):
            raise ValueError(f"k must be an integer in 0..{self.K - 1}, got {k!r}")
        k = int(k)
        if not 0 <= k < self.K:
            raise IndexError(f"sub-bath index {k} out of range 0..{self.K - 1}")
        r = np.empty(4, dtype=np.complex128)
        r[0] = 1.0
        r[1:] = self.bloch[k]
        return r

    def correlation(self, ops, k: int = 0) -> complex:
        """``Tr[ B^{phi_G} ... B^{phi_1} (Omega_k) ]`` from the transfer-tensor product.

        ``ops`` is the superoperator index sequence in **time order**
        ``[phi_1, ..., phi_G]`` (oldest first), ``G <= n_sites``.  Provided for
        verification against a brute-force superoperator chain.
        """
        ops = list(ops)
        if len(ops) > self.n_sites:
            raise ValueError(
                f"ops has {len(ops)} entries but the grid has only {self.n_sites} sites")
        T = self.transfer_for(k)
        v = self.boundary_vector(k)
        for g, phi in enumerate(ops, start=1):
            if isinstance(phi, bool) or not isinstance(phi, numbers.Integral) \
                    or not 0 <= int(phi) < self.d_phys:
                raise ValueError(f"phi must be an integer in 0..{self.d_phys - 1}, got {phi!r}")
            v = T[g - 1, int(phi)] @ v
        return complex(v[0])


class SeparableTDBathCorrelation(CumulantEngine):
    """Eq.-F1 transfer tensors for a time-dependent separable spin-1/2 bath (Dicke).

    Requires a model exposing ``bath_operator_at(k, t)``, ``bath_bloch_vectors()`` and a
    ``bath_params()`` carrying ``K`` plus the per-spin ``pump`` / ``emission`` /
    ``dephasing`` rates.
    """

    bath_type = "separable_td"

    def compute(self, model, T: float, eps: float, order: int = 2
                ) -> TimeDependentSeparableCorrelation:
        """Build the correlation description for ``model`` on the grid ``0..T`` step ``eps``.

        ``order`` is the **resolved** expansion order and must be passed explicitly by the
        pipeline builder -- it is part of the grid signature and decides the sub-step map,
        so re-reading ``model.time_step_order`` here could silently disagree with the
        order the evolution actually runs at.
        """
        self._check_model(model)
        n_steps = self._n_steps(T, eps)
        if isinstance(order, bool) or not isinstance(order, numbers.Integral) \
                or int(order) not in (1, 2):
            raise ValueError(f"order must be the integer 1 or 2, got {order!r}")
        for name in ("bath_operator_at", "bath_bloch_vectors"):
            if not callable(getattr(model, name, None)):
                raise NotImplementedError(
                    f"SeparableTDBathCorrelation needs a model exposing {name}() "
                    f"(a time-dependent separable bath, e.g. DickeModel)")
        bp = model.bath_params()
        bloch = np.asarray(model.bath_bloch_vectors(), dtype=np.float64)
        if bloch.shape != (bp.K, 3):
            raise ValueError(
                f"bath_bloch_vectors() must have shape ({bp.K}, 3), got {bloch.shape}")
        rates = np.stack(
            [np.asarray(bp.pump, dtype=np.float64),
             np.asarray(bp.emission, dtype=np.float64),
             np.asarray(bp.dephasing, dtype=np.float64)], axis=1)
        return TimeDependentSeparableCorrelation(
            eps=float(eps),
            n_steps=n_steps,
            order=int(order),
            bloch=bloch,
            rates=rates,
            bath_operator=model.bath_operator_at,
        )
