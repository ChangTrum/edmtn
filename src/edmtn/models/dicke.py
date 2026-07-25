"""Inhomogeneous Dicke model with local Lindblad dissipation (Layer 1).

A single cavity mode couples to ``K`` independent two-level systems::

    H = omega_c a^dag a + sum_k (omega_k / 2) sigma_{k,z}
        + sum_k g_k (a + a^dag) sigma_{k,x}

In the interaction picture of ``H_0 = omega_c a^dag a + sum_k (omega_k/2) sigma_{k,z}``
the coupling factorises into **one** system operator and one operator per sub-bath::

    H_I(t) = sum_k S(t) (x) B_k(t)
    S(t)   = a e^{-i omega_c t} + a^dag e^{i omega_c t}
    B_k(t) = g_k [ cos(omega_k t) sigma_{k,x} - sin(omega_k t) sigma_{k,y} ]

so the model has a **single coupling channel** (``d_phys = 2*1 + 1 = 3``) and a
**separable, time-dependent** bath -- hence ``bath_type = "separable_td"``.  Note the
bath operators are written in **Pauli** units (``sigma``), not in spin units
(``J = sigma/2``) as the Gaudin model uses; ``bath_operator_at`` is the single source
of truth for that convention.

Open-system extension.  Each dissipative channel is **local** -- to the cavity, or to
one bath spin::

    d rho / dt = -i [H_I(t), rho] + kappa D[a] rho
                 + sum_k ( w_k D[sigma_k^+] + gamma_k D[sigma_k^-]
                           + (gamma_k^phi / 2) D[sigma_{k,z}] ) rho
    D[L] rho = L rho L^dag - (1/2) { L^dag L, rho }

with ``Gamma_{k,1} = w_k + gamma_k`` and ``Gamma_{k,2} = Gamma_{k,1}/2 + gamma_k^phi``.
Locality is not a stylistic choice: a **collective** jump operator (e.g. ``D[J^-]``)
would correlate the sub-baths and invalidate the entire separable transfer-tensor
construction.  These particular jump operators are also invariant under the
interaction-picture transformation (``D[a e^{-i omega_c t}] = D[a]``,
``D[e^{i omega_k t} sigma_k^+] = D[sigma_k^+]``, ``D[sigma_{k,z}]`` unchanged), which is
what lets the dissipators be applied as time-independent channels; that is a property of
*these* operators, not a general fact.

The parameters layer up, so the simplest configuration is also the cleanest baseline:

1. **closed, homogeneous Dicke Hamiltonian** -- scalar ``coupling`` (``g_k = G/sqrt(K)``)
   and scalar ``omega``, every rate ``0``, vacuum cavity, infinite-temperature spins;
2. **inhomogeneous** -- per-spin ``coupling`` and/or ``omega`` arrays;
3. **non-equilibrium** -- any of ``kappa``, ``pump``, ``emission``, ``dephasing``.

**All rates default to zero**, so the default Hamiltonian really is the closed model of
the equilibrium derivation.  Keep the Hamiltonian and the *state* apart, though: the run
itself is a **quench** from ``vacuum (x) maximally-mixed``, not an equilibrium
calculation.  The familiar critical coupling ``G_c = sqrt(omega_0 omega_c)/2`` is the
**zero-temperature, thermodynamic-limit equilibrium ground-state** transition; it does
not describe this default quench from infinite-temperature spins, and the pipeline
neither computes nor verifies it.

The layers are independent switches, which is what makes **controlled comparison**
possible -- enable one at a time and compare against the baseline.  With several enabled
at once a difference cannot be attributed to any single one of them.

Fock truncation.  The cavity is represented in a truncated Fock space of dimension
``n_fock`` (levels ``0 .. n_fock - 1``).  This is a **numerical** parameter chosen to
balance cost against accuracy, and is deliberately *not* called ``n_max``: the highest
occupied photon number is a property of the physical state, may have no finite value at
all (a coherent or thermal state has an infinite tail), and is read off the photon-number
distribution :meth:`DickeModel.fock_populations` rather than specified up front.

Time discretisation.  The pipeline samples the frozen generator at the **midpoint**
``t_n^* = (n - 1/2) eps`` and places the dissipative channels in Strang half-steps at the
ends of each physical step.  Under stated conditions -- a fixed ``n_fock``, a smooth
bounded generator, the interaction-picture-invariant dissipators above, and no discarded
weight (so no compression, reference or round-off floor has been reached) --
``expansion_order = 2`` is then globally second order; the measured convergence orders on
the test configuration are 1.97 (order 2) and 1.02 (order 1).  Outside those conditions
the observed order degrades and the claim does not carry.  The derivation, the exact
discretisation maps and the full verification record are in
``docs/design/dicke-second-order-discretisation.md``.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field

import numpy as np

from .base import AbstractOQSModel

# Pauli matrices -- the bath operators of this model are written in Pauli units.
_SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

#: cavity initial states selectable by name (an explicit matrix is also accepted)
CAVITY_STATES = ("vacuum", "coherent", "thermal")

#: bath initial states selectable by name (an explicit ``(K, 3)`` Bloch array is also accepted)
BATH_STATES = ("inf", "thermal", "ground")

#: tolerance on ``||r_k|| <= 1`` for an explicit Bloch vector (a Bloch vector of length
#: slightly above 1 through round-off is accepted; a genuinely unphysical one is not)
_BLOCH_TOL = 1e-12


# -- parameter validation (module-private; Layer 1 keeps its own leaf checks rather
#    than importing the driver-layer validators, per the P0-2 decision) -----------
def _to_float(name: str, value) -> float:
    """Coerce a real ``value`` to float; a bool, a non-real, or a too-large Python int
    (a ``numbers.Real`` that overflows float64) becomes a ``ValueError`` rather than a
    leaked ``TypeError`` / ``OverflowError``."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be representable as a finite real number, got {value!r}") from exc


def _finite_float(name: str, value) -> float:
    v = _to_float(name, value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def _positive_finite(name: str, value) -> float:
    v = _to_float(name, value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return v


def _nonnegative_finite(name: str, value) -> float:
    v = _to_float(name, value)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return v


def _is_int(value) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _int_at_least(name: str, value, minimum: int) -> int:
    if not _is_int(value):
        raise ValueError(f"{name} must be an integer (not bool), got {value!r}")
    v = int(value)
    if v < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return v


def _to_complex(name: str, value) -> complex:
    """Coerce a real or complex ``value`` to a finite Python ``complex``."""
    if isinstance(value, bool) or not isinstance(value, numbers.Complex):
        raise ValueError(f"{name} must be a real or complex number, got {value!r}")
    try:
        c = complex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite complex number, got {value!r}") from exc
    if not (math.isfinite(c.real) and math.isfinite(c.imag)):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return c


def _per_spin(name: str, value, K: int, check) -> np.ndarray:
    """Broadcast a scalar, or validate a length-``K`` array, into a read-only ``float64``
    array of per-spin values.

    ``check(name, x) -> float`` is the leaf validator applied to **every** entry, so an
    illegal entry deep inside an array fails exactly like an illegal scalar.  The result
    is in the caller's order: **no sorting and no normalisation** is applied anywhere in
    this model, so index ``k`` means the same physical spin in every per-spin array.
    """
    if np.isscalar(value) or isinstance(value, numbers.Number):
        return np.full(K, check(name, value), dtype=np.float64)
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real scalar or a length-{K} array: {exc}") from exc
    if arr.shape != (K,):
        raise ValueError(
            f"{name} must be a real scalar or have shape ({K},), got shape {arr.shape}")
    out = np.array([check(f"{name}[{i}]", x) for i, x in enumerate(arr.tolist())],
                   dtype=np.float64)
    return out


@dataclass(frozen=True)
class DickeBathParams:
    """Bath parameters of :class:`DickeModel`.

    Every per-spin array is indexed by the **same** sub-bath index ``k`` and is used in
    the order given -- this model never sorts or renormalises a coupling profile, so
    ``couplings[k]``, ``omegas[k]``, the rates and ``bloch[k]`` all describe the same
    physical spin.  (Contrast :class:`~edmtn.models.gaudin.GaudinModel`, whose named
    profiles are sorted descending; mixing that with independent per-spin frequencies
    would silently pair the wrong parameters.)

    Attributes
    ----------
    K : int
        Number of bath spins.
    couplings : np.ndarray
        Per-spin couplings ``g_k`` (length ``K``).
    omegas : np.ndarray
        Per-spin level splittings ``omega_k`` (length ``K``).
    pump, emission, dephasing : np.ndarray
        Per-spin Lindblad rates ``w_k``, ``gamma_k``, ``gamma_k^phi`` (length ``K``,
        all ``>= 0``; all zero by default, giving a closed bath).
    bloch : np.ndarray
        Initial Bloch vectors ``r_k = (r_x, r_y, r_z)`` of shape ``(K, 3)``, so that
        ``Omega_k = (1/2)(I + r_k . sigma)``.
    temperature : float
        Bath temperature the state was built at: ``+inf`` for ``"inf"``, ``0.0`` for
        ``"ground"``, ``1/beta`` for ``"thermal"``, and ``nan`` for an explicit Bloch
        array (which need not correspond to any temperature).  Descriptive only -- the
        correlation engine reads ``bloch``, never this field.

    All arrays are **privately copied at construction and marked read-only**.
    """

    K: int
    couplings: np.ndarray = field(repr=False)
    omegas: np.ndarray = field(repr=False)
    pump: np.ndarray = field(repr=False)
    emission: np.ndarray = field(repr=False)
    dephasing: np.ndarray = field(repr=False)
    bloch: np.ndarray = field(repr=False)
    temperature: float = math.nan

    def __post_init__(self):
        for name in ("couplings", "omegas", "pump", "emission", "dephasing", "bloch"):
            arr = np.array(getattr(self, name), dtype=np.float64, copy=True)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)


class DickeModel(AbstractOQSModel):
    """Cavity mode coupled to ``K`` two-level systems, with optional local dissipation.

    With every rate left at ``0`` this is exactly the closed inhomogeneous Dicke model;
    with the defaults throughout (scalar ``coupling`` and ``omega``, vacuum cavity,
    infinite-temperature spins) it is a **closed homogeneous Dicke quench** from a
    ``vacuum (x) maximally-mixed`` product state.

    Out-of-range, non-finite or wrongly-shaped parameters raise ``ValueError`` here, at
    construction.  Capability limits of the downstream engines (rather than of the
    parameters) surface as ``NotImplementedError`` at solve time -- notably the
    ``separable_td`` pipeline publishes no time-resolved coupling-channel polarization.

    Parameters
    ----------
    K : int
        Number of bath spins; a strict non-``bool`` integer ``>= 1``.
    n_fock : int
        **Dimension** of the truncated cavity Fock space; a strict non-``bool`` integer
        ``>= 2``, giving levels ``0 .. n_fock - 1`` and ``system_dim = n_fock``.  This is
        a numerical truncation, *not* the highest occupied photon number -- check its
        adequacy with :meth:`fock_populations`, which returns the photon-number
        distribution itself rather than a threshold-dependent "maximum occupied level".
    coupling : float or array-like
        The per-spin couplings ``g_k``.  **A real scalar is the collective coupling**
        ``G``, giving the standard Dicke scaling ``g_k = G / sqrt(K)``; a length-``K``
        array is used **verbatim, in the order given** (any sign, no sorting, no
        normalisation).  There is deliberately no separate ``G`` argument: it would be
        inert whenever an explicit array is supplied.
    omega_c : float
        Cavity frequency; finite, ``> 0`` (default ``1.0``).
    omega : float or array-like
        Spin level splittings ``omega_k``; **finite and strictly positive** -- a scalar
        (broadcast to all ``K``, the homogeneous case) or a length-``K`` array (default
        ``1.0``).  Positivity is required, not cosmetic: ``bath_state="ground"`` means
        ``r_{k,z} = -1``, which is the ground state of ``(omega_k/2) sigma_z`` only for
        ``omega_k > 0``, and is degenerate (no unique ground state) at ``omega_k = 0``.
        A signed rotating-frame detuning would need those state definitions rewritten
        and is not accepted here.
    cavity_state : str or array-like
        ``"vacuum"`` (default), ``"coherent"``, ``"thermal"``, or an explicit
        ``(n_fock, n_fock)`` density matrix.  A named state is built **inside** the
        truncated space and then renormalised to unit trace, so a truncated coherent or
        thermal state is the exact state's projection rescaled -- not the exact state.
        An **explicit** matrix is checked here only for being numeric, correctly shaped
        and finite; its Hermiticity, unit trace and positive semidefiniteness are checked
        by :meth:`~edmtn.models.base.AbstractOQSModel.validate`, which
        :class:`~edmtn.driver.solver.EDMSolver` runs before building any pipeline.  So a
        non-physical explicit matrix still fails before any tensor is built, but at
        validation rather than at ``__init__``.
    cavity_params : dict, optional
        Extra keys for a named cavity state: ``alpha`` (complex) for ``"coherent"``,
        ``nbar`` (finite, ``>= 0``) for ``"thermal"``.  Ignored for ``"vacuum"`` and for
        an explicit matrix.
    bath_state : str or array-like
        ``"inf"`` (default; maximally mixed, ``r_k = 0``), ``"thermal"``
        (``r_{k,z} = -tanh(beta omega_k / 2)``), ``"ground"`` (``r_{k,z} = -1``), or an
        explicit ``(K, 3)`` array of Bloch vectors with ``||r_k|| <= 1``.
    bath_state_params : dict, optional
        Extra keys for a named bath state: ``beta`` (finite, ``> 0``, in units with
        ``k_B = 1``) for ``"thermal"``.  Ignored otherwise.
    kappa : float
        Cavity decay rate ``kappa`` in ``kappa D[a]``; finite, ``>= 0`` (default ``0``).
    pump, emission, dephasing : float or array-like
        Per-spin rates ``w_k`` (``D[sigma^+]``), ``gamma_k`` (``D[sigma^-]``) and
        ``gamma_k^phi`` (in ``(gamma^phi/2) D[sigma_z]``); finite and ``>= 0``, scalar or
        length ``K`` (all default ``0``).
    time_step_order : int
        Small-step expansion order used downstream: a strict non-``bool`` integer ``1``
        or ``2`` (default ``2``).
    """

    bath_type = "separable_td"

    def __init__(
        self,
        *,
        K: int,
        n_fock: int,
        coupling,
        omega_c: float = 1.0,
        omega=1.0,
        cavity_state="vacuum",
        cavity_params: dict | None = None,
        bath_state="inf",
        bath_state_params: dict | None = None,
        kappa: float = 0.0,
        pump=0.0,
        emission=0.0,
        dephasing=0.0,
        time_step_order: int = 2,
    ):
        self.K = _int_at_least("K", K, 1)
        # n_fock >= 2: a one-level "cavity" has a = 0 and no dynamics at all, which is a
        # degenerate configuration far more likely to be a typo than an intent.
        self.n_fock = _int_at_least("n_fock", n_fock, 2)
        if not _is_int(time_step_order) or int(time_step_order) not in (1, 2):
            raise ValueError(
                f"time_step_order must be the integer 1 or 2, got {time_step_order!r}")
        self.time_step_order = int(time_step_order)

        self.omega_c = _positive_finite("omega_c", omega_c)
        self.kappa = _nonnegative_finite("kappa", kappa)

        # -- per-spin parameters: same index k throughout, never sorted or renormalised --
        self._couplings = self._resolve_couplings(coupling, self.K)
        self._omegas = _per_spin("omega", omega, self.K, _positive_finite)
        self._pump = _per_spin("pump", pump, self.K, _nonnegative_finite)
        self._emission = _per_spin("emission", emission, self.K, _nonnegative_finite)
        self._dephasing = _per_spin("dephasing", dephasing, self.K, _nonnegative_finite)

        # -- initial states --
        self.cavity_params = dict(cavity_params or {})
        self.bath_state_params = dict(bath_state_params or {})
        self.cavity_state = cavity_state if isinstance(cavity_state, str) else "custom"
        self.bath_state = bath_state if isinstance(bath_state, str) else "custom"
        self._rho0 = self._resolve_cavity_state(cavity_state, self.cavity_params)
        bloch, temperature = self._resolve_bath_state(
            bath_state, self.bath_state_params, self._omegas)
        self._bloch = bloch

        self._bath = DickeBathParams(
            K=self.K,
            couplings=self._couplings,
            omegas=self._omegas,
            pump=self._pump,
            emission=self._emission,
            dephasing=self._dephasing,
            bloch=self._bloch,
            temperature=temperature,
        )

    # -- coupling / state resolution --------------------------------------

    @staticmethod
    def _resolve_couplings(coupling, K: int) -> np.ndarray:
        """``g_k`` from a scalar collective coupling ``G`` or from an explicit array."""
        if isinstance(coupling, str):
            raise ValueError(
                f"coupling must be a real scalar (the collective G) or a length-{K} "
                f"array, got the string {coupling!r}; there are no named profiles")
        if np.isscalar(coupling) or isinstance(coupling, numbers.Number):
            G = _finite_float("coupling", coupling)
            return np.full(K, G / math.sqrt(K), dtype=np.float64)
        return _per_spin("coupling", coupling, K, _finite_float)

    def _resolve_cavity_state(self, spec, params: dict) -> np.ndarray:
        """Initial cavity density matrix, as a ``(n_fock, n_fock)`` complex array."""
        d = self.n_fock
        if not isinstance(spec, str):
            try:
                rho = np.array(spec, dtype=np.complex128, copy=True)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"an explicit cavity_state must be a numeric ({d}, {d}) matrix") from exc
            if rho.shape != (d, d):
                raise ValueError(
                    f"an explicit cavity_state must have shape ({d}, {d}), got {rho.shape}")
            if not np.all(np.isfinite(rho)):
                raise ValueError("an explicit cavity_state must be finite")
            return rho  # Hermiticity / trace / PSD are checked by AbstractOQSModel.validate
        if spec not in CAVITY_STATES:
            raise ValueError(
                f"unknown cavity_state {spec!r}; choose from {CAVITY_STATES}, or pass an "
                f"explicit ({d}, {d}) density matrix")
        if spec == "vacuum":
            amps = np.zeros(d, dtype=np.complex128)
            amps[0] = 1.0
            return np.outer(amps, amps.conj())
        if spec == "coherent":
            alpha = _to_complex("cavity_params['alpha']", params.get("alpha", 0.0))
            # built by recursion c_n = c_{n-1} * alpha / sqrt(n): no factorial overflow
            amps = np.empty(d, dtype=np.complex128)
            amps[0] = 1.0
            for n in range(1, d):
                amps[n] = amps[n - 1] * alpha / math.sqrt(n)
            norm = float(np.linalg.norm(amps))
            if not math.isfinite(norm) or norm == 0.0:
                raise ValueError(
                    f"the truncated coherent state is not normalisable for "
                    f"alpha={alpha!r}, n_fock={d}")
            amps = amps / norm          # renormalised inside the truncated space
            return np.outer(amps, amps.conj())
        # thermal: p_n ~ (nbar / (1 + nbar))^n, renormalised inside the truncated space
        nbar = _nonnegative_finite("cavity_params['nbar']", params.get("nbar", 0.0))
        ratio = nbar / (1.0 + nbar)     # in [0, 1); nbar = 0 gives the vacuum
        weights = ratio ** np.arange(d, dtype=np.float64)
        total = float(weights.sum())
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError(
                f"the truncated thermal state is not normalisable for nbar={nbar!r}")
        return np.diag((weights / total).astype(np.complex128))

    def _resolve_bath_state(self, spec, params: dict, omegas: np.ndarray):
        """Return ``(bloch, temperature)`` for the ``K`` bath spins."""
        K = self.K
        if not isinstance(spec, str):
            try:
                r = np.array(spec, dtype=np.float64, copy=True)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"an explicit bath_state must be a real ({K}, 3) Bloch array") from exc
            if r.shape != (K, 3):
                raise ValueError(
                    f"an explicit bath_state must have shape ({K}, 3), got {r.shape}")
            if not np.all(np.isfinite(r)):
                raise ValueError("an explicit bath_state must be finite")
            norms = np.linalg.norm(r, axis=1)
            bad = np.nonzero(norms > 1.0 + _BLOCH_TOL)[0]
            if bad.size:
                raise ValueError(
                    f"Bloch vectors must satisfy ||r_k|| <= 1; entries {bad.tolist()} have "
                    f"norms up to {float(norms[bad].max()):.6g}")
            return r, math.nan
        if spec not in BATH_STATES:
            raise ValueError(
                f"unknown bath_state {spec!r}; choose from {BATH_STATES}, or pass an "
                f"explicit ({K}, 3) Bloch array")
        r = np.zeros((K, 3), dtype=np.float64)
        if spec == "inf":
            return r, math.inf                      # maximally mixed: Omega_k = I/2
        if spec == "ground":
            r[:, 2] = -1.0
            return r, 0.0
        beta = _positive_finite("bath_state_params['beta']", params.get("beta", 1.0))
        r[:, 2] = -np.tanh(0.5 * beta * omegas)     # <sigma_z> of exp(-beta omega sigma_z/2)
        return r, 1.0 / beta

    # -- cavity operators --------------------------------------------------

    def annihilation_operator(self) -> np.ndarray:
        """Truncated annihilation operator ``a`` (``a|n> = sqrt(n)|n-1>``)."""
        d = self.n_fock
        return np.diag(np.sqrt(np.arange(1, d, dtype=np.float64)), 1).astype(np.complex128)

    # -- system ------------------------------------------------------------

    @property
    def system_dim(self) -> int:
        return self.n_fock

    def system_hamiltonian(self) -> np.ndarray:
        return np.diag(self.omega_c * np.arange(self.n_fock, dtype=np.float64)
                       ).astype(np.complex128)

    def coupling_operators(self) -> list[np.ndarray]:
        """The single Schroedinger-picture coupling operator ``a + a^dag``.

        No ``1/sqrt(2)`` normalisation: the Hamiltonian's coupling term is
        ``g_k (a + a^dag) sigma_{k,x}``, so ``a + a^dag`` (not the normalised quadrature
        ``q = (a + a^dag)/sqrt(2)``) is what multiplies ``B_k``.
        """
        a = self.annihilation_operator()
        return [a + a.conj().T]

    def coupling_operators_at(self, t: float) -> list[np.ndarray]:
        """``S(t) = a e^{-i omega_c t} + a^dag e^{i omega_c t}`` in closed form.

        Overrides the base class's eigendecomposition of ``H_S`` -- the closed form is
        both cheaper and free of eigensolver round-off.  The two agree to machine
        precision (asserted in the model tests).
        """
        t = _finite_float("t", t)
        a = self.annihilation_operator()
        phase = np.exp(1j * self.omega_c * t)
        return [a * phase.conjugate() + a.conj().T * phase]

    def system_operators(self) -> dict[str, np.ndarray]:
        """Named cavity operators.

        ``x = a + a^dag`` and ``p = i(a^dag - a)`` are the **unnormalised** quadratures
        (the derivation note's ``q``/``p`` are these divided by ``sqrt(2)``).  ``P_top``
        projects onto the highest retained Fock level, so ``Tr[P_top rho]`` is the
        single-number version of the tail check that :meth:`fock_populations` resolves.
        """
        d = self.n_fock
        a = self.annihilation_operator()
        ad = a.conj().T
        P_top = np.zeros((d, d), dtype=np.complex128)
        P_top[d - 1, d - 1] = 1.0
        return {
            "I": np.eye(d, dtype=np.complex128),
            "a": a,
            "adag": ad,
            "n": ad @ a,
            "x": a + ad,
            "p": 1j * (ad - a),
            "P_top": P_top,
        }

    def initial_system_state(self) -> np.ndarray:
        return self._rho0.copy()

    # -- bath --------------------------------------------------------------

    def bath_params(self) -> DickeBathParams:
        return self._bath

    @property
    def couplings(self) -> np.ndarray:
        """Per-spin couplings ``g_k`` (length ``K``), in the model's stored order."""
        return self._bath.couplings

    @property
    def omegas(self) -> np.ndarray:
        """Per-spin level splittings ``omega_k`` (length ``K``), same order as ``couplings``."""
        return self._bath.omegas

    def bath_bloch_vectors(self) -> np.ndarray:
        """Initial Bloch vectors ``r_k`` of shape ``(K, 3)`` (read-only)."""
        return self._bath.bloch

    def bath_operator_at(self, k: int, t: float) -> np.ndarray:
        """Interaction-picture bath operator of sub-bath ``k``::

            B_k(t) = g_k [ cos(omega_k t) sigma_x - sin(omega_k t) sigma_y ]

        in **Pauli** units.  This method is the single source of truth for that
        convention: the correlation engine reads ``B_k`` from here rather than
        reconstructing it, so the model and the transfer tensors cannot drift apart.
        """
        k = _int_at_least("k", k, 0)
        if k >= self.K:
            raise ValueError(f"k must be in 0..{self.K - 1}, got {k}")
        t = _finite_float("t", t)
        wt = self.omegas[k] * t
        return self.couplings[k] * (math.cos(wt) * _SIGMA_X - math.sin(wt) * _SIGMA_Y)

    def memory_time(self) -> float | None:
        # no memory-time cutoff is imposed; with non-zero rates the bath correlation
        # decays on its own, but the pipeline never truncates the history.
        return None

    # -- diagnostics -------------------------------------------------------

    def fock_populations(self, rho) -> np.ndarray:
        """Photon-number distribution ``p_n = rho_nn`` of a cavity density matrix.

        This is the diagnostic for whether ``n_fock`` is wide enough: plot it and read
        off where the distribution has decayed.  Deliberately *not* a "maximum occupied
        level" -- that would need an arbitrary threshold, and for a coherent or thermal
        state no finite maximum exists at all.

        Works on a **backend-native** array: the diagonal is taken with array methods
        rather than ``np.asarray``, which CuPy refuses to convert implicitly, so a
        ``rho`` read straight off a GPU run is accepted and returns a GPU array.
        """
        shape = getattr(rho, "shape", None)
        if shape != (self.n_fock, self.n_fock):
            raise ValueError(
                f"rho must be a ({self.n_fock}, {self.n_fock}) cavity density matrix, "
                f"got shape {shape}")
        return rho.diagonal().real

    def __repr__(self) -> str:
        return (f"DickeModel(K={self.K}, n_fock={self.n_fock}, omega_c={self.omega_c!r}, "
                f"cavity_state={self.cavity_state!r}, bath_state={self.bath_state!r})")
