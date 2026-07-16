"""Gaudin / central-spin model (Layer 1).

A central spin-1/2 couples isotropically (Heisenberg) to ``K`` independent
bath spin-1/2, with no self-Hamiltonian on either side (paper Eq. 22)::

    H(t) = sum_{k=1}^{K} g_k  S . J_k,
    S . J_k = S_x J_{k;x} + S_y J_{k;y} + S_z J_{k;z}

Because neither the central spin nor the bath spins have a self-Hamiltonian:

* ``H_S = 0`` — the interaction picture is trivial, so the coupling operators
  ``S_alpha`` are time-independent;
* the bath is time-independent, giving an **infinite memory time**.

The bath is *separable* and *non-Gaussian*: each sub-bath is uncorrelated and the
cumulant expansion does not converge (all orders are non-zero), so the
downstream pipeline uses the analytic correlation-tensor MPS form (Eq. F1)
rather than cumulants.

The central spin starts polarised along ``+z`` (``rho(0) = S_z + 1/2``); each
bath spin starts at infinite temperature, i.e. maximally mixed (``I/2``,
unpolarised).  By default the couplings follow the paper's linearly decreasing
profile

    g_k = g * sqrt(6K / (2K^2 + 3K + 1)) * (K + 1 - k) / K,   k = 1..K,

normalised so that ``sum_k g_k^2 = g^2``.  Other profiles (``uniform``, ``exp``,
``random``, or an explicit array) are selectable via the ``coupling`` argument --
see :data:`COUPLING_PROFILES` -- to study how the EDM's structure depends on the
*shape* of the coupling distribution rather than the model itself.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field

import numpy as np

from .base import AbstractOQSModel

# spin-1/2 operators (S = sigma / 2); the bath spins J_k use the same operators.
_SX = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.complex128)
_SY = np.array([[0.0, -0.5j], [0.5j, 0.0]], dtype=np.complex128)
_SZ = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128)
_ID = np.eye(2, dtype=np.complex128)


# -- parameter validation (module-private leaf checks; Layer 1 keeps its own,
#    per the P0-2 decision to defer a shared validation module) -------------
def _to_float(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be representable as a finite real number, got {value!r}") from exc


def _positive_finite(name: str, value) -> float:
    v = _to_float(name, value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return v


def _finite_float(name: str, value) -> float:
    v = _to_float(name, value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def _is_int(value) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _positive_int(name: str, value) -> int:
    if not _is_int(value):
        raise ValueError(f"{name} must be a positive integer (not bool), got {value!r}")
    v = int(value)
    if v < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}")
    return v


def _validate_g_K(g, K) -> tuple[float, int]:
    """g / K validation shared by every public coupling-profile entry point (each is
    exported, so it cannot rely on GaudinModel having validated its arguments first)."""
    return _positive_finite("g", g), _positive_int("K", K)


def _normalise_descending(c: np.ndarray, g: float) -> np.ndarray:
    """Sort descending and rescale so ``sum_k g_k**2 == g**2``."""
    c = np.sort(np.asarray(c, dtype=np.float64))[::-1]
    if not np.all(np.isfinite(c)):
        raise ValueError("coupling profile has non-finite values")
    s = float(np.sum(c**2))
    if not math.isfinite(s) or s <= 0.0:
        raise ValueError("coupling profile has non-finite or zero norm")
    return c * (g / np.sqrt(s))


def linear_couplings(g: float, K: int) -> np.ndarray:
    """Paper's linearly decreasing coupling profile ``g_k`` (length ``K``, descending).

    Normalised so that ``sum_k g_k**2 == g**2``.
    """
    g, K = _validate_g_K(g, K)
    k = np.arange(1, K + 1, dtype=np.float64)
    norm = np.sqrt(6.0 * K / (2.0 * K**2 + 3.0 * K + 1.0))
    return g * norm * (K + 1.0 - k) / K


def uniform_couplings(g: float, K: int) -> np.ndarray:
    """Flat profile ``g_k = g / sqrt(K)`` (length ``K``); ``sum_k g_k**2 == g**2``."""
    g, K = _validate_g_K(g, K)
    return np.full(K, g / np.sqrt(K), dtype=np.float64)


def exponential_couplings(g: float, K: int, beta: float = 0.15) -> np.ndarray:
    """Geometric decay ``g_k ~ exp(-beta*k)`` (descending), normalised to ``g**2``."""
    g, K = _validate_g_K(g, K)
    beta = _positive_finite("beta", beta)
    c = np.exp(-beta * np.arange(K, dtype=np.float64))
    return _normalise_descending(c, g)


def random_couplings(g: float, K: int, seed: int = 0,
                     low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Disordered profile ``g_k ~ Uniform(low, high)`` (sorted descending), norm ``g**2``.

    The absolute scale of ``[low, high)`` is irrelevant after normalisation; only the
    *shape* of the draw matters.  ``seed`` selects the realisation.  Sorted, so the
    marginal magnitude spectrum (uniform order statistics) is what differs from other
    modes -- contrast with :func:`ou_couplings`, which keeps sequence correlation.
    """
    g, K = _validate_g_K(g, K)
    low = _finite_float("low", low)
    high = _finite_float("high", high)
    if not low < high:
        raise ValueError(f"low must be < high, got low={low!r}, high={high!r}")
    c = np.random.default_rng(seed).uniform(low, high, size=K)
    return _normalise_descending(c, g)


def ou_couplings(g: float, K: int, rho: float = 0.8, seed: int = 0) -> np.ndarray:
    """Correlated (OU / AR(1)) disordered profile, **NOT sorted** -- norm ``g**2``.

    ``c_k = rho * c_{k-1} + sqrt(1 - rho**2) * z_k`` with ``z ~ N(0, 1)`` -- the
    stationary Ornstein-Uhlenbeck / AR(1) process (unit marginal variance, lag-1
    correlation ``rho``); ``g_k = |c_k|`` then normalised.  Unlike the i.i.d.
    :func:`random_couplings`, this is **left in generation order** so the
    nearest-neighbour correlation along the sub-bath index survives (sorting would
    destroy it, collapsing every ``rho`` onto the same half-normal spectrum).  The
    fold therefore proceeds in sequence order and ``x = g_{L+1}^2 / gbar_L^2`` is
    non-monotonic -- meaningful for the scaling-law fit, not for a critical ``L``.
    """
    g, K = _validate_g_K(g, K)
    rho = _finite_float("rho", rho)
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"rho must be in [0, 1), got {rho!r}")
    rng = np.random.default_rng(seed)
    c = np.empty(K, dtype=np.float64)
    c[0] = rng.standard_normal()
    s = np.sqrt(1.0 - rho * rho)
    for k in range(1, K):
        c[k] = rho * c[k - 1] + s * rng.standard_normal()
    a = np.abs(c)
    nrm = float(np.sqrt(np.sum(a**2)))
    if nrm <= 0:
        raise ValueError("OU draw collapsed to zero norm")
    return a * (g / nrm)                       # NO sort: keep the correlation


COUPLING_PROFILES = {
    "linear": linear_couplings,
    "uniform": uniform_couplings,
    "exp": exponential_couplings,
    "random": random_couplings,
    "ou": ou_couplings,
}


def coupling_profile(kind: str, g: float, K: int, **params) -> np.ndarray:
    """Return the named coupling profile ``g_k`` (descending, ``sum g_k**2 == g**2``).

    ``kind`` is one of :data:`COUPLING_PROFILES`; ``params`` are the profile's own
    knobs (``beta`` for ``exp``; ``seed``/``low``/``high`` for ``random``).
    """
    try:
        fn = COUPLING_PROFILES[kind]
    except KeyError:
        raise ValueError(
            f"unknown coupling profile {kind!r}; choose from {sorted(COUPLING_PROFILES)}"
        ) from None
    return fn(g, K, **params)


@dataclass(frozen=True)
class GaudinBathParams:
    """Parameters of the Gaudin spin bath.

    Parameters
    ----------
    g : float
        Base coupling constant; ``sum_k g_k**2 == g**2``.  Sets the time unit.
    K : int
        Number of bath spin-1/2.
    couplings : ndarray
        The per-sub-bath couplings ``g_k`` (length ``K``), descending.
    temperature : float
        Bath temperature; ``inf`` (the only validated case) means each bath spin
        is maximally mixed (``I/2``).
    """

    g: float
    K: int
    couplings: np.ndarray = field(repr=False)
    temperature: float = np.inf

    def __post_init__(self):
        # own private read-only copy: never alias a caller-supplied array (frozen -> setattr)
        c = np.array(self.couplings, dtype=np.float64, copy=True)
        c.setflags(write=False)
        object.__setattr__(self, "couplings", c)


class GaudinModel(AbstractOQSModel):
    """Central spin-1/2 isotropically coupled to ``K`` bath spin-1/2.

    Parameters
    ----------
    g : float
        Base coupling constant (sets the time unit ``g^{-1}``).
    K : int
        Number of bath spins (paper uses ``K = 49``).
    time_step_order : int
        Small-step expansion order used downstream (default ``2``, as in the paper).
    coupling : str | array-like
        The per-sub-bath coupling profile ``g_k``.  Either a named profile from
        :data:`COUPLING_PROFILES` (``"linear"`` -- the paper default, ``"uniform"``,
        ``"exp"``, ``"random"``) or an explicit length-``K`` array of couplings.
        Named profiles are normalised so ``sum_k g_k**2 == g**2`` and returned
        descending; an explicit array is used verbatim (you own its normalisation).
    coupling_params : dict, optional
        Extra knobs for a named profile (``beta`` for ``"exp"``;
        ``seed``/``low``/``high`` for ``"random"``).  Ignored for explicit arrays.
    """

    bath_type = "separable"

    def __init__(self, g: float, K: int, time_step_order: int = 2, *,
                 coupling: str | np.ndarray = "linear",
                 coupling_params: dict | None = None):
        self.g = _positive_finite("g", g)
        self.K = _positive_int("K", K)
        if not _is_int(time_step_order) or int(time_step_order) not in (1, 2):
            raise ValueError(
                f"time_step_order must be the integer 1 or 2, got {time_step_order!r}")
        self.time_step_order = int(time_step_order)
        self.coupling_params = dict(coupling_params or {})
        if isinstance(coupling, str):
            self.coupling = coupling
            gk = coupling_profile(coupling, self.g, self.K, **self.coupling_params)
        else:
            self.coupling = "custom"
            # own read-only copy; illegal contents (complex, too-large int, non-finite)
            # become ValueError rather than a leaked OverflowError/TypeError.  Custom
            # couplings are used verbatim -- any sign, no normalisation imposed.
            try:
                gk = np.array(coupling, dtype=np.float64, copy=True)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"explicit couplings must be a real, finite, length-{self.K} array") from exc
            if gk.shape != (self.K,):
                raise ValueError(
                    f"explicit couplings must have length K={self.K}, got shape {gk.shape}"
                )
            if not np.all(np.isfinite(gk)):
                raise ValueError("explicit couplings must all be finite")
            gk.setflags(write=False)
        self._bath = GaudinBathParams(g=self.g, K=self.K, couplings=gk)

    # -- system ------------------------------------------------------------

    @property
    def system_dim(self) -> int:
        return 2

    def system_hamiltonian(self) -> np.ndarray:
        # no central-spin self-Hamiltonian (Eq. 22): H_S = 0
        return np.zeros((2, 2), dtype=np.complex128)

    def coupling_operators(self) -> list[np.ndarray]:
        # three isotropic channels: S_x, S_y, S_z
        return [_SX.copy(), _SY.copy(), _SZ.copy()]

    def coupling_operators_at(self, t: float) -> list[np.ndarray]:
        # H_S = 0 => interaction picture is trivial; operators are static.
        return self.coupling_operators()

    def system_operators(self) -> dict[str, np.ndarray]:
        return {"I": _ID.copy(), "Sx": _SX.copy(), "Sy": _SY.copy(), "Sz": _SZ.copy()}

    def initial_system_state(self) -> np.ndarray:
        # rho(0) = S_z + 1/2 = diag(1, 0), polarised along +z
        return _SZ + 0.5 * _ID

    # -- bath --------------------------------------------------------------

    def bath_params(self) -> GaudinBathParams:
        return self._bath

    def bath_spin_operators(self) -> list[np.ndarray]:
        """Single bath-spin operators ``[J_x, J_y, J_z]`` (spin-1/2, ``sigma/2``).

        The bath operator on channel ``alpha`` of sub-bath ``k`` is
        ``g_k * J_alpha``; the operators are identical for every ``k``.
        """
        return [_SX.copy(), _SY.copy(), _SZ.copy()]

    @property
    def couplings(self) -> np.ndarray:
        """The per-sub-bath couplings ``g_k`` (length ``K``)."""
        return self._bath.couplings

    def effective_coupling(self, L: int | None = None) -> float:
        """Effective coupling ``g_L = sqrt(sum_{k=1}^{L} g_k**2)`` of the first ``L`` sub-baths.

        The paper scales time by ``g_L * t`` (Figs. 6, 11, 12, where the bond-dimension
        growth collapses onto a universal curve).  For the **normalised named profiles**
        ``g_K = g`` (the distribution is normalised so ``sum_k g_k**2 == g**2``); for a
        **custom** array the couplings are used verbatim, so ``effective_coupling`` simply
        reports ``sqrt(sum g_k**2)`` of the supplied array -- not necessarily ``g``.
        ``L`` defaults to ``K`` and must otherwise be an integer in ``1..K``.
        """
        if L is None:
            L = self.K
        elif not _is_int(L) or not 1 <= int(L) <= self.K:
            raise ValueError(f"L must be None or an integer in 1..{self.K}, got {L!r}")
        L = int(L)
        return float(np.sqrt(np.sum(self._bath.couplings[:L] ** 2)))

    def memory_time(self) -> float | None:
        # bath spins have no self-Hamiltonian => time-independent => infinite memory
        return None
