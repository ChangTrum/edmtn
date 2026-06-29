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

from dataclasses import dataclass, field

import numpy as np

from .base import AbstractOQSModel

# spin-1/2 operators (S = sigma / 2); the bath spins J_k use the same operators.
_SX = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.complex128)
_SY = np.array([[0.0, -0.5j], [0.5j, 0.0]], dtype=np.complex128)
_SZ = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128)
_ID = np.eye(2, dtype=np.complex128)


def _check_K(K: int) -> None:
    if K < 1:
        raise ValueError("K must be a positive integer")


def _normalise_descending(c: np.ndarray, g: float) -> np.ndarray:
    """Sort descending and rescale so ``sum_k g_k**2 == g**2``."""
    c = np.sort(np.asarray(c, dtype=np.float64))[::-1]
    s = float(np.sum(c**2))
    if s <= 0:
        raise ValueError("coupling profile has zero norm")
    return c * (g / np.sqrt(s))


def linear_couplings(g: float, K: int) -> np.ndarray:
    """Paper's linearly decreasing coupling profile ``g_k`` (length ``K``, descending).

    Normalised so that ``sum_k g_k**2 == g**2``.
    """
    _check_K(K)
    k = np.arange(1, K + 1, dtype=np.float64)
    norm = np.sqrt(6.0 * K / (2.0 * K**2 + 3.0 * K + 1.0))
    return g * norm * (K + 1.0 - k) / K


def uniform_couplings(g: float, K: int) -> np.ndarray:
    """Flat profile ``g_k = g / sqrt(K)`` (length ``K``); ``sum_k g_k**2 == g**2``."""
    _check_K(K)
    return np.full(K, g / np.sqrt(K), dtype=np.float64)


def exponential_couplings(g: float, K: int, beta: float = 0.15) -> np.ndarray:
    """Geometric decay ``g_k ~ exp(-beta*k)`` (descending), normalised to ``g**2``."""
    _check_K(K)
    if beta <= 0:
        raise ValueError("beta must be positive")
    c = np.exp(-beta * np.arange(K, dtype=np.float64))
    return _normalise_descending(c, g)


def random_couplings(g: float, K: int, seed: int = 0,
                     low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Disordered profile ``g_k ~ Uniform(low, high)`` (sorted descending), norm ``g**2``.

    The absolute scale of ``[low, high)`` is irrelevant after normalisation; only the
    *shape* of the draw matters.  ``seed`` selects the realisation.
    """
    _check_K(K)
    c = np.random.default_rng(seed).uniform(low, high, size=K)
    return _normalise_descending(c, g)


COUPLING_PROFILES = {
    "linear": linear_couplings,
    "uniform": uniform_couplings,
    "exp": exponential_couplings,
    "random": random_couplings,
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
        if g <= 0:
            raise ValueError("g must be positive")
        if K < 1:
            raise ValueError("K must be a positive integer")
        if time_step_order not in (1, 2):
            raise ValueError("time_step_order must be 1 or 2")
        self.g = float(g)
        self.K = int(K)
        self.time_step_order = time_step_order
        self.coupling_params = dict(coupling_params or {})
        if isinstance(coupling, str):
            self.coupling = coupling
            gk = coupling_profile(coupling, self.g, self.K, **self.coupling_params)
        else:
            self.coupling = "custom"
            gk = np.asarray(coupling, dtype=np.float64)
            if gk.shape != (self.K,):
                raise ValueError(
                    f"explicit couplings must have length K={self.K}, got shape {gk.shape}"
                )
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

        The paper scales time by ``g_L * t`` (Figs. 6, 11, 12, where the
        bond-dimension growth collapses onto a universal curve); ``g_K = g`` by
        the normalisation of the coupling distribution.  ``L`` defaults to ``K``.
        """
        if L is None:
            L = self.K
        if not 1 <= L <= self.K:
            raise ValueError(f"L must be in 1..{self.K}, got {L}")
        return float(np.sqrt(np.sum(self._bath.couplings[:L] ** 2)))

    def memory_time(self) -> float | None:
        # bath spins have no self-Hamiltonian => time-independent => infinite memory
        return None
