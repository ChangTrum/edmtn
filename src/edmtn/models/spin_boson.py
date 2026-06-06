"""Spin-boson model (Layer 1).

A spin-1/2 with a transverse tunnelling term couples through its z-component to
a bosonic bath:

    H_S   = mu * S_x
    H(t)  = S_z(t) B(t),   B(t) = sum_k g_k (a_k^dag e^{i w_k t} + h.c.)

The bath coupling distribution is the spectral density
``J(w) = sum_k g_k^2 delta(w - w_k)``.  The (generalised Ohmic) family used here
is ``J(w) = 2 J0 * w_c^{1-s} * w^s * e^{-w / w_c}`` for ``w > 0`` (``s = 1`` is
Ohmic).  In the interaction picture the coupling operator becomes
``S_z(t) = cos(mu t) S_z + sin(mu t) S_y``.

The system starts fully polarised along ``+z`` and the bath in vacuum
(zero temperature).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import AbstractOQSModel

# spin-1/2 operators (S = sigma / 2)
_SX = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.complex128)
_SY = np.array([[0.0, -0.5j], [0.5j, 0.0]], dtype=np.complex128)
_SZ = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128)
_ID = np.eye(2, dtype=np.complex128)


@dataclass(frozen=True)
class SpinBosonBathParams:
    """Parameters of the (generalised Ohmic) bosonic bath.

    Parameters
    ----------
    J0 : float
        Dimensionless coupling constant.
    omega_c : float
        Cutoff frequency.
    s : float
        Spectral exponent: ``s = 1`` Ohmic, ``s < 1`` sub-Ohmic, ``s > 1``
        super-Ohmic.
    temperature : float
        Bath temperature (``0`` = vacuum).  Phase-1 validation targets ``T = 0``.
    """

    J0: float
    omega_c: float
    s: float = 1.0
    temperature: float = 0.0


class SpinBosonModel(AbstractOQSModel):
    """Spin-1/2 coupled to a Gaussian bosonic bath via ``S_z``.

    Parameters
    ----------
    J0 : float
        Dimensionless system-bath coupling strength.
    omega_c : float
        Bath cutoff frequency.
    mu : float
        Transverse tunnelling strength (``H_S = mu S_x``); sets the time unit.
    s : float
        Spectral exponent (default ``1.0``, Ohmic).
    temperature : float
        Bath temperature (default ``0.0``).
    time_step_order : int
        Small-step expansion order used downstream (default ``2``).
    """

    bath_type = "gaussian"

    def __init__(
        self,
        J0: float,
        omega_c: float,
        mu: float,
        s: float = 1.0,
        temperature: float = 0.0,
        time_step_order: int = 2,
    ):
        if omega_c <= 0:
            raise ValueError("omega_c must be positive")
        if mu <= 0:
            raise ValueError("mu must be positive")
        if time_step_order not in (1, 2):
            raise ValueError("time_step_order must be 1 or 2")
        self.mu = float(mu)
        self.time_step_order = time_step_order
        self._bath = SpinBosonBathParams(
            J0=float(J0), omega_c=float(omega_c), s=float(s), temperature=float(temperature)
        )

    # -- system ------------------------------------------------------------

    @property
    def system_dim(self) -> int:
        return 2

    def system_hamiltonian(self) -> np.ndarray:
        return self.mu * _SX

    def coupling_operators(self) -> list[np.ndarray]:
        # single-channel coupling through S_z
        return [_SZ.copy()]

    def coupling_operators_at(self, t: float) -> list[np.ndarray]:
        # closed form of e^{i mu S_x t} S_z e^{-i mu S_x t}
        return [np.cos(self.mu * t) * _SZ + np.sin(self.mu * t) * _SY]

    def system_operators(self) -> dict[str, np.ndarray]:
        return {"I": _ID.copy(), "Sx": _SX.copy(), "Sy": _SY.copy(), "Sz": _SZ.copy()}

    def initial_system_state(self) -> np.ndarray:
        # rho(0) = S_z + 1/2 = diag(1, 0), fully polarised along +z
        return _SZ + 0.5 * _ID

    # -- bath --------------------------------------------------------------

    def bath_params(self) -> SpinBosonBathParams:
        return self._bath

    def spectral_density(self, omega):
        """Spectral density ``J(omega)``, vectorised; zero for ``omega <= 0``."""
        p = self._bath
        omega = np.asarray(omega, dtype=np.float64)
        # clamp to >= 0 so the power/exp are evaluated on a safe branch, then
        # mask out the non-positive frequencies.
        w = np.where(omega > 0.0, omega, 0.0)
        j = 2.0 * p.J0 * p.omega_c ** (1.0 - p.s) * w ** p.s * np.exp(-w / p.omega_c)
        out = np.where(omega > 0.0, j, 0.0)
        return out if out.ndim else float(out)

    def memory_time(self) -> float | None:
        # the Ohmic correlation decays as a power law (no hard cutoff); the
        # bond dimension is controlled by truncation precision instead.
        return None
