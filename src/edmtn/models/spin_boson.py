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

The system starts fully polarised along ``+z``, and the bath at the model's
``temperature`` (default ``0`` -- vacuum).

**Layered capability contract.** The *model* accepts any finite ``temperature >= 0``:
it is a parameter container and does not itself decide what can be solved.  The
*Gaussian cumulant engine* (Layer 2) currently implements the **zero-temperature**
correlation only, so a non-zero temperature raises ``NotImplementedError`` at compute
time, not at construction.  Keep the two apart when reading errors: an out-of-range or
non-finite parameter is a ``ValueError`` from the model, an unsupported-but-legal
setting is a ``NotImplementedError`` from the engine, and legal parameters whose
correlation overflows float64 give a ``FloatingPointError``.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np

from .base import AbstractOQSModel

# spin-1/2 operators (S = sigma / 2)
_SX = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.complex128)
_SY = np.array([[0.0, -0.5j], [0.5j, 0.0]], dtype=np.complex128)
_SZ = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128)
_ID = np.eye(2, dtype=np.complex128)


# -- parameter validation (module-private; Layer 1 keeps its own leaf checks
#    rather than importing the driver-layer validators) --------------------
def _to_float(name: str, value) -> float:
    """Coerce a real ``value`` to float, turning a too-large Python int (which is a
    ``numbers.Real`` but overflows float64) into a ``ValueError`` rather than a raw
    ``OverflowError`` -- keeping the "illegal parameter -> ValueError" contract."""
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


def _nonnegative_finite(name: str, value) -> float:
    v = _to_float(name, value)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return v


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
        Bath temperature; finite and ``>= 0`` (``0`` = vacuum).  The *model* accepts any
        finite non-negative value, but the Gaussian cumulant engine currently implements
        the zero-temperature correlation only and raises ``NotImplementedError`` at compute
        time for a non-zero temperature.
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
        Dimensionless system-bath coupling strength; finite, ``>= 0``.  ``J0 = 0`` is a
        legal no-coupling baseline (the spectral density and Gaussian correlation are then
        exactly zero), not an invalid value.
    omega_c : float
        Bath cutoff frequency; finite, ``> 0``.
    mu : float
        Transverse tunnelling strength (``H_S = mu S_x``); finite, ``> 0``.  Sets the time unit.
    s : float
        Spectral exponent; finite, ``> 0`` (default ``1.0``, Ohmic).
    temperature : float
        Bath temperature; finite, ``>= 0`` (default ``0.0``).  Accepted here, but the Gaussian
        cumulant engine currently supports only ``0`` -- a non-zero value raises
        ``NotImplementedError`` when the correlation is computed, not at construction.
    time_step_order : int
        Small-step expansion order used downstream: a strict non-``bool`` integer ``1`` or ``2``
        (default ``2``).

    Out-of-range or non-finite parameters raise ``ValueError`` here; legal parameters whose
    correlation overflows float64 raise ``FloatingPointError`` at compute time.
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
        J0 = _nonnegative_finite("J0", J0)          # 0 = no bath coupling (kept as a baseline)
        omega_c = _positive_finite("omega_c", omega_c)
        mu = _positive_finite("mu", mu)
        s = _positive_finite("s", s)
        temperature = _nonnegative_finite("temperature", temperature)
        # model allows temperature >= 0; the Gaussian cumulant engine still rejects
        # temperature != 0 at compute time (finite-T correlation unsupported).
        if (isinstance(time_step_order, bool)
                or not isinstance(time_step_order, numbers.Integral)
                or int(time_step_order) not in (1, 2)):
            raise ValueError(
                f"time_step_order must be the integer 1 or 2, got {time_step_order!r}")
        self.mu = mu
        self.time_step_order = int(time_step_order)
        self._bath = SpinBosonBathParams(J0=J0, omega_c=omega_c, s=s, temperature=temperature)

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
        """Spectral density ``J(omega)``, vectorised; zero for ``omega <= 0``.

        Rejects non-finite ``omega`` (``ValueError``).  ``J0 == 0`` short-circuits to
        zero (no power/exp/gamma evaluated).  Huge-but-finite ``J0``/``omega_c``/``s``
        can overflow float64; a non-finite result is reported as ``FloatingPointError``
        rather than silently returned.  Finite non-positive ``omega`` still gives 0.
        """
        p = self._bath
        omega = np.asarray(omega, dtype=np.float64)
        if not np.all(np.isfinite(omega)):
            raise ValueError("omega must be finite")
        if p.J0 == 0.0:
            out = np.zeros_like(omega)
            return out if out.ndim else float(out)
        # clamp to >= 0 so the power/exp are evaluated on a safe branch, then mask out
        # the non-positive frequencies.  errstate silences the expected overflow warnings.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            w = np.where(omega > 0.0, omega, 0.0)
            try:
                j = (2.0 * p.J0 * np.float64(p.omega_c) ** (1.0 - p.s)
                     * w ** p.s * np.exp(-w / p.omega_c))
            except OverflowError as exc:
                raise FloatingPointError(
                    "spectral density overflowed; check J0, omega_c, s, and omega") from exc
            out = np.where(omega > 0.0, j, 0.0)
        if not np.all(np.isfinite(out)):
            raise FloatingPointError(
                "spectral density is non-finite; check J0, omega_c, s, and omega")
        return out if out.ndim else float(out)

    def memory_time(self) -> float | None:
        # the Ohmic correlation decays as a power law (no hard cutoff); the
        # bond dimension is controlled by truncation precision instead.
        return None
