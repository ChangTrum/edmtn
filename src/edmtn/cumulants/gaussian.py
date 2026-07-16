"""Gaussian-bath cumulant engine (Layer 2).

A Gaussian bath has only second-order cumulants, fully determined by the bath
correlation function

    f(tau) = integral_0^inf  J(omega) e^{-i omega tau} d omega

evaluated at zero temperature.  In the superoperator index convention used
downstream (index 1 = commutator action B^-, index 2 = anticommutator/mean-field
action B^+), the nonzero second-order cumulants between a later time and an
earlier time are

    C[2, 2] = Re f(lag)
    C[2, 1] = 2 Im f(lag)

so the kernel construction only needs ``Re f`` and ``2 Im f`` sampled at integer
step lags ``lag = m * eps``.

For the generalised Ohmic family ``J(omega) = 2 J0 w_c^{1-s} omega^s e^{-omega/w_c}``
the zero-temperature correlation has the closed form

    f(tau) = 2 J0 Gamma(s+1) w_c^2 / (1 + i w_c tau)^{s+1},

which this engine uses by default; a numerical Fourier integral of the model's
spectral density is available as a cross-check / fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .base import CumulantEngine


@dataclass(frozen=True)
class GaussianCumulants:
    """Second-order cumulants of a Gaussian bath on a discrete time grid.

    Attributes
    ----------
    eps : float
        Time step.
    n_steps : int
        Number of steps ``N`` (the grid covers lags ``0 .. N``).
    f : np.ndarray
        Complex array of shape ``(N + 1,)`` with ``f[m] = f(m * eps)``.
    """

    eps: float
    n_steps: int
    f: np.ndarray

    @property
    def re(self) -> np.ndarray:
        """``Re f`` at each integer step lag (the ``C[2, 2]`` cumulant)."""
        return self.f.real

    @property
    def im2(self) -> np.ndarray:
        """``2 Im f`` at each integer step lag (the ``C[2, 1]`` cumulant)."""
        return 2.0 * self.f.imag

    def f_at(self, lag: int) -> complex:
        """Correlation at integer step ``lag`` (i.e. physical time ``lag * eps``)."""
        return complex(self.f[lag])


class GaussianCumulantEngine(CumulantEngine):
    """Second-order cumulants of a zero-temperature Gaussian bath.

    Parameters
    ----------
    method : {'auto', 'analytic', 'numeric'}
        How to evaluate the correlation function.  ``'analytic'`` uses the
        closed-form generalised-Ohmic result; ``'numeric'`` integrates the
        model's spectral density with an oscillatory-weight quadrature;
        ``'auto'`` picks analytic.
    """

    bath_type = "gaussian"

    def __init__(self, method: str = "auto"):
        if method not in ("auto", "analytic", "numeric"):
            raise ValueError(f"unknown method {method!r}")
        self.method = method

    # -- public API --------------------------------------------------------

    def compute(self, model, T: float, eps: float) -> GaussianCumulants:
        """Sample the bath correlation on the grid ``f(m * eps)`` for ``m = 0..N``."""
        self._check_model(model)
        self._require_zero_temperature(model)
        n = self._n_steps(T, eps)
        lags = eps * np.arange(n + 1)
        f = np.asarray(self.correlation_function(model, lags), dtype=np.complex128)
        # final defensive guard: the analytic path already raises on overflow, but this
        # also catches the numeric method, subclasses, or future implementations that
        # return a non-finite correlation before it reaches the kernel.
        if not np.all(np.isfinite(f)):
            raise FloatingPointError(
                "Gaussian bath correlation f(tau) is non-finite; check the bath parameters")
        return GaussianCumulants(eps=eps, n_steps=n, f=f)

    def correlation_function(self, model, tau):
        """Evaluate ``f(tau)`` for scalar or array ``tau`` (physical time)."""
        method = self.method
        if method == "auto":
            method = "analytic"
        if method == "analytic":
            return self._analytic(model, tau)
        return self._numeric(model, tau)

    # -- implementations ---------------------------------------------------

    @staticmethod
    def _require_zero_temperature(model) -> None:
        temp = model.bath_params().temperature
        if temp != 0.0:
            raise NotImplementedError(
                "GaussianCumulantEngine currently supports zero temperature only "
                f"(got temperature={temp})"
            )

    @staticmethod
    def _analytic(model, tau):
        """Closed-form generalised-Ohmic correlation at zero temperature.

        ``J0 == 0`` short-circuits to zero (no gamma/power evaluated).  Huge-but-finite
        ``J0``/``omega_c``/``s`` can overflow float64 or ``math.gamma``; such overflow is
        reported as ``FloatingPointError`` rather than leaking a raw ``OverflowError`` or
        a non-finite array into the kernel.
        """
        p = model.bath_params()
        tau = np.asarray(tau, dtype=np.float64)
        if p.J0 == 0.0:
            f = np.zeros(tau.shape, dtype=np.complex128)
            return f if f.ndim else complex(f)
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                prefactor = 2.0 * p.J0 * math.gamma(p.s + 1.0) * np.float64(p.omega_c) ** 2
                f = prefactor / (1.0 + 1j * p.omega_c * tau) ** (p.s + 1.0)
        except OverflowError as exc:
            raise FloatingPointError(
                "Gaussian bath correlation overflowed; check J0, omega_c, and s") from exc
        f = np.asarray(f, dtype=np.complex128)
        if not np.all(np.isfinite(f)):
            raise FloatingPointError(
                "Gaussian bath correlation f(tau) is non-finite; check J0, omega_c, and s")
        return f if f.ndim else complex(f)

    @staticmethod
    def _numeric(model, tau):
        """Numerical Fourier integral of the model spectral density."""
        from scipy.integrate import quad

        taus = np.atleast_1d(np.asarray(tau, dtype=np.float64))
        J = model.spectral_density
        out = np.empty(taus.shape, dtype=np.complex128)
        for i, t in enumerate(taus):
            re, _ = quad(J, 0.0, np.inf, weight="cos", wvar=float(t), limit=200)
            si, _ = quad(J, 0.0, np.inf, weight="sin", wvar=float(t), limit=200)
            out[i] = re - 1j * si
        return out if np.ndim(tau) else complex(out[0])
