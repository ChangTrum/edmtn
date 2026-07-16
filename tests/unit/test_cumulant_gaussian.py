"""Unit tests for Layer 2 (Gaussian cumulant engine).

Validates the bath correlation function f(tau) against its closed form, the
analytic/numeric agreement, and the discrete-grid cumulant container.
"""

import math

import numpy as np
import pytest

from edmtn.cumulants import GaussianCumulantEngine, GaussianCumulants
from edmtn.models import SpinBosonModel


@pytest.fixture
def model():
    return SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0)


@pytest.fixture
def engine():
    return GaussianCumulantEngine()


def f_ref(model, tau):
    """Reference closed form 2 J0 Gamma(s+1) wc^2 / (1 + i wc tau)^(s+1)."""
    p = model.bath_params()
    return 2 * p.J0 * math.gamma(p.s + 1) * p.omega_c**2 / (1 + 1j * p.omega_c * tau) ** (p.s + 1)


# --------------------------------------------------------------------------
# correlation function
# --------------------------------------------------------------------------

def test_f_at_zero_equals_integral_of_J(model, engine):
    # f(0) = int_0^inf J(w) dw = 2 J0 wc^2 for Ohmic
    p = model.bath_params()
    f0 = engine.correlation_function(model, 0.0)
    assert np.isclose(f0.real, 2 * p.J0 * p.omega_c**2)
    assert np.isclose(f0.imag, 0.0, atol=1e-12)


@pytest.mark.parametrize("tau", [0.0, 0.02, 0.1, 0.5, 1.0, 3.0, 7.0])
def test_analytic_matches_reference(model, engine, tau):
    assert np.isclose(engine.correlation_function(model, tau), f_ref(model, tau))


@pytest.mark.parametrize("s", [0.5, 1.0, 1.5, 2.0])
def test_analytic_matches_numeric_various_s(s):
    m = SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0, s=s)
    ana = GaussianCumulantEngine(method="analytic")
    num = GaussianCumulantEngine(method="numeric")
    taus = np.array([0.0, 0.05, 0.3, 1.0, 2.5])
    fa = ana.correlation_function(m, taus)
    fn = num.correlation_function(m, taus)
    np.testing.assert_allclose(fa, fn, atol=1e-8, rtol=1e-6)


def test_correlation_vectorised(model, engine):
    taus = np.array([0.0, 0.1, 0.5, 1.0])
    f = engine.correlation_function(model, taus)
    assert f.shape == (4,)
    np.testing.assert_allclose(f, [f_ref(model, t) for t in taus])


def test_correlation_scalar_returns_complex(model, engine):
    out = engine.correlation_function(model, 0.5)
    assert isinstance(out, complex)


def test_correlation_decays(model, engine):
    taus = np.linspace(0, 10, 50)
    mag = np.abs(engine.correlation_function(model, taus))
    # magnitude is monotonically non-increasing for the Ohmic correlation
    assert np.all(np.diff(mag) <= 1e-9)


# --------------------------------------------------------------------------
# discrete-grid cumulants
# --------------------------------------------------------------------------

def test_compute_grid_shape(model, engine):
    cum = engine.compute(model, T=2.0, eps=0.01)
    assert isinstance(cum, GaussianCumulants)
    assert cum.n_steps == 200
    assert cum.f.shape == (201,)
    assert cum.eps == 0.01


def test_compute_values_on_grid(model, engine):
    eps = 0.05
    cum = engine.compute(model, T=1.0, eps=eps)
    for m in (0, 1, 5, 20):
        assert np.isclose(cum.f_at(m), f_ref(model, m * eps))


def test_re_and_im2_properties(model, engine):
    cum = engine.compute(model, T=1.0, eps=0.1)
    np.testing.assert_allclose(cum.re, cum.f.real)
    np.testing.assert_allclose(cum.im2, 2.0 * cum.f.imag)
    # at lag 0 the imaginary part vanishes
    assert np.isclose(cum.im2[0], 0.0, atol=1e-12)


def test_compute_matches_numeric_engine(model):
    ana = GaussianCumulantEngine(method="analytic").compute(model, T=1.0, eps=0.05)
    num = GaussianCumulantEngine(method="numeric").compute(model, T=1.0, eps=0.05)
    np.testing.assert_allclose(ana.f, num.f, atol=1e-8, rtol=1e-6)


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------

def test_bath_type_mismatch_raises(engine):
    class FakeModel:
        bath_type = "separable"

    with pytest.raises(ValueError):
        engine.compute(FakeModel(), T=1.0, eps=0.1)


def test_finite_temperature_not_supported(engine):
    hot = SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0, temperature=0.3)
    with pytest.raises(NotImplementedError):
        engine.compute(hot, T=1.0, eps=0.1)


def test_non_integer_steps_raises(model, engine):
    with pytest.raises(ValueError):
        engine.compute(model, T=1.0, eps=0.03)  # 1/0.03 not integer


@pytest.mark.parametrize("bad", [{"T": -1.0, "eps": 0.1}, {"T": 1.0, "eps": -0.1}])
def test_nonpositive_args_raise(model, engine, bad):
    with pytest.raises(ValueError):
        engine.compute(model, **bad)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        GaussianCumulantEngine(method="wavelet")


# --------------------------------------------------------------------------
# P0-2: J0=0 no-coupling baseline + overflow -> FloatingPointError
# --------------------------------------------------------------------------

def test_zero_coupling_correlation_is_zero(engine):
    # J0=0 short-circuits (no gamma/power); correlation strictly zero, finite
    m = SpinBosonModel(J0=0.0, omega_c=5.0, mu=1.0, s=1.0)
    cum = engine.compute(m, T=1.0, eps=0.1)
    assert np.all(np.isfinite(cum.f))
    np.testing.assert_allclose(cum.f, 0.0)


def test_zero_coupling_short_circuits_extreme_exponent(engine):
    # s=172 would overflow math.gamma(173); J0=0 must short-circuit before that
    m = SpinBosonModel(J0=0.0, omega_c=1.0, mu=1.0, s=172.0)
    cum = engine.compute(m, T=0.1, eps=0.1)
    np.testing.assert_allclose(cum.f, 0.0)


@pytest.mark.filterwarnings("error")  # no unhandled RuntimeWarning may leak
def test_correlation_overflow_omega_c(engine):
    # omega_c**2 overflows float64 -> FloatingPointError, not a raw OverflowError
    m = SpinBosonModel(J0=1.0, omega_c=1e200, mu=1.0, s=1.0)
    with pytest.raises(FloatingPointError):
        engine.compute(m, T=0.1, eps=0.1)


@pytest.mark.filterwarnings("error")
def test_correlation_overflow_gamma(engine):
    # math.gamma(s+1) overflows -> caught OverflowError -> FloatingPointError
    m = SpinBosonModel(J0=1.0, omega_c=1.0, mu=1.0, s=172.0)
    with pytest.raises(FloatingPointError):
        engine.compute(m, T=0.1, eps=0.1)


def test_compute_final_guard_catches_nonfinite_correlation():
    # isolate compute()'s final guard: a subclass returns a non-finite correlation
    # directly (bypassing the analytic path's own overflow check)
    class _NanEngine(GaussianCumulantEngine):
        def correlation_function(self, model, tau):
            return np.full(np.shape(tau), np.nan, dtype=np.complex128)

    m = SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0)
    with pytest.raises(FloatingPointError):
        _NanEngine().compute(m, T=0.2, eps=0.1)
