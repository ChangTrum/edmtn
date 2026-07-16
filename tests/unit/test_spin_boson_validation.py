"""SpinBosonModel parameter validation (P0-2).

The model's spectral density and bath correlation are defined directly by its
parameters, so an illegal / non-finite parameter must fail loudly at construction
(ValueError) rather than producing negative spectra, NaN correlations, or backend-
dependent failures downstream. Huge-but-finite parameters that overflow float64
(or math.gamma) are reported as FloatingPointError; J0=0 is a legal no-coupling
baseline that short-circuits to exactly zero.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from edmtn.models import SpinBosonModel

_BASE = dict(J0=0.5, omega_c=5.0, mu=1.0)


# -- parameter rejection matrix (ValueError) -------------------------------

@pytest.mark.parametrize("kw", [
    {"J0": -0.1}, {"J0": np.nan}, {"J0": np.inf},
    {"omega_c": 0}, {"omega_c": np.nan}, {"omega_c": np.inf},
    {"mu": 0}, {"mu": np.nan}, {"mu": np.inf},
    {"s": 0}, {"s": -1}, {"s": np.nan}, {"s": np.inf},
    {"temperature": -1}, {"temperature": np.nan}, {"temperature": np.inf},
    {"time_step_order": 1.0}, {"time_step_order": True},
    {"time_step_order": 0}, {"time_step_order": 3},
])
def test_bad_parameters_rejected(kw):
    with pytest.raises(ValueError):
        SpinBosonModel(**{**_BASE, **kw})


# -- legal values pass, incl. J0=0 baseline and the s menu -----------------

@pytest.mark.parametrize("kw", [
    {"J0": 0.0}, {"J0": 0.5},
    {"s": 0.5}, {"s": 1.0}, {"s": 2.0},
    {"temperature": 0.0}, {"temperature": 0.3},   # 0.3 builds; the engine rejects it later
    {"time_step_order": 1}, {"time_step_order": 2},
])
def test_legal_parameters_pass(kw):
    SpinBosonModel(**{**_BASE, **kw})


@pytest.mark.parametrize("param", ["J0", "omega_c", "mu", "s", "temperature"])
def test_huge_python_int_rejected_as_valueerror(param):
    # 10**400 is a numbers.Real that overflows float64; float() would leak a raw
    # OverflowError -- must be turned into ValueError, not leaked
    with pytest.raises(ValueError):
        SpinBosonModel(**{**_BASE, param: 10 ** 400})


def test_numpy_scalars_normalized_to_python():
    m = SpinBosonModel(
        J0=np.float64(0.5), omega_c=np.float64(5.0), mu=np.float64(1.0),
        s=np.float64(2.0), temperature=np.float64(0.0), time_step_order=np.int64(2),
    )
    p = m.bath_params()
    assert type(p.J0) is float and type(p.omega_c) is float
    assert type(p.s) is float and type(p.temperature) is float
    assert type(m.mu) is float
    assert type(m.time_step_order) is int and m.time_step_order == 2


# -- spectral_density: non-finite omega rejected, finite <=0 stays 0 --------

@pytest.mark.parametrize("omega", [np.nan, np.inf, -np.inf])
def test_spectral_density_nonfinite_omega_rejected(omega):
    m = SpinBosonModel(**_BASE)
    with pytest.raises(ValueError):
        m.spectral_density(omega)


def test_spectral_density_nonfinite_in_array_rejected():
    m = SpinBosonModel(**_BASE)
    with pytest.raises(ValueError):
        m.spectral_density(np.array([1.0, np.nan, 2.0]))


# -- spectral_density: finite params, finite omega, overflowing output ------

@pytest.mark.filterwarnings("error")  # no unhandled RuntimeWarning may leak
def test_spectral_density_output_overflow_raises():
    m = SpinBosonModel(J0=1e308, omega_c=1.0, mu=1.0, s=1.0)
    with pytest.raises(FloatingPointError):
        m.spectral_density(1.0)


# -- J0 = 0: strictly zero, no NaN/Inf, both shapes -------------------------

def test_zero_coupling_spectral_density_is_zero():
    m = SpinBosonModel(J0=0.0, omega_c=5.0, mu=1.0, s=1.0)
    assert m.spectral_density(1.0) == 0.0
    assert isinstance(m.spectral_density(1.0), float)
    out = m.spectral_density(np.array([-1.0, 0.0, 1.0]))
    assert_allclose(out, 0.0)
    assert np.all(np.isfinite(out))


def test_zero_coupling_short_circuits_extreme_exponent_spectral():
    # s=172 would overflow math.gamma downstream; spectral density has no gamma but
    # J0=0 must still return zero without evaluating the power/exp branch
    m = SpinBosonModel(J0=0.0, omega_c=1.0, mu=1.0, s=172.0)
    assert m.spectral_density(1.0) == 0.0


def test_zero_coupling_still_rejects_nonfinite_omega():
    # J0=0 does not excuse a non-finite frequency
    m = SpinBosonModel(J0=0.0, omega_c=1.0, mu=1.0, s=1.0)
    with pytest.raises(ValueError):
        m.spectral_density(np.nan)
    with pytest.raises(ValueError):
        m.spectral_density(np.inf)
