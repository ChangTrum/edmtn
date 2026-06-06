"""Unit tests for Layer 1 (physical models).

Covers the spin-boson model's operators, interaction-picture coupling, initial
state, spectral density, and the model registry.
"""

import numpy as np
import pytest

from edmtn.models import ModelRegistry, SpinBosonBathParams, SpinBosonModel
from edmtn.models.base import AbstractOQSModel

# spin-1/2 reference operators
SX = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
SY = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
SZ = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
ID = np.eye(2, dtype=complex)


@pytest.fixture
def model():
    return SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0)


# --------------------------------------------------------------------------
# basic structure
# --------------------------------------------------------------------------

def test_is_abstract_subclass(model):
    assert isinstance(model, AbstractOQSModel)


def test_system_dim(model):
    assert model.system_dim == 2


def test_bath_type_and_order(model):
    assert model.bath_type == "gaussian"
    assert model.time_step_order == 2


def test_system_hamiltonian(model):
    H = model.system_hamiltonian()
    np.testing.assert_allclose(H, 1.0 * SX)
    np.testing.assert_allclose(H, H.conj().T)  # Hermitian


def test_coupling_operators(model):
    ops = model.coupling_operators()
    assert len(ops) == 1
    np.testing.assert_allclose(ops[0], SZ)


def test_system_operators_dict(model):
    ops = model.system_operators()
    assert set(ops) == {"I", "Sx", "Sy", "Sz"}
    np.testing.assert_allclose(ops["Sx"], SX)
    np.testing.assert_allclose(ops["Sy"], SY)
    np.testing.assert_allclose(ops["Sz"], SZ)
    np.testing.assert_allclose(ops["I"], ID)


def test_initial_state(model):
    rho = model.initial_system_state()
    np.testing.assert_allclose(rho, np.diag([1.0, 0.0]))
    assert np.isclose(np.trace(rho), 1.0)
    np.testing.assert_allclose(rho, rho.conj().T)
    # positive semidefinite
    assert np.all(np.linalg.eigvalsh(rho) >= -1e-12)


def test_validate_passes(model):
    model.validate()  # should not raise


# --------------------------------------------------------------------------
# interaction picture
# --------------------------------------------------------------------------

def test_coupling_at_zero_is_static(model):
    np.testing.assert_allclose(model.coupling_operators_at(0.0)[0], SZ, atol=1e-12)


@pytest.mark.parametrize("t", [0.1, 0.7, 1.3, 2.5, np.pi])
def test_coupling_closed_form_matches_generic(model, t):
    # closed form cos(mu t) Sz + sin(mu t) Sy must equal e^{i H_S t} Sz e^{-i H_S t}
    analytic = model.coupling_operators_at(t)[0]
    generic = AbstractOQSModel.coupling_operators_at(model, t)[0]
    np.testing.assert_allclose(analytic, generic, atol=1e-12)


@pytest.mark.parametrize("t", [0.3, 1.1, 2.0])
def test_coupling_is_hermitian_and_unit_norm(model, t):
    S = model.coupling_operators_at(t)[0]
    np.testing.assert_allclose(S, S.conj().T, atol=1e-12)
    # rotation preserves the operator's eigenvalues (+/- 1/2)
    np.testing.assert_allclose(sorted(np.linalg.eigvalsh(S)), [-0.5, 0.5], atol=1e-12)


def test_interaction_picture_preserves_trace(model):
    S = model.interaction_picture_operator(SZ, 0.9)
    assert np.isclose(np.trace(S), np.trace(SZ))


# --------------------------------------------------------------------------
# spectral density
# --------------------------------------------------------------------------

def test_spectral_density_ohmic_values(model):
    p = model.bath_params()
    # J(w) = 2 J0 w e^{-w/wc} for s = 1
    for w in (0.5, 1.0, 3.0, 7.0):
        expected = 2 * p.J0 * w * np.exp(-w / p.omega_c)
        assert np.isclose(model.spectral_density(w), expected)


def test_spectral_density_nonpositive_is_zero(model):
    assert model.spectral_density(0.0) == 0.0
    assert model.spectral_density(-1.0) == 0.0


def test_spectral_density_vectorised(model):
    w = np.array([-1.0, 0.0, 1.0, 5.0])
    out = model.spectral_density(w)
    assert out.shape == (4,)
    assert out[0] == 0.0 and out[1] == 0.0
    assert out[2] > 0 and out[3] > 0


def test_spectral_density_scalar_returns_float(model):
    out = model.spectral_density(2.0)
    assert isinstance(out, float)


def test_spectral_density_peak_at_omega_c_for_ohmic(model):
    # d/dw [w e^{-w/wc}] = 0 at w = wc
    wc = model.bath_params().omega_c
    grid = np.linspace(0.01, 5 * wc, 20001)
    peak = grid[np.argmax(model.spectral_density(grid))]
    assert np.isclose(peak, wc, rtol=2e-3)


def test_sub_and_super_ohmic_exponent():
    sub = SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0, s=0.5)
    sup = SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0, s=2.0)
    w = 1.0
    # low-frequency scaling J ~ w^s
    assert sub.spectral_density(w) != sup.spectral_density(w)
    ratio = sub.spectral_density(2.0) / sub.spectral_density(1.0)
    # for sub-Ohmic s=0.5 at small w/wc the ratio is dominated by w^0.5 * exp
    assert ratio > 0


# --------------------------------------------------------------------------
# parameters and construction guards
# --------------------------------------------------------------------------

def test_bath_params_dataclass(model):
    p = model.bath_params()
    assert isinstance(p, SpinBosonBathParams)
    assert p.J0 == 0.5 and p.omega_c == 5.0 and p.s == 1.0 and p.temperature == 0.0


@pytest.mark.parametrize("kw", [{"omega_c": -1.0}, {"mu": 0.0}, {"time_step_order": 3}])
def test_invalid_construction(kw):
    base = dict(J0=0.5, omega_c=5.0, mu=1.0)
    base.update(kw)
    with pytest.raises(ValueError):
        SpinBosonModel(**base)


def test_memory_time_default(model):
    assert model.memory_time() is None


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def test_registry_has_spin_boson():
    assert "spin_boson" in ModelRegistry.available()


def test_registry_create():
    m = ModelRegistry.create("spin_boson", J0=0.3, omega_c=4.0, mu=1.0)
    assert isinstance(m, SpinBosonModel)
    assert m.bath_params().J0 == 0.3


def test_registry_unknown_raises():
    with pytest.raises(KeyError):
        ModelRegistry.create("not_a_model")
