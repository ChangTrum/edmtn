"""Unit tests for the Gaudin / central-spin model (Layer 1).

Covers the model's operators (isotropic 3-channel coupling, no self-Hamiltonian),
initial state, the linearly-decreasing coupling distribution and its
normalisation, the infinite memory time, and the registry entry.
"""

import numpy as np
import pytest

from edmtn.models import GaudinBathParams, GaudinModel, ModelRegistry, linear_couplings
from edmtn.models.base import AbstractOQSModel

# spin-1/2 reference operators
SX = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
SY = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
SZ = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
ID = np.eye(2, dtype=complex)


@pytest.fixture
def model():
    return GaudinModel(g=1.0, K=49)


# --------------------------------------------------------------------------
# basic structure
# --------------------------------------------------------------------------

def test_is_abstract_subclass(model):
    assert isinstance(model, AbstractOQSModel)


def test_system_dim(model):
    assert model.system_dim == 2


def test_bath_type_and_order(model):
    assert model.bath_type == "separable"
    assert model.time_step_order == 2


def test_no_self_hamiltonian(model):
    np.testing.assert_array_equal(model.system_hamiltonian(), np.zeros((2, 2)))


def test_three_isotropic_coupling_channels(model):
    ops = model.coupling_operators()
    assert len(ops) == 3
    np.testing.assert_allclose(ops[0], SX)
    np.testing.assert_allclose(ops[1], SY)
    np.testing.assert_allclose(ops[2], SZ)


def test_coupling_is_static_no_interaction_picture_rotation(model):
    # H_S = 0 => coupling operators are time-independent
    for t in (0.0, 1.3, 7.7):
        ops_t = model.coupling_operators_at(t)
        for got, ref in zip(ops_t, (SX, SY, SZ)):
            np.testing.assert_allclose(got, ref, atol=1e-12)


def test_coupling_at_matches_generic_with_zero_HS(model):
    # generic interaction-picture with H_S = 0 reduces to the static operators
    generic = AbstractOQSModel.coupling_operators_at(model, 2.0)
    for got, ref in zip(generic, (SX, SY, SZ)):
        np.testing.assert_allclose(got, ref, atol=1e-12)


def test_bath_spin_operators(model):
    Js = model.bath_spin_operators()
    assert len(Js) == 3
    np.testing.assert_allclose(Js[0], SX)
    np.testing.assert_allclose(Js[1], SY)
    np.testing.assert_allclose(Js[2], SZ)


def test_system_operators_dict(model):
    ops = model.system_operators()
    assert set(ops) == {"I", "Sx", "Sy", "Sz"}
    np.testing.assert_allclose(ops["Sx"], SX)
    np.testing.assert_allclose(ops["Sz"], SZ)


def test_initial_state(model):
    rho = model.initial_system_state()
    np.testing.assert_allclose(rho, np.diag([1.0, 0.0]))
    assert np.isclose(np.trace(rho), 1.0)
    assert np.all(np.linalg.eigvalsh(rho) >= -1e-12)


def test_validate_passes(model):
    model.validate()  # zero H_S is Hermitian; rho is a valid state


def test_memory_time_infinite(model):
    assert model.memory_time() is None


# --------------------------------------------------------------------------
# coupling distribution
# --------------------------------------------------------------------------

@pytest.mark.parametrize("K", [1, 4, 10, 49])
def test_couplings_normalised_to_g_squared(K):
    g = 1.7
    gk = linear_couplings(g, K)
    assert gk.shape == (K,)
    assert np.isclose(np.sum(gk**2), g**2)


def test_couplings_linearly_decreasing(model):
    gk = model.couplings
    assert gk.shape == (49,)
    assert np.all(np.diff(gk) < 0)            # strictly decreasing
    # equal spacing: g_k = g*norm*(K+1-k)/K is linear in k
    assert np.allclose(np.diff(gk, 2), 0.0, atol=1e-12)
    assert gk[-1] > 0                          # last coupling positive (k=K -> (K+1-K)/K)


def test_couplings_explicit_formula():
    g, K = 2.0, 5
    gk = linear_couplings(g, K)
    k = np.arange(1, K + 1)
    expected = g * np.sqrt(6 * K / (2 * K**2 + 3 * K + 1)) * (K + 1 - k) / K
    np.testing.assert_allclose(gk, expected)


def test_bath_params(model):
    p = model.bath_params()
    assert isinstance(p, GaudinBathParams)
    assert p.g == 1.0 and p.K == 49
    assert np.isinf(p.temperature)
    np.testing.assert_allclose(p.couplings, model.couplings)


def test_linear_couplings_rejects_bad_K():
    with pytest.raises(ValueError):
        linear_couplings(1.0, 0)


# --------------------------------------------------------------------------
# effective coupling g_L (paper time-scaling for Figs. 6/11/12)
# --------------------------------------------------------------------------

def test_effective_coupling_full_equals_g(model):
    # g_K = g by normalisation
    assert model.effective_coupling() == pytest.approx(1.0)
    assert model.effective_coupling(49) == pytest.approx(1.0)


def test_effective_coupling_formula_and_monotonic(model):
    gk = model.couplings
    for L in (1, 5, 20, 49):
        assert model.effective_coupling(L) == pytest.approx(np.sqrt(np.sum(gk[:L] ** 2)))
    vals = [model.effective_coupling(L) for L in range(1, 50)]
    assert np.all(np.diff(vals) > 0)  # strictly increasing in L


@pytest.mark.parametrize("L", [0, 50, -1])
def test_effective_coupling_rejects_bad_L(model, L):
    with pytest.raises(ValueError):
        model.effective_coupling(L)


# --------------------------------------------------------------------------
# construction guards
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [{"g": 0.0}, {"g": -1.0}, {"K": 0}, {"time_step_order": 3}])
def test_invalid_construction(kw):
    base = dict(g=1.0, K=49)
    base.update(kw)
    with pytest.raises(ValueError):
        GaudinModel(**base)


# --------------------------------------------------------------------------
# selectable coupling profiles
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["linear", "uniform", "exp", "random"])
def test_named_profiles_normalised_and_descending(kind):
    g, K = 1.3, 24
    m = GaudinModel(g=g, K=K, coupling=kind)
    gk = m.couplings
    assert gk.shape == (K,)
    assert m.coupling == kind
    np.testing.assert_allclose(np.sum(gk**2), g**2, rtol=1e-12)  # sum g_k^2 = g^2
    assert np.all(np.diff(gk) <= 1e-12)                          # descending


def test_default_is_linear():
    a = GaudinModel(g=1.0, K=20).couplings
    b = GaudinModel(g=1.0, K=20, coupling="linear").couplings
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a, linear_couplings(1.0, 20))


def test_uniform_profile_is_flat():
    gk = GaudinModel(g=2.0, K=16, coupling="uniform").couplings
    np.testing.assert_allclose(gk, 2.0 / np.sqrt(16))


def test_exp_profile_beta_controls_decay():
    slow = GaudinModel(g=1.0, K=30, coupling="exp", coupling_params={"beta": 0.05}).couplings
    fast = GaudinModel(g=1.0, K=30, coupling="exp", coupling_params={"beta": 0.5}).couplings
    # both normalised; faster decay concentrates weight -> larger leading coupling
    assert fast[0] > slow[0]
    np.testing.assert_allclose(np.sum(fast**2), 1.0, rtol=1e-12)


def test_random_profile_seed_reproducible_and_varies():
    a = GaudinModel(g=1.0, K=20, coupling="random", coupling_params={"seed": 7}).couplings
    b = GaudinModel(g=1.0, K=20, coupling="random", coupling_params={"seed": 7}).couplings
    c = GaudinModel(g=1.0, K=20, coupling="random", coupling_params={"seed": 8}).couplings
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_explicit_coupling_array_used_verbatim():
    custom = np.linspace(1.0, 0.1, 12)
    m = GaudinModel(g=1.0, K=12, coupling=custom)
    assert m.coupling == "custom"
    np.testing.assert_array_equal(m.couplings, custom)


def test_explicit_coupling_wrong_length_rejected():
    with pytest.raises(ValueError):
        GaudinModel(g=1.0, K=12, coupling=np.ones(5))


def test_unknown_profile_rejected():
    with pytest.raises(ValueError):
        GaudinModel(g=1.0, K=10, coupling="nonsense")


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def test_registry_has_gaudin():
    assert "gaudin" in ModelRegistry.available()


def test_registry_create():
    m = ModelRegistry.create("gaudin", g=0.5, K=10)
    assert isinstance(m, GaudinModel)
    assert m.K == 10 and m.g == 0.5
