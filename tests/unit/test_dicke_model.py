"""Layer-1 contract of :class:`~edmtn.models.dicke.DickeModel`.

Guards the pieces the rest of the Dicke pipeline reads directly: the single coupling
channel (``d_phys = 3``), the closed-form interaction-picture ``S(t)``, the **Pauli-unit**
bath operator ``B_k(t)``, the per-spin parameter alignment (never sorted), the four cavity
and four bath initial states, and the ``bath_params()`` container that
``validate_separable_bath_kernel`` reads ``K`` from.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from edmtn.models import DickeModel, ModelRegistry
from edmtn.models.base import AbstractOQSModel

_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_SY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)


def _model(**kw):
    base = dict(K=3, n_fock=4, coupling=0.6)
    base.update(kw)
    return DickeModel(**base)


# -- structure ---------------------------------------------------------------------

def test_model_validates_and_declares_a_single_channel():
    m = _model()
    m.validate()                                    # raises if the model is malformed
    assert m.bath_type == "separable_td"
    assert m.system_dim == 4                        # n_fock IS the dimension
    assert len(m.coupling_operators()) == 1         # single channel -> d_phys = 3
    assert m.time_step_order == 2


def test_registered_under_dicke():
    assert ModelRegistry.is_registered("dicke")
    m = ModelRegistry.create("dicke", K=2, n_fock=3, coupling=0.4)
    assert isinstance(m, DickeModel)


def test_hamiltonian_and_coupling_operator_conventions():
    m = _model(omega_c=1.7)
    np.testing.assert_allclose(np.diag(m.system_hamiltonian()).real,
                               1.7 * np.arange(4), atol=1e-14)
    a = m.annihilation_operator()
    # a|n> = sqrt(n)|n-1>, so a[n-1, n] = sqrt(n)
    np.testing.assert_allclose(np.diag(a, 1).real, np.sqrt([1.0, 2.0, 3.0]), atol=1e-14)
    # the coupling operator is a + a^dag with NO 1/sqrt(2)
    np.testing.assert_allclose(m.coupling_operators()[0], a + a.conj().T, atol=1e-14)


def test_number_operator_and_commutator():
    m = _model(n_fock=6)
    a = m.annihilation_operator()
    ops = m.system_operators()
    np.testing.assert_allclose(ops["n"], a.conj().T @ a, atol=1e-14)
    # [a, a^dag] = I except in the top row, where the truncation bites
    comm = a @ a.conj().T - a.conj().T @ a
    np.testing.assert_allclose(np.diag(comm).real[:-1], np.ones(5), atol=1e-14)
    assert np.isclose(np.diag(comm).real[-1], -5.0)   # truncation defect, not a bug
    assert set(ops) == {"I", "a", "adag", "n", "x", "p", "P_top"}
    assert ops["P_top"][5, 5] == 1.0 and np.trace(ops["P_top"]).real == 1.0


# -- interaction picture -----------------------------------------------------------

@pytest.mark.parametrize("t", [0.0, 0.37, 2.5, -1.25])
def test_closed_form_St_matches_the_base_class_eigendecomposition(t):
    """The override must be an optimisation, not a different operator."""
    m = _model(omega_c=1.3)
    closed = m.coupling_operators_at(t)[0]
    base = AbstractOQSModel.coupling_operators_at(m, t)[0]
    np.testing.assert_allclose(closed, base, atol=1e-13)


def test_St_is_hermitian_and_reduces_to_a_plus_adag_at_zero():
    m = _model(omega_c=0.9)
    for t in (0.0, 0.8):
        S = m.coupling_operators_at(t)[0]
        np.testing.assert_allclose(S, S.conj().T, atol=1e-14)
    np.testing.assert_allclose(m.coupling_operators_at(0.0)[0],
                               m.coupling_operators()[0], atol=1e-14)


@pytest.mark.parametrize("t", [0.0, 0.41, 1.9])
def test_bath_operator_is_the_pauli_unit_closed_form(t):
    m = _model(K=2, coupling=[0.35, 0.22], omega=[0.8, 1.3])
    for k in range(2):
        expected = m.couplings[k] * (math.cos(m.omegas[k] * t) * _SX
                                     - math.sin(m.omegas[k] * t) * _SY)
        np.testing.assert_allclose(m.bath_operator_at(k, t), expected, atol=1e-14)
    # Pauli units, NOT spin units: ||B_k(0)|| is g_k, not g_k/2
    assert np.isclose(abs(m.bath_operator_at(0, 0.0)[0, 1]), 0.35)


@pytest.mark.parametrize("k", [True, -1, 2, 2.0, "0"])
def test_bath_operator_rejects_an_illegal_sub_bath_index(k):
    m = _model(K=2, coupling=0.3)
    with pytest.raises(ValueError):
        m.bath_operator_at(k, 0.1)


# -- per-spin parameters -----------------------------------------------------------

def test_scalar_coupling_is_the_collective_G():
    m = _model(K=4, coupling=0.8)
    np.testing.assert_allclose(m.couplings, np.full(4, 0.8 / 2.0), atol=1e-14)
    assert np.isclose(float(np.sum(m.couplings ** 2)), 0.8 ** 2)   # sum g_k^2 = G^2


def test_explicit_arrays_are_used_verbatim_and_stay_aligned():
    """The whole point of not reusing Gaudin's sorted profiles: index k is one spin."""
    g = [0.1, 0.9, -0.4]
    w = [2.0, 0.5, 1.25]
    m = _model(K=3, coupling=g, omega=w, emission=[0.01, 0.02, 0.03])
    np.testing.assert_allclose(m.couplings, g, atol=1e-14)       # no sorting
    np.testing.assert_allclose(m.omegas, w, atol=1e-14)          # no re-ordering
    np.testing.assert_allclose(m.bath_params().emission, [0.01, 0.02, 0.03], atol=1e-14)
    # and B_k pairs the k-th coupling with the k-th frequency
    np.testing.assert_allclose(m.bath_operator_at(1, 0.0), 0.9 * _SX, atol=1e-14)


def test_bath_params_container_is_read_only_and_carries_K():
    m = _model(K=3)
    bp = m.bath_params()
    assert bp.K == 3                                  # validate_separable_bath_kernel reads this
    for name in ("couplings", "omegas", "pump", "emission", "dephasing", "bloch"):
        arr = getattr(bp, name)
        assert not arr.flags.writeable, name
    assert bp.bloch.shape == (3, 3)


def test_supplied_arrays_are_copied_not_aliased():
    g = np.array([0.1, 0.2, 0.3])
    m = _model(K=3, coupling=g)
    g[0] = 99.0
    assert m.couplings[0] == 0.1


# -- initial states ----------------------------------------------------------------

def test_vacuum_is_the_default_cavity_state():
    m = _model()
    rho = m.initial_system_state()
    assert np.isclose(rho[0, 0].real, 1.0) and np.isclose(np.trace(rho).real, 1.0)


def test_truncated_coherent_state_is_renormalised_and_matches_the_poisson_curve():
    alpha = 0.7 + 0.3j
    m = _model(n_fock=12, cavity_state="coherent", cavity_params={"alpha": alpha})
    m.validate()
    p = m.fock_populations(m.initial_system_state())
    assert np.isclose(p.sum(), 1.0)                   # renormalised inside the truncation
    # shape matches |alpha|^{2n}/n! up to the truncation's renormalisation constant
    n = np.arange(12)
    poisson = abs(alpha) ** (2 * n) / np.array([math.factorial(int(i)) for i in n])
    np.testing.assert_allclose(p / p[0], poisson / poisson[0], rtol=1e-10)


def test_truncated_thermal_cavity_state_is_geometric():
    nbar = 0.8
    m = _model(n_fock=10, cavity_state="thermal", cavity_params={"nbar": nbar})
    m.validate()
    p = m.fock_populations(m.initial_system_state())
    assert np.isclose(p.sum(), 1.0)
    np.testing.assert_allclose(p[1:] / p[:-1], nbar / (1.0 + nbar), rtol=1e-12)


def test_explicit_cavity_state_round_trips():
    rho = np.diag([0.5, 0.3, 0.2, 0.0]).astype(np.complex128)
    m = _model(cavity_state=rho)
    m.validate()
    np.testing.assert_allclose(m.initial_system_state(), rho, atol=1e-14)
    assert m.cavity_state == "custom"


def test_bath_states_give_the_documented_bloch_vectors():
    inf = _model(K=2, bath_state="inf")
    np.testing.assert_allclose(inf.bath_bloch_vectors(), np.zeros((2, 3)), atol=1e-14)
    assert inf.bath_params().temperature == math.inf

    gnd = _model(K=2, bath_state="ground")
    np.testing.assert_allclose(gnd.bath_bloch_vectors()[:, 2], [-1.0, -1.0], atol=1e-14)
    assert gnd.bath_params().temperature == 0.0

    beta, w = 2.0, [0.8, 1.2]
    th = _model(K=2, omega=w, bath_state="thermal", bath_state_params={"beta": beta})
    np.testing.assert_allclose(th.bath_bloch_vectors()[:, 2],
                               -np.tanh(0.5 * beta * np.asarray(w)), atol=1e-14)
    assert np.isclose(th.bath_params().temperature, 0.5)


def test_explicit_bloch_vectors_round_trip_and_reject_unphysical_ones():
    r = np.array([[0.2, -0.3, 0.5], [0.0, 0.4, -0.4]])
    m = _model(K=2, bath_state=r)
    np.testing.assert_allclose(m.bath_bloch_vectors(), r, atol=1e-14)
    assert math.isnan(m.bath_params().temperature)     # no temperature is implied
    with pytest.raises(ValueError, match=r"\|\|r_k\|\| <= 1"):
        _model(K=2, bath_state=[[0.0, 0.0, 1.0], [0.9, 0.9, 0.0]])


# -- parameter validation ----------------------------------------------------------

@pytest.mark.parametrize("kw", [
    dict(K=0), dict(K=True), dict(K=2.0),
    dict(n_fock=1), dict(n_fock=True), dict(n_fock=3.5),      # n_fock >= 2
    dict(omega_c=0.0), dict(omega_c=-1.0), dict(omega_c=math.inf),
    dict(kappa=-0.1), dict(kappa=math.nan),
    dict(pump=-1e-9), dict(emission=[0.1, -0.1, 0.1]), dict(dephasing=math.inf),
    dict(coupling=math.nan), dict(coupling="linear"),         # no named profiles
    dict(omega=0.0), dict(omega=-1.0),                        # omega must be > 0 ...
    dict(omega=[1.0, 0.0, 1.0]), dict(omega=[1.0, -2.0, 1.0]),  # ... entry by entry too
    dict(coupling=[0.1, 0.2]),                                # wrong length for K=3
    dict(omega=[1.0, 2.0]),                                   # wrong length
    dict(time_step_order=3), dict(time_step_order=True),
    dict(cavity_state="squeezed"), dict(bath_state="hot"),
    dict(cavity_state=np.zeros((3, 3))),                      # wrong shape for n_fock=4
    dict(bath_state=np.zeros((2, 3))),                        # wrong shape for K=3
    dict(bath_state_params={"beta": 0.0}, bath_state="thermal"),
    dict(cavity_params={"nbar": -1.0}, cavity_state="thermal"),
    dict(cavity_params={"alpha": math.inf}, cavity_state="coherent"),
])
def test_illegal_parameters_raise_value_error(kw):
    with pytest.raises(ValueError):
        _model(**kw)


def test_huge_integer_parameter_is_a_value_error_not_an_overflow():
    with pytest.raises(ValueError):
        _model(omega_c=10 ** 400)


# -- diagnostics -------------------------------------------------------------------

def test_fock_populations_sums_to_the_trace_and_rejects_a_wrong_shape():
    m = _model(n_fock=5, cavity_state="thermal", cavity_params={"nbar": 1.5})
    rho = m.initial_system_state()
    p = m.fock_populations(rho)
    assert p.shape == (5,)
    assert np.isclose(p.sum(), np.trace(rho).real)
    with pytest.raises(ValueError, match="cavity density matrix"):
        m.fock_populations(np.eye(3, dtype=np.complex128))


@pytest.mark.parametrize("bad", [(3, 4), (5, 4), (5,), (5, 5, 1)])
def test_fock_populations_rejects_a_non_square_or_wrong_sized_matrix(bad):
    """A rectangular (n_fock, m) matrix has a diagonal of the right length -- checking the
    diagonal alone would let it through, so the full shape must be checked first."""
    m = _model(n_fock=5)
    with pytest.raises(ValueError, match="cavity density matrix"):
        m.fock_populations(np.zeros(bad, dtype=np.complex128))


def test_ground_bath_state_is_the_ground_state_of_the_spin_hamiltonian():
    """``r_z = -1`` is the ground state of ``(omega_k/2) sigma_z`` only for positive omega,
    which is exactly why omega is required to be strictly positive."""
    m = _model(K=2, omega=[0.5, 2.0], bath_state="ground")
    np.testing.assert_allclose(m.bath_bloch_vectors()[:, 2], [-1.0, -1.0], atol=1e-14)
    # thermal agrees with it in the zero-temperature limit, for the same reason
    cold = _model(K=2, omega=[0.5, 2.0], bath_state="thermal",
                  bath_state_params={"beta": 500.0})
    np.testing.assert_allclose(cold.bath_bloch_vectors()[:, 2], [-1.0, -1.0], atol=1e-9)


def test_fock_populations_does_not_force_a_numpy_conversion():
    """Must work on a backend-native array: no np.asarray (CuPy refuses that implicitly).

    Emulated with a minimal duck-typed array so the guard is exercised on CPU too.
    """
    class _NoNumpy:
        def __init__(self, diag):
            self._diag = np.asarray(diag)
            self.shape = (self._diag.size, self._diag.size)

        def diagonal(self):
            return self._diag

        def __array__(self, *a, **k):                       # what CuPy refuses to do
            raise TypeError("implicit conversion to a NumPy array is not allowed")

    m = _model(n_fock=4)
    p = m.fock_populations(_NoNumpy(np.array([0.5, 0.25, 0.25, 0.0])))
    np.testing.assert_allclose(p, [0.5, 0.25, 0.25, 0.0], atol=1e-14)


def test_defaults_are_the_closed_model():
    """Every rate zero by default -- the closed inhomogeneous Dicke model."""
    bp = _model().bath_params()
    for name in ("pump", "emission", "dephasing"):
        np.testing.assert_allclose(getattr(bp, name), 0.0, atol=0.0)
    assert _model().kappa == 0.0
