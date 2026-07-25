"""Time-dependent separable-bath correlation engine (Layer 2).

Every reference in this module is built **independently** of the code under test:

* the bath channel is compared against ``scipy.linalg.expm`` of the generator, and
  against a Lindblad channel assembled from the jump operators ``sigma^+``, ``sigma^-``,
  ``sigma_z`` acting on ``vec(rho)``;
* the transfer tensors are compared against the closed-form matrices of the derivation;
* the whole transfer-tensor chain is compared against a brute-force superoperator product
  on 2x2 matrices, with the dissipative channel interleaved by hand in the Strang pattern.

The brute force is what would catch a reversed ``D @ A`` / ``A @ D``, a wrong sub-step
map, or a stray second-order coefficient leaking onto the bath side.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
from scipy.linalg import expm

from edmtn.cumulants import (
    SeparableTDBathCorrelation,
    TimeDependentSeparableCorrelation,
    bath_channel_matrix,
    relaxation_factor,
)
from edmtn.models import DickeModel, GaudinModel

_I2 = np.eye(2, dtype=np.complex128)
_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_SY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_SZ = np.diag([1.0, -1.0]).astype(np.complex128)
_SP = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
_SM = _SP.conj().T
_SIGMA = [_I2, _SX, _SY, _SZ]


def _dissipator(L):
    M = L.conj().T @ L
    return np.kron(L, L.conj()) - 0.5 * (np.kron(M, _I2) + np.kron(_I2, M.T))


def _lindblad_channel(pump, emission, dephasing, dt):
    """Independent single-spin channel on row-major ``vec(rho)``, from the jump operators."""
    gen = pump * _dissipator(_SP) + emission * _dissipator(_SM) + 0.5 * dephasing * _dissipator(_SZ)
    return expm(dt * gen)


def _brute_force_correlation(model, k, eps, order, ops):
    """``Tr[B^{phi_G} ... B^{phi_1}(Omega_k)]`` with the channels interleaved by hand."""
    bp = model.bath_params()
    rates = (float(bp.pump[k]), float(bp.emission[k]), float(bp.dephasing[k]))
    Eh = _lindblad_channel(*rates, 0.5 * eps)
    r = model.bath_bloch_vectors()[k]
    X = 0.5 * (_I2 + r[0] * _SX + r[1] * _SY + r[2] * _SZ)
    chan = lambda Y: (Eh @ Y.reshape(-1)).reshape(2, 2)      # noqa: E731

    for g, phi in enumerate(ops, start=1):
        n = (g - 1) // order + 1
        B = model.bath_operator_at(k, (n - 0.5) * eps)       # midpoint, shared by sub-steps
        if phi == 0:
            bphi = lambda Y: Y                               # noqa: E731
        elif phi == 1:
            bphi = lambda Y: -1j * (B @ Y - Y @ B)           # noqa: E731
        else:
            bphi = lambda Y: 0.5 * (B @ Y + Y @ B)           # noqa: E731
        if order == 1:
            X = chan(bphi(chan(X)))                          # D_h A D_h
        elif (g - 1) % 2 == 0:
            X = bphi(chan(X))                                # A D_h   (earlier sub-step)
        else:
            X = chan(bphi(X))                                # D_h A   (later sub-step)
    return complex(np.trace(X))


def _model(**kw):
    base = dict(K=2, n_fock=3, coupling=[0.35, 0.22], omega=[0.8, 1.3])
    base.update(kw)
    return DickeModel(**base)


def _corr(model, T=0.4, eps=0.1, order=2):
    return SeparableTDBathCorrelation().compute(model, T=T, eps=eps, order=order)


# -- the relaxation factor and the bath channel ------------------------------------

def test_relaxation_factor_limit_and_agreement_with_the_naive_form():
    assert relaxation_factor(0.0, 0.037) == 0.037                 # exact limit, no 0/0
    for gamma, dt in ((0.7, 0.05), (3.0, 0.2)):
        naive = (1.0 - math.exp(-gamma * dt)) / gamma
        assert math.isclose(relaxation_factor(gamma, dt), naive, rel_tol=1e-12)


def test_relaxation_factor_is_stable_for_a_tiny_rate():
    """The naive (1 - exp(-x))/gamma loses digits here; expm1 does not."""
    gamma, dt = 1e-14, 0.05
    assert math.isclose(relaxation_factor(gamma, dt), dt, rel_tol=1e-9)


@pytest.mark.parametrize("rates", [(0.05, 0.11, 0.04), (0.0, 0.0, 0.3), (0.9, 0.0, 0.0)])
def test_bath_channel_matches_an_independent_matrix_exponential(rates):
    pump, emission, dephasing = rates
    dt = 0.025
    g1 = pump + emission
    g2 = 0.5 * g1 + dephasing
    gen = np.array([[0.0, 0.0, 0.0, 0.0],
                    [0.0, -g2, 0.0, 0.0],
                    [0.0, 0.0, -g2, 0.0],
                    [pump - emission, 0.0, 0.0, -g1]], dtype=np.float64)
    np.testing.assert_allclose(bath_channel_matrix(*rates, dt), expm(dt * gen), atol=1e-13)


@pytest.mark.parametrize("rates", [(0.05, 0.11, 0.04), (0.0, 0.0, 0.25)])
def test_bath_channel_equals_the_lindblad_channel_in_the_pauli_basis(rates):
    """Independent route: act with the jump-operator channel, then read the coefficients."""
    dt = 0.03
    E = _lindblad_channel(*rates, dt)
    D = bath_channel_matrix(*rates, dt)
    for r in ([0.2, -0.3, 0.5], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.6, 0.1, -0.2]):
        X = 0.5 * (_I2 + r[0] * _SX + r[1] * _SY + r[2] * _SZ)
        Xp = (E @ X.reshape(-1)).reshape(2, 2)
        direct = np.array([np.trace(s @ Xp) for s in _SIGMA])
        np.testing.assert_allclose(direct, D @ np.array([1.0, *r], dtype=np.complex128),
                                   atol=1e-13)


def test_zero_rates_give_the_identity_channel():
    np.testing.assert_allclose(bath_channel_matrix(0.0, 0.0, 0.0, 0.05),
                               np.eye(4, dtype=np.complex128), atol=0.0)


@pytest.mark.parametrize("args", [
    (-1.0, 0.1), (math.nan, 0.1), (math.inf, 0.1), (True, 0.1),
    (0.5, -0.1), (0.5, math.nan), (10 ** 400, 0.1), (0.5, 10 ** 400), ("1", 0.1),
])
def test_relaxation_factor_rejects_illegal_arguments(args):
    """Exported, so illegal input must not reach the exponential: a negative rate would
    give an amplifying `exp(+|gamma| dt) > 1`."""
    with pytest.raises(ValueError):
        relaxation_factor(*args)


@pytest.mark.parametrize("args", [
    (-0.5, 0.0, 0.0, 0.1), (0.0, -0.5, 0.0, 0.1), (0.0, 0.0, -0.5, 0.1),
    (0.0, 0.0, 0.0, -0.1), (math.nan, 0.0, 0.0, 0.1), (0.0, math.inf, 0.0, 0.1),
    (True, 0.0, 0.0, 0.1), (10 ** 400, 0.0, 0.0, 0.1), (0.0, 0.0, 0.0, "0.1"),
])
def test_bath_channel_rejects_illegal_arguments(args):
    """A negative rate previously produced D[3,3] = 1.05 -- a non-physical amplification."""
    with pytest.raises(ValueError):
        bath_channel_matrix(*args)


def test_zero_rate_and_zero_dt_stay_legal():
    assert relaxation_factor(0.0, 0.05) == 0.05
    assert relaxation_factor(0.5, 0.0) == 0.0
    np.testing.assert_allclose(bath_channel_matrix(0.0, 0.0, 0.0, 0.0),
                               np.eye(4, dtype=np.complex128), atol=0.0)


def test_pure_dephasing_branch():
    """Gamma_1 = 0 with dephasing > 0: no affine shift, but transverse damping."""
    D = bath_channel_matrix(0.0, 0.0, 0.3, 0.1)
    assert np.all(np.isfinite(D))
    assert D[3, 0] == 0.0 and np.isclose(D[3, 3].real, 1.0)
    assert np.isclose(D[1, 1].real, math.exp(-0.3 * 0.1))


# -- the transfer tensors ----------------------------------------------------------

@pytest.mark.parametrize("t", [0.05, 0.37])
def test_transfer_tensor_matches_the_derivations_closed_form(t):
    """A_k(t) element by element against the three matrices of the derivation."""
    g, w = 0.35, 0.8
    m = _model(coupling=[g, 0.22], omega=[w, 1.3])
    # eps and order chosen so that sub-step g = 1 samples exactly the midpoint t
    corr = _corr(m, T=2 * t, eps=2 * t, order=1)
    A = corr.transfer_for(0)[0]                       # zero rates -> D = I, so this is A_k
    c, s = math.cos(w * t), math.sin(w * t)
    np.testing.assert_allclose(A[0], np.eye(4), atol=1e-14)
    np.testing.assert_allclose(A[1], 2 * g * np.array([[0, 0, 0, 0],
                                                       [0, 0, 0, -s],
                                                       [0, 0, 0, -c],
                                                       [0, s, c, 0]]), atol=1e-14)
    np.testing.assert_allclose(A[2], g * np.array([[0, c, -s, 0],
                                                   [c, 0, 0, 0],
                                                   [-s, 0, 0, 0],
                                                   [0, 0, 0, 0]]), atol=1e-14)


def test_sample_time_is_the_shared_midpoint():
    corr = _corr(_model(), T=0.4, eps=0.1, order=2)
    assert corr.n_sites == 8
    # both sub-steps of physical step n share t_n* = (n - 1/2) eps
    got = [corr.sample_time(g) for g in range(1, 9)]
    np.testing.assert_allclose(got, [0.05, 0.05, 0.15, 0.15, 0.25, 0.25, 0.35, 0.35], atol=1e-14)
    corr1 = _corr(_model(), T=0.4, eps=0.1, order=1)
    np.testing.assert_allclose([corr1.sample_time(g) for g in range(1, 5)],
                               [0.05, 0.15, 0.25, 0.35], atol=1e-14)


def test_grid_signature_distinguishes_grids_that_share_n_sites():
    m = _model()
    a = _corr(m, T=0.4, eps=0.1, order=1)      # n_sites = 4
    b = _corr(m, T=0.2, eps=0.1, order=2)      # n_sites = 4 as well
    c = _corr(m, T=0.8, eps=0.2, order=1)      # n_sites = 4 as well
    assert a.n_sites == b.n_sites == c.n_sites == 4
    assert a.grid_signature != b.grid_signature != c.grid_signature
    assert a.grid_signature != c.grid_signature


def test_strang_placement_of_the_bath_channel():
    """earlier = A D_h, later = D_h A, order 1 = D_h A D_h -- not the other way round."""
    eps = 0.1
    m = _model(pump=[0.05, 0.02], emission=[0.11, 0.07], dephasing=[0.04, 0.09])
    bp = m.bath_params()
    D_h = bath_channel_matrix(bp.pump[0], bp.emission[0], bp.dephasing[0], 0.5 * eps)
    closed = _model()                                    # same B_k, no rates -> bare A

    for order in (1, 2):
        T = _corr(m, T=0.4, eps=eps, order=order).transfer_for(0)
        A = _corr(closed, T=0.4, eps=eps, order=order).transfer_for(0)
        for g in range(1, T.shape[0] + 1):
            if order == 1:
                expected = D_h @ A[g - 1] @ D_h
            elif (g - 1) % 2 == 0:
                expected = A[g - 1] @ D_h
            else:
                expected = D_h @ A[g - 1]
            np.testing.assert_allclose(T[g - 1], expected, atol=1e-14)
            if order == 2 and (g - 1) % 2 == 0:          # the reversed order really differs
                assert not np.allclose(T[g - 1], D_h @ A[g - 1], atol=1e-8)


def test_transfer_carries_no_second_order_coefficients():
    """c_1 / c_2 live on the system side only; both sub-steps share the bare A_k."""
    corr = _corr(_model(), T=0.4, eps=0.1, order=2)      # zero rates -> D = I
    T = corr.transfer_for(0)
    for n in range(4):
        np.testing.assert_allclose(T[2 * n], T[2 * n + 1], atol=1e-14)


def test_boundary_vector_is_the_bloch_vector_with_unit_trace():
    inf = _corr(_model(bath_state="inf"))
    np.testing.assert_allclose(inf.boundary_vector(0), [1, 0, 0, 0], atol=1e-14)
    r = np.array([[0.2, -0.3, 0.5], [0.0, 0.4, -0.4]])
    custom = _corr(_model(bath_state=r))
    np.testing.assert_allclose(custom.boundary_vector(1), [1.0, *r[1]], atol=1e-14)


# -- the whole chain against brute force -------------------------------------------

@pytest.mark.parametrize("order", [1, 2])
def test_closed_chain_matches_a_brute_force_superoperator_product(order):
    m = _model(bath_state=[[0.2, -0.3, 0.5], [0.0, 0.4, -0.4]])
    corr = _corr(m, T=0.4, eps=0.1, order=order)
    rng = random.Random(0)
    for _ in range(20):
        ops = [rng.randrange(3) for _ in range(corr.n_sites)]
        for k in range(2):
            assert abs(corr.correlation(ops, k)
                       - _brute_force_correlation(m, k, 0.1, order, ops)) < 1e-12


@pytest.mark.parametrize("order", [1, 2])
def test_dissipative_chain_matches_a_brute_force_superoperator_product(order):
    """The reference interleaves an independent Lindblad channel, never bath_channel_matrix."""
    m = _model(bath_state=[[0.2, -0.3, 0.5], [0.0, 0.4, -0.4]],
               pump=[0.05, 0.02], emission=[0.11, 0.07], dephasing=[0.04, 0.09])
    corr = _corr(m, T=0.4, eps=0.1, order=order)
    rng = random.Random(1)
    for _ in range(20):
        ops = [rng.randrange(3) for _ in range(corr.n_sites)]
        for k in range(2):
            assert abs(corr.correlation(ops, k)
                       - _brute_force_correlation(m, k, 0.1, order, ops)) < 1e-12


def test_all_identity_chain_is_the_unit_trace_when_closed():
    corr = _corr(_model(bath_state=[[0.2, -0.3, 0.5], [0.0, 0.4, -0.4]]))
    assert abs(corr.correlation([0] * corr.n_sites, 0) - 1.0) < 1e-14


# -- engine / container contracts --------------------------------------------------

def test_engine_rejects_a_wrong_bath_type():
    with pytest.raises(ValueError, match="bath_type"):
        SeparableTDBathCorrelation().compute(GaudinModel(g=1.0, K=2), T=0.4, eps=0.1, order=2)


@pytest.mark.parametrize("order", [0, 3, True, 2.0, None])
def test_engine_rejects_an_illegal_order(order):
    with pytest.raises(ValueError):
        SeparableTDBathCorrelation().compute(_model(), T=0.4, eps=0.1, order=order)


def test_engine_rejects_a_non_integer_time_grid():
    with pytest.raises(ValueError):
        SeparableTDBathCorrelation().compute(_model(), T=0.35, eps=0.1, order=2)


@pytest.mark.parametrize("k", [-1, 2, True, 1.0])
def test_transfer_and_boundary_reject_an_illegal_sub_bath_index(k):
    corr = _corr(_model())
    with pytest.raises((ValueError, IndexError)):
        corr.transfer_for(k)
    with pytest.raises((ValueError, IndexError)):
        corr.boundary_vector(k)


def test_container_arrays_are_read_only_copies():
    corr = _corr(_model())
    assert not corr.bloch.flags.writeable
    assert not corr.rates.flags.writeable


def test_correlation_rejects_an_over_long_or_illegal_operator_sequence():
    corr = _corr(_model(), T=0.2, eps=0.1, order=1)      # n_sites = 2
    with pytest.raises(ValueError, match="only 2 sites"):
        corr.correlation([0, 0, 0])
    with pytest.raises(ValueError, match="phi"):
        corr.correlation([0, 3])


def test_container_rejects_a_malformed_construction():
    with pytest.raises(ValueError, match=r"bloch must have shape"):
        TimeDependentSeparableCorrelation(
            eps=0.1, n_steps=2, order=2, bloch=np.zeros(3), rates=np.zeros(3),
            bath_operator=lambda k, t: _SX)
    with pytest.raises(ValueError, match="order must be"):
        TimeDependentSeparableCorrelation(
            eps=0.1, n_steps=2, order=3, bloch=np.zeros((1, 3)), rates=np.zeros((1, 3)),
            bath_operator=lambda k, t: _SX)
