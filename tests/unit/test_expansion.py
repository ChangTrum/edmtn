"""Unit tests for Layer 4b (time-step expansion / system superoperators)."""

import numpy as np
import pytest
from scipy.linalg import expm

from edmtn.expansion import (
    FirstOrderExpander,
    SecondOrderExpander,
    anticommutator_superoperator,
    apply_superoperator,
    commutator_superoperator,
    first_order_superoperators,
)
from edmtn.models import SpinBosonModel

SX = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
SY = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
SZ = np.array([[0.5, 0], [0, -0.5]], dtype=complex)


def random_rho(seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


# --------------------------------------------------------------------------
# superoperator building blocks
# --------------------------------------------------------------------------

def test_commutator_superoperator_action():
    rho = random_rho()
    got = apply_superoperator(commutator_superoperator(SZ), rho)
    expected = -1j * (SZ @ rho - rho @ SZ)
    np.testing.assert_allclose(got, expected, atol=1e-14)


def test_anticommutator_superoperator_action():
    rho = random_rho()
    got = apply_superoperator(anticommutator_superoperator(SZ), rho)
    expected = 0.5 * (SZ @ rho + rho @ SZ)
    np.testing.assert_allclose(got, expected, atol=1e-14)


def test_commutator_with_complex_operator():
    # S_y has imaginary entries -> exercises the transpose (not conjugate)
    rho = random_rho(3)
    got = apply_superoperator(commutator_superoperator(SY), rho)
    np.testing.assert_allclose(got, -1j * (SY @ rho - rho @ SY), atol=1e-14)


# --------------------------------------------------------------------------
# first-order superoperators
# --------------------------------------------------------------------------

def test_phys_dim_single_channel():
    S = first_order_superoperators([SZ], eps=0.1)
    assert S.shape == (3, 4, 4)


def test_phys_dim_multi_channel():
    S = first_order_superoperators([SX, SY, SZ], eps=0.1)
    assert S.shape == (7, 4, 4)  # 2*3 + 1


def test_identity_superoperator():
    S = first_order_superoperators([SZ], eps=0.1)
    np.testing.assert_allclose(S[0], np.eye(4))
    rho = random_rho()
    np.testing.assert_allclose(apply_superoperator(S[0], rho), rho, atol=1e-14)


def test_index_convention():
    # phi=1 -> eps * S^+ (anticommutator); phi=2 -> eps * S^- (commutator)
    eps = 0.1
    S = first_order_superoperators([SZ], eps=eps)
    np.testing.assert_allclose(S[1], eps * anticommutator_superoperator(SZ), atol=1e-14)
    np.testing.assert_allclose(S[2], eps * commutator_superoperator(SZ), atol=1e-14)


def test_commutator_traceless_action():
    eps = 0.1
    S = first_order_superoperators([SZ], eps=eps)
    rho = random_rho()
    out = apply_superoperator(S[2], rho)  # -i eps [Sz, rho]
    assert np.isclose(np.trace(out), 0.0, atol=1e-14)


def test_anticommutator_trace_action():
    eps = 0.1
    S = first_order_superoperators([SZ], eps=eps)
    rho = random_rho()
    out = apply_superoperator(S[1], rho)  # eps/2 {Sz, rho}
    assert np.isclose(np.trace(out), eps * np.trace(SZ @ rho), atol=1e-14)


def test_superoperators_preserve_hermiticity():
    eps = 0.1
    S = first_order_superoperators([SZ], eps=eps)
    rho = random_rho()
    for phi in range(3):
        out = apply_superoperator(S[phi], rho)
        np.testing.assert_allclose(out, out.conj().T, atol=1e-14)


def test_empty_coupling_raises():
    with pytest.raises(ValueError):
        first_order_superoperators([], eps=0.1)


# --------------------------------------------------------------------------
# expanders
# --------------------------------------------------------------------------

def test_first_order_expander():
    exp = FirstOrderExpander()
    step = exp.build([SZ], eps=0.1)
    assert step.order == 1
    assert step.phys_dim == 3 and step.d == 2
    assert len(step.families) == 1


def test_second_order_expander_structure():
    exp = SecondOrderExpander()
    step = exp.build([SZ], eps=0.1)
    assert step.order == 2
    assert len(step.families) == 2
    S1, S2 = step.families
    # identity preserved in both families
    np.testing.assert_allclose(S1[0], np.eye(4))
    np.testing.assert_allclose(S2[0], np.eye(4))


def test_second_order_coefficients():
    eps = 0.1
    base = first_order_superoperators([SZ], eps=eps)
    S1, S2 = SecondOrderExpander().build([SZ], eps=eps).families
    np.testing.assert_allclose(S1[1:], (1 - 1j) / 2 * base[1:], atol=1e-14)
    np.testing.assert_allclose(S2[1:], (1 + 1j) / 2 * base[1:], atol=1e-14)


def test_second_order_split_matches_expm():
    # [I + (1+i)/2 eps L][I + (1-i)/2 eps L] reproduces e^{eps L} to O(eps^3),
    # validated on a toy single-mode Liouvillian L = -i[H, .], H = Sz (x) B.
    rng = np.random.default_rng(1)
    B = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    B = B + B.conj().T
    H = np.kron(SZ, B)  # 4x4 total Hamiltonian
    L = commutator_superoperator(H)  # -i[H, .], 16x16 Liouvillian

    def split(eps):
        I = np.eye(L.shape[0])
        return (I + (1 + 1j) / 2 * eps * L) @ (I + (1 - 1j) / 2 * eps * L)

    errs = []
    for eps in (0.1, 0.05, 0.025):
        errs.append(np.linalg.norm(split(eps) - expm(eps * L)))
    errs = np.array(errs)
    # halving eps should cut the error by ~2^3 = 8 (third-order accuracy)
    ratios = errs[:-1] / errs[1:]
    assert np.all(ratios > 6.0)


def test_first_order_split_is_lower_order():
    # first order [I + eps L] reproduces e^{eps L} only to O(eps^2)
    rng = np.random.default_rng(2)
    L = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
    errs = []
    for eps in (0.1, 0.05):
        errs.append(np.linalg.norm((np.eye(6) + eps * L) - expm(eps * L)))
    ratio = errs[0] / errs[1]
    assert 3.0 < ratio < 5.0  # ~2^2 = 4


# --------------------------------------------------------------------------
# integration with the model (interaction picture)
# --------------------------------------------------------------------------

def test_build_at_uses_interaction_picture():
    model = SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0)
    eps = 0.1
    t = 0.7
    step = SecondOrderExpander().build_at(model, t, eps)
    Sz_t = model.coupling_operators_at(t)[0]
    expected_base = first_order_superoperators([Sz_t], eps)
    np.testing.assert_allclose(step.families[1][1:], (1 + 1j) / 2 * expected_base[1:], atol=1e-14)


def test_phys_dim_matches_kernel_dphys():
    # single-channel spin-boson expander must match the Gaussian kernel d_phys = 3
    step = FirstOrderExpander().build_at(SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0), 0.0, 0.1)
    assert step.phys_dim == 3
