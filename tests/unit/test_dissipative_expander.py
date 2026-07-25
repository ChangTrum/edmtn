"""Cavity amplitude-damping channel and its Strang placement (Layer 4b).

Two things are guarded here.

* The **channel** ``amplitude_damping_matrix`` is checked against two *independent*
  constructions -- ``scipy.linalg.expm`` of the Lindblad generator, and an explicit Kraus
  sum built from ``V_m`` -- never against itself.  Trace preservation on the truncated
  space and the exact semigroup property are what the Strang half-step placement relies
  on, so both are asserted directly.
* The **placement** is checked by comparing the produced families against hand-written
  matrix products, separately for order 1 and order 2, including that the ``phi = 0``
  identity entry is damped as well.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm

from edmtn.expansion import (
    DissipativeExpander,
    FirstOrderExpander,
    SecondOrderExpander,
    amplitude_damping_matrix,
    cavity_damping_channel,
)

_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def _annihilation(d):
    return np.diag(np.sqrt(np.arange(1, d, dtype=np.float64)), 1).astype(np.complex128)


def _lindblad_generator(d, kappa):
    """``kappa D[a]`` as a row-major superoperator matrix, built independently."""
    a = _annihilation(d)
    n = a.conj().T @ a
    I = np.eye(d, dtype=np.complex128)
    return kappa * (np.kron(a, a.conj()) - 0.5 * (np.kron(n, I) + np.kron(I, n.T)))


def _kraus_sum(d, kappa, dt):
    """``sum_m V_m rho V_m^dag`` as a matrix, from the Kraus operators directly."""
    a = _annihilation(d)
    x = math.exp(-dt * kappa)
    V0 = np.diag(x ** (0.5 * np.arange(d, dtype=np.float64))).astype(np.complex128)
    out = np.zeros((d * d, d * d), dtype=np.complex128)
    apow = np.eye(d, dtype=np.complex128)
    for m in range(d):
        Vm = math.sqrt((1.0 - x) ** m / math.factorial(m)) * (V0 @ apow)
        out += np.kron(Vm, Vm.conj())
        apow = apow @ a
    return out


# -- the channel -------------------------------------------------------------------

@pytest.mark.parametrize("d,kappa,dt", [(3, 0.5, 0.02), (5, 0.37, 0.043), (6, 1.3, 0.11)])
def test_channel_matches_an_independent_matrix_exponential(d, kappa, dt):
    np.testing.assert_allclose(
        amplitude_damping_matrix(d, kappa, dt),
        expm(dt * _lindblad_generator(d, kappa)), atol=1e-12)


@pytest.mark.parametrize("d,kappa,dt", [(4, 0.6, 0.05), (6, 0.2, 0.3)])
def test_channel_matches_an_independent_kraus_sum(d, kappa, dt):
    np.testing.assert_allclose(
        amplitude_damping_matrix(d, kappa, dt), _kraus_sum(d, kappa, dt), atol=1e-13)


@pytest.mark.parametrize("d,kappa,dt", [(3, 0.9, 0.07), (7, 0.15, 0.25)])
def test_channel_is_exactly_trace_preserving_on_the_truncated_space(d, kappa, dt):
    """The environment closing of the EDM is vec(I)^T, so this is load-bearing."""
    vecI = np.eye(d, dtype=np.complex128).reshape(-1)
    np.testing.assert_allclose(vecI @ amplitude_damping_matrix(d, kappa, dt), vecI, atol=1e-13)


@pytest.mark.parametrize("d,kappa,eps", [(4, 0.45, 0.06), (6, 0.8, 0.02)])
def test_channel_is_an_exact_semigroup(d, kappa, eps):
    """Two half-steps compose into one full step, so the per-step dissipative
    increment is exactly ``eps`` -- the property the Strang placement depends on."""
    half = amplitude_damping_matrix(d, kappa, 0.5 * eps)
    np.testing.assert_allclose(half @ half, amplitude_damping_matrix(d, kappa, eps), atol=1e-13)


@pytest.mark.parametrize("d", [2, 5])
def test_zero_rate_or_zero_time_is_the_exact_identity(d):
    eye = np.eye(d * d, dtype=np.complex128)
    assert np.array_equal(amplitude_damping_matrix(d, 0.0, 0.1), eye)   # bit-for-bit
    assert np.array_equal(amplitude_damping_matrix(d, 0.7, 0.0), eye)


def test_channel_is_completely_positive():
    """Choi matrix positive semidefinite -- a wrong sqrt/binomial would break this."""
    d, kappa, dt = 4, 0.6, 0.08
    M = amplitude_damping_matrix(d, kappa, dt)
    # Choi = sum_{ij} |i><j| (x) E(|i><j|); reshape the row-major superoperator
    choi = M.reshape(d, d, d, d).transpose(0, 2, 1, 3).reshape(d * d, d * d)
    assert np.linalg.eigvalsh(0.5 * (choi + choi.conj().T)).min() > -1e-12


@pytest.mark.parametrize("args", [
    (0, 0.5, 0.1), (True, 0.5, 0.1), (2.5, 0.5, 0.1),
    (3, -0.1, 0.1), (3, math.nan, 0.1), (3, 0.5, -0.1), (3, 0.5, math.inf),
    (3, True, 0.1), (3, 0.5, True),
    (3, 10 ** 400, 0.1), (3, 0.5, 10 ** 400),   # real, but overflows float64
    (3, "0.5", 0.1),
])
def test_channel_rejects_illegal_arguments(args):
    with pytest.raises(ValueError):
        amplitude_damping_matrix(*args)


# -- the Strang placement ----------------------------------------------------------

def test_order_two_places_one_half_channel_at_each_end():
    """early site = S_1 M(h), late site = M(h) S_2  ->  Q = M(h) S_2 S_1 M(h)."""
    eps, kappa, d = 0.05, 0.3, 2
    base = SecondOrderExpander()
    wrapped = DissipativeExpander(base, cavity_damping_channel(kappa))
    plain = base.build([_SX], eps)
    damped = wrapped.build([_SX], eps)
    M = amplitude_damping_matrix(d, kappa, 0.5 * eps)      # h = eps/2, NOT eps/order
    np.testing.assert_allclose(damped.families[0], plain.families[0] @ M, atol=1e-14)
    np.testing.assert_allclose(damped.families[1], M @ plain.families[1], atol=1e-14)


def test_order_one_places_a_half_channel_on_each_side_of_the_single_site():
    eps, kappa = 0.05, 0.3
    base = FirstOrderExpander()
    wrapped = DissipativeExpander(base, cavity_damping_channel(kappa))
    plain = base.build([_SX], eps)
    damped = wrapped.build([_SX], eps)
    M = amplitude_damping_matrix(2, kappa, 0.5 * eps)
    assert len(damped.families) == 1
    np.testing.assert_allclose(damped.families[0], M @ plain.families[0] @ M, atol=1e-14)


def test_half_step_is_eps_over_two_at_both_orders():
    """A `eps/order` half-step would silently halve the damping at order 2."""
    eps, kappa = 0.08, 0.9
    M_half = amplitude_damping_matrix(2, kappa, eps / 2.0)
    M_quarter = amplitude_damping_matrix(2, kappa, eps / 4.0)
    fam = DissipativeExpander(SecondOrderExpander(),
                              cavity_damping_channel(kappa)).build([_SX], eps).families
    plain = SecondOrderExpander().build([_SX], eps).families
    np.testing.assert_allclose(fam[1], M_half @ plain[1], atol=1e-14)
    assert not np.allclose(fam[1], M_quarter @ plain[1], atol=1e-10)


def test_the_identity_entry_is_damped_too():
    """phi = 0 carries the dissipation as well -- it is not left as the identity."""
    eps, kappa = 0.05, 0.4
    damped = DissipativeExpander(SecondOrderExpander(),
                                 cavity_damping_channel(kappa)).build([_SX], eps)
    M = amplitude_damping_matrix(2, kappa, 0.5 * eps)
    np.testing.assert_allclose(damped.families[0][0], M, atol=1e-14)
    np.testing.assert_allclose(damped.families[1][0], M, atol=1e-14)
    assert not np.allclose(damped.families[0][0], np.eye(4), atol=1e-8)


def test_zero_kappa_leaves_the_families_bit_for_bit_unchanged():
    eps = 0.05
    base = SecondOrderExpander()
    plain = base.build([_SX], eps)
    damped = DissipativeExpander(base, cavity_damping_channel(0.0)).build([_SX], eps)
    for a, b in zip(plain.families, damped.families):
        assert np.array_equal(a, b)


def test_wrapper_reports_the_base_order_and_metadata():
    for base, order in ((FirstOrderExpander(), 1), (SecondOrderExpander(), 2)):
        w = DissipativeExpander(base, cavity_damping_channel(0.2))
        assert w.order == order
        st = w.build([_SX], 0.05)
        assert st.order == order and st.d == 2 and st.phys_dim == 3


def test_build_at_threads_through_the_wrapper():
    """The evolution engine calls build_at(model, t, eps); the wrapper must inherit it."""
    from edmtn.models import DickeModel

    m = DickeModel(K=1, n_fock=3, coupling=0.4, omega_c=1.1)
    w = DissipativeExpander(SecondOrderExpander(), cavity_damping_channel(0.25))
    st = w.build_at(m, 0.35, 0.05)
    expected = w.build(m.coupling_operators_at(0.35), 0.05)
    for a, b in zip(st.families, expected.families):
        np.testing.assert_allclose(a, b, atol=1e-14)


def test_channel_is_built_once_per_grid():
    calls = []

    def counting(d, dt):
        calls.append((d, dt))
        return amplitude_damping_matrix(d, 0.3, dt)

    w = DissipativeExpander(SecondOrderExpander(), counting)
    for _ in range(5):
        w.build([_SX], 0.05)
    assert len(calls) == 1 and calls[0] == (2, 0.025)


class _BadOrder:
    order = 3


@pytest.mark.parametrize("bad_base", [None, object(), _BadOrder()])
def test_wrapper_rejects_a_base_without_a_usable_order(bad_base):
    with pytest.raises(ValueError):
        DissipativeExpander(bad_base, cavity_damping_channel(0.1))


def test_wrapper_rejects_a_non_callable_or_misshaped_channel():
    with pytest.raises(ValueError):
        DissipativeExpander(SecondOrderExpander(), "not callable")
    bad = DissipativeExpander(SecondOrderExpander(), lambda d, dt: np.eye(3))
    with pytest.raises(ValueError, match="must return"):
        bad.build([_SX], 0.05)
