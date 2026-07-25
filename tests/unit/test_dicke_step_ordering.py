"""One physical step, assembled from the production pieces, against hand-written formulas.

No tensor network is involved: this isolates the **operator ordering** of the Strang
discretisation, which is the part most easily reversed because two orderings coexist in
the code (site tensors are stored newest-first, while the rightmost factor acts first).

The target maps, with ``h = eps/2``, ``t_n^* = (n - 1/2) eps``, and the rightmost factor
acting first::

    order 2:   Q_n = E_h  F_{2,n}  F_{1,n}  E_h
    order 1:   Q_n = E_h  F_{1,n}  E_h
    E_h = M(h) (x) D_k(h)

Checked separately for the system-side factor ``M(h)`` and the bath-side factor
``D_k(h)``, and then jointly on the tensor-product step operator -- the joint check is
what would catch the two sides disagreeing about which sub-step is the earlier one.
"""

from __future__ import annotations

import numpy as np
import pytest

from edmtn.cumulants import SeparableTDBathCorrelation, bath_channel_matrix
from edmtn.expansion import (
    DissipativeExpander,
    FirstOrderExpander,
    SecondOrderExpander,
    amplitude_damping_matrix,
    cavity_damping_channel,
)
from edmtn.models import DickeModel

EPS = 0.1
N_FOCK = 3
RATES = dict(kappa=0.23, pump=0.05, emission=0.11, dephasing=0.04)


def _model(**kw):
    base = dict(K=1, n_fock=N_FOCK, coupling=0.4, omega=0.9, omega_c=1.3)
    base.update(kw)
    return DickeModel(**base)


def _pieces(order):
    """Production families / transfers, plus their undamped counterparts and the channels."""
    damped_model = _model(**RATES)
    closed_model = _model()
    t_mid = 0.5 * EPS                                  # physical step n = 1

    base = SecondOrderExpander() if order == 2 else FirstOrderExpander()
    plain = base.build(closed_model.coupling_operators_at(t_mid), EPS).families
    damped = DissipativeExpander(base, cavity_damping_channel(RATES["kappa"])).build(
        damped_model.coupling_operators_at(t_mid), EPS).families

    corr_c = SeparableTDBathCorrelation().compute(closed_model, T=EPS, eps=EPS, order=order)
    corr_d = SeparableTDBathCorrelation().compute(damped_model, T=EPS, eps=EPS, order=order)
    A = corr_c.transfer_for(0)                          # bare A_k(t_n^*), oldest first
    Ad = corr_d.transfer_for(0)                         # with the Strang half-channels

    M_h = amplitude_damping_matrix(N_FOCK, RATES["kappa"], 0.5 * EPS)
    D_h = bath_channel_matrix(RATES["pump"], RATES["emission"], RATES["dephasing"], 0.5 * EPS)
    return plain, damped, A, Ad, M_h, D_h


def _system_step(families, order):
    """``sum over sub-step families`` as a map on (phi_2, phi_1) -> system matrix."""
    if order == 1:
        return {(p,): families[0][p] for p in range(3)}
    return {(p2, p1): families[1][p2] @ families[0][p1]
            for p2 in range(3) for p1 in range(3)}


def _bath_step(transfer, order):
    if order == 1:
        return {(p,): transfer[0][p] for p in range(3)}
    return {(p2, p1): transfer[1][p2] @ transfer[0][p1]
            for p2 in range(3) for p1 in range(3)}


# -- system side -------------------------------------------------------------------

def test_system_side_order_two_is_Mh_S2_S1_Mh():
    plain, damped, _, _, M_h, _ = _pieces(2)
    got = _system_step(damped, 2)
    for (p2, p1), value in got.items():
        expected = M_h @ plain[1][p2] @ plain[0][p1] @ M_h
        np.testing.assert_allclose(value, expected, atol=1e-13, err_msg=f"phi=({p2},{p1})")


def test_system_side_order_one_is_Mh_S_Mh():
    plain, damped, _, _, M_h, _ = _pieces(1)
    for (p,), value in _system_step(damped, 1).items():
        np.testing.assert_allclose(value, M_h @ plain[0][p] @ M_h, atol=1e-13)


def test_system_side_rejects_the_naive_placement():
    """M_h S2 M_h S1 differs -- otherwise the test above would be vacuous."""
    plain, damped, _, _, M_h, _ = _pieces(2)
    naive = M_h @ plain[1][1] @ M_h @ plain[0][1]
    assert not np.allclose(_system_step(damped, 2)[(1, 1)], naive, atol=1e-9)


# -- bath side ---------------------------------------------------------------------

def test_bath_side_order_two_is_Dh_A_A_Dh():
    _, _, A, Ad, _, D_h = _pieces(2)
    for (p2, p1), value in _bath_step(Ad, 2).items():
        expected = D_h @ A[1][p2] @ A[0][p1] @ D_h
        np.testing.assert_allclose(value, expected, atol=1e-13, err_msg=f"phi=({p2},{p1})")


def test_bath_side_order_one_is_Dh_A_Dh():
    _, _, A, Ad, _, D_h = _pieces(1)
    for (p,), value in _bath_step(Ad, 1).items():
        np.testing.assert_allclose(value, D_h @ A[0][p] @ D_h, atol=1e-13)


def test_bath_side_rejects_the_reversed_half_channels():
    """A D_h on the later site (instead of D_h A) must be distinguishable."""
    _, _, A, Ad, _, D_h = _pieces(2)
    reversed_ = A[1][1] @ D_h @ D_h @ A[0][1]
    assert not np.allclose(_bath_step(Ad, 2)[(1, 1)], reversed_, atol=1e-9)


def test_bath_side_uses_the_same_bare_transfer_in_both_sub_steps():
    """No c_1 / c_2 on the bath side: A is identical in the two algebraic sub-steps."""
    _, _, A, _, _, _ = _pieces(2)
    np.testing.assert_allclose(A[0], A[1], atol=1e-14)


# -- the two sides together --------------------------------------------------------

@pytest.mark.parametrize("order", [1, 2])
def test_tensor_product_step_matches_E_h_F_F_E_h(order):
    """The joint check: both sides must agree on which sub-step is the earlier one."""
    plain, damped, A, Ad, M_h, D_h = _pieces(order)
    sys_p, bath_p = _system_step(damped, order), _bath_step(Ad, order)
    sys_r, bath_r = _system_step(plain, order), _bath_step(A, order)

    dim = (N_FOCK ** 2) * 4
    produced = np.zeros((dim, dim), dtype=np.complex128)
    for key in sys_p:
        produced += np.kron(sys_p[key], bath_p[key])

    bare = np.zeros((dim, dim), dtype=np.complex128)
    for key in sys_r:
        bare += np.kron(sys_r[key], bath_r[key])

    E_h = np.kron(M_h, D_h)
    np.testing.assert_allclose(produced, E_h @ bare @ E_h, atol=1e-12)


def test_the_joint_step_is_not_reproduced_by_swapping_the_two_sub_steps():
    """Guards the joint check: exchanging early/late on ONE side must break it."""
    plain, damped, A, Ad, M_h, D_h = _pieces(2)
    dim = (N_FOCK ** 2) * 4
    produced = sum(np.kron(_system_step(damped, 2)[k], _bath_step(Ad, 2)[k])
                   for k in _system_step(damped, 2))
    swapped_bath = {(p2, p1): Ad[0][p2] @ Ad[1][p1] for p2 in range(3) for p1 in range(3)}
    wrong = sum(np.kron(_system_step(damped, 2)[k], swapped_bath[k]) for k in swapped_bath)
    assert produced.shape == (dim, dim)
    assert not np.allclose(produced, wrong, atol=1e-9)


@pytest.mark.parametrize("order", [1, 2])
def test_closed_model_is_unaffected_by_the_channel_factors(order):
    """All rates zero: the Strang factors are exactly the identity, no placement to get wrong."""
    closed = _model()
    t_mid = 0.5 * EPS
    base = SecondOrderExpander() if order == 2 else FirstOrderExpander()
    plain = base.build(closed.coupling_operators_at(t_mid), EPS).families
    damped = DissipativeExpander(base, cavity_damping_channel(0.0)).build(
        closed.coupling_operators_at(t_mid), EPS).families
    for a, b in zip(plain, damped):
        assert np.array_equal(a, b)
