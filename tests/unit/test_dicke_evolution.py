"""Algebraic anchor: the uncompressed Dicke EDM against an independent full-Hilbert reference.

This is the test that decides whether the tensor-network contraction is right.  The
reference is built **from scratch** in this module -- a dense Liouvillian on the full
cavity-and-spins space, propagated with the same discretisation -- and imports **nothing**
from the pipeline it checks: not ``amplitude_damping_matrix``, not the transfer-tensor
builder, not ``DissipativeExpander``.  It reads only the model's *physical* inputs
(dimensions, couplings, frequencies, rates, initial states) and rebuilds ``B_k(t)``,
the dissipators and the step map itself, so a shared sign or ordering error cannot pass
on both sides at once.

Scope, stated plainly: this checks the **tensor algebra**, not physical accuracy.  Both
sides use the identical time discretisation, so agreement says the contraction reproduces
that discretisation exactly -- it says nothing about how close the discretisation is to
the true dynamics.  That is measured separately, against a continuous-time reference, in
``test_dicke_convergence_order.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from edmtn.driver import solve
from edmtn.models import DickeModel

_I2 = np.eye(2, dtype=np.complex128)
_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_SY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_SZ = np.diag([1.0, -1.0]).astype(np.complex128)
_SP = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
_SM = _SP.conj().T
_C1, _C2 = (1.0 - 1.0j) / 2.0, (1.0 + 1.0j) / 2.0

#: the contract the pipeline must meet (the measured agreement is ~3e-15)
ANCHOR_TOL = 1e-10


def _kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def reference_reduced_state(model, T, eps, order, sample="mid"):
    """Dense full-Hilbert propagation of the same discretisation, built independently.

    ``sample`` selects the freezing point of ``H_I``; ``"mid"`` is the pipeline's rule and
    the others exist only so the tests can show the anchor is sensitive to it.
    """
    d, K = model.n_fock, model.K
    bp = model.bath_params()
    dim = d * 2 ** K
    Id = np.eye(dim, dtype=np.complex128)

    a1 = np.diag(np.sqrt(np.arange(1, d, dtype=np.float64)), 1).astype(np.complex128)
    cav = lambda o: _kron_all([o] + [_I2] * K)                              # noqa: E731
    spin = lambda k, o: _kron_all(                                          # noqa: E731
        [np.eye(d, dtype=np.complex128)] + [o if j == k else _I2 for j in range(K)])
    a = cav(a1)

    def dissipator(L):
        M = L.conj().T @ L
        return np.kron(L, L.conj()) - 0.5 * (np.kron(M, Id) + np.kron(Id, M.T))

    LD = model.kappa * dissipator(a)
    for k in range(K):
        LD += (bp.pump[k] * dissipator(spin(k, _SP))
               + bp.emission[k] * dissipator(spin(k, _SM))
               + 0.5 * bp.dephasing[k] * dissipator(spin(k, _SZ)))

    def commutator_superop(t):
        S = a * np.exp(-1j * model.omega_c * t) + a.conj().T * np.exp(1j * model.omega_c * t)
        # B_k rebuilt from the couplings, NOT read from model.bath_operator_at
        B = sum(bp.couplings[k] * (np.cos(bp.omegas[k] * t) * spin(k, _SX)
                                   - np.sin(bp.omegas[k] * t) * spin(k, _SY))
                for k in range(K))
        H = S @ B
        return -1j * (np.kron(H, Id) - np.kron(Id, H.T))

    rho0 = _kron_all([model.initial_system_state()]
                     + [0.5 * (_I2 + r[0] * _SX + r[1] * _SY + r[2] * _SZ)
                        for r in model.bath_bloch_vectors()])
    v = rho0.reshape(-1)
    E_h = expm(0.5 * eps * LD)                       # Strang half-step, both orders
    Idl = np.eye(dim * dim, dtype=np.complex128)
    offset = {"mid": 0.5, "right": 1.0, "left": 0.0}[sample]

    for n in range(1, int(round(T / eps)) + 1):
        A = commutator_superop((n - offset) * eps)
        v = E_h @ v
        if order == 2:
            v = (Idl + _C2 * eps * A) @ ((Idl + _C1 * eps * A) @ v)
        else:
            v = (Idl + eps * A) @ v
        v = E_h @ v

    full = v.reshape(dim, dim).reshape(d, 2 ** K, d, 2 ** K)
    return np.einsum("imjm->ij", full)               # partial trace over the spins


def _edm_reduced_state(model, T, eps, order):
    """Lossless EDM: cutoff 0 with no bond cap discards nothing."""
    res = solve(model, T=T, eps=eps, expansion_order=order, cutoff=0.0, max_bond=None)
    return np.asarray(res.final_density_matrix)


ASYMMETRIC = dict(
    K=3, n_fock=4,
    coupling=[0.35, -0.22, 0.3],            # distinct magnitudes AND a sign
    omega=[0.8, 1.3, 0.55],                 # distinct frequencies
    omega_c=1.1,
    cavity_state="coherent", cavity_params={"alpha": 0.3 + 0.2j},   # complex alpha
    bath_state=[[0.2, -0.3, 0.5], [0.0, 0.4, -0.4], [0.1, 0.1, 0.9]],  # off-axis Bloch
)
RATES = dict(kappa=0.23, pump=[0.05, 0.02, 0.03],
             emission=[0.11, 0.07, 0.04], dephasing=[0.04, 0.09, 0.02])


# -- the matrix --------------------------------------------------------------------

@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("rates", [False, True], ids=["closed", "dissipative"])
@pytest.mark.parametrize("bath_state", ["inf", "thermal", "ground"])
@pytest.mark.parametrize("cavity_state", ["vacuum", "coherent"])
def test_uncompressed_edm_matches_the_full_hilbert_reference(
        order, rates, bath_state, cavity_state):
    kw = dict(K=3, n_fock=4, coupling=[0.35, -0.22, 0.3], omega=[0.8, 1.3, 0.55],
              omega_c=1.1, bath_state=bath_state, cavity_state=cavity_state)
    if bath_state == "thermal":
        kw["bath_state_params"] = {"beta": 1.7}
    if cavity_state == "coherent":
        kw["cavity_params"] = {"alpha": 0.3 + 0.2j}
    if rates:
        kw.update(RATES)
    model = DickeModel(**kw)
    got = _edm_reduced_state(model, T=0.4, eps=0.1, order=order)
    ref = reference_reduced_state(model, T=0.4, eps=0.1, order=order)
    assert np.max(np.abs(got - ref)) < ANCHOR_TOL


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("T", [0.2, 0.5])
def test_fully_asymmetric_anchor_at_several_times(order, T):
    """Everything distinct at once, so no symmetry can hide a transposed or reversed factor."""
    model = DickeModel(**ASYMMETRIC, **RATES)
    got = _edm_reduced_state(model, T=T, eps=0.1, order=order)
    ref = reference_reduced_state(model, T=T, eps=0.1, order=order)
    assert np.max(np.abs(got - ref)) < ANCHOR_TOL


def test_a_single_step_already_agrees():
    """Smallest possible grid: isolates the step map from any accumulation effect."""
    model = DickeModel(**ASYMMETRIC, **RATES)
    for order in (1, 2):
        got = _edm_reduced_state(model, T=0.1, eps=0.1, order=order)
        ref = reference_reduced_state(model, T=0.1, eps=0.1, order=order)
        assert np.max(np.abs(got - ref)) < ANCHOR_TOL


def test_sub_baths_folds_the_first_L_spins_only():
    """The partial fold must equal the reference for a model holding only those spins."""
    model = DickeModel(**ASYMMETRIC, **RATES)
    res = solve(model, T=0.3, eps=0.1, expansion_order=2, cutoff=0.0, max_bond=None,
                sub_baths=2)
    two = DickeModel(K=2, n_fock=4, coupling=ASYMMETRIC["coupling"][:2],
                     omega=ASYMMETRIC["omega"][:2], omega_c=1.1,
                     cavity_state="coherent", cavity_params={"alpha": 0.3 + 0.2j},
                     bath_state=ASYMMETRIC["bath_state"][:2],
                     kappa=RATES["kappa"], pump=RATES["pump"][:2],
                     emission=RATES["emission"][:2], dephasing=RATES["dephasing"][:2])
    ref = reference_reduced_state(two, T=0.3, eps=0.1, order=2)
    assert np.max(np.abs(np.asarray(res.final_density_matrix) - ref)) < ANCHOR_TOL


# -- the anchor must be able to fail -----------------------------------------------

@pytest.mark.parametrize("sample", ["right", "left"])
def test_the_anchor_is_sensitive_to_the_sampling_point(sample):
    """A reference frozen at an endpoint instead of the midpoint must NOT match.

    Without this, the anchor could pass while both sides quietly agreed on the wrong rule.
    """
    model = DickeModel(**ASYMMETRIC, **RATES)
    got = _edm_reduced_state(model, T=0.4, eps=0.1, order=2)
    wrong = reference_reduced_state(model, T=0.4, eps=0.1, order=2, sample=sample)
    assert np.max(np.abs(got - wrong)) > 1e-6


def test_the_anchor_is_sensitive_to_the_physical_parameters():
    """Perturbing one coupling must break the match, so the reference is not degenerate."""
    model = DickeModel(**ASYMMETRIC, **RATES)
    got = _edm_reduced_state(model, T=0.4, eps=0.1, order=2)
    perturbed = dict(ASYMMETRIC)
    perturbed["coupling"] = [0.35, -0.22, 0.31]          # third spin only
    ref = reference_reduced_state(DickeModel(**perturbed, **RATES), T=0.4, eps=0.1, order=2)
    assert np.max(np.abs(got - ref)) > 1e-8


def test_the_anchor_is_sensitive_to_the_dissipation():
    model = DickeModel(**ASYMMETRIC, **RATES)
    got = _edm_reduced_state(model, T=0.4, eps=0.1, order=2)
    closed = reference_reduced_state(DickeModel(**ASYMMETRIC), T=0.4, eps=0.1, order=2)
    assert np.max(np.abs(got - closed)) > 1e-4


def test_reference_conserves_the_trace():
    """Sanity on the reference itself, independently of the pipeline."""
    ref = reference_reduced_state(DickeModel(**ASYMMETRIC, **RATES), T=0.4, eps=0.1, order=2)
    assert abs(complex(np.trace(ref)).real - 1.0) < 1e-9
