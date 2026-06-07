"""Unit tests for Layer 2 (separable-bath correlation engine, Gaudin).

The decisive correctness check is that the per-sub-bath transfer MPS (Eq. F1, in
the superoperator-index convention) reproduces the exact time-ordered bath
correlation ``Tr[B^{phi_T}...B^{phi_1}(Omega_k)]`` for arbitrary superoperator
index sequences -- independently of any downstream picking convention.
"""

import itertools

import numpy as np
import pytest

from edmtn.cumulants import SeparableBathCorrelation, SeparableCorrelation
from edmtn.cumulants.base import CumulantEngine
from edmtn.models import GaudinModel, SpinBosonModel

# spin-1/2 operators
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
J = [X / 2, Y / 2, Z / 2]


def _superop(phi, X, gk):
    """Bath superoperator ``B^phi`` acting on operator ``X`` (0=I, 2a-1=B^-, 2a=B^+)."""
    if phi == 0:
        return X
    alpha = (phi + 1) // 2 - 1  # phi 1,2 -> 0 ; 3,4 -> 1 ; 5,6 -> 2
    B = gk * J[alpha]
    if phi % 2 == 1:  # B^- commutator
        return -1j * (B @ X - X @ B)
    return 0.5 * (B @ X + X @ B)  # B^+ mean field


def reference_correlation(ops, gk):
    """Brute-force Tr[B^{phi_T} . ... . B^{phi_1}(Omega_k)], Omega_k = I/2.

    ``ops`` is the time-ordered index list ``[phi_1, ..., phi_T]`` (oldest first).
    """
    rho = 0.5 * I2
    for phi in ops:  # apply earliest (phi_1) first
        rho = _superop(phi, rho, gk)
    return np.trace(rho)


@pytest.fixture
def model():
    return GaudinModel(g=1.0, K=49)


@pytest.fixture
def engine():
    return SeparableBathCorrelation()


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------

def test_engine_bath_type_and_subclass(engine):
    assert engine.bath_type == "separable"
    assert isinstance(engine, CumulantEngine)


def test_rejects_non_separable_model(engine):
    with pytest.raises(ValueError):
        engine.compute(SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0), T=1.0, eps=0.1)


def test_output_shape_and_metadata(model, engine):
    corr = engine.compute(model, T=3.0, eps=0.03)
    assert isinstance(corr, SeparableCorrelation)
    assert corr.K == model.K == 49
    assert corr.d_phys == 7 and corr.bond_dim == 4   # 2*3+1 channels, D_a for spin-1/2
    assert corr.transfer.shape == (49, 7, 4, 4)
    assert corr.n_steps == 100                       # 3.0 / 0.03
    np.testing.assert_allclose(corr.couplings, model.couplings)


def test_transfer_independent_of_time_grid(model, engine):
    # the MPS is uniform in time: transfer tensors must not depend on T/eps
    a = engine.compute(model, T=3.0, eps=0.03).transfer
    b = engine.compute(model, T=10.0, eps=0.1).transfer
    np.testing.assert_allclose(a, b)


# --------------------------------------------------------------------------
# Eq. F1 transfer-tensor values (superoperator convention)
# --------------------------------------------------------------------------

def test_identity_channel_is_identity_matrix(model, engine):
    # B^0 = I  =>  A[0][a, a'] = (1/2) Tr[sigma_a sigma_a'] = delta_{a, a'}
    corr = engine.compute(model, T=1.0, eps=1.0)
    for k in (0, 10, 48):
        np.testing.assert_allclose(corr.transfer_for(k)[0], np.eye(4), atol=1e-12)


def test_single_step_correlations(model, engine):
    corr = engine.compute(model, T=1.0, eps=1.0)
    # T = 1: C = Tr[B^phi(I/2)].  Identity -> 1; B^- (traceless commutator) -> 0;
    # B^+ -> Tr[g J_a]/... = 0 (traceless).
    assert corr.correlation([0], k=5) == pytest.approx(1.0)
    for phi in range(1, 7):
        assert abs(corr.correlation([phi], k=5)) < 1e-12


def test_transfer_matches_explicit_superoperator(model, engine):
    # spot-check A[phi] against the brute-force superoperator matrix in sigma basis
    corr = engine.compute(model, T=1.0, eps=1.0)
    sigma = [I2, X, Y, Z]
    for k in (3, 20):
        gk = model.couplings[k]
        A = corr.transfer_for(k)
        for phi in range(7):
            for a in range(4):
                for ap in range(4):
                    ref = 0.5 * np.trace(sigma[a] @ _superop(phi, sigma[ap], gk))
                    assert A[phi, a, ap] == pytest.approx(ref, abs=1e-12)


# --------------------------------------------------------------------------
# decisive check: MPS reproduces the exact multi-time correlation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("k", [0, 7, 24, 48])
def test_mps_matches_bruteforce_all_sequences_T2(model, engine, k):
    corr = engine.compute(model, T=3.0, eps=0.03)
    gk = model.couplings[k]
    for ops in itertools.product(range(7), repeat=2):  # 49 sequences
        got = corr.correlation(ops, k=k)
        ref = reference_correlation(ops, gk)
        assert got == pytest.approx(ref, abs=1e-12)


def test_mps_matches_bruteforce_random_longer(model, engine):
    corr = engine.compute(model, T=8.0, eps=0.04)
    rng = np.random.default_rng(0)
    for _ in range(300):
        T = int(rng.integers(1, 7))
        ops = list(rng.integers(0, 7, size=T))
        k = int(rng.integers(0, model.K))
        got = corr.correlation(ops, k=k)
        ref = reference_correlation(ops, model.couplings[k])
        assert got == pytest.approx(ref, abs=1e-12)


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------

def test_rejects_finite_temperature(engine):
    """A separable model at finite temperature is not supported."""

    class _FiniteTempGaudin(GaudinModel):
        def bath_params(self):
            p = super().bath_params()
            from edmtn.models.gaudin import GaudinBathParams

            return GaudinBathParams(g=p.g, K=p.K, couplings=p.couplings, temperature=1.0)

    with pytest.raises(NotImplementedError):
        engine.compute(_FiniteTempGaudin(g=1.0, K=4), T=1.0, eps=0.1)
