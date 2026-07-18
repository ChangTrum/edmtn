"""Unit tests for Layer 5 (separable-bath evolution engine, Gaudin).

The decisive correctness check: the *uncompressed* separable EDM must reproduce
the exact Trotterised reduced dynamics of the full central-spin + K-bath-spin
system (same expansion order), to machine precision.  Then compression with a
zero cutoff must preserve it, and a hard cutoff must keep the trace.
"""

import numpy as np
import pytest

from edmtn.evolution import EDMMPS, SeparableBathEvolution, SeparableEvolutionResult
from edmtn.expansion import FirstOrderExpander, SecondOrderExpander
from edmtn.kernels import SeparableKernelEngine
from edmtn.models import GaudinModel

# spin-1/2 operators
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
S_ALPHA = [X / 2, Y / 2, Z / 2]  # spin-1/2 system & bath operators


# --------------------------------------------------------------------------
# brute-force reference: exact Trotterised full-system reduced dynamics
# --------------------------------------------------------------------------

def exact_separable_rho(model, eps, n_steps, order):
    """Reduced ``rho(T)`` from the full (2^{K+1}) Trotter evolution.

    H = sum_k g_k sum_alpha S_alpha (system) (x) J_{k,alpha} (bath spin k); the
    per-step Liouville propagator matches the EDM expansion order, so the
    uncompressed EDM must reproduce this exactly.
    """
    g = model.couplings
    K = model.K
    D = 2 ** (K + 1)
    Id = np.eye(D, dtype=complex)

    # H = sum_k g_k sum_alpha (S_alpha on factor 0)(J_alpha on factor k+1)
    H = np.zeros((D, D), dtype=complex)
    for k in range(K):
        for alpha in range(3):
            ops = [I2] * (K + 1)
            ops[0] = S_ALPHA[alpha]      # system
            ops[k + 1] = S_ALPHA[alpha]  # bath spin k
            term = ops[0]
            for o in ops[1:]:
                term = np.kron(term, o)
            H += g[k] * term

    # Liouvillian  H^- chi = -i [H, chi]  on row-major vec(chi)
    Hm = -1j * (np.kron(H, Id) - np.kron(Id, H.T))
    Iv = np.eye(D * D, dtype=complex)
    if order == 1:
        M = Iv + eps * Hm
    else:
        c1, c2 = (1 - 1j) / 2, (1 + 1j) / 2
        M = (Iv + c2 * eps * Hm) @ (Iv + c1 * eps * Hm)

    # chi(0) = rho_sys(0) (x) (I/2)^{(x)K}
    chi = model.initial_system_state().astype(complex)
    for _ in range(K):
        chi = np.kron(chi, I2 / 2)
    vec = chi.reshape(-1)
    for _ in range(n_steps):
        vec = M @ vec
    chi_T = vec.reshape(D, D)

    # partial trace over the K bath spins (keep factor 0 = system)
    chi4 = chi_T.reshape(2, 2 ** K, 2, 2 ** K)
    return np.einsum("ibkb->ik", chi4)


def run_edm(model, eps, n_steps, order, **kw):
    expander = SecondOrderExpander() if order == 2 else FirstOrderExpander()
    engine = SeparableKernelEngine.from_model(model, T=eps * n_steps, eps=eps)
    evo = SeparableBathEvolution(expander=expander)
    return evo.run(model, engine, eps, n_steps, record_rho=True, **kw)


# --------------------------------------------------------------------------
# physics: uncompressed EDM == exact Trotter dynamics
# --------------------------------------------------------------------------

@pytest.mark.parametrize("K", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2])
def test_uncompressed_matches_exact_trotter(K, order):
    model = GaudinModel(g=0.7, K=K)
    eps, n_steps = 0.1, 3
    res = run_edm(model, eps, n_steps, order, compress=False)
    rho = res.density_matrices[-1]              # rho_{L=K}(T)
    ref = exact_separable_rho(model, eps, n_steps, order)
    np.testing.assert_allclose(rho, ref, atol=1e-10)


@pytest.mark.parametrize("order", [1, 2])
def test_intermediate_subbath_matches_partial_bath(order):
    # rho recorded after L sub-baths must equal the exact dynamics of the
    # central spin coupled to ONLY the first L bath spins.
    model = GaudinModel(g=0.6, K=3)
    eps, n_steps = 0.1, 2
    res = run_edm(model, eps, n_steps, order, compress=False, record_every=1)
    for idx, L in enumerate(res.recorded_L):
        partial = GaudinModel(g=0.6, K=3)
        # exact reference using only the first L couplings
        sub = _SubModel(model, L)
        ref = exact_separable_rho(sub, eps, n_steps, order)
        np.testing.assert_allclose(res.density_matrices[idx], ref, atol=1e-10)


class _SubModel:
    """View of a Gaudin model restricted to its first ``L`` sub-baths in STORED order
    (strongest-first only because this test uses the sorted default `linear` profile)."""

    def __init__(self, model, L):
        self._couplings = model.couplings[:L]
        self.K = L

    @property
    def couplings(self):
        return self._couplings

    def initial_system_state(self):
        return np.diag([1.0, 0.0]).astype(complex)


# --------------------------------------------------------------------------
# compression
# --------------------------------------------------------------------------

def test_zero_cutoff_compression_preserves_state():
    # P1-13: compress=False now GENUINELY skips compression (exponential bonds); compress=True,
    # cutoff=0 does an exact canonicalise + full-SVD recompression. The two exact
    # representations must agree on the reduced state and the recorded fold axis (their bond
    # dimensions may differ -- different-rank exact representations).
    model = GaudinModel(g=0.7, K=2)
    eps, n_steps = 0.1, 3
    raw = run_edm(model, eps, n_steps, order=2, compress=False)
    comp = run_edm(model, eps, n_steps, order=2, compress=True, cutoff=0.0)
    np.testing.assert_allclose(comp.density_matrices[-1], raw.density_matrices[-1], atol=1e-9)
    assert comp.recorded_L == raw.recorded_L
    assert comp.n_sub_baths == raw.n_sub_baths


def test_trace_preserved_under_hard_cutoff():
    model = GaudinModel(g=1.0, K=10)
    res = run_edm(model, eps=0.05, n_steps=8, order=2, compress=True,
                  cutoff=1e-6, max_bond=64)
    rho = res.density_matrices[-1]
    assert abs(np.trace(rho) - 1.0) < 1e-3
    # Hermitian, physical
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-6)


def test_polarization_decays_from_half():
    # <S_z(T)> = Tr[S_z rho]; starts at +1/2, the spin bath depolarises it.
    model = GaudinModel(g=1.0, K=20)
    res = run_edm(model, eps=0.05, n_steps=40, order=2, compress=True,
                  cutoff=1e-6, max_bond=120)
    rho = res.density_matrices[-1]
    sz = float(np.trace(Z / 2 @ rho).real)
    assert -0.05 < sz < 0.5   # decayed below initial 0.5, not blown up


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------

def test_result_structure_and_recording():
    model = GaudinModel(g=0.8, K=6)
    res = run_edm(model, eps=0.1, n_steps=3, order=1, compress=True,
                  cutoff=1e-8, record_every=2)
    assert isinstance(res, SeparableEvolutionResult)
    assert res.n_sub_baths == 6
    assert res.recorded_L[-1] == 6              # last always recorded
    assert res.recorded_L == [2, 4, 6]
    assert len(res.bond_dims) == len(res.recorded_L)
    assert len(res.density_matrices) == len(res.recorded_L)
    assert isinstance(res.mps, EDMMPS)


def test_bond_dim_grows_with_sub_baths():
    model = GaudinModel(g=1.0, K=12)
    res = run_edm(model, eps=0.05, n_steps=10, order=2, compress=True,
                  cutoff=1e-6, max_bond=200, record_every=1)
    # bond dimension is non-trivial and respects the cap
    assert max(res.bond_dims) > 4
    assert max(res.bond_dims) <= 200
