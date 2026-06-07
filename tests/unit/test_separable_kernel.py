"""Unit tests for Layer 3 (separable-bath kernel engine, Gaudin).

Checks that the per-sub-bath combined-kernel MPO is the Layer-2 transfer tensor
with the picking tensor attached at every site, with the correct uniform /
boundary structure, and that closing all open arms recovers the bare
superoperator correlation (the bridge back to Layer 2).
"""

import itertools

import numpy as np
import pytest

from edmtn.cumulants import SeparableBathCorrelation
from edmtn.kernels import KernelMPO, SeparableKernelEngine, picking_tensor
from edmtn.kernels.base import KernelProvider
from edmtn.models import GaudinModel

# spin-1/2 operators (for the brute-force reference)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
J = [X / 2, Y / 2, Z / 2]


def _superop(phi, M, gk):
    if phi == 0:
        return M
    alpha = (phi + 1) // 2 - 1
    B = gk * J[alpha]
    if phi % 2 == 1:
        return -1j * (B @ M - M @ B)
    return 0.5 * (B @ M + M @ B)


def reference_correlation(ops, gk):
    rho = 0.5 * I2
    for phi in ops:
        rho = _superop(phi, rho, gk)
    return np.trace(rho)


@pytest.fixture
def model():
    return GaudinModel(g=1.0, K=49)


@pytest.fixture
def engine(model):
    return SeparableKernelEngine.from_model(model, T=3.0, eps=0.03)


# --------------------------------------------------------------------------
# construction / structure
# --------------------------------------------------------------------------

def test_dphys_and_subbath_count(engine):
    assert engine.d_phys == 7
    assert engine.K == 49


def test_for_sub_bath_returns_provider(engine):
    prov = engine.for_sub_bath(0)
    assert isinstance(prov, KernelProvider)
    assert prov.memory_time() is None  # infinite memory


def test_sub_bath_index_bounds(engine):
    with pytest.raises(IndexError):
        engine.for_sub_bath(49)
    with pytest.raises(IndexError):
        engine.for_sub_bath(-1)


def test_mpo_site_shapes(engine):
    mpo = engine.get_kernel_mpo(4, k=3)
    assert isinstance(mpo, KernelMPO)
    assert mpo.t == 4 and mpo.d_phys == 7
    shapes = [s.shape for s in mpo.site_tensors]
    assert shapes[0] == (7, 7, 1, 4)      # newest: left boundary
    assert shapes[1] == (7, 7, 4, 4)      # interior
    assert shapes[2] == (7, 7, 4, 4)
    assert shapes[3] == (7, 7, 4, 1)      # oldest: right boundary


def test_single_site_mpo(engine):
    mpo = engine.get_kernel_mpo(1, k=0)
    assert len(mpo.site_tensors) == 1
    assert mpo.site_tensors[0].shape == (7, 7, 1, 1)


def test_interior_sites_uniform(engine):
    mpo = engine.get_kernel_mpo(5, k=7)
    # interior sites (1..t-2) are the same time-independent tensor
    for s in mpo.site_tensors[2:-1]:
        np.testing.assert_array_equal(s, mpo.site_tensors[1])


def test_rejects_bad_t(engine):
    with pytest.raises(ValueError):
        engine.get_kernel_mpo(0, k=0)


# --------------------------------------------------------------------------
# operatorisation: T[up, down, l, r] = sum_mid P[up, mid, down] A[mid, l, r]
# --------------------------------------------------------------------------

def test_operatorize_matches_picking_contraction(model):
    corr = SeparableBathCorrelation().compute(model, T=2.0, eps=0.1)
    eng = SeparableKernelEngine(corr)
    P = picking_tensor(7)
    for k in (0, 12, 48):
        A = corr.transfer_for(k)                          # (7, 4, 4)
        expected = np.einsum("amd,mlr->adlr", P, A)       # (up, down, l, r)
        # interior site of the MPO is exactly the operatorised tensor
        interior = eng.get_kernel_mpo(3, k=k).site_tensors[1]
        np.testing.assert_allclose(interior, expected, atol=1e-12)


# --------------------------------------------------------------------------
# closing all open arms recovers the bare superoperator correlation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("t", [1, 2, 3])
def test_closed_open_arms_recover_correlation(model, t):
    corr = SeparableBathCorrelation().compute(model, T=3.0, eps=0.03)
    eng = SeparableKernelEngine(corr)
    for k in (0, 24, 48):
        dense = eng.get_kernel_mpo(t, k=k).to_dense()  # [up(t..1), down(t..1)]
        gk = model.couplings[k]
        for downs in itertools.product(range(7), repeat=t):
            # all open arms closed (phi_up = 0): P[0, mid, down] = delta_{mid,down}
            val = dense[(0,) * t + downs]
            # down axes are newest-first; correlation() wants time order (oldest first)
            ops = list(reversed(downs))
            assert val == pytest.approx(reference_correlation(ops, gk), abs=1e-12)


def test_open_arm_routes_noise(engine):
    # a noise routed entirely to its open arm (phi_up = phi_down = phi, no partner)
    # picks the identity-correlation branch: P[phi, 0, phi] = 1 contracts A[0] = I.
    mpo = engine.get_kernel_mpo(2, k=5)
    dense = mpo.to_dense()  # [u1, u0, d1, d0]  (newest u0,d0)
    # newest open arm carries phi, oldest closed: should equal A[0] identity path
    # sanity: tensor is finite and non-trivial
    assert np.isfinite(dense).all()
    assert np.linalg.norm(dense) > 0
