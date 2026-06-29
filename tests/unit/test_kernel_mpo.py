"""Unit tests for Layer 3 (Gaussian combined-kernel MPO).

The combined kernel is validated against an *independent* reference built from
the brute-force Wick expansion of the bath correlation tensor and the picking
tensor: the kernel must satisfy the advance relation

    C^{(t)} = K_t . C^{(t-1)}

where C^{(t)} is the open-armed correlation tensor computed directly from the
cumulants.
"""

import itertools

import numpy as np
import pytest

from edmtn.cumulants import GaussianCumulantEngine
from edmtn.kernels import GaussianKernelEngine, KernelMPO, picking_tensor
from edmtn.kernels.base import KernelProvider
from edmtn.models import SpinBosonModel

D = 3  # single-channel superoperator dimension


@pytest.fixture
def cum():
    model = SpinBosonModel(J0=0.7, omega_c=4.0, mu=1.0)
    return GaussianCumulantEngine().compute(model, T=1.0, eps=0.1)


@pytest.fixture
def engine(cum):
    return GaussianKernelEngine(cum)


# --------------------------------------------------------------------------
# independent reference oracles
# --------------------------------------------------------------------------

def cum_pair(cum, later, earlier, lag):
    """Pairwise cumulant; nonzero only when the later index is B^+ (=2)."""
    if later == 2 and earlier == 2:
        return complex(cum.re[lag])
    if later == 2 and earlier == 1:
        return complex(cum.im2[lag])
    return 0.0


def wick_correlation(cum, phis):
    """Brute-force C_{Phi} for phis = (phi_t, ..., phi_1) (latest first).

    Gaussian bath: sum over all perfect matchings of the nonzero-index
    positions, product of pairwise cumulants; zero index = identity.
    """
    nz = [i for i, p in enumerate(phis) if p != 0]
    if len(nz) == 0:
        return 1.0 + 0j
    if len(nz) % 2 == 1:
        return 0.0 + 0j

    def matchings(items):
        if not items:
            yield []
            return
        a = items[0]
        for k in range(1, len(items)):
            b = items[k]
            rest = items[1:k] + items[k + 1:]
            for m in matchings(rest):
                yield [(a, b)] + m

    total = 0.0 + 0j
    for m in matchings(nz):
        term = 1.0 + 0j
        for i, k in m:  # i < k => position i is later (latest-first ordering)
            term *= cum_pair(cum, phis[i], phis[k], lag=k - i)
        total += term
    return total


def open_armed_correlation(cum, t):
    """Dense C^{Phi'(t)}_{Phi(t)} via picking tensor applied to the Wick correlation.

    Axis order: [phi_up(t), ..., phi_up(1), phi_down(t), ..., phi_down(1)].
    """
    P = picking_tensor(D)  # P[up, mid, down]
    out = np.zeros((D,) * (2 * t), dtype=np.complex128)
    rng = range(D)
    for ups in itertools.product(rng, repeat=t):
        for downs in itertools.product(rng, repeat=t):
            acc = 0.0 + 0j
            for mids in itertools.product(rng, repeat=t):
                w = wick_correlation(cum, mids)
                if w == 0:
                    continue
                fac = 1.0 + 0j
                for j in range(t):
                    fac *= P[ups[j], mids[j], downs[j]]
                acc += fac * w
            out[ups + downs] = acc
    return out


def advance(K_dense, C_prev, t):
    """Contract C^{(t)} = K_t . C^{(t-1)} per the advance relation."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    ups = list(letters[:t])                 # phi_up(t..1)
    d_new = letters[t]                      # phi_down(t) free
    if t == 1:
        # K has axes [up, down]; C_prev is the scalar 1
        return K_dense * C_prev
    mids = list(letters[t + 1 : t + 1 + (t - 1)])       # contracted
    es = list(letters[t + 1 + (t - 1) : t + 1 + 2 * (t - 1)])  # phi_down(t-1..1) free
    k_sub = "".join(ups) + d_new + "".join(mids)
    c_sub = "".join(mids) + "".join(es)
    out_sub = "".join(ups) + d_new + "".join(es)
    import opt_einsum as oe  # lazy: opt_einsum isn't in every env
    return oe.contract(f"{k_sub},{c_sub}->{out_sub}", K_dense, C_prev, optimize="auto")


# --------------------------------------------------------------------------
# structural tests
# --------------------------------------------------------------------------

def test_picking_tensor_definition():
    P = picking_tensor(D)
    # null index maps to (0, 0)
    assert P[0, 0, 0] == 1
    assert np.all(P[1:, :, 0] == 0) and np.all(P[:, 1:, 0] == 0)
    # nonzero down -> either open arm (up=down, mid=0) or cumulant (up=0, mid=down)
    for down in (1, 2):
        assert P[down, 0, down] == 1
        assert P[0, down, down] == 1


def test_is_kernel_provider(engine):
    assert isinstance(engine, KernelProvider)


def test_kernel_shapes(engine):
    K = engine.get_kernel_mpo(4)
    assert isinstance(K, KernelMPO)
    assert K.t == 4 and K.d_phys == D
    assert len(K.site_tensors) == 4
    # boundary bonds are 1, interior bonds are 2
    assert K.site_tensors[0].shape == (D, D, 1, 2)   # newest
    assert K.site_tensors[1].shape == (D, D, 2, 2)   # interior
    assert K.site_tensors[2].shape == (D, D, 2, 2)   # interior
    assert K.site_tensors[3].shape == (D, D, 2, 1)   # oldest


def test_bond_dimension_is_two(engine):
    K = engine.get_kernel_mpo(5)
    # fixed lateral bond dimension 2 (Gaussian closed form)
    assert max(K.bond_dims) == 2


def test_t1_kernel_is_identity(engine):
    K = engine.get_kernel_mpo(1)
    dense = K.to_dense()  # shape (D, D)
    np.testing.assert_allclose(dense, np.eye(D), atol=1e-14)


# --------------------------------------------------------------------------
# correctness: advance relation against the Wick reference
# --------------------------------------------------------------------------

@pytest.mark.parametrize("t", [2, 3, 4])
def test_advance_relation_matches_reference(cum, engine, t):
    K_dense = engine.get_kernel_mpo(t).to_dense()
    C_prev = (
        np.array(1.0 + 0j) if t == 1 else open_armed_correlation(cum, t - 1)
    )
    C_t_ref = open_armed_correlation(cum, t)
    C_t_kernel = advance(K_dense, C_prev, t)
    np.testing.assert_allclose(C_t_kernel, C_t_ref, atol=1e-12)


def test_closing_arms_gives_physical_correlation(cum, engine):
    # build C^{(2)} via the kernel, close all open arms with delta^0, and check
    # against the brute-force physical correlation TrB[B B Omega].
    t = 2
    C2 = advance(engine.get_kernel_mpo(t).to_dense(), open_armed_correlation(cum, 1), t)
    # close upper arms: select phi_up = 0 on every site (axes 0..t-1)
    closed = C2[0, 0]  # phi_up(2)=0, phi_up(1)=0  -> indexed downs (phi_down2, phi_down1)
    for d2 in range(D):
        for d1 in range(D):
            ref = wick_correlation(cum, (d2, d1))
            assert np.isclose(closed[d2, d1], ref, atol=1e-12)


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------

def test_step_too_large_raises(engine, cum):
    with pytest.raises(ValueError):
        engine.get_kernel_mpo(cum.n_steps + 2)


def test_step_zero_raises(engine):
    with pytest.raises(ValueError):
        engine.get_kernel_mpo(0)


def test_from_model_constructor():
    model = SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0)
    engine = GaussianKernelEngine.from_model(model, T=0.5, eps=0.1)
    K = engine.get_kernel_mpo(3)
    assert K.t == 3
