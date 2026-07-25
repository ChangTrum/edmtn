"""Time-dependent separable-bath kernel engine (Layer 3).

The two things that can silently go wrong here are the **site ordering** (the correlation
array is oldest-first, the MPO is newest-first) and the **boundaries** (newest left index
fixed to 0, oldest right index contracted with the bath Bloch vector).  Both are checked
against explicitly recomputed references, and the all-arms-closed contraction is compared
against the Layer-2 correlation for random operator sequences.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from edmtn.kernels import SeparableTDKernelEngine
from edmtn.kernels.base import picking_tensor
from edmtn.models import DickeModel


def _model(**kw):
    base = dict(K=2, n_fock=3, coupling=[0.35, 0.22], omega=[0.8, 1.3],
                bath_state=[[0.2, -0.3, 0.5], [0.0, 0.4, -0.4]])
    base.update(kw)
    return DickeModel(**base)


def _dissipative_model():
    return _model(pump=[0.05, 0.02], emission=[0.11, 0.07], dephasing=[0.04, 0.09])


def _engine(model, T=0.4, eps=0.1, order=2):
    return SeparableTDKernelEngine.from_model(model, T=T, eps=eps, order=order)


def _closed_arms_product(sites, ops, n_sites):
    """Contract the MPO with ``phi_up = 0`` everywhere and ``phi_down`` from ``ops``."""
    acc = np.array([[1.0 + 0.0j]])
    for p, T in enumerate(sites):
        g = n_sites - p                     # site p carries sub-step g
        acc = acc @ T[0, ops[g - 1]]
    return complex(acc[0, 0])


# -- structure ---------------------------------------------------------------------

@pytest.mark.parametrize("order,n_sites", [(1, 4), (2, 8)])
def test_site_shapes_and_boundaries(order, n_sites):
    ke = _engine(_model(), order=order)
    assert ke.n_sites == n_sites
    sites = ke.get_kernel_mpo(n_sites, 0).site_tensors
    assert len(sites) == n_sites
    assert sites[0].shape == (3, 3, 1, 4)          # newest: left lateral index fixed to 0
    assert sites[-1].shape == (3, 3, 4, 1)         # oldest: right index contracted with r_k
    for T in sites[1:-1]:
        assert T.shape == (3, 3, 4, 4)


def test_single_site_grid_has_both_boundaries():
    ke = _engine(_model(), T=0.1, eps=0.1, order=1)
    sites = ke.get_kernel_mpo(1, 0).site_tensors
    assert len(sites) == 1 and sites[0].shape == (3, 3, 1, 1)


def test_sites_are_the_reverse_of_the_oldest_first_transfer_array():
    """site_tensors[p] == operatorised(transfer[n_sites - 1 - p]), boundaries aside."""
    ke = _engine(_dissipative_model())
    P = picking_tensor(3)
    A = ke.corr.transfer_for(1)                                  # oldest first
    op = np.einsum("amd,gmlr->gadlr", P, A)
    sites = ke.get_kernel_mpo(ke.n_sites, 1).site_tensors
    n = ke.n_sites
    for p in range(1, n - 1):                                    # bulk: untouched by boundaries
        np.testing.assert_allclose(sites[p], op[n - 1 - p], atol=1e-14)
    np.testing.assert_allclose(sites[0], op[n - 1][:, :, 0:1, :], atol=1e-14)
    r = ke.corr.boundary_vector(1)
    np.testing.assert_allclose(sites[-1][..., 0], np.tensordot(op[0], r, axes=([3], [0])),
                               atol=1e-14)


def test_reversing_the_site_order_really_changes_the_tensors():
    """Guards the test above from being vacuous on a time-uniform kernel."""
    ke = _engine(_model())
    sites = ke.get_kernel_mpo(ke.n_sites, 0).site_tensors
    assert not np.allclose(sites[1], sites[-2], atol=1e-8)


# -- contraction against Layer 2 ---------------------------------------------------

@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("dissipative", [False, True])
def test_all_arms_closed_reproduces_the_bare_correlation(order, dissipative):
    model = _dissipative_model() if dissipative else _model()
    ke = _engine(model, order=order)
    rng = random.Random(3)
    for _ in range(15):
        ops = [rng.randrange(3) for _ in range(ke.n_sites)]
        for k in range(ke.K):
            sites = ke.get_kernel_mpo(ke.n_sites, k).site_tensors
            got = _closed_arms_product(sites, ops, ke.n_sites)
            assert abs(got - ke.corr.correlation(ops, k)) < 1e-12


def test_maximally_mixed_bath_reduces_to_slicing_the_oldest_index():
    """r_k = (1, 0, 0, 0) must reproduce the time-independent engine's boundary."""
    ke = _engine(_model(bath_state="inf"))
    P = picking_tensor(3)
    op = np.einsum("amd,gmlr->gadlr", P, ke.corr.transfer_for(0))
    oldest = ke.get_kernel_mpo(ke.n_sites, 0).site_tensors[-1]
    np.testing.assert_allclose(oldest, op[0][:, :, :, 0:1], atol=1e-14)


def test_the_bloch_boundary_actually_matters():
    """A different initial bath state must change the oldest site (not a no-op arm)."""
    a = _engine(_model(bath_state="inf")).get_kernel_mpo(8, 0).site_tensors[-1]
    b = _engine(_model(bath_state="ground")).get_kernel_mpo(8, 0).site_tensors[-1]
    assert not np.allclose(a, b, atol=1e-8)


# -- the grid signature ------------------------------------------------------------

def test_check_grid_accepts_the_grid_it_was_built_for():
    ke = _engine(_model(), T=0.4, eps=0.1, order=2)
    ke.check_grid(0.1, 4, 2)                    # must not raise
    assert ke.grid_signature == (0.1, 4, 2)


@pytest.mark.parametrize("grid", [(0.2, 4, 2), (0.1, 8, 2), (0.1, 4, 1)])
def test_check_grid_rejects_every_component_mismatch(grid):
    """n_sites alone cannot distinguish these: (0.1,4,1) and (0.1,2,2) both give 4 sites."""
    ke = _engine(_model(), T=0.4, eps=0.1, order=2)
    with pytest.raises(ValueError, match="time-grid mismatch"):
        ke.check_grid(*grid)


def test_two_grids_sharing_n_sites_have_different_signatures():
    m = _model()
    a = _engine(m, T=0.4, eps=0.1, order=1)     # 4 sites
    b = _engine(m, T=0.2, eps=0.1, order=2)     # 4 sites
    assert a.n_sites == b.n_sites
    assert a.grid_signature != b.grid_signature
    with pytest.raises(ValueError, match="time-grid mismatch"):
        a.check_grid(*b.grid_signature)


@pytest.mark.parametrize("t", [4, 16, 0, True, 8.0])
def test_get_kernel_mpo_rejects_a_site_count_it_was_not_built_for(t):
    ke = _engine(_model(), order=2)             # 8 sites
    with pytest.raises(ValueError):
        ke.get_kernel_mpo(t, 0)


# -- interface ---------------------------------------------------------------------

@pytest.mark.parametrize("k", [-1, 2, True, 1.0])
def test_for_sub_bath_rejects_an_illegal_index(k):
    ke = _engine(_model())
    with pytest.raises((ValueError, IndexError)):
        ke.for_sub_bath(k)


def test_engine_exposes_the_interface_the_evolution_validates():
    ke = _engine(_model())
    assert ke.d_phys == 3                       # 2 * n_ch + 1 for the single Dicke channel
    assert ke.K == 2
    assert callable(ke.for_sub_bath)
    assert ke.memory_time() is None
    assert ke.for_sub_bath(0).memory_time() is None


def test_sub_bath_providers_are_built_independently():
    """Lazily built per sub-bath: distinct arrays, distinct content."""
    ke = _engine(_model())
    a = ke.for_sub_bath(0).get_kernel_mpo(8, ).site_tensors[3]
    b = ke.for_sub_bath(1).get_kernel_mpo(8).site_tensors[3]
    assert not np.allclose(a, b, atol=1e-8)     # different g_k / omega_k
