"""Unit tests for Layer 5 (single-bath EDM-MPS evolution).

The MPS engine is validated against an *independent* brute-force reference that
contracts the dense open-armed correlation tensor (built from the already-tested
Layer-3 kernel) with the time-ordered system-superoperator product -- no MPS
machinery involved.  We check:

* the per-step contraction is exact (un-compressed MPS == dense reference);
* compression with ``cutoff = 0`` (keep all) preserves the result;
* compression with a real cutoff stays accurate and keeps the trace;
* the bond dimension grows at most linearly (the paper's central claim).
"""

import numpy as np
import pytest

from edmtn.cumulants import GaussianCumulantEngine
from edmtn.evolution import (
    EDMMPS,
    SingleBathEvolution,
    dense_reduced_density_matrix,
)
from edmtn.evolution import mps_utils
from edmtn.expansion import FirstOrderExpander
from edmtn.kernels import GaussianKernelEngine
from edmtn.models import SpinBosonModel


@pytest.fixture
def setup():
    model = SpinBosonModel(J0=0.7, omega_c=4.0, mu=1.0)
    eps = 0.1
    T = 2.0
    cum = GaussianCumulantEngine().compute(model, T=T, eps=eps)
    engine = GaussianKernelEngine(cum)
    return model, engine, eps


def _sfamilies(model, eps, n):
    exp = FirstOrderExpander()
    return [exp.build_at(model, k * eps, eps).families[0] for k in range(1, n + 1)]


def _reference_rho(model, engine, eps, n):
    d = model.system_dim
    rho0 = model.initial_system_state().reshape(-1).astype(np.complex128)
    sf = _sfamilies(model, eps, n)
    return dense_reduced_density_matrix(engine, sf, rho0, n, d)


# --------------------------------------------------------------------------
# structural
# --------------------------------------------------------------------------

def test_first_step_is_identity(setup):
    model, engine, eps = setup
    ev = SingleBathEvolution()
    res = ev.run(model, engine, eps, n_steps=1, record_rho=True)
    # t = 1: rho(1) = S^0 rho0 = rho0 (single noise cannot pair)
    np.testing.assert_allclose(res.density_matrices[0], model.initial_system_state(), atol=1e-12)


def test_accepts_second_order_expander(setup):
    from edmtn.expansion import SecondOrderExpander

    # second order is supported (driven on a doubled sub-step grid)
    ev = SingleBathEvolution(expander=SecondOrderExpander())
    assert ev.expander.order == 2


def test_mps_is_edmmps(setup):
    model, engine, eps = setup
    res = SingleBathEvolution().run(model, engine, eps, n_steps=3)
    assert isinstance(res.mps, EDMMPS)
    assert res.mps.num_sites == 3


# --------------------------------------------------------------------------
# correctness: un-compressed MPS reproduces the dense reference
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_uncompressed_matches_reference(setup, n):
    model, engine, eps = setup
    res = SingleBathEvolution().run(
        model, engine, eps, n_steps=n, record_rho=True, compress=False
    )
    ref = _reference_rho(model, engine, eps, n)
    np.testing.assert_allclose(res.density_matrices[-1], ref, atol=1e-11)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_exact_compression_matches_reference(setup, n):
    # cutoff = 0 keeps every singular value -> SVD sweep is exact
    model, engine, eps = setup
    res = SingleBathEvolution().run(
        model, engine, eps, n_steps=n, record_rho=True, compress=True, cutoff=0.0
    )
    ref = _reference_rho(model, engine, eps, n)
    np.testing.assert_allclose(res.density_matrices[-1], ref, atol=1e-11)


def test_every_step_matches_reference(setup):
    model, engine, eps = setup
    res = SingleBathEvolution().run(
        model, engine, eps, n_steps=5, record_rho=True, compress=True, cutoff=0.0
    )
    for n in range(1, 6):
        ref = _reference_rho(model, engine, eps, n)
        np.testing.assert_allclose(res.density_matrices[n - 1], ref, atol=1e-11)


# --------------------------------------------------------------------------
# physics-flavoured checks
# --------------------------------------------------------------------------

def test_trace_preserved(setup):
    model, engine, eps = setup
    res = SingleBathEvolution().run(
        model, engine, eps, n_steps=8, record_rho=True, cutoff=1e-8
    )
    for rho in res.density_matrices:
        assert np.isclose(np.trace(rho), 1.0, atol=1e-6)


def test_hermitian_density_matrix(setup):
    model, engine, eps = setup
    res = SingleBathEvolution().run(model, engine, eps, n_steps=6, record_rho=True)
    rho = res.density_matrices[-1]
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-10)


def test_compression_accurate_vs_exact(setup):
    model, engine, eps = setup
    exact = SingleBathEvolution().run(
        model, engine, eps, n_steps=8, record_rho=True, cutoff=0.0
    )
    approx = SingleBathEvolution().run(
        model, engine, eps, n_steps=8, record_rho=True, cutoff=1e-7
    )
    np.testing.assert_allclose(
        approx.density_matrices[-1], exact.density_matrices[-1], atol=1e-5
    )


# --------------------------------------------------------------------------
# bond-dimension growth
# --------------------------------------------------------------------------

def test_bond_dimension_grows_at_most_linearly(setup):
    model, engine, eps = setup
    res = SingleBathEvolution().run(model, engine, eps, n_steps=15, cutoff=1e-6)
    bonds = res.bond_dims
    # "at most linear" growth (Theorem 2 / Fig. 4b): each step adds a *bounded*
    # number of independent EDM directions, so consecutive increments are capped
    # by a constant independent of time (here d**2 x d_phys channels).
    d_phys = res.mps.d_phys
    cap = model.system_dim ** 2 * d_phys
    increments = np.diff([1] + bonds)
    assert increments.max() <= cap
    # cumulative growth therefore stays under a linear envelope
    for t, D in enumerate(bonds, start=1):
        assert D <= cap * t


def test_truncation_reduces_bond_vs_exact(setup):
    model, engine, eps = setup
    exact = SingleBathEvolution().run(model, engine, eps, n_steps=10, cutoff=0.0)
    approx = SingleBathEvolution().run(model, engine, eps, n_steps=10, cutoff=1e-5)
    assert approx.mps.max_bond <= exact.mps.max_bond


def test_max_bond_cap_respected(setup):
    model, engine, eps = setup
    cap = 6
    res = SingleBathEvolution().run(
        model, engine, eps, n_steps=10, cutoff=1e-10, max_bond=cap
    )
    assert res.mps.max_bond <= cap


# --------------------------------------------------------------------------
# mps utilities
# --------------------------------------------------------------------------

def test_open_arm_tensor_closes_to_reduced(setup):
    model, engine, eps = setup
    n = 3
    res = SingleBathEvolution().run(model, engine, eps, n_steps=n, compress=False)
    full = res.mps.open_arm_tensor()  # (d_phys,)*n + (d, d)
    closed = full[(0,) * n]
    np.testing.assert_allclose(closed, res.mps.reduced_density_matrix(), atol=1e-12)


def test_compress_keeps_norm_with_zero_cutoff(setup):
    model, engine, eps = setup
    res = SingleBathEvolution().run(model, engine, eps, n_steps=4, compress=False)
    mps = res.mps.copy()
    rho_before = mps.reduced_density_matrix()
    mps_utils.compress(mps, cutoff=0.0)
    rho_after = mps.reduced_density_matrix()
    np.testing.assert_allclose(rho_after, rho_before, atol=1e-11)
