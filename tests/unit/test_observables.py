"""Unit tests for Layer 6 (observable extraction).

The central correctness check cross-validates the two independent extraction
routes for the coupling-channel polarization:

* the Eq.-F2 environment sweep over a *single* final EDM, and
* ``Tr[S_z(t) rho(t)]`` over the recorded reduced density matrices.

These are mathematically identical for the *uncompressed* EDM, so the exact
check uses ``compress=False`` (small ``N``); a separate test confirms they stay
close under real compression.
"""

import numpy as np
import pytest

from edmtn.cumulants import GaussianCumulantEngine
from edmtn.evolution import SingleBathEvolution
from edmtn.kernels import GaussianKernelEngine
from edmtn.models import SpinBosonModel
from edmtn.observables import (
    ObservableExtractor,
    is_converged,
    max_history_deviation,
    saturated,
)


def _build(J0=0.6, omega_c=5.0, mu=1.0, eps=0.05, N=20):
    model = SpinBosonModel(J0=J0, omega_c=omega_c, mu=mu)
    cum = GaussianCumulantEngine().compute(model, T=N * eps, eps=eps)
    engine = GaussianKernelEngine(cum)
    return model, engine, eps, N


@pytest.fixture
def run():
    return _build()


def _sz_at(model, t):
    return model.coupling_operators_at(t)[0]


# --------------------------------------------------------------------------
# single-time
# --------------------------------------------------------------------------

def test_density_matrix_and_trace(run):
    model, engine, eps, N = run
    res = SingleBathEvolution().run(model, engine, eps, n_steps=N, cutoff=1e-6)
    rho = ObservableExtractor.density_matrix(res.mps)
    assert rho.shape == (2, 2)
    assert np.isclose(ObservableExtractor.trace(res.mps), 1.0, atol=1e-6)
    assert ObservableExtractor.trace_deviation(res.mps) < 1e-6


def test_expectation_matches_manual(run):
    model, engine, eps, N = run
    res = SingleBathEvolution().run(model, engine, eps, n_steps=N, cutoff=1e-6)
    rho = ObservableExtractor.density_matrix(res.mps)
    op = _sz_at(model, N * eps)
    assert np.isclose(
        ObservableExtractor.expectation(res.mps, op), np.trace(op @ rho), atol=1e-12
    )


# --------------------------------------------------------------------------
# Eq. F2 history vs recorded-rho method
# --------------------------------------------------------------------------

def test_f2_history_matches_recorded_exact():
    # mathematically identical for the uncompressed EDM
    model, engine, eps, _ = _build(N=7)
    res = SingleBathEvolution().run(
        model, engine, eps, n_steps=7, record_rho=True, compress=False
    )
    t_f2, v_f2 = ObservableExtractor.coupling_polarization_history(res.mps, eps)
    t_rec, v_rec = ObservableExtractor.expectation_history(
        res.density_matrices, res.times, lambda t: _sz_at(model, t)
    )
    np.testing.assert_allclose(t_f2, t_rec, atol=1e-12)
    np.testing.assert_allclose(v_f2, v_rec.real, atol=1e-9)


def test_f2_history_matches_recorded_compressed(run):
    model, engine, eps, N = run
    res = SingleBathEvolution().run(
        model, engine, eps, n_steps=N, record_rho=True, cutoff=1e-7
    )
    t_f2, v_f2 = ObservableExtractor.coupling_polarization_history(res.mps, eps)
    _, v_rec = ObservableExtractor.expectation_history(
        res.density_matrices, res.times, lambda t: _sz_at(model, t)
    )
    np.testing.assert_allclose(v_f2, v_rec.real, atol=1e-4)


def test_history_starts_polarized_and_decays(run):
    model, engine, eps, N = run
    res = SingleBathEvolution().run(model, engine, eps, n_steps=N, cutoff=1e-6)
    _, v = ObservableExtractor.coupling_polarization_history(res.mps, eps)
    # initial polarization +1/2, monotone decay for this (near-overdamped) J0
    assert np.isclose(v[0], 0.5, atol=2e-2)
    assert v[-1] < v[0]
    assert np.all(v <= 0.5 + 1e-9) and np.all(v >= -0.5 - 1e-9)


def test_history_values_real(run):
    model, engine, eps, N = run
    res = SingleBathEvolution().run(model, engine, eps, n_steps=N, cutoff=1e-6)
    _, v = ObservableExtractor.coupling_polarization_history(res.mps, eps)
    assert v.dtype == np.float64


def test_channel_out_of_range_raises(run):
    model, engine, eps, N = run
    res = SingleBathEvolution().run(model, engine, eps, n_steps=5, cutoff=1e-6)
    with pytest.raises(ValueError):
        ObservableExtractor.coupling_polarization_history(res.mps, eps, channel=2)


# --------------------------------------------------------------------------
# convergence diagnostics
# --------------------------------------------------------------------------

def test_timestep_convergence_history():
    model, engine, eps, N = _build(eps=0.05, N=16)
    coarse = SingleBathEvolution().run(model, engine, eps, n_steps=N, cutoff=1e-6)
    tc, vc = ObservableExtractor.coupling_polarization_history(coarse.mps, eps)

    eps2 = eps / 2
    cum2 = GaussianCumulantEngine().compute(model, T=N * eps, eps=eps2)
    engine2 = GaussianKernelEngine(cum2)
    fine = SingleBathEvolution().run(model, engine2, eps2, n_steps=2 * N, cutoff=1e-6)
    tf, vf = ObservableExtractor.coupling_polarization_history(fine.mps, eps2)

    dev = max_history_deviation(tc, vc, tf, vf)
    # first-order step error: small but nonzero
    assert dev < 5e-2
    assert is_converged(tc, vc, tf, vf, tol=5e-2)


def test_saturated_detector():
    assert saturated([1, 5, 10, 20, 20, 20, 20])
    assert not saturated([1, 5, 10, 20, 25, 30, 36])
    assert not saturated([1, 2])
