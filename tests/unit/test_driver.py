"""Unit tests for Layer 7 (driver / orchestration)."""

import numpy as np
import pytest

from edmtn.driver import (
    EDMSolver,
    SolverConfig,
    SolverResult,
    available_pipelines,
    solve,
)
from edmtn.models import SpinBosonModel


@pytest.fixture
def model():
    return SpinBosonModel(J0=0.6, omega_c=5.0, mu=1.0)


# --------------------------------------------------------------------------
# configuration / pipeline selection
# --------------------------------------------------------------------------

def test_config_n_steps():
    cfg = SolverConfig(eps=0.05, T=1.0)
    assert cfg.n_steps == 20


def test_gaussian_pipeline_registered():
    assert "gaussian" in available_pipelines()


def test_unknown_bath_type_raises(model):
    class Weird(SpinBosonModel):
        bath_type = "exotic"

    w = Weird(J0=0.5, omega_c=5.0, mu=1.0)
    with pytest.raises(NotImplementedError):
        EDMSolver.from_model(w, T=0.5, eps=0.1)


def test_second_order_runs(model):
    # second order is supported end-to-end through the driver
    res = EDMSolver.from_model(model, T=0.4, eps=0.05, expansion_order=2).solve()
    assert res.times.shape == (8,)
    assert np.isclose(res.polarization[0], 0.5, atol=5e-2)


# --------------------------------------------------------------------------
# solve
# --------------------------------------------------------------------------

def test_solve_returns_result(model):
    res = EDMSolver.from_model(model, T=1.0, eps=0.05, cutoff=1e-6).solve()
    assert isinstance(res, SolverResult)
    assert res.times.shape == res.polarization.shape == (20,)
    assert len(res.bond_dims) == 20
    # default now inherits the model's time_step_order (=2) -> 2 sub-step sites per step
    assert res.mps.num_sites == 40


def test_solve_times_ascending_and_endpoint(model):
    res = EDMSolver.from_model(model, T=1.0, eps=0.05).solve()
    assert np.all(np.diff(res.times) > 0)
    assert np.isclose(res.times[0], 0.05)
    assert np.isclose(res.times[-1], 1.0)


def test_polarization_physical(model):
    res = EDMSolver.from_model(model, T=2.0, eps=0.05, cutoff=1e-6).solve()
    v = res.polarization
    assert np.isclose(v[0], 0.5, atol=2e-2)
    assert v[-1] < v[0]
    assert np.all(v <= 0.5 + 1e-9) and np.all(v >= -0.5 - 1e-9)


def test_convenience_solve(model):
    res = solve(model, T=0.5, eps=0.05, cutoff=1e-6)
    assert isinstance(res, SolverResult)
    assert res.times.shape == (10,)


def test_custom_observable_matches_manual(model):
    # custom observable via recorded rho; cross-check vs the built-in
    # coupling-channel (S_z) history at the same compression
    def sz(t):
        return model.coupling_operators_at(t)[0]

    res = EDMSolver.from_model(
        model, T=1.0, eps=0.05, cutoff=1e-7, record_rho=True
    ).solve(observables={"Sz": sz})
    assert "Sz" in res.observables
    np.testing.assert_allclose(
        res.observables["Sz"].real, res.polarization, atol=1e-4
    )


def test_custom_observable_requires_no_flag(model):
    # passing observables implicitly enables rho recording
    res = EDMSolver.from_model(model, T=0.5, eps=0.05).solve(
        observables={"Sz": lambda t: model.coupling_operators_at(t)[0]}
    )
    assert res.observables["Sz"].shape == (10,)


def test_max_bond_cap(model):
    res = EDMSolver.from_model(
        model, T=1.0, eps=0.05, cutoff=1e-10, max_bond=8
    ).solve()
    assert res.max_bond <= 8


# --------------------------------------------------------------------------
# convergence helper
# --------------------------------------------------------------------------

def test_timestep_convergence(model):
    solver = EDMSolver.from_model(model, T=0.8, eps=0.05, cutoff=1e-6)
    res = solver.timestep_convergence(tol=5e-2)
    assert res.deviation < 5e-2
    assert res.converged is True
    # legacy 2-tuple unpack stays supported
    dev, ok = res
    assert dev == res.deviation and ok is res.converged
    assert res[0] == res.deviation and res[1] is res.converged and len(res) == 2
