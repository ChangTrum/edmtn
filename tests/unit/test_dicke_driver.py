"""Driver contract of the ``separable_td`` (Dicke) pipeline.

Covers the entry behaviour that differs from the other pipelines -- no coupling-channel
polarization, the ``channel`` resolution that still enforces the strict type/range
contract, ``timestep_convergence`` refusing up front rather than leaking a ``None``, and
the grid hook that rejects a kernel built for a different time grid.
"""

from __future__ import annotations

import numpy as np
import pytest

from edmtn.driver import EDMSolver, solve
from edmtn.driver.auto_config import SolverConfig, available_pipelines, build_pipeline
from edmtn.evolution.separable_bath import SeparableBathEvolution
from edmtn.kernels import SeparableTDKernelEngine
from edmtn.models import DickeModel


def _model(**kw):
    base = dict(K=3, n_fock=4, coupling=0.5, omega=1.0)
    base.update(kw)
    return DickeModel(**base)


def _dissipative(**kw):
    return _model(kappa=0.15, pump=0.02, emission=0.05, dephasing=0.03, **kw)


def _solve(model, **kw):
    opts = dict(T=0.3, eps=0.1, cutoff=1e-10)
    opts.update(kw)
    return solve(model, **opts)


# -- pipeline selection ------------------------------------------------------------

def test_pipeline_is_registered_and_selected_by_bath_type():
    assert "separable_td" in available_pipelines()
    ke, ev = build_pipeline(_dissipative(), SolverConfig(eps=0.1, T=0.3))
    assert isinstance(ke, SeparableTDKernelEngine)
    assert isinstance(ev, SeparableBathEvolution)
    assert ev.expander.order == 2


def test_kernel_is_built_from_the_resolved_order_not_the_model_attribute():
    """expansion_order=1 on a model declaring 2 must give an order-1 kernel."""
    model = _model(time_step_order=2)
    ke, ev = build_pipeline(model, SolverConfig(eps=0.1, T=0.3, expansion_order=1))
    assert ke.grid_signature == (0.1, 3, 1)
    assert ev.expander.order == 1


# -- results -----------------------------------------------------------------------

@pytest.mark.parametrize("order", [1, 2])
def test_end_to_end_solve_returns_a_physical_final_state(order):
    model = _dissipative()
    res = _solve(model, expansion_order=order)
    np.testing.assert_allclose(res.times, [0.1, 0.2, 0.3], atol=1e-12)
    assert res.polarization is None                       # no channel history on this pipeline
    assert res.density_matrices is None                   # per-L states are rho_L(T), not rho(t)
    rho = res.final_density_matrix
    assert rho is not None and rho.shape == (4, 4)
    assert abs(complex(np.trace(rho)).real - 1.0) < 1e-8
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-8)
    p = model.fock_populations(rho)
    assert p.min() > -1e-10 and abs(p.sum() - 1.0) < 1e-8
    assert res.expansion_order == order
    assert res.backend.startswith("cpu")


def test_per_L_axis_fields_are_populated():
    res = _solve(_dissipative())
    assert res.sub_bath_counts == [1, 2, 3]
    assert len(res.sub_bath_bond_dims) == len(res.sub_bath_counts)
    assert len(res.truncation_errors) == len(res.sub_bath_counts)
    assert res.bond_dims == res.sub_bath_bond_dims        # documented legacy alias
    assert res.sub_baths_used == 3
    assert res.final_time_bond_dims == res.mps.bond_dims


def test_record_rho_records_the_per_L_states_and_agrees_with_the_final_one():
    res = _solve(_dissipative(), record_rho=True)
    assert res.sub_bath_final_density_matrices is not None
    assert len(res.sub_bath_final_density_matrices) == len(res.sub_bath_counts)
    np.testing.assert_allclose(res.final_density_matrix,
                               res.sub_bath_final_density_matrices[-1], atol=0.0)


def test_sub_baths_limits_the_fold_and_the_final_state_says_so():
    res = _solve(_dissipative(), sub_baths=2)
    assert res.sub_baths_used == 2                       # final_density_matrix is rho_2(T)
    assert res.sub_bath_counts[-1] == 2
    full = _solve(_dissipative())
    assert not np.allclose(res.final_density_matrix, full.final_density_matrix, atol=1e-8)


def test_closed_and_dissipative_runs_differ():
    """Guards against the rates being silently dropped somewhere in the wiring."""
    closed = _solve(_model()).final_density_matrix
    damped = _solve(_dissipative()).final_density_matrix
    assert not np.allclose(closed, damped, atol=1e-6)


def test_finite_step_negativity_shrinks_with_eps():
    """The truncated expansion is not completely positive at finite ``eps``.

    A coarse step can leave a small negative eigenvalue in the reduced state; it is a
    discretisation artifact, not a broken contraction, and it must **shrink as ``eps``
    shrinks**.  Refining by 4x here takes it from order 1e-6 to non-negative, which is the
    falsifiable half of the claim -- a systematic negativity would not improve.
    """
    def min_eig(eps):
        rho = np.asarray(_solve(_dissipative(), eps=eps, cutoff=1e-12).final_density_matrix)
        return float(np.linalg.eigvalsh(0.5 * (rho + rho.conj().T)).min())

    coarse, fine = min_eig(0.1), min_eig(0.025)
    assert coarse > -1e-4                       # small even at the coarse step
    assert fine > coarse                        # and it improves under refinement
    assert fine > -1e-9                         # essentially gone by eps = T/12


def test_custom_observables_are_refused_before_the_evolution_runs():
    solver = EDMSolver.from_model(_dissipative(), T=0.3, eps=0.1)
    with pytest.raises(NotImplementedError, match="not supported"):
        solver.solve({"n": lambda t: np.eye(4)})


# -- the channel contract ----------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, 2, True, 1.5, "1", None.__class__])
def test_illegal_channels_still_raise_value_error(bad):
    """The strict type/range contract survives: only a LEGAL channel gets the capability error."""
    with pytest.raises(ValueError):
        _solve(_model(), channel=bad)


def test_the_one_legal_channel_raises_not_implemented_with_an_honest_message():
    with pytest.raises(NotImplementedError) as exc:
        _solve(_model(), channel=1)
    message = str(exc.value)
    assert "coupling operator" in message                 # the model HAS one ...
    assert "coupling-channel history" in message          # ... the pipeline lacks the history
    assert "time-INDEPENDENT" in message                  # and states the actual reason
    assert "final_density_matrix" in message              # and points at what to read


def test_channel_none_is_the_default_and_solves():
    res = _solve(_model())
    assert res.polarization is None and res.final_density_matrix is not None


@pytest.mark.parametrize("channel", [None, 1])
def test_timestep_convergence_refuses_up_front(channel):
    """Legal input, missing capability -- for both the default and the one legal channel."""
    solver = EDMSolver.from_model(_dissipative(), T=0.3, eps=0.1)
    with pytest.raises(NotImplementedError, match="final_density_matrix"):
        solver.timestep_convergence(channel=channel)


@pytest.mark.parametrize("bad", [0, -1, 2, True, 1.5, "1"])
def test_timestep_convergence_keeps_the_strict_channel_contract(bad):
    """The capability gate must not swallow the type/range contract that solve() enforces:
    an illegal channel is a ValueError here too, not a NotImplementedError."""
    solver = EDMSolver.from_model(_dissipative(), T=0.3, eps=0.1)
    with pytest.raises(ValueError):
        solver.timestep_convergence(channel=bad)


def test_track_two_still_rejects_this_bath_type():
    """Track 2 is Gaudin / ``bath_type='separable'`` only.

    Asserted precisely rather than as "some exception": the guard is the first statement
    of ``solve_cutensornet``, before any CuPy/cuQuantum import, so a missing-GPU
    ``ImportError`` would be passing for the wrong reason on a CPU box.
    """
    solver = EDMSolver(_model(), SolverConfig(eps=0.1, T=0.3, backend="hpc"))
    with pytest.raises(NotImplementedError, match="separable baths"):
        solver.solve()


# -- the grid hook -----------------------------------------------------------------

def test_evolution_rejects_a_kernel_built_for_a_different_grid():
    model = _dissipative()
    kernel = SeparableTDKernelEngine.from_model(model, T=0.3, eps=0.1, order=2)
    evolution = SeparableBathEvolution()
    with pytest.raises(ValueError, match="time-grid mismatch"):
        evolution.run(model, kernel, eps=0.2, n_steps=3, compress=False)


def test_the_grid_hook_catches_a_mismatch_that_shares_the_site_count():
    """order=1,N=4 and order=2,N=2 both give 4 sites -- the count alone cannot tell."""
    from edmtn.expansion import FirstOrderExpander

    model = _dissipative()
    kernel = SeparableTDKernelEngine.from_model(model, T=0.2, eps=0.1, order=2)   # 4 sites
    evolution = SeparableBathEvolution(expander=FirstOrderExpander())             # order 1
    with pytest.raises(ValueError, match="time-grid mismatch"):
        evolution.run(model, kernel, eps=0.1, n_steps=4, compress=False)          # 4 sites too


def test_midpoint_sampling_is_a_no_op_for_a_static_coupling():
    """Why moving the shared Layer-5 sampling point to the midpoint cannot move Gaudin.

    ``SeparableBathEvolution._build_system_mps`` now samples at ``(n - 1/2) eps`` instead
    of ``n eps``.  Gaudin has ``H_S = 0``, so its interaction-picture operators are
    constant and the site tensors are literally unchanged; only a model with a
    time-dependent coupling (Dicke) sees any difference.
    """
    from edmtn.models import GaudinModel

    gaudin = GaudinModel(g=1.0, K=3)
    baseline = gaudin.coupling_operators_at(0.0)
    for t in (0.05, 0.1, 1.7, 42.0):
        for got, want in zip(gaudin.coupling_operators_at(t), baseline):
            assert np.array_equal(got, want)

    dicke = _model()                                    # ... and for Dicke it does matter
    assert not np.allclose(dicke.coupling_operators_at(0.05)[0],
                           dicke.coupling_operators_at(0.1)[0], atol=1e-8)


def test_matching_grid_passes_the_hook():
    model = _dissipative()
    kernel = SeparableTDKernelEngine.from_model(model, T=0.2, eps=0.1, order=2)
    out = SeparableBathEvolution().run(model, kernel, eps=0.1, n_steps=2, compress=False)
    assert out.n_sub_baths == 3
