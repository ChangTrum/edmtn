"""``SolverResult.final_density_matrix`` -- the one reduced state every pipeline returns.

Added so that a solve which publishes no polarization (``separable_td``) still returns a
physical result, but it is filled on **all** pipelines so callers never have to reach into
``result.mps``.  The contract checked here: it is present without ``record_rho``, it agrees
with whatever per-axis reduced state the pipeline already publishes, it costs no second
contraction, and it keeps the backend-native array type.
"""

from __future__ import annotations

import numpy as np
import pytest

from edmtn.driver import solve
from edmtn.models import DickeModel, GaudinModel, SpinBosonModel


def _spin_boson():
    return SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0)


def _gaudin():
    return GaudinModel(g=1.0, K=3)


def _dicke():
    return DickeModel(K=3, n_fock=4, coupling=0.5, kappa=0.15, emission=0.05)


@pytest.mark.parametrize("model_fn,kw", [
    (_spin_boson, dict(expansion_order=2)),
    (_gaudin, dict(expansion_order=2, channel=3)),
    (_dicke, dict(expansion_order=2)),
])
def test_present_and_physical_without_record_rho(model_fn, kw):
    res = solve(model_fn(), T=0.3, eps=0.1, cutoff=1e-10, record_rho=False, **kw)
    rho = res.final_density_matrix
    assert rho is not None
    assert abs(complex(np.trace(rho)).real - 1.0) < 1e-8
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-8)
    # Positivity only up to the finite-step non-CP artifact of the truncated expansion:
    # a coarse eps can leave a small negative eigenvalue that vanishes as eps -> 0 (see
    # test_dicke_driver.test_finite_step_negativity_shrinks_with_eps).  This bound catches
    # a genuinely broken state without pretending the map is CP at finite eps.
    assert np.linalg.eigvalsh(0.5 * (rho + rho.conj().T)).min() > -1e-4


def test_single_bath_agrees_with_the_last_recorded_time_slice():
    res = solve(_spin_boson(), T=0.3, eps=0.1, expansion_order=2, cutoff=1e-10, record_rho=True)
    assert res.density_matrices is not None
    np.testing.assert_allclose(res.final_density_matrix, res.density_matrices[-1], atol=0.0)


@pytest.mark.parametrize("model_fn,kw", [
    (_gaudin, dict(channel=3)),
    (_dicke, dict()),
])
def test_separable_agrees_with_the_last_recorded_sub_bath_state(model_fn, kw):
    res = solve(model_fn(), T=0.3, eps=0.1, expansion_order=2, cutoff=1e-10,
                record_rho=True, **kw)
    assert res.sub_bath_final_density_matrices is not None
    np.testing.assert_allclose(res.final_density_matrix,
                               res.sub_bath_final_density_matrices[-1], atol=0.0)


def test_recording_does_not_change_the_final_state():
    """Reusing a recorded state must give the same answer as contracting once."""
    a = solve(_gaudin(), T=0.3, eps=0.1, expansion_order=2, cutoff=1e-10,
              record_rho=False, channel=3)
    b = solve(_gaudin(), T=0.3, eps=0.1, expansion_order=2, cutoff=1e-10,
              record_rho=True, channel=3)
    np.testing.assert_allclose(a.final_density_matrix, b.final_density_matrix, atol=1e-12)


def test_gaudin_final_polarization_point_is_consistent_with_the_field():
    """p_T and final_density_matrix now share one contraction -- they must still agree."""
    res = solve(_gaudin(), T=0.3, eps=0.1, expansion_order=2, cutoff=1e-10, channel=3)
    model = _gaudin()
    Sz = model.coupling_operators_at(0.3)[2]
    expected = float(np.trace(Sz @ np.asarray(res.final_density_matrix)).real)
    assert abs(res.polarization[-1] - expected) < 1e-12


def test_separable_field_is_the_state_of_the_folded_sub_baths_only():
    """With sub_baths < K it is rho_L(T), not the full-K result -- as documented."""
    partial = solve(_gaudin(), T=0.3, eps=0.1, expansion_order=2, cutoff=1e-10,
                    sub_baths=1, channel=3)
    full = solve(_gaudin(), T=0.3, eps=0.1, expansion_order=2, cutoff=1e-10, channel=3)
    assert partial.sub_baths_used == 1 and full.sub_baths_used == 3
    assert not np.allclose(partial.final_density_matrix, full.final_density_matrix, atol=1e-6)


def test_field_defaults_to_none_for_a_hand_built_result():
    """It must be a defaulted field, or existing manual SolverResult(...) calls break."""
    from edmtn.driver.solver import SolverResult

    res = SolverResult(times=[], polarization=[], bond_dims=[], truncation_errors=[],
                       expansion_order=2)
    assert res.final_density_matrix is None


@pytest.mark.gpu
def test_gpu_final_state_stays_on_device_and_feeds_the_fock_diagnostic():
    """CPU-passing cannot prove the array type survives; and fock_populations must not
    force a NumPy conversion, which CuPy refuses."""
    model = _dicke()
    res = solve(model, T=0.2, eps=0.1, expansion_order=2, cutoff=1e-10, backend="gpu")
    assert res.backend.startswith("gpu")
    rho = res.final_density_matrix
    assert type(rho).__module__.split(".")[0] == "cupy"
    p = model.fock_populations(rho)
    assert type(p).__module__.split(".")[0] == "cupy"
    cpu = solve(model, T=0.2, eps=0.1, expansion_order=2, cutoff=1e-10)
    np.testing.assert_allclose(np.asarray(rho.get()), np.asarray(cpu.final_density_matrix),
                               atol=1e-10)
