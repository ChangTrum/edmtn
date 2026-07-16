"""HPC track (cuQuantum 2D contraction) — logic tests via the NumPy executor.

These run locally (no GPU / cuQuantum): the `executor="numpy"` path exercises the
2D assembly, the per-step ρ(t) history, the error metrics, and the driver wiring.
The cuQuantum executor (exact, single- and multi-GPU) is validated on c1 by
``examples/track2/cutensornet_sanity.py``. Track 2 is the exact route only — the truncated
regime lives in Track 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from edmtn.driver.auto_config import SolverConfig
from edmtn.driver.solver import EDMSolver
from edmtn.evolution.cutensornet import _make_expander, solve_cutensornet
from edmtn.evolution.separable_bath import SeparableBathEvolution
from edmtn.kernels.separable_mpo import SeparableKernelEngine
from edmtn.models import GaudinModel


def _track1_final_rho(model, order, eps, N):
    ke = SeparableKernelEngine.from_model(model, N * eps, eps)
    res = SeparableBathEvolution(_make_expander(order)).run(model, ke, eps, N, compress=False)
    return res.mps.reduced_density_matrix()


@pytest.mark.parametrize("order,K,N", [(1, 2, 3), (1, 3, 3), (2, 2, 2)])
def test_exact_numpy_matches_track1(order, K, N):
    model = GaudinModel(g=1.0, K=K, time_step_order=order)
    cfg = SolverConfig(eps=0.1, T=N * 0.1, expansion_order=order, backend="hpc")
    out = solve_cutensornet(model, cfg, channel=3, executor="numpy")
    ref = _track1_final_rho(model, order, 0.1, N)
    assert np.max(np.abs(out["final_rho"] - ref)) < 1e-10


def test_density_matrices_and_metrics_returned():
    model = GaudinModel(g=1.0, K=3, time_step_order=1)
    cfg = SolverConfig(eps=0.1, T=0.4, expansion_order=1, backend="hpc")
    out = solve_cutensornet(model, cfg, channel=3, executor="numpy")
    assert len(out["density_matrices"]) == cfg.n_steps        # ρ(t) history is first-class
    assert out["final_rho"] is out["density_matrices"][-1]
    m = out["error_metrics"]
    assert m["hermiticity"] < 1e-12 and m["trace_dev"] < 1e-12  # exact: hermitian, trace-preserving
    # polarization derived only when a channel is given
    assert out["polarization"] is not None and len(out["polarization"]) == cfg.n_steps
    no_chan = solve_cutensornet(model, cfg, channel=None, executor="numpy")
    assert no_chan["polarization"] is None and no_chan["density_matrices"] is not None


def test_solver_hpc_skips_track1_pipeline():
    s = EDMSolver(GaudinModel(g=1.0, K=2), SolverConfig(eps=0.1, T=0.2, backend="hpc"))
    assert s.evolution is None and s.kernel_engine is None
    assert s.config.pathfinder == "cuquantum" and s.config.time_windows is None


def test_auto_backend_removed():
    # rejected at config construction now (centralized SolverConfig validation)
    with pytest.raises(ValueError):
        SolverConfig(eps=0.1, T=0.2, backend="auto")


def test_windows_not_yet_implemented():
    # rejected at config construction now, for any non-None time_windows
    with pytest.raises(NotImplementedError):
        SolverConfig(eps=0.1, T=0.2, backend="hpc", time_windows=2)
