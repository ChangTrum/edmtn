"""Track 1 / Track 2 share one public time axis (P0-6).

Separable (Gaudin) Track 1 used to return the Eq.-F2 sweep on the axis
``[0, eps, ..., T-eps]`` while spin-boson, Track 2 and the SolverResult docstring use
``[eps, 2eps, ..., T]``. So ``cpu.polarization[i]`` and ``hpc.polarization[i]`` were one
step apart. Track 1 now drops p(0) and appends ``Tr[S_a(T) rho(T)]`` from the final MPS, so
both backends agree index-for-index. Covers the N=1 boundary (which also exercises the
single-site QuimbEDM.fold fix). Track 2 runs on the NumPy executor (no GPU needed).
"""

from __future__ import annotations

import numpy as np
import pytest

import edmtn.evolution.cutensornet as ctn
from edmtn.driver import EDMSolver
from edmtn.expansion.first_order import FirstOrderExpander
from edmtn.evolution.separable_bath import SeparableBathEvolution
from edmtn.kernels.separable_mpo import SeparableKernelEngine
from edmtn.models import GaudinModel


def _force_numpy(monkeypatch):
    real = ctn.solve_cutensornet
    monkeypatch.setattr(ctn, "solve_cutensornet",
                        lambda *a, **k: real(*a, **{**k, "executor": "numpy"}))


def _track1(K, T, eps, order, sb):
    # exact CPU (cutoff=0, no bond cap, full SVD) to match Track 2's exact contraction
    return EDMSolver.from_model(GaudinModel(g=0.7, K=K), T=T, eps=eps, expansion_order=order,
                                cutoff=0.0, max_bond=None, compress_decomp="exact",
                                sub_baths=sb).solve(channel=3)


def _track2(monkeypatch, K, T, eps, order, sb):
    _force_numpy(monkeypatch)
    return EDMSolver.from_model(GaudinModel(g=0.7, K=K), T=T, eps=eps, expansion_order=order,
                                backend="hpc", sub_baths=sb).solve(channel=3)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("K", [1, 2, 3])
@pytest.mark.parametrize("sub_baths", ["one", "all"])
def test_track1_track2_axis_and_polarization_match(monkeypatch, order, K, sub_baths):
    sb = 1 if sub_baths == "one" else K
    T, eps = 0.3, 0.1                                   # N = 3
    t1 = _track1(K, T, eps, order, sb)
    t2 = _track2(monkeypatch, K, T, eps, order, sb)
    np.testing.assert_allclose(t1.times, t2.times)
    np.testing.assert_allclose(t1.polarization, t2.polarization, atol=1e-10)


# -- N = 1 boundary (also drives the single-site fold) ---------------------

@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("K", [1, 2, 3])
@pytest.mark.parametrize("sub_baths", ["one", "all"])
def test_n1_track1_track2(monkeypatch, order, K, sub_baths):
    sb = 1 if sub_baths == "one" else K
    T, eps = 0.1, 0.1                                   # N = 1  (T == eps)
    t1 = _track1(K, T, eps, order, sb)                  # order-1 -> single-site fold path
    assert t1.times.shape == t1.polarization.shape == (1,)
    assert np.isclose(t1.times[0], eps) and np.isclose(t1.times[0], T)
    t2 = _track2(monkeypatch, K, T, eps, order, sb)
    np.testing.assert_allclose(t1.times, t2.times)
    np.testing.assert_allclose(t1.polarization, t2.polarization, atol=1e-10)


# -- public-result invariants (no float == on the time axis) ---------------

@pytest.mark.parametrize("order", [1, 2])
def test_separable_public_axis_invariants(order):
    T, eps, n_steps = 0.5, 0.1, 5
    res = EDMSolver.from_model(GaudinModel(g=0.7, K=2), T=T, eps=eps,
                               expansion_order=order, cutoff=0.0).solve(channel=3)
    np.testing.assert_allclose(res.times, eps * np.arange(1, n_steps + 1))
    assert np.isclose(res.times[0], eps)
    assert np.isclose(res.times[-1], T)
    assert res.times.shape == res.polarization.shape == (n_steps,)


# -- direct single-site fold regression (the dangling-a0 bug) ---------------

def test_single_site_fold_has_no_dangling_index():
    # order-1, n_steps=1 -> n_sites=1 -> QuimbEDM.fold single-site branch. Before the fix
    # to_edmmps() raised on an unhandled a0; here the fold + to_edmmps must succeed and the
    # final EDM-MPS carries exactly one site with a well-formed 2x2 reduced state.
    model = GaudinModel(g=0.7, K=3)
    eng = SeparableKernelEngine.from_model(model, T=0.1, eps=0.1)
    ev = SeparableBathEvolution(expander=FirstOrderExpander()).run(model, eng, 0.1, 1)
    assert ev.mps.num_sites == 1
    rho = np.asarray(ev.mps.reduced_density_matrix())
    assert rho.shape == (2, 2)
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-9)   # trace-preserving
