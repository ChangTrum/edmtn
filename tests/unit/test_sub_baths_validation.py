"""Strict sub_baths validation (P0-10).

The evolution layers used to do ``min(int(sub_baths), K)`` -- silently truncating a float
(2.9->2), coercing a bool (True->1), and clamping an over-range value (99->K) -- which changes
how many bath spins are actually in the Hamiltonian with no error. A shared
``validate_sub_baths(sub_baths, K)`` now rejects those uniformly across Track 1, Track 2 and the
direct evolution entry points, resolves None->K once, and the actual folded L is recorded on
``result.sub_baths_used`` (so a None or out-of-range request can't masquerade as K).
"""

from __future__ import annotations

import numpy as np
import pytest

import edmtn.evolution.cutensornet as ctn
from edmtn.driver import EDMSolver, SolverConfig
from edmtn.evolution.cutensornet import _make_expander, build_2d_network
from edmtn.evolution.separable_bath import SeparableBathEvolution
from edmtn.expansion.first_order import FirstOrderExpander
from edmtn.kernels.separable_mpo import SeparableKernelEngine
from edmtn.models import GaudinModel, SpinBosonModel, validate_sub_baths


def _force_numpy(monkeypatch):
    real = ctn.solve_cutensornet
    monkeypatch.setattr(ctn, "solve_cutensornet",
                        lambda *a, **k: real(*a, **{**k, "executor": "numpy"}))


# -- the shared helper -----------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, 4, 2.9, True, "3"])
def test_validate_sub_baths_rejects(bad):
    with pytest.raises(ValueError):
        validate_sub_baths(bad, 3)


def test_validate_sub_baths_accepts_and_normalizes():
    assert validate_sub_baths(None, 3) == 3          # None -> all K
    assert validate_sub_baths(1, 3) == 1
    assert validate_sub_baths(3, 3) == 3
    v = validate_sub_baths(np.int64(2), 3)
    assert v == 2 and type(v) is int                 # NumPy int -> Python int


# -- driver paths: over-K rejected on BOTH tracks (no silent all-K) --------

@pytest.mark.parametrize("backend", [None, "hpc"], ids=["track1", "track2"])
def test_solve_rejects_sub_baths_over_K(monkeypatch, backend):
    if backend == "hpc":
        _force_numpy(monkeypatch)
    kw = dict(T=0.2, eps=0.1, expansion_order=2, sub_baths=4)   # K=3 -> 4 is out of range
    if backend:
        kw["backend"] = backend
    with pytest.raises(ValueError):
        EDMSolver.from_model(GaudinModel(g=1.0, K=3), **kw).solve(channel=3)


# -- direct evolution entry points: no silent clamp ------------------------

def test_separable_run_rejects_over_K():
    model = GaudinModel(g=1.0, K=3)
    eng = SeparableKernelEngine.from_model(model, T=0.2, eps=0.1)
    with pytest.raises(ValueError):
        SeparableBathEvolution(expander=FirstOrderExpander()).run(model, eng, 0.1, 2, sub_baths=99)


def test_build_2d_network_rejects_over_K():
    with pytest.raises(ValueError):
        build_2d_network(GaudinModel(g=1.0, K=3), _make_expander(2), 0.1, 2, sub_baths=99)


def test_direct_cutensornet_sub_baths_fail_fast(monkeypatch):
    made, rdm = [], []
    monkeypatch.setattr(ctn, "_make_expander", lambda o: made.append(o))
    monkeypatch.setattr(ctn, "reduced_density_matrix", lambda *a, **k: rdm.append(1))
    cfg = SolverConfig(eps=0.1, T=0.2, backend="hpc", sub_baths=4)
    with pytest.raises(ValueError):
        ctn.solve_cutensornet(GaudinModel(g=1.0, K=3), cfg, channel=3, executor="numpy")
    assert made == [] and rdm == []           # no expander / contraction started


# -- the actual folded L is recorded (None can't masquerade as K) ----------

@pytest.mark.parametrize("backend", [None, "hpc"], ids=["track1", "track2"])
@pytest.mark.parametrize("sub_baths,expected", [(2, 2), (None, 3)])
def test_sub_baths_used_recorded(monkeypatch, backend, sub_baths, expected):
    if backend == "hpc":
        _force_numpy(monkeypatch)
    kw = dict(T=0.2, eps=0.1, expansion_order=2, sub_baths=sub_baths)
    if backend:
        kw["backend"] = backend
    res = EDMSolver.from_model(GaudinModel(g=1.0, K=3), **kw).solve(channel=3)
    assert res.sub_baths_used == expected


def test_sub_baths_used_none_for_spinboson():
    res = EDMSolver.from_model(SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0),
                               T=0.2, eps=0.1, expansion_order=2).solve()
    assert res.sub_baths_used is None


def test_track2_sub_baths_used_matches_network_meta(monkeypatch):
    _force_numpy(monkeypatch)
    model = GaudinModel(g=1.0, K=3)
    out = ctn.solve_cutensornet(model, SolverConfig(eps=0.1, T=0.2, backend="hpc", sub_baths=2),
                                channel=3, executor="numpy")
    *_, meta = build_2d_network(model, _make_expander(2), 0.1, 2, sub_baths=2)
    assert out["sub_baths_used"] == meta["n_fold"] == 2


# -- convergence metadata records the ACTUAL L (from the results) ----------

@pytest.mark.parametrize("sub_baths,expected", [(2, 2), (None, 3)])
def test_convergence_records_actual_sub_baths(sub_baths, expected):
    solver = EDMSolver.from_model(GaudinModel(g=1.0, K=3), T=0.4, eps=0.1,
                                  expansion_order=2, cutoff=0.0, sub_baths=sub_baths)
    m = solver.timestep_convergence(channel=3).metadata
    assert m["coarse_sub_baths_used"] == expected
    assert m["fine_sub_baths_used"] == expected


def test_convergence_sub_baths_none_for_spinboson():
    solver = EDMSolver.from_model(SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0),
                                  T=0.4, eps=0.1, expansion_order=2)
    m = solver.timestep_convergence().metadata
    assert m["coarse_sub_baths_used"] is None and m["fine_sub_baths_used"] is None
