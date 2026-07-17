"""SolverResult axis semantics (P0-8).

Every array field names its own horizontal axis, so a caller never has to inspect the
internal ``evolution`` object to know what an index means:
  * ``density_matrices``               -> rho(t), aligned 1:1 with ``times`` (else None)
  * ``time_bond_dims``                 -> max bond per physical time step (aligned with ``times``)
  * ``sub_bath_counts/_bond_dims``     -> the separable Track-1 fold axis L, D_L
  * ``sub_bath_final_density_matrices``-> rho_L(T) on the fold axis
  * ``final_time_bond_dims``           -> the final EDM-MPS internal bonds along the time chain
Track 1's per-L states (rho_L(T)) must NEVER leak into the top-level ``density_matrices``.
"""

from __future__ import annotations

import pytest

import edmtn.evolution.cutensornet as ctn
from edmtn.driver import EDMSolver
from edmtn.models import GaudinModel, SpinBosonModel


def _sb():
    return SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0)


def _force_numpy(monkeypatch):
    real = ctn.solve_cutensornet
    monkeypatch.setattr(ctn, "solve_cutensornet",
                        lambda *a, **k: real(*a, **{**k, "executor": "numpy"}))


# -- spin-boson: density_matrices exposed whenever rho(t) is ACTUALLY available --------

def test_spinboson_order1_no_record_rho_has_no_density():
    res = EDMSolver.from_model(_sb(), T=0.3, eps=0.1, expansion_order=1, cutoff=1e-6).solve()
    assert res.density_matrices is None
    assert len(res.time_bond_dims) == len(res.times)


def test_spinboson_order1_record_rho_exposes_density():
    res = EDMSolver.from_model(_sb(), T=0.3, eps=0.1, expansion_order=1,
                               cutoff=1e-6, record_rho=True).solve()
    assert len(res.density_matrices) == len(res.times)
    assert len(res.time_bond_dims) == len(res.times)


def test_spinboson_order2_exposes_density_without_record_rho():
    # 2nd order records rho(t) internally to build the polarization; the field must expose
    # it as-is, NOT `ev.density_matrices if cfg.record_rho else None`
    res = EDMSolver.from_model(_sb(), T=0.3, eps=0.1, expansion_order=2, cutoff=1e-6).solve()
    assert len(res.density_matrices) == len(res.times)


def test_spinboson_custom_observable_exposes_density():
    m = _sb()
    res = EDMSolver.from_model(m, T=0.3, eps=0.1, expansion_order=1, cutoff=1e-6).solve(
        observables={"Sz": lambda t: m.coupling_operators_at(t)[0]})
    assert len(res.density_matrices) == len(res.times)


def test_spinboson_field_mapping():
    res = EDMSolver.from_model(_sb(), T=0.3, eps=0.1, expansion_order=2,
                               cutoff=1e-6, record_rho=True).solve()
    assert res.time_bond_dims == res.evolution.bond_dims
    assert res.final_time_bond_dims == res.mps.bond_dims
    assert len(res.time_bond_dims) == len(res.times)
    assert len(res.density_matrices) == len(res.times)
    assert res.sub_bath_counts is None
    assert res.sub_bath_bond_dims is None
    assert res.sub_bath_final_density_matrices is None
    assert res.bond_dims == res.time_bond_dims          # legacy alias here


# -- Gaudin Track 1: per-L records live only in sub_bath_*; density_matrices is None ---

@pytest.mark.parametrize("record_rho", [True, False])
def test_gaudin_track1_axes(record_rho):
    model = GaudinModel(g=0.8, K=3)
    res = EDMSolver.from_model(model, T=0.3, eps=0.1, expansion_order=2,
                               cutoff=0.0, record_rho=record_rho).solve(channel=3)
    assert res.density_matrices is None                 # rho_L(T) is NOT rho(t)
    assert res.time_bond_dims is None
    assert res.sub_bath_counts == res.evolution.recorded_L
    assert res.sub_bath_bond_dims == res.evolution.bond_dims
    assert len(res.sub_bath_counts) == len(res.sub_bath_bond_dims)
    assert res.final_time_bond_dims == res.mps.bond_dims
    assert res.bond_dims == res.sub_bath_bond_dims       # legacy alias here
    if record_rho:
        assert len(res.sub_bath_final_density_matrices) == len(res.sub_bath_counts)
    else:
        assert res.sub_bath_final_density_matrices is None


# -- Gaudin Track 2 (hpc via NumPy executor): rho(t) only in density_matrices ----------

def test_gaudin_track2_axes(monkeypatch):
    _force_numpy(monkeypatch)
    model = GaudinModel(g=0.8, K=2)
    res = EDMSolver.from_model(model, T=0.2, eps=0.1, expansion_order=2,
                               backend="hpc").solve(channel=3)
    assert len(res.density_matrices) == len(res.times)
    assert res.time_bond_dims is None
    assert res.sub_bath_counts is None
    assert res.sub_bath_bond_dims is None
    assert res.sub_bath_final_density_matrices is None
    assert res.final_time_bond_dims is None
    assert res.bond_dims == []


# -- truncation_errors are honest None placeholders, not fabricated zeros (P0-9) --------

def test_spinboson_truncation_errors_are_none_placeholders():
    # cutoff>0 + a real bond cap: truncation genuinely happens, yet the field is None
    # ("not measured"), never a fabricated 0.0 ("confirmed lossless")
    res = EDMSolver.from_model(_sb(), T=0.3, eps=0.1, expansion_order=2,
                               cutoff=1e-6, max_bond=8).solve()
    assert len(res.truncation_errors) == len(res.times)
    assert all(x is None for x in res.truncation_errors)
    assert res.truncation_errors == res.evolution.truncation_errors


def test_gaudin_track1_truncation_errors_are_none_placeholders():
    model = GaudinModel(g=0.8, K=3)
    res = EDMSolver.from_model(model, T=0.3, eps=0.1, expansion_order=2,
                               cutoff=1e-6, max_bond=32).solve(channel=3)
    assert len(res.truncation_errors) == len(res.sub_bath_counts)
    assert all(x is None for x in res.truncation_errors)
    assert res.truncation_errors == res.evolution.truncation_errors


def test_gaudin_track2_truncation_errors_empty(monkeypatch):
    _force_numpy(monkeypatch)
    res = EDMSolver.from_model(GaudinModel(g=0.8, K=2), T=0.2, eps=0.1,
                               expansion_order=2, backend="hpc").solve(channel=3)
    assert res.truncation_errors == []
