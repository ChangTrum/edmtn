"""Order resolution (P0-4): one effective Trotter order across config / model / kernel /
expander / result.

The config default (``expansion_order=None``) inherits the model's ``time_step_order``;
an explicit config value overrides it. The resolved order is used identically by the
kernel, the expander, the site count, and the recorded ``result.expansion_order`` -- so the
result metadata always matches the algorithm that produced it. Verified for spin-boson,
Gaudin (Track 1) and Gaudin (Track 2 / hpc via the NumPy executor). The inherit/override
matrix uses BOTH model orders so nothing can pass by hard-coding the default to 2.
"""

from __future__ import annotations

import pytest

import edmtn.evolution.cutensornet as ctn
from edmtn.driver import EDMSolver, SolverConfig, build_pipeline
from edmtn.driver.auto_config import resolve_expansion_order
from edmtn.evolution.cutensornet import build_2d_network
from edmtn.models import GaudinModel, SpinBosonModel

# (model_order, config_order, resolved) -- covers inherit both ways + override both ways
_MATRIX = [(1, None, 1), (2, None, 2), (2, 1, 1), (1, 2, 2)]


def _spin_boson(order):
    return SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0, time_step_order=order)


def _gaudin(order):
    return GaudinModel(g=1.0, K=2, time_step_order=order)


# -- the resolve helper + config default -----------------------------------

@pytest.mark.parametrize("model_order,cfg_order,resolved", _MATRIX)
def test_resolve_expansion_order(model_order, cfg_order, resolved):
    cfg = SolverConfig(eps=0.1, T=0.2, expansion_order=cfg_order)
    assert resolve_expansion_order(_gaudin(model_order), cfg) == resolved


def test_config_default_order_is_none():
    assert SolverConfig(eps=0.1, T=0.2).expansion_order is None


# -- Track 1: config / expander / result / site count all agree ------------

@pytest.mark.parametrize("make", [_spin_boson, _gaudin], ids=["spinboson", "gaudin"])
@pytest.mark.parametrize("model_order,cfg_order,resolved", _MATRIX)
def test_track1_single_order_everywhere(make, model_order, cfg_order, resolved):
    model = make(model_order)
    n_steps = 3
    solver = EDMSolver.from_model(model, T=0.3, eps=0.1, expansion_order=cfg_order)
    assert solver.config.expansion_order == resolved
    assert solver.evolution.expander.order == resolved
    # Gaudin: read S_z (channel 3), the decaying signal; <S_x>/<S_y> are ~0 by symmetry
    # and would trip the extractor's relative imaginary-part guard. Spin-boson: S_z is ch 1.
    channel = 3 if isinstance(model, GaudinModel) else 1
    res = solver.solve(channel=channel)
    assert res.expansion_order == resolved
    assert res.mps.num_sites == resolved * n_steps          # order-2 doubles the sites


@pytest.mark.parametrize("model_order,cfg_order,resolved", _MATRIX)
def test_gaussian_kernel_matches_expander(model_order, cfg_order, resolved):
    # the Gaussian kernel carries its own order; it must equal the expander's
    solver = EDMSolver.from_model(_spin_boson(model_order), T=0.3, eps=0.1,
                                  expansion_order=cfg_order)
    assert solver.kernel_engine.order == solver.evolution.expander.order == resolved


def test_separable_kernel_has_no_order_field():
    # Gaudin's separable kernel is order-free (order lives only in the expander), so there
    # is nothing for it to disagree with -- documented so the missing attribute is intended
    solver = EDMSolver.from_model(_gaudin(2), T=0.3, eps=0.1)
    assert not hasattr(solver.kernel_engine, "order")
    assert solver.evolution.expander.order == solver.config.expansion_order == 2


# -- build_pipeline is a public entry point and must resolve too -----------

@pytest.mark.parametrize("make,has_kernel_order", [(_spin_boson, True), (_gaudin, False)],
                         ids=["spinboson", "gaudin"])
def test_build_pipeline_inherits_model_order(make, has_kernel_order):
    model = make(2)
    cfg = SolverConfig(eps=0.1, T=0.2)            # expansion_order defaults to None
    assert cfg.expansion_order is None
    kernel, evolution = build_pipeline(model, cfg)
    assert evolution.expander.order == model.time_step_order
    if has_kernel_order:
        assert kernel.order == model.time_step_order
    # the solver stores a resolved config; the original frozen config is untouched
    solver = EDMSolver(model, cfg)
    assert solver.config.expansion_order == model.time_step_order
    assert cfg.expansion_order is None


# -- Track 2 (hpc) via the NumPy executor (no GPU/cuQuantum needed) ---------

@pytest.mark.parametrize("model_order,cfg_order,resolved", _MATRIX)
def test_track2_records_resolved_order(monkeypatch, model_order, cfg_order, resolved):
    real = ctn.solve_cutensornet
    monkeypatch.setattr(ctn, "solve_cutensornet",
                        lambda *a, **k: real(*a, **{**k, "executor": "numpy"}))
    model = _gaudin(model_order)
    res = EDMSolver.from_model(model, T=0.2, eps=0.1, backend="hpc",
                               expansion_order=cfg_order).solve(channel=3)
    assert res.expansion_order == resolved


@pytest.mark.parametrize("order", [1, 2])
def test_track2_site_count_doubles(order):
    from edmtn.evolution.cutensornet import _make_expander  # noqa: PLC0415
    *_, meta = build_2d_network(_gaudin(2), _make_expander(order), 0.1, 3, sub_baths=None)
    assert meta["n_sites"] == order * 3


# -- direct solve_cutensornet() fallback (bypasses EDMSolver's resolution) --
# guards the duplicated Layer-5 resolution that keeps the two paths from drifting

@pytest.mark.parametrize("model_order", [1, 2])
def test_direct_cutensornet_inherits_model_order(monkeypatch, model_order):
    model = _gaudin(model_order)
    cfg = SolverConfig(eps=0.1, T=0.2, backend="hpc")
    assert cfg.expansion_order is None                       # not pre-resolved
    seen = {}
    real_make = ctn._make_expander

    def spy(order):
        seen["order"] = order
        return real_make(order)

    monkeypatch.setattr(ctn, "_make_expander", spy)
    out = ctn.solve_cutensornet(model, cfg, channel=3, executor="numpy")
    assert seen["order"] == model_order                      # inherited model order, no EDMSolver
    assert len(out["times"]) == cfg.n_steps


@pytest.mark.parametrize("bad_order", [True, 1.0, 0, 3])
def test_direct_cutensornet_rejects_bad_model_order(bad_order):
    model = _gaudin(2)
    model.time_step_order = bad_order                        # bypass GaudinModel validation
    cfg = SolverConfig(eps=0.1, T=0.2, backend="hpc")
    with pytest.raises(ValueError):
        ctn.solve_cutensornet(model, cfg, channel=3, executor="numpy")
