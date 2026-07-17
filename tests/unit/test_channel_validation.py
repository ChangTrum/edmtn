"""Unified channel validation (P0-5).

A 1-based ``channel`` is validated once, with the SAME shared helper
(``models.base.validate_channel``), at all three public entry points -- the solver,
the HPC solve, and the observable extractor -- BEFORE any array indexing. So an
illegal channel (0, negative, out-of-range, bool, float, string) raises ``ValueError``
identically on every order/backend, instead of ``channel=0`` silently selecting the
last channel via a negative index, or a float/string leaking IndexError/TypeError.

The mapping ``channel c -> operators[c-1]`` is proven with sentinel operators (distinct,
predictable expectation values), not with Gaudin's real S_x/S_y (both ~0 by symmetry,
which would pass even if the two channels were swapped).
"""

from __future__ import annotations

import numpy as np
import pytest

import edmtn.evolution.cutensornet as ctn
from edmtn.driver import EDMSolver, SolverConfig
from edmtn.models import GaudinModel, SpinBosonModel, validate_channel
from edmtn.observables.extractor import ObservableExtractor

_SB_BAD = [0, -1, 1.0, 1.5, True, "3", 2]      # spin-boson: 1 channel, so 2 is out of range
_G_BAD = [0, -1, 1.0, 1.5, True, "3", 4]       # Gaudin: 3 channels, so 4 is out of range


def _spin_boson(order=2):
    return SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0, time_step_order=order)


def _gaudin(order=2):
    return GaudinModel(g=1.0, K=2, time_step_order=order)


def _force_numpy(monkeypatch):
    real = ctn.solve_cutensornet
    monkeypatch.setattr(ctn, "solve_cutensornet",
                        lambda *a, **k: real(*a, **{**k, "executor": "numpy"}))


# -- the shared helper itself ----------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, 1.0, 1.5, True, "3", None])
def test_validate_channel_rejects(bad):
    with pytest.raises(ValueError):
        validate_channel(bad, 3)


def test_validate_channel_range():
    with pytest.raises(ValueError):
        validate_channel(4, 3)
    with pytest.raises(ValueError):
        validate_channel(2, 1)


def test_validate_channel_normalizes_numpy_int():
    c = validate_channel(np.int64(2), 3)
    assert c == 2 and type(c) is int


# -- EDMSolver.solve, Track 1, both orders (spin-boson + Gaudin) -----------

@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("bad", _SB_BAD)
def test_solve_spinboson_rejects(order, bad):
    solver = EDMSolver.from_model(_spin_boson(order), T=0.2, eps=0.1)
    with pytest.raises(ValueError):
        solver.solve(channel=bad)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("bad", _G_BAD)
def test_solve_gaudin_rejects(order, bad):
    solver = EDMSolver.from_model(_gaudin(order), T=0.2, eps=0.1)
    with pytest.raises(ValueError):
        solver.solve(channel=bad)


# -- Gaudin Track 2 via the hpc path (numpy executor) ----------------------

@pytest.mark.parametrize("bad", _G_BAD)
def test_solve_hpc_rejects(monkeypatch, bad):
    _force_numpy(monkeypatch)
    solver = EDMSolver.from_model(_gaudin(2), T=0.2, eps=0.1, backend="hpc")
    with pytest.raises(ValueError):
        solver.solve(channel=bad)


# -- direct solve_cutensornet + fail-fast (nothing runs before the error) --

@pytest.mark.parametrize("bad", _G_BAD)
def test_direct_cutensornet_rejects_and_is_fail_fast(monkeypatch, bad):
    made, rdm = [], []
    monkeypatch.setattr(ctn, "_make_expander", lambda o: made.append(o))
    monkeypatch.setattr(ctn, "reduced_density_matrix", lambda *a, **k: rdm.append(1))
    cfg = SolverConfig(eps=0.1, T=0.2, backend="hpc")
    with pytest.raises(ValueError):
        ctn.solve_cutensornet(_gaudin(2), cfg, channel=bad, executor="numpy")
    assert made == [] and rdm == []          # no expander / contraction started


# -- direct extractor entry ------------------------------------------------

@pytest.fixture(scope="module")
def gaudin_mps():
    # a real Gaudin EDM-MPS (d_phys=7 -> 3 channels), built once
    return EDMSolver.from_model(_gaudin(2), T=0.2, eps=0.1).solve(channel=3).mps


@pytest.mark.parametrize("bad", [0, -1, 1.0, 1.5, True, "3", 4])
def test_extractor_rejects_bad_channel(gaudin_mps, bad):
    with pytest.raises(ValueError):
        ObservableExtractor.coupling_polarization_history(gaudin_mps, 0.1, channel=bad, order=2)


# -- fail-fast for EDMSolver.solve (evolution / Track-2 does not start) -----

def test_solve_track1_failfast(monkeypatch):
    solver = EDMSolver.from_model(_gaudin(2), T=0.2, eps=0.1)
    monkeypatch.setattr(solver.evolution, "run",
                        lambda *a, **k: pytest.fail("evolution.run must not start on a bad channel"))
    with pytest.raises(ValueError):
        solver.solve(channel=0)


def test_solve_hpc_failfast(monkeypatch):
    called = []
    monkeypatch.setattr(ctn, "solve_cutensornet", lambda *a, **k: called.append(1))
    solver = EDMSolver.from_model(_gaudin(2), T=0.2, eps=0.1, backend="hpc")
    with pytest.raises(ValueError):
        solver.solve(channel=0)
    assert called == []                       # Track-2 solve never dispatched


# -- valid channels run ----------------------------------------------------

def test_valid_channel_spinboson():
    res = EDMSolver.from_model(_spin_boson(2), T=0.2, eps=0.1).solve(channel=1)
    assert res.polarization is not None


def test_valid_channel_gaudin_track1():
    # channel 3 (S_z) is the decaying signal; S_x/S_y ~0 trip the extractor imaginary guard
    res = EDMSolver.from_model(_gaudin(2), T=0.2, eps=0.1).solve(channel=3)
    assert res.polarization is not None


def test_valid_numpy_int_channel():
    res = EDMSolver.from_model(_gaudin(2), T=0.2, eps=0.1).solve(channel=np.int64(3))
    assert res.polarization is not None


@pytest.mark.parametrize("c", [1, 2, 3])
def test_valid_channels_gaudin_track2(monkeypatch, c):
    _force_numpy(monkeypatch)
    res = EDMSolver.from_model(_gaudin(2), T=0.2, eps=0.1, backend="hpc").solve(channel=c)
    assert res.polarization is not None and len(res.polarization) == 2   # N = T/eps


def test_direct_cutensornet_channel_none_skips():
    cfg = SolverConfig(eps=0.1, T=0.2, backend="hpc")
    out = ctn.solve_cutensornet(_gaudin(2), cfg, channel=None, executor="numpy")
    assert out["polarization"] is None and out["density_matrices"] is not None


# -- mapping proof: channel c -> operators[c-1] (sentinel operators) --------

def test_track2_channel_maps_to_operator(monkeypatch):
    # three sentinel operators with distinct Tr[op @ rho] (rho = I/2): 5, 10, 15
    A = np.array([[10.0, 0], [0, 0]], dtype=complex)   # Tr[A @ I/2] = 5
    B = np.array([[0, 0], [0, 20.0]], dtype=complex)    # Tr[B @ I/2] = 10
    C = np.array([[30.0, 0], [0, 0]], dtype=complex)    # Tr[C @ I/2] = 15
    rho = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
    model = _gaudin(2)                                   # real coupling_operators() -> 3 channels
    monkeypatch.setattr(model, "coupling_operators_at", lambda t: [A, B, C])
    monkeypatch.setattr(ctn, "reduced_density_matrix",
                        lambda *a, **k: (rho, {"hermiticity": 0.0, "trace_dev": 0.0}))
    cfg = SolverConfig(eps=0.1, T=0.2, backend="hpc")
    for c, expected in {1: 5.0, 2: 10.0, 3: 15.0}.items():
        out = ctn.solve_cutensornet(model, cfg, channel=c, executor="numpy")
        assert np.allclose(out["polarization"], expected), f"channel {c} selected the wrong operator"
