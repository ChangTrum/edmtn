"""Model validation + the automatic solver-side check (P1-12).

``EDMSolver`` now calls ``model.validate()`` in ``__init__`` -- BEFORE resolving the config or
building any pipeline/kernel -- so a malformed model fails loudly at construction on both
tracks instead of surfacing as a deep cumulant/kernel error.  ``validate()`` itself is extended
to check finiteness, positive-semidefiniteness, a real unit trace, a positive-integer
dimension and a non-empty coupling set, always raising ``ValueError`` (never a leaked
``AttributeError`` / ``TypeError`` / ``LinAlgError``).  Temperature stays an engine-capability
gate (``NotImplementedError``) but now admits only real ``+inf``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import edmtn.driver.solver as solver_mod
import edmtn.evolution.cutensornet as ctn
from edmtn.cumulants.separable import SeparableBathCorrelation
from edmtn.driver import EDMSolver
from edmtn.driver.auto_config import SolverConfig
from edmtn.models import GaudinModel, SpinBosonModel


def _sb():
    return SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0)


def _gaudin():
    return GaudinModel(g=0.8, K=2)


# -- malformed models: override exactly one method to inject one defect ----------------------

class _NanH(SpinBosonModel):
    def system_hamiltonian(self):
        H = np.array(super().system_hamiltonian(), dtype=complex)
        H[0, 0] = np.nan
        return H


class _InfH(SpinBosonModel):
    def system_hamiltonian(self):
        H = np.array(super().system_hamiltonian(), dtype=complex)
        H[0, 0] = np.inf
        return H


class _NonNumericH(SpinBosonModel):
    def system_hamiltonian(self):
        return np.array([["a", "b"], ["c", "d"]])  # string dtype, (2, 2)


class _NanRho(SpinBosonModel):
    def initial_system_state(self):
        r = np.array(super().initial_system_state(), dtype=complex)
        r[0, 0] = np.nan
        return r


class _NonNumericRho(SpinBosonModel):
    def initial_system_state(self):
        return np.array([[object(), 0], [0, object()]], dtype=object)


class _TraceRealOff(SpinBosonModel):
    def initial_system_state(self):
        return np.array([[0.7, 0.0], [0.0, 0.7]], dtype=complex)  # trace 1.4


class _TraceImag(SpinBosonModel):
    def initial_system_state(self):
        return np.array([[0.5 + 0.2j, 0.0], [0.0, 0.5]], dtype=complex)  # trace 1 + 0.2i


class _NonPSDRho(SpinBosonModel):
    def initial_system_state(self):
        return np.array([[1.1, 0.0], [0.0, -0.1]], dtype=complex)  # trace 1, min eig -0.1


class _TinyNegEigRho(SpinBosonModel):
    def initial_system_state(self):
        return np.array([[1.0 + 1e-9, 0.0], [0.0, -1e-9]], dtype=complex)  # min eig -1e-9 (allowed)


class _InfCoupling(SpinBosonModel):
    def coupling_operators(self):
        S = np.array(super().coupling_operators()[0], dtype=complex)
        S[0, 0] = np.inf
        return [S]


class _EmptyCoupling(SpinBosonModel):
    def coupling_operators(self):
        return []


class _NonIterableCoupling(SpinBosonModel):
    def coupling_operators(self):
        return 42  # not iterable


class _BadDimZero(SpinBosonModel):
    @property
    def system_dim(self):
        return 0


class _BadDimBool(SpinBosonModel):
    @property
    def system_dim(self):
        return True


class _BadDimFloat(SpinBosonModel):
    @property
    def system_dim(self):
        return 1.0


class _NpIntDim(SpinBosonModel):
    @property
    def system_dim(self):
        return np.int64(2)  # a genuine integer -> must PASS


# malformed variants of BOTH base models (for the solver-construction test)
class _NanH_Gaudin(GaudinModel):
    def system_hamiltonian(self):
        H = np.array(super().system_hamiltonian(), dtype=complex)
        H[0, 0] = np.nan
        return H


# -- validate() rejects each malformed model with a field-named ValueError -------------------

@pytest.mark.parametrize("factory,match", [
    (lambda: _BadDimZero(J0=0.5, omega_c=5.0, mu=1.0), "system_dim"),
    (lambda: _BadDimBool(J0=0.5, omega_c=5.0, mu=1.0), "system_dim"),
    (lambda: _BadDimFloat(J0=0.5, omega_c=5.0, mu=1.0), "system_dim"),
    (lambda: _NanH(J0=0.5, omega_c=5.0, mu=1.0), "system_hamiltonian"),
    (lambda: _InfH(J0=0.5, omega_c=5.0, mu=1.0), "system_hamiltonian"),
    (lambda: _NonNumericH(J0=0.5, omega_c=5.0, mu=1.0), "system_hamiltonian"),
    (lambda: _NanRho(J0=0.5, omega_c=5.0, mu=1.0), "initial_system_state"),
    (lambda: _NonNumericRho(J0=0.5, omega_c=5.0, mu=1.0), "initial_system_state"),
    (lambda: _TraceRealOff(J0=0.5, omega_c=5.0, mu=1.0), "trace"),
    (lambda: _TraceImag(J0=0.5, omega_c=5.0, mu=1.0), "imaginary"),
    (lambda: _NonPSDRho(J0=0.5, omega_c=5.0, mu=1.0), "eigenvalue"),
    (lambda: _EmptyCoupling(J0=0.5, omega_c=5.0, mu=1.0), "coupling_operators"),
    (lambda: _NonIterableCoupling(J0=0.5, omega_c=5.0, mu=1.0), "coupling_operators"),
    (lambda: _InfCoupling(J0=0.5, omega_c=5.0, mu=1.0), "coupling operator 0"),
])
def test_validate_rejects_malformed(factory, match):
    with pytest.raises(ValueError, match=match):
        factory().validate()


def test_validate_accepts_normal_models():
    _sb().validate()
    _gaudin().validate()


def test_validate_accepts_numpy_int_dim():
    _NpIntDim(J0=0.5, omega_c=5.0, mu=1.0).validate()


def test_validate_psd_tolerance_boundary():
    # min eigenvalue just inside the tolerance floor is allowed ...
    _TinyNegEigRho(J0=0.5, omega_c=5.0, mu=1.0).validate()
    # ... but a genuine negative eigenvalue is rejected
    with pytest.raises(ValueError, match="eigenvalue"):
        _NonPSDRho(J0=0.5, omega_c=5.0, mu=1.0).validate()


# -- EDMSolver runs validate() at construction, before the pipeline --------------------------

def _spy_pipeline(monkeypatch):
    calls = {"resolve": 0, "build": 0}
    monkeypatch.setattr(solver_mod, "resolve_config_for_model",
                        lambda *a, **k: calls.__setitem__("resolve", calls["resolve"] + 1))
    monkeypatch.setattr(solver_mod, "build_pipeline",
                        lambda *a, **k: calls.__setitem__("build", calls["build"] + 1))
    return calls


@pytest.mark.parametrize("model", [
    _NanH(J0=0.5, omega_c=5.0, mu=1.0),          # malformed spin-boson (Track 1: gaussian)
    _NanH_Gaudin(g=0.8, K=2),                    # malformed Gaudin (Track 1: separable)
])
def test_track1_solver_rejects_before_pipeline(monkeypatch, model):
    calls = _spy_pipeline(monkeypatch)
    with pytest.raises(ValueError):
        EDMSolver(model, SolverConfig(eps=0.1, T=0.3))
    assert calls == {"resolve": 0, "build": 0}   # neither config-resolve nor pipeline reached


def test_track2_solver_rejects_malformed_at_construction(monkeypatch):
    # hpc defers kernel build to solve(), but a MALFORMED model must still be rejected at
    # construction (validate() runs before the hpc branch) -- not deferred to .solve()
    calls = _spy_pipeline(monkeypatch)
    with pytest.raises(ValueError):
        EDMSolver(_NanH_Gaudin(g=0.8, K=2), SolverConfig(eps=0.1, T=0.2, backend="hpc"))
    assert calls == {"resolve": 0, "build": 0}


def test_timestep_convergence_survives_double_validation():
    # coarse + fine each build an EDMSolver (validate() runs twice); result still computes
    solver = EDMSolver.from_model(_gaudin(), T=0.3, eps=0.1)
    tc = solver.timestep_convergence(tol=1e-3, channel=3)
    assert tc.deviation >= 0.0


# -- temperature: engine gate admits only real +inf, on both tracks --------------------------

def _gaudin_temp(temp):
    class _M(GaudinModel):
        def bath_params(self):
            return replace(super().bath_params(), temperature=temp)
    return _M(g=0.8, K=2)


def _force_numpy(monkeypatch):
    real = ctn.solve_cutensornet
    monkeypatch.setattr(ctn, "solve_cutensornet",
                        lambda *a, **k: real(*a, **{**k, "executor": "numpy"}))


def test_separable_compute_accepts_pos_inf():
    corr = SeparableBathCorrelation().compute(_gaudin_temp(np.inf), T=0.5, eps=0.1)
    assert corr.K == 2


@pytest.mark.parametrize("temp", [-np.inf, 50.0, np.nan, True])
def test_separable_compute_rejects_bad_temperature(temp):
    with pytest.raises(NotImplementedError, match="temperature"):
        SeparableBathCorrelation().compute(_gaudin_temp(temp), T=0.5, eps=0.1)


@pytest.mark.parametrize("temp", [-np.inf, 50.0, np.nan])
def test_track1_solver_rejects_bad_temperature(temp):
    # Track 1 builds the kernel in __init__, so the temperature gate fires at construction
    with pytest.raises(NotImplementedError, match="temperature"):
        EDMSolver.from_model(_gaudin_temp(temp), T=0.4, eps=0.1)


@pytest.mark.parametrize("temp", [-np.inf, 50.0, np.nan])
def test_track2_solver_rejects_bad_temperature_at_solve(monkeypatch, temp):
    # Track 2 defers the kernel to solve(); construction succeeds (validate passes), the
    # temperature gate fires in the deferred cuTensorNet build (NumPy executor, no GPU)
    _force_numpy(monkeypatch)
    solver = EDMSolver.from_model(_gaudin_temp(temp), T=0.2, eps=0.1, backend="hpc")
    with pytest.raises(NotImplementedError, match="temperature"):
        solver.solve(channel=3)
