"""Convergence-order anchor: the Dicke discretisation against a *continuous-time* reference.

Deliberately separate from ``test_dicke_evolution.py``.  That module's reference applies
the **same** discretisation as the pipeline, so agreement there proves the tensor
contraction and says nothing about accuracy.  Here the reference is the solution of the
continuous-time master equation (``solve_ivp``/DOP853 at ``rtol=1e-13``, ``atol=1e-15``),
so the error really is discretisation error and its scaling in ``eps`` is measurable.

What is measured, and under which conditions:

* fixed finite Fock truncation, held constant while ``eps`` is refined;
* ``cutoff = 0`` with ``max_bond = None``: the compression sweep still **runs**, as an
  exact no-discard recompression -- that is not the same as skipping compression, but it
  discards nothing, so no truncation error mixes into the measurement;
* error norm: maximum elementwise absolute deviation of the final **reduced** (cavity)
  density matrix;
* four step sizes, least-squares fit of ``log(error)`` against ``log(eps)``;
* the smallest error is ~3e-6, several orders above the reference solver's own ~1e-13
  floor, so the fit is not measuring round-off.

Measured on this configuration: **order 1 -> 1.02**, **order 2 -> 1.97**.  The gates below
sit a little under those, since the fitted value approaches its limit from below as the
window is refined.  Sensitivity of the result to the *sampling point* is established
separately in ``test_dicke_evolution.test_the_anchor_is_sensitive_to_the_sampling_point``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from edmtn.driver import solve
from edmtn.models import DickeModel

_I2 = np.eye(2, dtype=np.complex128)
_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_SY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_SZ = np.diag([1.0, -1.0]).astype(np.complex128)
_SP = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
_SM = _SP.conj().T

T_FINAL = 0.6
STEP_COUNTS = (8, 16, 32, 64)          # eps = T/N, well inside the asymptotic regime

BASE = dict(K=2, n_fock=3, coupling=[0.35, -0.22], omega=[0.8, 1.3], omega_c=1.1,
            cavity_state="coherent", cavity_params={"alpha": 0.3 + 0.2j},
            bath_state=[[0.2, -0.3, 0.5], [0.0, 0.4, -0.4]])
RATES = dict(kappa=0.23, pump=[0.05, 0.02], emission=[0.11, 0.07], dephasing=[0.04, 0.09])


def _kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def _continuous_reference(model, T):
    """Solve the continuous-time master equation and reduce to the cavity."""
    d, K = model.n_fock, model.K
    bp = model.bath_params()
    dim = d * 2 ** K
    Id = np.eye(dim, dtype=np.complex128)
    a1 = np.diag(np.sqrt(np.arange(1, d, dtype=np.float64)), 1).astype(np.complex128)
    cav = lambda o: _kron_all([o] + [_I2] * K)                              # noqa: E731
    spin = lambda k, o: _kron_all(                                          # noqa: E731
        [np.eye(d, dtype=np.complex128)] + [o if j == k else _I2 for j in range(K)])
    a = cav(a1)

    def dissipator(L):
        M = L.conj().T @ L
        return np.kron(L, L.conj()) - 0.5 * (np.kron(M, Id) + np.kron(Id, M.T))

    LD = model.kappa * dissipator(a)
    for k in range(K):
        LD += (bp.pump[k] * dissipator(spin(k, _SP))
               + bp.emission[k] * dissipator(spin(k, _SM))
               + 0.5 * bp.dephasing[k] * dissipator(spin(k, _SZ)))

    def generator(t):
        S = a * np.exp(-1j * model.omega_c * t) + a.conj().T * np.exp(1j * model.omega_c * t)
        B = sum(bp.couplings[k] * (np.cos(bp.omegas[k] * t) * spin(k, _SX)
                                   - np.sin(bp.omegas[k] * t) * spin(k, _SY))
                for k in range(K))
        H = S @ B
        return -1j * (np.kron(H, Id) - np.kron(Id, H.T)) + LD

    rho0 = _kron_all([model.initial_system_state()]
                     + [0.5 * (_I2 + r[0] * _SX + r[1] * _SY + r[2] * _SZ)
                        for r in model.bath_bloch_vectors()])
    sol = solve_ivp(lambda t, y: generator(t) @ y, [0.0, T], rho0.reshape(-1),
                    rtol=1e-13, atol=1e-15, method="DOP853")
    full = sol.y[:, -1].reshape(dim, dim).reshape(d, 2 ** K, d, 2 ** K)
    return np.einsum("imjm->ij", full)


_REFERENCE_CACHE: dict = {}


def _reference(dissipative):
    key = bool(dissipative)
    if key not in _REFERENCE_CACHE:
        kw = dict(BASE, **RATES) if dissipative else dict(BASE)
        _REFERENCE_CACHE[key] = _continuous_reference(DickeModel(**kw), T_FINAL)
    return _REFERENCE_CACHE[key]


def _error_curve(order, dissipative):
    kw = dict(BASE, **RATES) if dissipative else dict(BASE)
    model = DickeModel(**kw)
    ref = _reference(dissipative)
    eps_values, errors = [], []
    for N in STEP_COUNTS:
        eps = T_FINAL / N
        res = solve(model, T=T_FINAL, eps=eps, expansion_order=order,
                    cutoff=0.0, max_bond=None)          # nothing discarded
        errors.append(float(np.max(np.abs(np.asarray(res.final_density_matrix) - ref))))
        eps_values.append(eps)
    return np.asarray(eps_values), np.asarray(errors)


def _fitted_order(eps_values, errors):
    return float(np.polyfit(np.log(eps_values), np.log(errors), 1)[0])


@pytest.mark.parametrize("dissipative", [False, True], ids=["closed", "dissipative"])
def test_first_order_is_globally_first_order(dissipative):
    eps_values, errors = _error_curve(1, dissipative)
    p = _fitted_order(eps_values, errors)
    assert 0.85 <= p <= 1.25, f"fitted order {p:.3f}, errors {errors}"


@pytest.mark.parametrize("dissipative", [False, True], ids=["closed", "dissipative"])
def test_second_order_is_globally_second_order(dissipative):
    """Midpoint sampling plus Strang placement; either one alone caps this at 1."""
    eps_values, errors = _error_curve(2, dissipative)
    p = _fitted_order(eps_values, errors)
    assert p >= 1.9, f"fitted order {p:.3f}, errors {errors}"
    # and the trend is towards 2 from below, not away from it
    pairwise = np.log(errors[:-1] / errors[1:]) / np.log(2.0)
    assert pairwise[-1] > pairwise[0], f"pairwise orders {pairwise}"


def test_second_order_is_far_more_accurate_than_first_at_the_same_step():
    """A direct consequence of the order gap; fails if order 2 silently degrades to 1."""
    _, e1 = _error_curve(1, True)
    _, e2 = _error_curve(2, True)
    assert np.all(e2 < e1)
    assert e2[-1] < e1[-1] / 100.0


def test_the_measurement_is_above_the_reference_error_floor():
    """The fit must not be reading round-off: the smallest error stays far above ~1e-13."""
    _, errors = _error_curve(2, True)
    assert errors.min() > 1e-9
    assert errors.max() < 1e-2                # ... and the coarse end is still a small error


def test_reference_is_trace_preserving_and_differs_between_the_two_configurations():
    """Guards the reference itself, and that turning the rates on actually changes it."""
    closed, damped = _reference(False), _reference(True)
    for rho in (closed, damped):
        assert abs(complex(np.trace(rho)).real - 1.0) < 1e-9
    assert np.max(np.abs(closed - damped)) > 1e-3
