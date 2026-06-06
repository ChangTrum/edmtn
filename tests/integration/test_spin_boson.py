"""Integration test: spin-boson dynamics vs the EDM paper (Fig. 4).

A full-resolution reproduction of Fig. 4 (``eps = 0.01``, ``mu t`` up to ~17,
six coupling strengths) is far too expensive for the test suite -- the EDM
algorithm is inherently ``O(N^2)`` in the number of steps.  ``examples/
reproduce_fig4.py`` runs that at publication settings.

Here we verify, at a moderate (still second-order) resolution, the *qualitative*
physics the paper reports:

* the spin starts fully polarised, ``<S_z(0)> = 1/2``;
* the relaxation crosses over from **damped oscillation** (weak coupling, the
  curve dips below zero) to **overdamped decay** (strong coupling, no dip) near
  ``J0 ~ 0.5`` (Fig. 4a);
* ``<S_z(t)>`` stays physical, ``|<S_z>| <= 1/2`` (second order, no first-order
  overshoot);
* the EDM bond dimension grows at most linearly and then saturates (Fig. 4b).
"""

import numpy as np
import pytest

from edmtn.driver import EDMSolver
from edmtn.models import SpinBosonModel

pytestmark = pytest.mark.integration


def _solve(J0, T=4.0, eps=0.04):
    model = SpinBosonModel(J0=J0, omega_c=5.0, mu=1.0)
    return EDMSolver.from_model(
        model, T=T, eps=eps, expansion_order=2, cutoff=1e-5
    ).solve()


def test_weak_coupling_oscillates():
    res = _solve(0.1)
    v = res.polarization
    assert np.isclose(v[0], 0.5, atol=3e-2)
    # damped oscillation: the curve dips below zero
    assert v.min() < -0.02
    assert np.all(v <= 0.5 + 5e-3) and np.all(v >= -0.5 - 5e-3)


def test_strong_coupling_overdamped():
    res = _solve(1.0)
    v = res.polarization
    assert np.isclose(v[0], 0.5, atol=3e-2)
    # overdamped: no oscillation below zero, stays physical
    assert v.min() > -0.01
    assert np.all(v <= 0.5 + 5e-3)


def test_crossover_between_regimes():
    # weak dips well below the strong-coupling minimum
    weak = _solve(0.1).polarization
    strong = _solve(1.0).polarization
    assert weak.min() < strong.min() - 0.05


def test_bond_dimension_linear_then_saturates():
    res = _solve(0.7)
    bonds = np.asarray(res.bond_dims)
    # at most linear: per-step increment bounded by a constant (d^2 * d_phys)
    cap = 2 ** 2 * 3
    assert np.diff(np.concatenate([[1], bonds])).max() <= cap
    # saturation: late-time growth is much slower than early-time growth
    early = bonds[len(bonds) // 4] - bonds[0]
    late = bonds[-1] - bonds[3 * len(bonds) // 4]
    assert late <= early
