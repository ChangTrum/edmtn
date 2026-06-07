"""Integration smoke test: the spin-boson (Gaussian) pipeline runs end-to-end.

Not a reproduction -- that lives in ``examples/reproduce_fig4.py``.  Here we just
check, at a small/fast resolution, that the whole stack (model -> cumulants ->
kernel -> expansion -> evolution -> observables -> driver) runs and returns
physically sane output for both expansion orders.
"""

import numpy as np
import pytest

from edmtn.driver import EDMSolver
from edmtn.models import SpinBosonModel

pytestmark = pytest.mark.integration


def _solve(J0, *, order, T=1.5, eps=0.05):
    model = SpinBosonModel(J0=J0, omega_c=5.0, mu=1.0)
    return EDMSolver.from_model(
        model, T=T, eps=eps, expansion_order=order, cutoff=1e-5
    ).solve()


@pytest.mark.parametrize("order", [1, 2])
def test_pipeline_runs_and_is_physical(order):
    res = _solve(0.5, order=order)
    n = res.times.size
    assert n > 0 and res.polarization.shape == (n,)
    # starts fully polarised; stays a physical spin-1/2 expectation
    assert np.isclose(res.polarization[0], 0.5, atol=5e-2)
    assert np.all(res.polarization <= 0.5 + 1e-2)
    assert np.all(res.polarization >= -0.5 - 1e-2)
    # bond dimension is positive and bounded by the per-step cap d**2 * d_phys
    bonds = np.asarray(res.bond_dims)
    assert np.all(bonds >= 1)
    assert np.diff(np.concatenate([[1], bonds])).max() <= 2 ** 2 * 3


def test_weak_coupling_relaxes_faster():
    # spin-boson crossover (Fig. 4a): weak coupling undergoes coherent
    # oscillation and dips faster, while strong coupling is overdamped and stays
    # near the initial polarization -- so at a fixed short time weak < strong.
    weak = _solve(0.2, order=2).polarization[-1]
    strong = _solve(1.0, order=2).polarization[-1]
    assert weak < strong
