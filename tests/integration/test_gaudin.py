"""Integration smoke test: the Gaudin (separable) pipeline runs end-to-end.

Not a reproduction -- Fig. 6/11/12 live in ``examples/reproduce_fig6.py``.  Here
we check, at a small/fast size, that the separable outer-loop stack (Gaudin model
-> separable correlation -> separable kernel -> outer-loop evolution ->
all-times polarization -> driver) runs and returns physically sane output, and
that the ``sub_baths`` (first-L-spins) control works.
"""

import numpy as np
import pytest

from edmtn.driver import EDMSolver
from edmtn.models import GaudinModel

pytestmark = pytest.mark.integration

Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _host(a):
    """NumPy view of an array that may live on the GPU (auto backend)."""
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _solve(K, *, order=2, T=1.0, eps=0.1, sub_baths=None):
    model = GaudinModel(g=1.0, K=K)
    return EDMSolver.from_model(
        model, T=T, eps=eps, expansion_order=order,
        cutoff=1e-6, max_bond=64, record_rho=True, sub_baths=sub_baths,
    ).solve(channel=3)  # channel 3 = S_z


@pytest.mark.parametrize("order", [1, 2])
def test_pipeline_runs_and_depolarises(order):
    res = _solve(8, order=order)
    n = res.times.size
    assert n > 0 and res.polarization.shape == (n,)
    # central spin starts polarised along +z and the spin bath depolarises it
    assert np.isclose(res.polarization[0], 0.5, atol=1e-6)
    assert res.polarization[-1] < res.polarization[0]
    # stays physical
    assert np.all(res.polarization <= 0.5 + 1e-3)
    assert np.all(res.polarization >= -0.5 - 1e-3)


def test_bond_dims_and_final_state_sane():
    res = _solve(8)
    bonds = np.asarray(res.bond_dims)        # per sub-bath L
    assert len(bonds) == 8
    assert np.all(bonds >= 1) and np.all(bonds <= 64)
    assert len(res.mps.bond_dims) == res.mps.num_sites - 1   # D_t available
    rho = _host(res.evolution.density_matrices[-1])           # may be on GPU (auto)
    assert abs(np.trace(rho) - 1.0) < 1e-3
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-6)


def test_sub_baths_folds_only_first_L():
    res = _solve(8, sub_baths=4)
    assert res.evolution.n_sub_baths == 4
    assert res.evolution.recorded_L[-1] == 4
    assert len(res.bond_dims) == 4
