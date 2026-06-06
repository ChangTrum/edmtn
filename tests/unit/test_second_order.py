"""Unit tests for the second-order (doubled sub-step grid) EDM evolution.

Validations:

* **self-consistency** -- the MPS contraction equals an independent dense
  reference built from the same second-order kernel and the alternating
  ``S_1 / S_2`` superoperator families (small ``N``, uncompressed);
* **accuracy** -- at a coarse step the second-order polarization is much closer
  to a fine-step reference than the first-order one (higher Trotter order);
* **artifact removal** -- second order does not overshoot ``|<S_z>| <= 1/2`` at
  strong coupling / long time, where first order does.
"""

import numpy as np
import pytest

from edmtn.cumulants import GaussianCumulantEngine
from edmtn.evolution import SingleBathEvolution, dense_reduced_density_matrix
from edmtn.expansion import FirstOrderExpander, SecondOrderExpander
from edmtn.kernels import GaussianKernelEngine
from edmtn.models import SpinBosonModel


def _engine(model, eps, N, order):
    cum = GaussianCumulantEngine().compute(model, T=N * eps, eps=eps)
    return GaussianKernelEngine(cum, order=order)


def _so_sfamilies(model, eps, N):
    """Alternating [S1(t1), S2(t1), S1(t2), S2(t2), ...] over 2N sub-steps."""
    exp = SecondOrderExpander()
    fam = []
    for n in range(1, N + 1):
        fams = exp.build_at(model, n * eps, eps).families
        fam.append(fams[0])  # S_1 (odd sub-step)
        fam.append(fams[1])  # S_2 (even sub-step)
    return fam


# --------------------------------------------------------------------------
# self-consistency: MPS == dense reference on the sub-step grid
# --------------------------------------------------------------------------

@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_second_order_mps_matches_dense(N):
    model = SpinBosonModel(J0=0.7, omega_c=4.0, mu=1.0)
    eps = 0.1
    engine = _engine(model, eps, N, order=2)

    res = SingleBathEvolution(expander=SecondOrderExpander()).run(
        model, engine, eps, n_steps=N, record_rho=True, compress=False
    )
    rho_mps = res.density_matrices[-1]

    sf = _so_sfamilies(model, eps, N)
    rho0 = model.initial_system_state().reshape(-1).astype(np.complex128)
    rho_ref = dense_reduced_density_matrix(engine, sf, rho0, 2 * N, model.system_dim)

    np.testing.assert_allclose(rho_mps, rho_ref, atol=1e-11)


def test_second_order_trace_and_hermitian():
    model = SpinBosonModel(J0=0.8, omega_c=5.0, mu=1.0)
    eps = 0.05
    engine = _engine(model, eps, 20, order=2)
    res = SingleBathEvolution(expander=SecondOrderExpander()).run(
        model, engine, eps, n_steps=20, record_rho=True, cutoff=1e-7
    )
    for rho in res.density_matrices:
        assert np.isclose(np.trace(rho), 1.0, atol=1e-6)
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-8)


# --------------------------------------------------------------------------
# accuracy: second order beats first order against a fine reference
# --------------------------------------------------------------------------

def _sz_history(model, eps, N, order, cutoff=1e-9):
    engine = _engine(model, eps, N, order=order)
    exp = SecondOrderExpander() if order == 2 else FirstOrderExpander()
    res = SingleBathEvolution(expander=exp).run(
        model, engine, eps, n_steps=N, record_rho=True, cutoff=cutoff
    )
    sz = np.array(
        [np.trace(model.coupling_operators_at(t)[0] @ r).real
         for t, r in zip(res.times, res.density_matrices)]
    )
    return np.array(res.times), sz


def test_second_order_more_accurate_than_first():
    model = SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0)
    T = 0.6
    # fine second-order reference
    t_ref, sz_ref = _sz_history(model, 0.01, int(round(T / 0.01)), order=2, cutoff=1e-7)
    # coarse first vs second order
    t1, sz1 = _sz_history(model, 0.06, int(round(T / 0.06)), order=1, cutoff=1e-7)
    t2, sz2 = _sz_history(model, 0.06, int(round(T / 0.06)), order=2, cutoff=1e-7)

    def at(tt, ss, t):
        return ss[int(np.argmin(np.abs(tt - t)))]

    probe = [0.12, 0.3, 0.48, 0.6]
    err1 = max(abs(at(t1, sz1, t) - at(t_ref, sz_ref, t)) for t in probe)
    err2 = max(abs(at(t2, sz2, t) - at(t_ref, sz_ref, t)) for t in probe)
    assert err2 < err1
    assert err2 < 0.3 * err1  # markedly better


# --------------------------------------------------------------------------
# artifact removal: no unphysical overshoot at strong coupling
# --------------------------------------------------------------------------

def test_no_overshoot_strong_coupling():
    # first order overshoots |<S_z>| > 1/2 by mu*t ~ 5 at strong coupling;
    # second order must stay physical.
    model = SpinBosonModel(J0=1.2, omega_c=5.0, mu=1.0)
    eps = 0.05
    N = int(round(4.0 / eps))
    _, sz = _sz_history(model, eps, N, order=2, cutoff=1e-5)
    assert np.all(sz <= 0.5 + 5e-3)
    assert np.all(sz >= -0.5 - 5e-3)
