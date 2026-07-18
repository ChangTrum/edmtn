"""Real truncation metric: max per-bond discarded weight (P1-15).

P0-9 replaced a fabricated ``0.0`` with ``None`` ("not measured").  Now the exact paths
report the genuine quantity:

    w_max = max_b ( sum_{i discarded at bond b} sigma_i**2 )

i.e. the discarded WEIGHT.  Note quimb's ``info["error"]`` is the discarded 2-NORM
``sqrt(sum sigma**2)``, so the weight is ``error**2``; and the ``dm`` path splits the reduced
density matrix whose eigenvalues are ``lambda = sigma**2``, so its weight is
``sum(lambda_discarded)`` directly (squaring quimb's eigh error would give ``sum(sigma**4)``,
a different quantity).  ``rsvd`` reports ``None``: its randomized sketch never sees the tail
of the spectrum it omitted, so it cannot measure the discarded weight at all.
"""

from __future__ import annotations

import numpy as np
import pytest
import quimb.tensor as qtn

from edmtn.evolution import SeparableBathEvolution, SingleBathEvolution
from edmtn.evolution.mps_utils import EDMMPS
from edmtn.evolution.quimb_decomp import register_eigh_metric_driver
from edmtn.evolution.quimb_edm import QuimbEDM, _TruncationAccumulator
from edmtn.expansion import FirstOrderExpander, SecondOrderExpander
from edmtn.kernels import SeparableKernelEngine
from edmtn.models import GaudinModel


def _edm(n=5, chi=8, seed=0):
    """A random EDM whose bonds are genuinely truncatable."""
    rng = np.random.default_rng(seed)
    d, d_phys, d2 = 2, 7, 4
    tensors, left = [], d2
    for p in range(n):
        right = d2 if p == n - 1 else chi
        tensors.append((rng.standard_normal((d_phys, left, right))
                        + 1j * rng.standard_normal((d_phys, left, right))).astype(np.complex128))
        left = right
    return QuimbEDM.from_edmmps(
        EDMMPS(tensors=tensors, d=d, d_phys=d_phys, rho0_vec=np.ones(d2, np.complex128)))


# -- 1. the decisive check: a known spectrum, truncated by hand ------------------------------

def test_known_spectrum_reports_exact_discarded_weight():
    """Keep 2 of sigma=[4,3,2] -> reported weight must be exactly sigma_3**2 = 4."""
    rng = np.random.default_rng(1)
    U, _ = np.linalg.qr(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))
    V, _ = np.linalg.qr(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))
    sigma = np.array([4.0, 3.0, 2.0])
    A = U @ np.diag(sigma) @ V.conj().T

    acc = _TruncationAccumulator("error")
    qtn.Tensor(A, inds=("a", "b")).split(
        left_inds=("a",), max_bond=2, cutoff=0.0, absorb="both", info=acc)
    assert acc.n_splits >= 1
    assert acc.max_weight == pytest.approx(sigma[2] ** 2)


def test_accumulators_are_isolated():
    """Independent accumulators never contaminate each other (no shared/global state)."""
    a, b = _TruncationAccumulator("error"), _TruncationAccumulator("error")
    a["error"] = 3.0        # weight 9
    b["error"] = 1.0        # weight 1
    a["error"] = 2.0        # max stays 9
    assert a.max_weight == pytest.approx(9.0)
    assert b.max_weight == pytest.approx(1.0)
    assert a.n_splits == 2 and b.n_splits == 1


def test_accumulator_rejects_non_finite():
    acc = _TruncationAccumulator("error")
    with pytest.raises(FloatingPointError):
        acc["error"] = np.inf


# -- 2. routing regression: zipup TOP-LEVEL vs direct NESTED (must not be merged) ------------

@pytest.mark.parametrize("method", ["zipup", "direct", "dm"])
def test_each_method_records_a_positive_weight_when_truncating(method):
    """Every exact method must capture a real weight; a merged/incorrect info route for any
    one of them silently yields 0.0 here."""
    q = _edm()
    out = q.compress(cutoff=0.0, cutoff_mode="rel", method=method, max_bond=2, decomp="exact")
    assert out.max_discarded_weight is not None
    assert out.max_discarded_weight > 0.0


@pytest.mark.parametrize("method", ["zipup", "direct", "dm"])
def test_no_truncation_reports_exact_zero(method):
    q = _edm()
    out = q.compress(cutoff=0.0, cutoff_mode="rel", method=method, max_bond=None, decomp="exact")
    assert out.max_discarded_weight == 0.0


def test_methods_agree_on_the_same_physical_quantity():
    """dm's sum(lambda_discarded) must equal the SVD paths' sum(sigma**2), not sum(sigma**4)."""
    q = _edm()
    w = {m: q.compress(cutoff=0.0, cutoff_mode="rel", method=m, max_bond=2,
                       decomp="exact").max_discarded_weight
         for m in ("zipup", "direct", "dm")}
    assert w["direct"] == pytest.approx(w["zipup"], rel=1e-6)
    assert w["dm"] == pytest.approx(w["zipup"], rel=1e-6)


# -- 3. dm adapter against a known PSD spectrum ----------------------------------------------

def test_dm_adapter_known_eigenvalues_and_matches_builtin_eigh():
    """lambda=[0.5,0.3,0.2], keep 2 -> weight must be 0.2 (sum lambda), NOT 0.04 (sum lambda**2)."""
    name = register_eigh_metric_driver()
    rng = np.random.default_rng(2)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))
    lam = np.array([0.5, 0.3, 0.2])
    M = Q @ np.diag(lam) @ Q.conj().T
    M = 0.5 * (M + M.conj().T)  # exactly hermitian

    acc = _TruncationAccumulator("discarded_weight")
    ours = qtn.Tensor(M, inds=("a", "b")).split(
        left_inds=("a",), method=name, max_bond=2, cutoff=0.0, absorb="both",
        positive=1, info=acc)
    assert acc.max_weight == pytest.approx(lam[2], rel=1e-8)      # 0.2, not 0.04

    ref = qtn.Tensor(M, inds=("a", "b")).split(
        left_inds=("a",), method="eigh", max_bond=2, cutoff=0.0, absorb="both", positive=1)
    np.testing.assert_allclose(
        (ours ^ ...).data, (ref ^ ...).data, atol=1e-10)  # same compressed operator


# -- 4. rsvd cannot measure it: None, never a fabricated number ------------------------------

@pytest.mark.parametrize("method", ["zipup", "direct"])
def test_rsvd_reports_none(method):
    q = _edm()
    out = q.compress(cutoff=0.0, cutoff_mode="rel", method=method, max_bond=2,
                     decomp="rsvd", decomp_q=2)
    assert out.max_discarded_weight is None


# -- 5. no compression at all -> a genuine 0.0 ----------------------------------------------

def test_single_site_and_untouched_containers_are_zero():
    q1 = _edm(n=1)
    assert q1.max_discarded_weight == 0.0                       # from_edmmps
    out = q1.compress(cutoff=0.0, cutoff_mode="rel", method="zipup", max_bond=2, decomp="exact")
    assert out.max_discarded_weight == 0.0                      # n <= 1 -> nothing to compress


# -- 6. more aggressive truncation is never cheaper (same input, same method) ----------------

def test_tighter_max_bond_discards_at_least_as_much():
    q = _edm()
    w2 = q.compress(cutoff=0.0, cutoff_mode="rel", method="zipup", max_bond=2,
                    decomp="exact").max_discarded_weight
    w4 = q.compress(cutoff=0.0, cutoff_mode="rel", method="zipup", max_bond=4,
                    decomp="exact").max_discarded_weight
    assert w2 >= w4


# -- 7. outer aggregation: axes + nothing dropped --------------------------------------------

def _spy_weights(monkeypatch):
    real = QuimbEDM.compress
    seen: list = []

    def spy(self, *a, **k):
        out = real(self, *a, **k)
        seen.append(out.max_discarded_weight)
        return out

    monkeypatch.setattr(QuimbEDM, "compress", spy)
    return seen


@pytest.mark.parametrize("order", [1, 2])
def test_single_bath_axis_and_aggregation(monkeypatch, order):
    from edmtn.cumulants import GaussianCumulantEngine
    from edmtn.kernels import GaussianKernelEngine
    from edmtn.models import SpinBosonModel

    model = SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0)
    cum = GaussianCumulantEngine().compute(model, T=0.5, eps=0.1)
    engine = GaussianKernelEngine(cum, order=order)
    evo = SingleBathEvolution(
        expander=SecondOrderExpander() if order == 2 else FirstOrderExpander())

    seen = _spy_weights(monkeypatch)
    res = evo.run(model, engine, 0.1, 4, cutoff=1e-8, max_bond=4, compress=True)

    assert len(res.truncation_errors) == len(res.times)        # per PHYSICAL step, not sub-step
    assert all(isinstance(w, float) for w in res.truncation_errors)
    # order 2 runs two sub-steps per physical step: nothing may be lost in the aggregation
    assert max(res.truncation_errors) == pytest.approx(max(w for w in seen if w is not None))


def _separable(record_every, **kw):
    model = GaudinModel(g=0.8, K=4)
    engine = SeparableKernelEngine.from_model(model, T=0.3, eps=0.1)
    evo = SeparableBathEvolution(expander=FirstOrderExpander())
    return evo.run(model, engine, 0.1, 3, record_every=record_every, **kw)


def test_separable_axis_and_record_every_keeps_intermediate_folds():
    every1 = _separable(1, cutoff=1e-8, max_bond=4, compress=True)
    every2 = _separable(2, cutoff=1e-8, max_bond=4, compress=True)

    assert len(every1.truncation_errors) == len(every1.recorded_L)
    assert len(every2.truncation_errors) == len(every2.recorded_L)

    # each record_every=2 entry is the max over its interval of the per-fold values
    by_L = dict(zip(every1.recorded_L, every1.truncation_errors))
    prev = 0
    for L, w in zip(every2.recorded_L, every2.truncation_errors):
        expected = max(by_L[k] for k in range(prev + 1, L + 1))
        assert w == pytest.approx(expected)
        prev = L


def test_separable_compress_false_is_zero():
    res = _separable(1, compress=False)
    assert all(w == 0.0 for w in res.truncation_errors)


def test_separable_rsvd_is_none():
    model = GaudinModel(g=0.8, K=3)
    engine = SeparableKernelEngine.from_model(model, T=0.3, eps=0.1)
    evo = SeparableBathEvolution(expander=FirstOrderExpander(), compress_decomp="rsvd")
    res = evo.run(model, engine, 0.1, 3, cutoff=1e-8, max_bond=4, compress=True)
    assert all(w is None for w in res.truncation_errors)


# -- 8. GPU: backend scalars must convert without leaving the device -------------------------

@pytest.mark.gpu
def test_accumulator_handles_cupy_scalars():  # pragma: no cover - GPU node only
    import cupy as cp  # noqa: PLC0415

    acc = _TruncationAccumulator("error")
    acc["error"] = cp.asarray(3.0)                  # 0-d device scalar
    acc["error"] = cp.asarray([1.0, 2.0])           # batched -> max on device, then .item()
    assert acc.max_weight == pytest.approx(9.0)
    assert isinstance(acc.max_weight, float)


# -- dm supports exact+quimb only: rejected before any tensor work ---------------------------

@pytest.mark.parametrize("kw", [
    dict(decomp="rsvd", decomp_q=2),
    dict(canon="householder"),
    dict(canon="cholqr"),
])
def test_direct_compress_rejects_dm_combination(kw):
    q = _edm()
    with pytest.raises(ValueError, match="dm"):
        q.compress(cutoff=0.0, cutoff_mode="rel", method="dm", max_bond=2, **kw)


def test_direct_compress_rejects_dm_combination_even_single_site():
    """The guard fires before the n <= 1 early return: an illegal combination is
    illegal regardless of chain length."""
    q = _edm(n=1)
    with pytest.raises(ValueError, match="dm"):
        q.compress(cutoff=0.0, cutoff_mode="rel", method="dm", max_bond=2, decomp="rsvd")


def test_dm_exact_quimb_solve_completes_with_numeric_metric():
    """Positive control: the guard must not break the legal dm path, which still
    reports a numeric (never None) truncation metric."""
    from edmtn.driver import solve  # noqa: PLC0415
    from edmtn.models import GaudinModel  # noqa: PLC0415
    res = solve(GaudinModel(g=1.0, K=2), T=0.2, eps=0.1, channel=3,
                compress_method="dm")
    assert res.truncation_errors, "expected at least one recorded metric entry"
    assert all(isinstance(w, float) for w in res.truncation_errors)
