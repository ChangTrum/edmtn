"""Causal-prefix time-resolved reads: ``record_time_reads`` and ``dm_tracking``.

The acceptance contract of ``docs/design/causal-prefix-time-reads.md`` section 7.4.  The
scheme reads ``rho(t_n)`` at every physical step off ONE ``K``-sub-bath folding run; this
file pins the identity, the cut mapping, the result-axis contract, the entry-point rules
and the compression semantics.

Two things are deliberately *not* asserted, because they are not true:

* ``dm_tracking`` is not bit-identical to quimb's ``dm``.  Both feed ``lambda = sigma**2``
  to quimb's own trimming, so they share the **rank-selection policy**; their environments
  differ (quimb builds left environments from the original chain, the tracking sweep
  canonicalises first), so their trajectories need not coincide.  End-to-end accuracy is
  therefore verified against an *uncompressed* reference, never by an inequality against
  ``dm``.
* Raw eigenvectors and un-gauge-fixed tensor elements are never compared: an
  eigendecomposition fixes eigenvectors only up to a phase and up to rotations within a
  degenerate subspace.
"""

from __future__ import annotations

import numpy as np
import pytest

from edmtn.driver import solve
from edmtn.evolution.prefix_reads import PrefixTerminators, physical_cuts
from edmtn.evolution.quimb_edm import QuimbEDM
from edmtn.evolution.separable_bath import SeparableBathEvolution
from edmtn.expansion import FirstOrderExpander
from edmtn.expansion.second_order import SecondOrderExpander
from edmtn.kernels import SeparableKernelEngine, SeparableTDKernelEngine
from edmtn.models import DickeModel, GaudinModel, SpinBosonModel

EPS = 0.1


def _dicke(K=3, **kw):
    base = dict(K=K, n_fock=3, coupling=0.7, omega=1.3, omega_c=0.9,
                kappa=0.21, pump=0.11, emission=0.17, dephasing=0.13,
                bath_state=np.tile([0.2, -0.3, 0.5], (K, 1)))
    base.update(kw)
    return DickeModel(**base)


def _engine(model, T, order):
    if model.bath_type == "separable_td":
        return SeparableTDKernelEngine.from_model(model, T=T, eps=EPS, order=order)
    return SeparableKernelEngine.from_model(model, T=T, eps=EPS)


def _run(model, order, n_steps, **kw):
    expander = SecondOrderExpander() if order == 2 else FirstOrderExpander()
    method = "dm_tracking" if (kw.get("record_time_reads")
                               and kw.get("compress", True)) else "zipup"
    ev = SeparableBathEvolution(expander=expander, compress_method=method)
    return ev.run(model, _engine(model, n_steps * EPS, order), eps=EPS,
                  n_steps=n_steps, **kw)


# -- 1/2. the cut mapping ----------------------------------------------------------

def test_physical_cuts_skip_the_mid_strang_boundaries():
    assert physical_cuts(4, 1) == [1, 2, 3, 4]
    assert physical_cuts(4, 2) == [2, 4, 6, 8]


def test_order_two_never_creates_an_odd_terminator():
    """Asserted on the container, not the output: an odd cut must not exist at all.

    Filtering at the end would be untestable from outside -- and a half-step read is not
    self-diagnosing, since its trace is still 1 to machine precision.
    """
    terms = PrefixTerminators(9, n_steps=4, order=2, like=np.zeros(1, dtype=np.complex128))
    assert sorted(terms.terms) == [2, 4, 6, 8]
    assert all(m % 2 == 0 for m in terms.terms)


# -- 3/4. the identity, against independent runs -----------------------------------

@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("build", [_dicke, lambda: GaudinModel(g=0.7, K=3)],
                         ids=["dicke", "gaudin"])
def test_uncompressed_prefix_reads_equal_independent_runs(order, build):
    """The core claim, and it is bath-type agnostic -- Gaudin and Dicke share the engine."""
    model, n = build(), 4
    got = _run(model, order, n, record_time_reads=True, compress=False).time_density_matrices
    assert len(got) == n
    for k in range(1, n + 1):
        ref = _run(model, order, k, compress=False).mps.reduced_density_matrix()
        np.testing.assert_allclose(np.asarray(got[k - 1]), np.asarray(ref), atol=1e-12)


@pytest.mark.parametrize("order", [1, 2])
def test_last_read_is_the_ordinary_reduced_density_matrix(order):
    """``l_M = I``: the newest cut is the output leg, so the last read is free."""
    res = _run(_dicke(), order, 4, record_time_reads=True, cutoff=1e-12)
    np.testing.assert_allclose(np.asarray(res.time_density_matrices[-1]),
                               np.asarray(res.mps.reduced_density_matrix()), atol=1e-13)


# -- 5/6/13. the driver axis contract ----------------------------------------------

@pytest.mark.parametrize("order", [1, 2])
def test_density_matrices_is_the_time_axis(order):
    res = solve(_dicke(), T=0.5, eps=EPS, expansion_order=order, cutoff=1e-12,
                record_time_reads=True, compress_method="dm_tracking")
    assert res.density_matrices is not None
    assert len(res.density_matrices) == len(res.times) == 5
    np.testing.assert_allclose(np.asarray(res.density_matrices[-1]),
                               np.asarray(res.final_density_matrix), atol=1e-13)
    assert res.compression_method_used == "dm_tracking"
    for rho in res.density_matrices:
        rho = np.asarray(rho)
        assert abs(complex(np.trace(rho)).real - 1.0) < 1e-8
        # hermiticity is a TRUNCATION-level property here, not an exactness one: measured
        # 3e-11 at order 1 and 2.4e-07 at order 2 for this cutoff (order 2 has twice the
        # sites and the complex c_1/c_2 coefficients).  The tolerance follows the
        # measurement rather than the other way round.
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-6)


def test_the_default_solve_is_unchanged():
    """Flag off: no time axis, no terminator, the ordinary compression path."""
    res = solve(_dicke(), T=0.3, eps=EPS, cutoff=1e-12)
    assert res.density_matrices is None
    assert res.final_density_matrix is not None
    assert res.compression_method_used == "zipup"


def test_the_two_axes_are_orthogonal():
    """``record_rho`` is the per-``L`` axis, ``record_time_reads`` the per-``t`` one."""
    res = solve(_dicke(), T=0.3, eps=EPS, cutoff=1e-12, record_rho=True,
                record_time_reads=True, compress_method="dm_tracking")
    assert len(res.density_matrices) == 3                     # 3 time steps
    assert len(res.sub_bath_final_density_matrices) == 3      # 3 sub-baths
    np.testing.assert_allclose(np.asarray(res.density_matrices[-1]),
                               np.asarray(res.sub_bath_final_density_matrices[-1]), atol=1e-13)


def test_sub_baths_gives_the_partial_fold_history():
    """``sub_baths=L`` means ``rho_L(t)`` -- what that option already means at the final time."""
    kw = dict(T=0.3, eps=EPS, cutoff=1e-12, record_time_reads=True,
              compress_method="dm_tracking")
    partial = solve(_dicke(), sub_baths=2, **kw)
    full = solve(_dicke(), **kw)
    assert partial.sub_baths_used == 2
    assert not np.allclose(np.asarray(partial.density_matrices[-1]),
                           np.asarray(full.density_matrices[-1]), atol=1e-8)


# -- 7/8. the other pipelines ------------------------------------------------------

def test_compress_false_needs_no_tracking_method():
    """No compression, no basis change -- so any otherwise-valid configuration is fine."""
    res = _run(_dicke(), 1, 3, record_time_reads=True, compress=False)
    assert res.time_density_matrices is not None


def test_single_bath_satisfies_the_request_without_any_prefix_machinery():
    """First order, no ``record_rho``, no observables: the case that yields ``None`` today."""
    model = SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0)
    assert solve(model, T=0.3, eps=EPS, expansion_order=1).density_matrices is None
    res = solve(model, T=0.3, eps=EPS, expansion_order=1, record_time_reads=True)
    assert res.density_matrices is not None
    assert len(res.density_matrices) == len(res.times)


# -- 9. the entry-point contract ---------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"record_time_reads": True},                                    # implicit zipup
    {"record_time_reads": True, "compress_method": "dm"},
    {"record_time_reads": True, "compress_method": "direct"},
    {"compress_method": "dm_tracking"},                             # without the flag
    {"record_time_reads": True, "compress_method": "dm_tracking", "preset": "balanced"},
    {"record_time_reads": True, "compress_method": "dm_tracking", "compress_decomp": "rsvd"},
])
def test_incompatible_configurations_raise_before_any_tensor(kwargs):
    with pytest.raises(ValueError):
        solve(_dicke(), T=0.3, eps=EPS, cutoff=1e-12, **kwargs)


def test_dm_tracking_never_reaches_quimbs_method_registry():
    """``compress_method`` is forwarded verbatim, so the name must be intercepted first.

    Without the interception in :meth:`QuimbEDM.compress` this leaks as a bare ``KeyError``.
    """
    from edmtn.evolution.mps_utils import EDMMPS

    rng = np.random.default_rng(0)
    mps = EDMMPS(tensors=[rng.normal(size=(3, 4, 4)) + 0j for _ in range(3)],
                 d=2, d_phys=3, rho0_vec=rng.normal(size=4) + 0j)
    edm = QuimbEDM.from_edmmps(mps)
    with pytest.raises(ValueError, match="go together"):
        edm.compress(cutoff=0.0, cutoff_mode="rel", method="dm_tracking", max_bond=None)
    with pytest.raises(ValueError, match="go together"):
        edm.compress(cutoff=0.0, cutoff_mode="rel", method="dm", max_bond=None,
                     terminators=object())


# -- 10. rank-selection policy, shared with the dm path ----------------------------

@pytest.mark.parametrize("mode", ["abs", "rel", "sum2", "rsum2", "sum1", "rsum1"])
@pytest.mark.parametrize("cutoff", [1e-3, 1e-6])
def test_rank_policy_matches_the_dm_path(mode, cutoff):
    """Same rank and same discarded weight as the driver quimb's ``dm`` compression uses.

    A deliberately **non-degenerate** spectrum, so the comparison cannot be confounded by
    the rotation freedom inside a degenerate eigenspace.  Compared: the retained rank and
    the discarded weight -- never the eigenvectors.
    """
    from quimb.tensor import decomp

    from edmtn.evolution.prefix_reads import _retained_rank
    from edmtn.evolution.quimb_decomp import register_eigh_metric_driver

    lam = np.array([1.0, 3e-2, 7e-4, 2e-5, 8e-7, 1e-9, 4e-12], dtype=np.float64)
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.normal(size=(lam.size, lam.size)) + 0j)
    rho = (q * lam) @ q.conj().T
    rho = 0.5 * (rho + rho.conj().T)

    # ours
    s, u = np.linalg.eigh(rho)
    s = np.clip(np.flip(s).real, 0.0, None)
    u = np.flip(u, axis=-1)
    ours = _retained_rank(s, u, cutoff, mode, None, np)
    ours_weight = float(s[ours:].sum())

    # the dm path's own driver, on the same matrix
    name = register_eigh_metric_driver()
    info = {"discarded_weight": None}
    out = decomp._SPLIT_FNS[name](
        rho, cutoff=cutoff, cutoff_mode=decomp._CUTOFF_MODE_MAP[mode], max_bond=-1,
        absorb=None, renorm=0, positive=1, info=info)
    theirs = out[0].shape[-1]

    assert ours == theirs
    assert ours_weight == pytest.approx(float(info["discarded_weight"]), abs=1e-18)


def test_max_bond_overrides_the_cutoff_in_the_rank_policy():
    from edmtn.evolution.prefix_reads import _retained_rank

    lam = np.array([1.0, 1e-1, 1e-2, 1e-3, 1e-4], dtype=np.float64)
    u = np.eye(lam.size, dtype=np.complex128)
    assert _retained_rank(lam, u, 1e-12, "rel", None, np) == 5
    assert _retained_rank(lam, u, 1e-12, "rel", 2, np) == 2


# -- 11. end-to-end accuracy, verified independently of dm -------------------------

def _reads(order, model=None, n=6, **kw):
    return [np.asarray(x) for x in
            _run(model if model is not None else _dicke(), order, n,
                 record_time_reads=True, **kw).time_density_matrices]


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("build", [_dicke, lambda: GaudinModel(g=0.7, K=3)],
                         ids=["dicke", "gaudin"])
def test_no_discard_reads_match_the_uncompressed_reference(order, build):
    """``dm_tracking`` time reads, for BOTH separable bath types -- not Gaudin by proxy.

    Gaudin's other coverage goes through the polarization; this compares the thing the
    feature actually produces, ``time_density_matrices``.
    """
    model = build()
    exact = _reads(order, model, compress=False)
    got = _reads(order, model, cutoff=0.0, max_bond=None)
    assert len(got) == len(exact) == 6
    for a, b in zip(got, exact):
        np.testing.assert_allclose(a, b, atol=1e-11)


@pytest.mark.parametrize("cutoff, max_bond, bound", [
    # Pre-registered from measurement, with ~8x headroom.  Measured worst deviation from
    # the uncompressed reads: 1.16e-07, 3.78e-07, 1.28e-06 respectively.
    (1e-14, None, 1e-6),
    (1e-12, None, 3e-6),
    (None, 24, 1e-5),        # max_bond, not cutoff, is the binding constraint here
])
def test_truncated_reads_stay_within_their_registered_bound(cutoff, max_bond, bound):
    """Each configuration has its own absolute bound against the UNCOMPRESSED reference.

    Not an inequality against quimb's ``dm``: there is no general relation between two
    algorithms' mutual difference and their individual errors, and ``cutoff`` is a local
    spectral rule rather than a bound on the final state.
    """
    exact = _reads(1, compress=False)
    res = _run(_dicke(), 1, 6, record_time_reads=True,
               cutoff=0.0 if cutoff is None else cutoff, max_bond=max_bond)
    got = [np.asarray(x) for x in res.time_density_matrices]
    worst = max(float(np.abs(a - b).max()) for a, b in zip(got, exact))
    assert worst < bound
    # the case must actually truncate, or the bound proves nothing
    assert max(w for w in res.truncation_errors if w is not None) > 0.0
    for rho in got:
        assert abs(complex(np.trace(rho)).real - 1.0) < 1e-6
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-8)   # measured <= 6.7e-10


# -- 12. terminator transport ------------------------------------------------------

def test_transport_is_not_vacuous():
    """Resizing a terminator instead of transporting it must break the reads."""
    terms = PrefixTerminators(4, n_steps=2, order=1, like=np.zeros(1, dtype=np.complex128))
    before = terms.terms[1].copy()
    factor = np.linalg.qr(np.random.default_rng(0).normal(size=(4, 3)))[0] + 0j
    terms.transport(1, factor)
    assert terms.terms[1].shape == (4, 3)
    assert not np.allclose(terms.terms[1], before[:, :3])


@pytest.mark.gpu
def test_terminator_transport_on_the_cupy_backend():
    cupy = pytest.importorskip("cupy")
    like = cupy.zeros(1, dtype=cupy.complex128)
    terms = PrefixTerminators(4, n_steps=2, order=1, like=like)
    assert type(terms.terms[1]).__module__.split(".")[0] == "cupy"
    res = solve(_dicke(), T=0.3, eps=EPS, cutoff=1e-12, backend="gpu",
                record_time_reads=True, compress_method="dm_tracking")
    assert res.density_matrices is not None and len(res.density_matrices) == 3


# -- the two defects the audit found -----------------------------------------------

def test_single_bath_refuses_dm_tracking_before_the_kernel_is_built(monkeypatch):
    """It has no terminators to transport, so the name must never reach the evolution.

    Two earlier versions of this rule were not good enough: the config layer let it through
    entirely (``QuimbEDM.compress`` then raised at the SECOND time site, mid-evolution), and
    then the engine caught it only after the driver had already built the kernel.  The spy
    is the point of this test -- asserting "it raises" would pass in all three versions.
    """
    import edmtn.driver.auto_config as ac

    calls = []
    monkeypatch.setattr(ac.GaussianKernelEngine, "from_model",
                        classmethod(lambda cls, *a, **k: calls.append(1)))
    model = SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0)
    with pytest.raises(ValueError, match="only implemented by the separable-bath"):
        solve(model, T=0.2, eps=EPS, record_time_reads=True,
              compress_method="dm_tracking")
    assert calls == []                       # the pipeline was never constructed


def test_single_bath_direct_run_also_refuses_dm_tracking():
    """The engine keeps its own check, for callers that bypass the driver entirely."""
    from edmtn.evolution.single_bath import SingleBathEvolution

    ev = SingleBathEvolution(compress_method="dm_tracking")
    with pytest.raises(ValueError, match="only implemented by the separable-bath"):
        ev.run(SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0), None, eps=EPS, n_steps=2)


@pytest.mark.parametrize("order, n_steps, expected", [
    (1, 1, None),        # a single site: nothing to compress
    (1, 3, "zipup"),     # compression ran
    (2, 1, "zipup"),     # order 2 already compresses at the second SUB-step
])
def test_single_bath_reports_the_compression_that_ran(order, n_steps, expected):
    """``compression_method_used`` must be honest on this pipeline too, not always None."""
    res = solve(SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0),
                T=n_steps * EPS, eps=EPS, expansion_order=order)
    assert res.compression_method_used == expected


def test_non_finite_truncation_metric_raises_like_the_ordinary_paths():
    """``max(0.0, nan)`` is ``0.0``, so a NaN weight would be reported as a healthy zero."""
    import edmtn.evolution.prefix_reads as pr
    from edmtn.evolution.mps_utils import EDMMPS

    rng = np.random.default_rng(3)
    tensors = [rng.normal(size=(3, 4, 4)) + 0j for _ in range(3)]
    terms = PrefixTerminators(4, n_steps=3, order=1, like=tensors[0])
    original = pr._scalar
    pr._scalar = lambda v: float("nan")
    try:
        with pytest.raises(FloatingPointError, match="non-finite/negative"):
            pr.tracking_compress(tensors, terms, cutoff=1e-3, cutoff_mode="rel",
                                 max_bond=2)
    finally:
        pr._scalar = original


def test_compression_method_used_reports_only_a_compression_that_ran():
    """``None`` when nothing compressed -- the field says *used*, not *requested*."""
    res = _run(_dicke(), 1, 3, record_time_reads=True, compress=False)
    assert res.compression_method_used is None
    res = _run(_dicke(), 1, 3, record_time_reads=True, cutoff=1e-12)
    assert res.compression_method_used == "dm_tracking"
