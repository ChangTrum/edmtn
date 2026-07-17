"""`compress=False` genuinely skips compression on BOTH pipelines (P1-13).

Previously the separable (Gaudin) path ran ``compress=False`` as a zero-cutoff exact
*recompression* -- it still canonicalised + SVD'd -- so the flag meant something different
than on the single-bath path (which truly skipped).  ``QuimbEDM.fold`` is now split into a
lossless ``fold_raw`` (MPO x MPS growth + multibond fusion) and the separate ``compress``;
``SeparableBathEvolution.run`` calls ``compress`` only when ``compress=True``.

These tests assert, via a pass-through spy on ``QuimbEDM.compress``, the EXACT number of
compression calls on each pipeline; that ``compress=False`` leaves the raw
``D_initial * D_a**L`` bond growth uncompressed; and that the ``fold``/``fold_raw`` split
itself is correct (fold_raw is pure and never compresses, fold == fold_raw + one compress).
"""

from __future__ import annotations

import numpy as np
import pytest

from edmtn.cumulants import GaussianCumulantEngine
from edmtn.evolution import SeparableBathEvolution, SingleBathEvolution
from edmtn.evolution.quimb_edm import QuimbEDM
from edmtn.expansion import FirstOrderExpander, SecondOrderExpander
from edmtn.kernels import GaussianKernelEngine, SeparableKernelEngine
from edmtn.models import GaudinModel, SpinBosonModel


def _spy_compress(monkeypatch):
    """Count QuimbEDM.compress calls while passing through to the real method."""
    real = QuimbEDM.compress
    calls = {"n": 0}

    def spy(self, *a, **k):
        calls["n"] += 1
        return real(self, *a, **k)

    monkeypatch.setattr(QuimbEDM, "compress", spy)
    return calls


# -- single-bath: exact call count = max(order*n_steps - 1, 0) -------------------------------

def _single(order):
    model = SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0)
    cum = GaussianCumulantEngine().compute(model, T=0.6, eps=0.1)
    engine = GaussianKernelEngine(cum, order=order)
    expander = SecondOrderExpander() if order == 2 else FirstOrderExpander()
    return model, engine, SingleBathEvolution(expander=expander)


@pytest.mark.parametrize("order,n_steps", [(1, 4), (2, 3), (1, 1)])
def test_single_bath_compress_true_call_count(monkeypatch, order, n_steps):
    model, engine, evo = _single(order)
    calls = _spy_compress(monkeypatch)
    evo.run(model, engine, 0.1, n_steps, cutoff=1e-8, compress=True)
    # first site leaves num_sites == 1 (no compress); every later sub-step compresses once
    assert calls["n"] == max(order * n_steps - 1, 0)


@pytest.mark.parametrize("order,n_steps", [(1, 4), (2, 3), (1, 1)])
def test_single_bath_compress_false_never_compresses(monkeypatch, order, n_steps):
    model, engine, evo = _single(order)
    calls = _spy_compress(monkeypatch)
    evo.run(model, engine, 0.1, n_steps, compress=False)
    assert calls["n"] == 0


# -- separable (Gaudin): compress=False -> 0, compress=True -> n_fold -------------------------

def _separable(K=2, order=1):
    model = GaudinModel(g=0.7, K=K)
    engine = SeparableKernelEngine.from_model(model, T=0.3, eps=0.1)
    expander = SecondOrderExpander() if order == 2 else FirstOrderExpander()
    return model, engine, SeparableBathEvolution(expander=expander)


@pytest.mark.parametrize("compress,expected", [(False, 0), (True, 2)])
def test_gaudin_compress_call_count(monkeypatch, compress, expected):
    # n_sites = order*n_steps = 1*3 = 3 > 1, so compress=True is a REAL compression (not the
    # n<=1 early-return); K=2 -> n_fold=2
    model, engine, evo = _separable(K=2, order=1)
    calls = _spy_compress(monkeypatch)
    evo.run(model, engine, 0.1, 3, compress=compress)
    assert calls["n"] == expected


# -- structural: compress=False leaves the raw D_initial * D_a**L bond growth ----------------

def test_gaudin_compress_false_raw_bond_growth():
    model = GaudinModel(g=0.7, K=2)
    eps, n_steps, order = 0.1, 2, 1
    engine = SeparableKernelEngine.from_model(model, T=eps * n_steps, eps=eps)
    evo = SeparableBathEvolution(expander=FirstOrderExpander())
    d, d_phys = model.system_dim, engine.d_phys
    rho0 = model.initial_system_state().reshape(-1).astype(np.complex128)
    base = evo._build_system_mps(model, eps, n_steps, order, d, d_phys, rho0, lambda a: a)

    raw = evo.run(model, engine, eps, n_steps, compress=False)
    n_fold, d_a = model.K, engine.corr.bond_dim
    expected = [dim * d_a ** n_fold for dim in base.bond_dims]
    assert raw.mps.bond_dims == expected                   # exact lossless growth, no compression

    # a zero-cutoff exact compression cannot exceed the raw bonds, and here it strictly reduces
    comp = evo.run(model, engine, eps, n_steps, compress=True, cutoff=0.0)
    assert all(c <= r for c, r in zip(comp.mps.bond_dims, raw.mps.bond_dims))
    assert any(c < r for c, r in zip(comp.mps.bond_dims, raw.mps.bond_dims))


# -- the fold/fold_raw split itself ----------------------------------------------------------

def _gaudin_edm_and_mpo(K=2, order=1, n_steps=2):
    model = GaudinModel(g=0.7, K=K)
    eps = 0.1
    engine = SeparableKernelEngine.from_model(model, T=eps * n_steps, eps=eps)
    expander = SecondOrderExpander() if order == 2 else FirstOrderExpander()
    evo = SeparableBathEvolution(expander=expander)
    d, d_phys = model.system_dim, engine.d_phys
    n_sites = order * n_steps
    rho0 = model.initial_system_state().reshape(-1).astype(np.complex128)
    base = evo._build_system_mps(model, eps, n_steps, order, d, d_phys, rho0, lambda a: a)
    q = QuimbEDM.from_edmmps(base)
    mpo = list(engine.for_sub_bath(0).get_kernel_mpo(n_sites).site_tensors)
    return q, mpo


def test_fold_raw_is_pure_and_never_compresses(monkeypatch):
    q, mpo = _gaudin_edm_and_mpo()
    before_bonds = list(q.bond_dims)
    before_rho = np.array(q.reduced_density_matrix())
    calls = _spy_compress(monkeypatch)
    raw = q.fold_raw(mpo)
    assert calls["n"] == 0                                  # fold_raw never compresses
    assert q.bond_dims == before_bonds                      # original object untouched ...
    np.testing.assert_allclose(np.array(q.reduced_density_matrix()), before_rho)
    assert all(rb >= b for rb, b in zip(raw.bond_dims, before_bonds))  # ... and raw grew the bonds


def test_fold_is_fold_raw_plus_one_compress(monkeypatch):
    q, mpo = _gaudin_edm_and_mpo()
    calls = _spy_compress(monkeypatch)
    folded = q.fold(mpo, cutoff=1e-12, cutoff_mode="rel", method="zipup", max_bond=None)
    assert calls["n"] == 1                                  # fold == fold_raw + exactly one compress
    manual = q.fold_raw(mpo).compress(
        cutoff=1e-12, cutoff_mode="rel", method="zipup", max_bond=None)
    np.testing.assert_allclose(
        np.array(folded.reduced_density_matrix()),
        np.array(manual.reduced_density_matrix()), atol=1e-12)


def test_fold_raw_single_site_to_edmmps():
    q, mpo = _gaudin_edm_and_mpo(K=1, order=1, n_steps=1)   # n_sites == 1
    edm = q.fold_raw(mpo).to_edmmps()                       # no dangling lateral a0
    assert edm.num_sites == 1
