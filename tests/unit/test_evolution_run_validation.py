"""Entry validation for the direct evolution ``run()`` methods (P1-11).

``SingleBathEvolution.run`` and ``SeparableBathEvolution.run`` are public and callable
without the driver (reference checks, tests, notebooks).  When they are, they must reject an
illegal argument with a clear ``ValueError`` at the entry point -- BEFORE any tensor is
built or the kernel is read -- instead of returning an all-``t=0`` trajectory, an empty
result, a ``ZeroDivisionError`` (``record_every=0``), a ``range()`` ``TypeError``
(``n_steps=2.5``), a non-finite time grid (``eps=1e308``) or a deep quimb error (bad
``cutoff_mode``).

Every illegal-input test asserts both (a) the ``ValueError`` and (b) via spies that *nothing
was constructed*: for single-bath ``convert`` / ``expander.build_at`` / ``QuimbEDM.empty``
were never called; for separable ``convert`` / ``_build_system_mps`` / ``QuimbEDM.from_edmmps``;
and on a kernel mismatch the kernel's ``get_kernel_mpo`` / ``for_sub_bath`` were never read.
"""

from __future__ import annotations

import numpy as np
import pytest

from edmtn.cumulants import GaussianCumulantEngine
from edmtn.driver import auto_config
from edmtn.evolution import SeparableBathEvolution, SingleBathEvolution, quimb_edm
from edmtn.evolution import _validation
from edmtn.evolution._validation import CUTOFF_MODES
from edmtn.expansion import SecondOrderExpander
from edmtn.kernels import GaussianKernelEngine, SeparableKernelEngine
from edmtn.models import GaudinModel, SpinBosonModel


# -- builders (small, valid model+kernel pairs) ----------------------------------------------

def _single(order: int = 1):
    model = SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0)
    cum = GaussianCumulantEngine().compute(model, T=0.5, eps=0.1)
    engine = GaussianKernelEngine(cum, order=order)
    return model, engine


def _separable(K: int = 1):
    model = GaudinModel(g=0.8, K=K)
    engine = SeparableKernelEngine.from_model(model, T=0.5, eps=0.1)
    return model, engine


def _single_kwargs(**over):
    base = dict(eps=0.1, n_steps=2, cutoff=1e-8, max_bond=8,
                record_rho=False, compress=True, cutoff_mode="rel")
    base.update(over)
    return base


def _separable_kwargs(**over):
    base = dict(eps=0.1, n_steps=1, cutoff=1e-8, max_bond=8, record_rho=False,
                compress=True, cutoff_mode="rel", record_every=1)
    base.update(over)
    return base


# -- spies: prove no construction started ----------------------------------------------------

def _spy_single(monkeypatch, evo, engine):
    calls = dict(convert=0, build_at=0, empty=0, get_kernel_mpo=0)
    monkeypatch.setattr(type(evo.expander), "build_at",
                        lambda *a, **k: calls.__setitem__("build_at", calls["build_at"] + 1))
    monkeypatch.setattr(quimb_edm.QuimbEDM, "empty",
                        lambda *a, **k: calls.__setitem__("empty", calls["empty"] + 1))
    monkeypatch.setattr(engine, "get_kernel_mpo",
                        lambda *a, **k: calls.__setitem__("get_kernel_mpo", calls["get_kernel_mpo"] + 1))

    def convert(a):
        calls["convert"] += 1
        return a

    return calls, convert


def _spy_separable(monkeypatch, evo, engine):
    calls = dict(convert=0, build_system_mps=0, from_edmmps=0, for_sub_bath=0)
    monkeypatch.setattr(evo, "_build_system_mps",
                        lambda *a, **k: calls.__setitem__("build_system_mps", calls["build_system_mps"] + 1))
    monkeypatch.setattr(quimb_edm.QuimbEDM, "from_edmmps",
                        lambda *a, **k: calls.__setitem__("from_edmmps", calls["from_edmmps"] + 1))
    monkeypatch.setattr(engine, "for_sub_bath",
                        lambda *a, **k: calls.__setitem__("for_sub_bath", calls["for_sub_bath"] + 1))

    def convert(a):
        calls["convert"] += 1
        return a

    return calls, convert


# -- shared illegal-scalar matrix ------------------------------------------------------------

_COMMON_BAD = [
    ("eps", 0), ("eps", -1.0), ("eps", float("nan")), ("eps", float("inf")), ("eps", 10 ** 400),
    ("n_steps", 0), ("n_steps", -1), ("n_steps", 2.5), ("n_steps", True),
    ("cutoff", -1.0), ("cutoff", float("nan")), ("cutoff", float("inf")), ("cutoff", 10 ** 400),
    ("max_bond", 0), ("max_bond", -1), ("max_bond", 2.5), ("max_bond", True),
    ("record_rho", 0), ("record_rho", 1), ("record_rho", "yes"), ("record_rho", np.bool_(True)),
    ("compress", 0), ("compress", 1), ("compress", "yes"), ("compress", np.bool_(True)),
    ("cutoff_mode", "relative"), ("cutoff_mode", "xyz"), ("cutoff_mode", 3),
]


@pytest.mark.parametrize("field,bad", _COMMON_BAD)
def test_single_rejects_bad_scalar(monkeypatch, field, bad):
    model, engine = _single()
    evo = SingleBathEvolution()
    calls, convert = _spy_single(monkeypatch, evo, engine)
    with pytest.raises(ValueError):
        evo.run(model, engine, convert=convert, **_single_kwargs(**{field: bad}))
    assert calls == dict(convert=0, build_at=0, empty=0, get_kernel_mpo=0)


@pytest.mark.parametrize("field,bad", _COMMON_BAD + [
    ("record_every", 0), ("record_every", -1), ("record_every", 2.5), ("record_every", True),
])
def test_separable_rejects_bad_scalar(monkeypatch, field, bad):
    model, engine = _separable()
    evo = SeparableBathEvolution()
    calls, convert = _spy_separable(monkeypatch, evo, engine)
    with pytest.raises(ValueError):
        evo.run(model, engine, convert=convert, **_separable_kwargs(**{field: bad}))
    assert calls == dict(convert=0, build_system_mps=0, from_edmmps=0, for_sub_bath=0)


# -- final time grid must be finite (eps finite + n_steps int is not enough) ------------------

def test_single_rejects_nonfinite_time_grid(monkeypatch):
    model, engine = _single()
    evo = SingleBathEvolution()
    calls, convert = _spy_single(monkeypatch, evo, engine)
    with pytest.raises(ValueError, match="final time"):
        evo.run(model, engine, convert=convert, **_single_kwargs(eps=1e308, n_steps=2))
    assert calls == dict(convert=0, build_at=0, empty=0, get_kernel_mpo=0)


def test_separable_rejects_nonfinite_time_grid(monkeypatch):
    model, engine = _separable()
    evo = SeparableBathEvolution()
    calls, convert = _spy_separable(monkeypatch, evo, engine)
    with pytest.raises(ValueError, match="final time"):
        evo.run(model, engine, convert=convert, **_separable_kwargs(eps=1e308, n_steps=2))
    assert calls == dict(convert=0, build_system_mps=0, from_edmmps=0, for_sub_bath=0)


# -- single-bath order mismatch (BOTH directions) --------------------------------------------

def test_single_rejects_second_order_evo_first_order_kernel(monkeypatch):
    model, engine = _single(order=1)                        # first-order kernel
    evo = SingleBathEvolution(expander=SecondOrderExpander())  # second-order evolution
    calls, convert = _spy_single(monkeypatch, evo, engine)
    with pytest.raises(ValueError, match="order"):
        evo.run(model, engine, convert=convert, **_single_kwargs())
    assert calls == dict(convert=0, build_at=0, empty=0, get_kernel_mpo=0)


def test_single_rejects_first_order_evo_second_order_kernel(monkeypatch):
    model, engine = _single(order=2)                        # second-order kernel
    evo = SingleBathEvolution()                              # first-order evolution (default)
    calls, convert = _spy_single(monkeypatch, evo, engine)
    with pytest.raises(ValueError, match="order"):
        evo.run(model, engine, convert=convert, **_single_kwargs())
    assert calls == dict(convert=0, build_at=0, empty=0, get_kernel_mpo=0)


# -- strict integer order: True / 1.0 must not slip through as 1 (True==1, 1.0==1) -----------


class _FakeExpander:
    """Stand-in expander with a corruptible ``order`` (build_at present only so a spy can
    patch it; it must never actually run because entry validation rejects the order first)."""

    def __init__(self, order):
        self.order = order

    def build_at(self, *a, **k):  # pragma: no cover - guarded by validation
        raise AssertionError("build_at should not be called")


@pytest.mark.parametrize("bad_order", [True, 1.0])
def test_single_rejects_nonint_kernel_order(monkeypatch, bad_order):
    model, engine = _single(order=bad_order)                # kernel with bool/float order
    evo = SingleBathEvolution()                             # genuine int-order expander
    calls, convert = _spy_single(monkeypatch, evo, engine)
    with pytest.raises(ValueError, match="order"):
        evo.run(model, engine, convert=convert, **_single_kwargs())
    assert calls == dict(convert=0, build_at=0, empty=0, get_kernel_mpo=0)


@pytest.mark.parametrize("bad_order", [True, 1.0])
def test_single_rejects_nonint_expander_order(monkeypatch, bad_order):
    model, engine = _single()
    evo = SingleBathEvolution(expander=_FakeExpander(bad_order))
    calls, convert = _spy_single(monkeypatch, evo, engine)
    with pytest.raises(ValueError, match="order"):
        evo.run(model, engine, convert=convert, **_single_kwargs())
    assert calls == dict(convert=0, build_at=0, empty=0, get_kernel_mpo=0)


@pytest.mark.parametrize("bad_order", [True, 1.0])
def test_separable_rejects_nonint_expander_order(monkeypatch, bad_order):
    model, engine = _separable()
    evo = SeparableBathEvolution(expander=_FakeExpander(bad_order))
    calls, convert = _spy_separable(monkeypatch, evo, engine)
    with pytest.raises(ValueError, match="order"):
        evo.run(model, engine, convert=convert, **_separable_kwargs())
    assert calls == dict(convert=0, build_system_mps=0, from_edmmps=0, for_sub_bath=0)


def test_validate_expansion_order_normalizes_and_rejects():
    assert type(_validation.validate_expansion_order("order", np.int64(1))) is int
    assert type(_validation.validate_expansion_order("order", np.int64(2))) is int
    for bad in [True, 1.0, 0, 3, np.float64(2.0), "1", None]:
        with pytest.raises(ValueError, match="order"):
            _validation.validate_expansion_order("order", bad)


# -- structural model/kernel mismatch --------------------------------------------------------

def test_single_rejects_d_phys_mismatch(monkeypatch):
    # Gaussian kernel (d_phys=3) fed a Gaudin model (3 channels -> expects d_phys=7)
    _, gaussian_engine = _single()
    gaudin_model = GaudinModel(g=0.8, K=2)
    evo = SingleBathEvolution()
    calls, convert = _spy_single(monkeypatch, evo, gaussian_engine)
    with pytest.raises(ValueError, match="d_phys"):
        evo.run(gaudin_model, gaussian_engine, convert=convert, **_single_kwargs())
    assert calls == dict(convert=0, build_at=0, empty=0, get_kernel_mpo=0)


def test_separable_rejects_K_mismatch(monkeypatch):
    # the mandated real regression: a K=3 model with a kernel built from a K=2 model.
    # Both have d_phys=7, so a d_phys-only check would MISS this -- the K check must catch it.
    model = GaudinModel(g=0.8, K=3)
    kernel = SeparableKernelEngine.from_model(GaudinModel(g=0.8, K=2), T=0.5, eps=0.1)
    evo = SeparableBathEvolution()
    calls, convert = _spy_separable(monkeypatch, evo, kernel)
    with pytest.raises(ValueError, match="K="):
        evo.run(model, kernel, convert=convert, **_separable_kwargs())
    assert calls == dict(convert=0, build_system_mps=0, from_edmmps=0, for_sub_bath=0)


# -- validator units: normalization, bool rejection, overflow, structural helpers ------------

class _FakeEngine:
    def __init__(self, d_phys, K=1, order=1, with_mpo=True, with_sub=True):
        self.d_phys = d_phys
        self.K = K
        self.order = order
        if with_mpo:
            self.get_kernel_mpo = lambda *a, **k: None
        if with_sub:
            self.for_sub_bath = lambda *a, **k: None


def test_validators_normalize_numpy_types():
    assert type(_validation.validate_positive_int("n", np.int64(3))) is int
    assert type(_validation.validate_positive_finite_float("x", np.float64(0.1))) is float
    assert type(_validation.validate_nonnegative_finite_float("x", np.float64(0.0))) is float
    assert type(_validation.validate_optional_positive_int("m", np.int64(4))) is int


def test_validators_reject_bool_as_number():
    with pytest.raises(ValueError):
        _validation.validate_positive_int("n", True)
    with pytest.raises(ValueError):
        _validation.validate_positive_finite_float("x", True)
    with pytest.raises(ValueError):
        _validation.validate_nonnegative_finite_float("x", False)
    with pytest.raises(ValueError):
        _validation.validate_optional_positive_int("m", True)


def test_validators_reject_huge_int_overflow():
    with pytest.raises(ValueError):
        _validation.validate_positive_finite_float("eps", 10 ** 400)
    with pytest.raises(ValueError):
        _validation.validate_nonnegative_finite_float("cutoff", 10 ** 400)


def test_kernel_dim_helper_rejects_wrong_d_phys():
    with pytest.raises(ValueError, match="d_phys"):
        _validation.validate_single_bath_kernel(
            SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0), _FakeEngine(d_phys=5), 1)
    with pytest.raises(ValueError, match="d_phys"):
        _validation.validate_separable_bath_kernel(
            GaudinModel(g=0.8, K=3), _FakeEngine(d_phys=5, K=3))


def test_kernel_helper_rejects_missing_interface():
    with pytest.raises(ValueError, match="get_kernel_mpo"):
        _validation.validate_single_bath_kernel(
            SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0), _FakeEngine(d_phys=3, with_mpo=False), 1)
    with pytest.raises(ValueError, match="for_sub_bath"):
        _validation.validate_separable_bath_kernel(
            GaudinModel(g=0.8, K=3), _FakeEngine(d_phys=7, K=3, with_sub=False))


# -- cutoff_mode contract: driver and evolution share one set --------------------------------

def test_cutoff_mode_set_is_shared_with_driver():
    # same object -> the two contracts cannot drift
    assert auto_config._CUTOFF_MODES is CUTOFF_MODES


# -- legal boundaries keep working -----------------------------------------------------------

def test_single_legal_boundaries_run():
    model, engine = _single()
    res = SingleBathEvolution().run(model, engine, eps=0.1, n_steps=1, cutoff=0.0, max_bond=None)
    assert len(res.times) == 1


def test_single_numpy_scalars_normalize_and_run():
    model, engine = _single()
    res = SingleBathEvolution().run(
        model, engine, eps=np.float64(0.1), n_steps=np.int64(2),
        cutoff=np.float64(1e-8), max_bond=np.int64(8))
    assert len(res.times) == 2


@pytest.mark.parametrize("mode", CUTOFF_MODES)
def test_single_every_legal_cutoff_mode_runs(mode):
    model, engine = _single()
    res = SingleBathEvolution().run(model, engine, eps=0.1, n_steps=2, cutoff=1e-8, cutoff_mode=mode)
    assert len(res.times) == 2


def test_separable_legal_boundaries_run():
    model, engine = _separable(K=1)
    res = SeparableBathEvolution().run(
        model, engine, eps=0.1, n_steps=1, cutoff=0.0, max_bond=None, record_every=1)
    assert res.n_sub_baths == 1
    assert res.mps is not None


def test_separable_numpy_scalars_normalize_and_run():
    model, engine = _separable(K=1)
    res = SeparableBathEvolution().run(
        model, engine, eps=np.float64(0.1), n_steps=np.int64(1),
        cutoff=np.float64(1e-8), max_bond=np.int64(8), record_every=np.int64(1))
    assert res.n_sub_baths == 1


@pytest.mark.parametrize("mode", CUTOFF_MODES)
def test_separable_every_legal_cutoff_mode_runs(mode):
    model, engine = _separable(K=1)
    res = SeparableBathEvolution().run(
        model, engine, eps=0.1, n_steps=1, cutoff=1e-8, cutoff_mode=mode, record_every=1)
    assert res.n_sub_baths == 1
