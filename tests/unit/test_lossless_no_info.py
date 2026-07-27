"""No truncation metric is requested when nothing can be discarded.

``cutoff = 0`` with ``max_bond = None`` gives quimb ``truncation = False``
(``parse_split_opts``), and an ``absorb`` of left/right then resolves ``method='auto'`` to
**qr** instead of svd (``parse_method_absorb``).  A QR takes no ``info``, and quimb's
generic ``qr_stabilized`` forwards ``**kwargs`` straight into ``xp.linalg.qr``:

    Q, R = xp.linalg.qr(x, **kwargs)          # quimb/tensor/decomp.py

so an injected ``info`` reaches ``cupy.linalg.qr`` and raises
``TypeError: qr() got an unexpected keyword argument 'info'``.  On NumPy the same call
survives only by accident -- ``qr_stabilized_numpy`` drops ``**kwargs`` for 2-d input on its
way to the numba kernel -- which is why this was invisible on CPU and fatal on GPU
(``test_gpu_hpc.py`` parity tests).

The fix does not patch quimb or teach ``cupy.linalg.qr`` to swallow the argument: in this
regime the discarded weight is exactly ``0.0``, so no accumulator is created and no ``info``
is injected.  A rank-limiting ``max_bond`` still truncates at ``cutoff = 0`` and is
deliberately NOT covered by the short circuit.
"""

from __future__ import annotations

import numpy as np
import pytest

from edmtn.evolution.mps_utils import EDMMPS
from edmtn.evolution.quimb_edm import QuimbEDM

METHODS = ("zipup", "dm", "direct")


def _mps(n=5, chi=6, d_phys=3, d=2, seed=0):
    rng = np.random.default_rng(seed)
    d2 = d * d
    tensors = []
    for p in range(n):
        left = d2 if p == 0 else chi
        right = d2 if p == n - 1 else chi
        tensors.append(rng.normal(size=(d_phys, left, right))
                       + 1j * rng.normal(size=(d_phys, left, right)))
    return EDMMPS(tensors=tensors, d=d, d_phys=d_phys,
                  rho0_vec=rng.normal(size=d2) + 0j)


def _compress(**kw):
    opts = dict(cutoff=0.0, cutoff_mode="rel", max_bond=None, decomp="exact")
    opts.update(kw)
    return QuimbEDM.from_edmmps(_mps()).compress(**opts)


# -- the regime that could not truncate --------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_lossless_reports_exactly_zero_not_none(method):
    """``0.0`` is the honest answer -- nothing was discarded, and that is measured."""
    assert _compress(method=method).max_discarded_weight == 0.0


@pytest.mark.parametrize("method", METHODS)
def test_no_info_reaches_the_split_when_nothing_can_be_discarded(method):
    """The spy is the point: without it, a passing CPU run proves nothing about CuPy.

    Intercepts ``Tensor.split``, the call every method actually goes through, and asserts
    the spy **fired** before asserting anything about what it saw.  Patching
    ``decomp.qr_stabilized`` does not work and is the trap this test was written into
    once already: quimb dispatches through the ``decomp._SPLIT_FNS`` registry, which still
    holds the original function object, so the spy never runs and
    ``all(... for kw in seen)`` is vacuously true over an empty list -- a check that cannot
    fail is not evidence.
    """
    import quimb.tensor as qtn

    seen = []
    original = qtn.Tensor.split

    def spy(self, *args, **kwargs):
        seen.append(dict(kwargs))
        return original(self, *args, **kwargs)

    qtn.Tensor.split = spy
    try:
        _compress(method=method)
    finally:
        qtn.Tensor.split = original
    assert seen, "the spy never fired -- it is not on the real call path"
    offenders = [kw for kw in seen if "info" in kw]
    assert not offenders, offenders


def test_lossless_compression_is_still_lossless():
    """The short circuit must not change the numbers, only what is asked of the split."""
    base = QuimbEDM.from_edmmps(_mps()).to_edmmps().reduced_density_matrix()
    for method in METHODS:
        got = _compress(method=method).to_edmmps().reduced_density_matrix()
        np.testing.assert_allclose(np.asarray(got), np.asarray(base), atol=1e-9)


# -- the regimes that CAN truncate keep the real metric ----------------------------

@pytest.mark.parametrize("method", METHODS)
def test_rank_limit_at_zero_cutoff_still_measures(method):
    """``cutoff = 0`` does NOT imply lossless: ``max_bond`` truncates on its own."""
    weight = _compress(method=method, max_bond=2).max_discarded_weight
    assert weight is not None and weight > 0.0


@pytest.mark.parametrize("method", METHODS)
def test_positive_cutoff_still_measures(method):
    """A cutoff loose enough to actually bite -- a random MPS has a flat spectrum, so a
    tight relative cutoff discards nothing and would make this assertion vacuous."""
    res = _compress(method=method, cutoff=0.5, max_bond=None)
    assert max(res.bond_dims) < 6                    # it really did truncate
    assert res.max_discarded_weight > 0.0


# -- the documented rsvd semantics are untouched -----------------------------------

@pytest.mark.parametrize("method", ("zipup", "direct"))
def test_rsvd_still_reports_none(method):
    """rSVD never sees the tail it omitted, so it has no honest weight to report."""
    assert _compress(method=method, decomp="rsvd", max_bond=2).max_discarded_weight is None
