"""The coupling-polarization imaginary-part guard: absolute floor OR relative tolerance.

The guard exists to catch a *gross* leak -- a wrong selector index or convention gives an
imaginary part of order the signal.  A purely relative rule cannot do that on a channel
whose whole history is zero, which channels 1/2 can be exactly, by symmetry: there
``values`` is pure roundoff and its imaginary part is no smaller than its real part, so the
test compares noise with noise and its outcome is decided by whatever floor happens to be
in the expression.

Measured on an identically-zero Gaudin channel (``g=0.7, K=2, eps=0.1, N=3, order=1,
channel=1, cutoff=0, max_bond=None``), where every gauge-invariant contraction -- the
``phi=0`` reduced state *and* every open-arm amplitude -- agreed with an uncompressed run to
``8.95e-16``:

    compress=False   max|Im| = 0.00e+00
    zipup            max|Im| = 9.22e-17
    dm_tracking      max|Im| = 3.20e-15

The old floor was ``1e-15`` (from ``rel * (real_max + 1e-12)``), so the third of those
raised on a result that was numerically identical to the other two.  ``_IMAG_ABS_TOL =
1e-12`` keeps ~300x margin over that measured spread while staying far below the O(1) scale
of the non-zero trajectories the guard polices.
"""

from __future__ import annotations

import numpy as np
import pytest

from edmtn.evolution.separable_bath import SeparableBathEvolution
from edmtn.expansion import FirstOrderExpander
from edmtn.expansion.second_order import SecondOrderExpander
from edmtn.kernels import SeparableKernelEngine
from edmtn.models import GaudinModel
from edmtn.observables import ObservableExtractor
from edmtn.observables.extractor import (
    _IMAG_ABS_TOL,
    _IMAG_REL_TOL,
    _check_imaginary_part,
)

EPS = 0.1
N = 3


def _mps(method, order, **extra):
    model = GaudinModel(g=0.7, K=2)
    kernel = SeparableKernelEngine.from_model(model, T=N * EPS, eps=EPS)
    expander = SecondOrderExpander() if order == 2 else FirstOrderExpander()
    ev = SeparableBathEvolution(expander=expander,
                                compress_method=method if method else "zipup")
    kw = dict(eps=EPS, n_steps=N, cutoff=0.0, max_bond=None)
    if method is None:
        kw["compress"] = False
    return ev.run(model, kernel, **kw, **extra).mps


def _pol(mps, order, channel=1):
    return ObservableExtractor.coupling_polarization_history(
        mps, EPS, channel=channel, order=order)[1]


# -- 1/2. the real zero-signal cases that used to raise -----------------------------

@pytest.mark.parametrize("order", [1, 2])
def test_zero_signal_dm_tracking_does_not_raise(order):
    """The reported failure, both orders.

    Order 1 scales the arm contraction by ``1/eps``; order 2 by ``(1+1j)/eps``.  Both
    amplify the roundoff of an identically-zero channel, so both need the absolute floor.
    """
    pol = _pol(_mps("dm_tracking", order, record_time_reads=True), order)
    assert np.max(np.abs(pol)) < _IMAG_ABS_TOL


# -- 3. the three paths agree on the zero signal ------------------------------------

@pytest.mark.parametrize("order", [1, 2])
def test_zero_signal_agrees_across_compression_paths(order):
    """``compress=False`` is the lossless baseline; zipup and dm_tracking are compared to it.

    zipup passing is a control, not a substitute for the uncompressed reference.
    """
    base = _pol(_mps(None, order), order)
    zipup = _pol(_mps("zipup", order), order)
    track = _pol(_mps("dm_tracking", order, record_time_reads=True), order)
    assert np.max(np.abs(base)) == 0.0                      # exactly zero, not merely small
    for other in (zipup, track):
        np.testing.assert_allclose(other, base, atol=1e-12)


# -- 4. a genuinely non-zero channel is untouched -----------------------------------

def test_nonzero_channel_is_governed_by_the_relative_tolerance():
    """channel=3 (``S_z``) is O(1) here, so the relative term decides and the floor is inert."""
    model = GaudinModel(g=0.7, K=2)
    kernel = SeparableKernelEngine.from_model(model, T=N * EPS, eps=EPS)
    ev = SeparableBathEvolution(expander=FirstOrderExpander())
    mps = ev.run(model, kernel, eps=EPS, n_steps=N, cutoff=0.0, max_bond=None).mps
    pol = _pol(mps, 1, channel=3)
    real_max = float(np.max(np.abs(pol)))
    assert real_max > 1e-3                                   # a real signal, not noise
    assert _IMAG_REL_TOL * real_max > _IMAG_ABS_TOL          # relative term dominates


# -- 5/6. the guard can still fail, and its boundary is where it says --------------

def _guard(imag_max, real_max):
    """Run the SHIPPED guard on a synthetic spectrum.  Rejected -> True.

    Calls the production helper, not a copy of the formula: a copy would keep passing with
    the guard deleted, which is the failure mode this file exists to prevent.
    """
    values = np.array([real_max + 1j * imag_max], dtype=np.complex128)
    try:
        _check_imaginary_part(values)
    except ValueError:
        return True
    return False


def test_gross_imaginary_leakage_still_raises():
    """An O(1) imaginary part on an O(1) signal -- the wrong-index case the guard is for."""
    assert _guard(imag_max=0.5, real_max=1.0)


def test_the_guard_message_carries_the_numbers():
    """So a future failure is diagnosable without re-instrumenting the extractor."""
    with pytest.raises(ValueError) as exc:
        _check_imaginary_part(np.array([1.0 + 0.5j], dtype=np.complex128))
    message = str(exc.value)
    for token in ("max|Im|", "max|Re|", "threshold", "rel_tol", "abs_tol"):
        assert token in message, token


@pytest.mark.parametrize("imag_max, real_max, rejected", [
    (1e-13, 0.0, False),          # below the absolute floor, zero signal -> pass
    (1e-11, 0.0, True),           # above the absolute floor, zero signal -> reject
    (1e-9, 1.0, False),           # below the relative tolerance on an O(1) signal
    (1e-2, 1.0, True),            # above the relative tolerance on an O(1) signal
    (1e-11, 1.0, False),          # floor is inert once the relative term dominates
])
def test_threshold_boundary(imag_max, real_max, rejected):
    assert _guard(imag_max, real_max) is rejected


def test_the_floor_is_what_changed():
    """Guards against a silent revert: the old rule rejected the measured dm_tracking case."""
    measured_imag, measured_real = 3.1951e-15, 2.3011e-15
    old_threshold = _IMAG_REL_TOL * (measured_real + 1e-12)
    assert measured_imag > old_threshold                     # the old rule raised ...
    assert not _guard(measured_imag, measured_real)          # ... the new one does not
