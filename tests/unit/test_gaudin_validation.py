"""GaudinModel / coupling-profile parameter validation + coupling-array isolation (P0-3).

Two threads:
  * Parameters (g, K, order, custom couplings, profile knobs, effective_coupling L) that
    define the spectra/transfer tensors must be rejected loudly (ValueError) at the entry
    point, not silently truncated (K=2.9) or leaked as NaN/Inf couplings or raw
    Overflow/TypeError.
  * The per-sub-bath coupling array (and the correlation transfer tensor) must never be
    aliased across layers: mutating a caller's array, or a stored array, must not change
    a model / correlation / kernel after construction. Stored arrays are read-only copies.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from edmtn.cumulants import SeparableCorrelation
from edmtn.kernels.separable_mpo import SeparableKernelEngine
from edmtn.models import (
    GaudinModel,
    coupling_profile,
    exponential_couplings,
    linear_couplings,
    ou_couplings,
    random_couplings,
    uniform_couplings,
)

_BASE = dict(g=1.0, K=2)


# -- model parameter rejection matrix (ValueError) -------------------------

@pytest.mark.parametrize("kw", [
    {"g": np.nan}, {"g": np.inf}, {"g": 0},
    {"K": 2.9}, {"K": True}, {"K": 0},
    {"time_step_order": 1.0}, {"time_step_order": True},
    {"coupling": [1.0, np.nan]}, {"coupling": [1.0, np.inf]},
    {"coupling": [10 ** 400, 1.0]},            # too-large int -> ValueError, not OverflowError
    {"coupling": [1 + 2j, 1.0]},               # complex -> ValueError, not TypeError
    {"coupling": np.ones(5)},                  # wrong length for K=2
])
def test_bad_model_parameters_rejected(kw):
    with pytest.raises(ValueError):
        GaudinModel(**{**_BASE, **kw})


@pytest.mark.parametrize("cp", [
    ("exp", {"beta": np.nan}), ("exp", {"beta": np.inf}), ("exp", {"beta": 0}),
    ("ou", {"rho": np.nan}), ("ou", {"rho": 1.0}), ("ou", {"rho": -0.1}),
    ("random", {"low": np.nan}), ("random", {"high": np.inf}),
    ("random", {"low": 1.0, "high": 1.0}),     # low < high required
])
def test_bad_profile_params_rejected(cp):
    kind, params = cp
    with pytest.raises(ValueError):
        GaudinModel(g=1.0, K=3, coupling=kind, coupling_params=params)


@pytest.mark.parametrize("L", [1.5, True, 0, 3])
def test_effective_coupling_bad_L_rejected(L):
    m = GaudinModel(g=1.0, K=2)
    with pytest.raises(ValueError):
        m.effective_coupling(L)


# -- public profile functions validate g / K on their own ------------------

@pytest.mark.parametrize("call", [
    lambda: linear_couplings(np.nan, 2),
    lambda: uniform_couplings(np.inf, 2),
    lambda: random_couplings(np.nan, 2),
    lambda: ou_couplings(np.inf, 2),
    lambda: exponential_couplings(1.0, 2.9),     # non-integer K
    lambda: linear_couplings(1.0, True),         # bool K
    lambda: coupling_profile("linear", np.nan, 2),
])
def test_public_profiles_reject_bad_g_K(call):
    with pytest.raises(ValueError):
        call()


# -- coupling-array isolation ----------------------------------------------

def test_model_couplings_not_aliased_to_source():
    source = np.array([0.8, 0.6])
    model = GaudinModel(g=1.0, K=2, coupling=source)
    before = model.couplings.copy()
    source[0] = 8.0
    assert_allclose(model.couplings, before)


def test_model_couplings_read_only():
    model = GaudinModel(g=1.0, K=2, coupling=[0.8, 0.6])
    with pytest.raises(ValueError):
        model.couplings[0] = 5.0


def test_kernel_correlation_isolated_from_external_mutation():
    source = np.array([0.8, 0.6])
    model = GaudinModel(g=1.0, K=2, coupling=source)
    kernel = SeparableKernelEngine.from_model(model, T=0.2, eps=0.1)
    c_before = kernel.corr.couplings.copy()
    t_before = kernel.corr.transfer.copy()
    source[0] = 8.0                              # mutate every external array we own
    assert_allclose(model.couplings, [0.8, 0.6])
    assert_allclose(kernel.corr.couplings, c_before)
    assert_allclose(kernel.corr.transfer, t_before)
    with pytest.raises(ValueError):
        kernel.corr.couplings[0] = 1.0
    with pytest.raises(ValueError):
        kernel.corr.transfer[0, 0, 0, 0] = 1.0


def test_separable_correlation_copies_and_freezes_arrays():
    coup = np.array([0.8, 0.6])
    trans = np.zeros((2, 7, 4, 4), dtype=np.complex128)
    trans[0, 2, 0, 1] = 0.16
    corr = SeparableCorrelation(eps=0.1, n_steps=2, couplings=coup, transfer=trans)
    # mutating the originals must not touch the stored copies
    coup[0] = 8.0
    trans[0, 2, 0, 1] = 4.16
    assert corr.couplings[0] == 0.8
    assert corr.transfer[0, 2, 0, 1] == 0.16
    # stored arrays (and views of them) are read-only
    with pytest.raises(ValueError):
        corr.couplings[0] = 1.0
    with pytest.raises(ValueError):
        corr.transfer[0, 2, 0, 1] = 1.0
    with pytest.raises(ValueError):
        corr.transfer_for(0)[2, 0, 1] = 1.0


# -- legality: negative / un-normalised custom couplings --------------------

def test_negative_custom_couplings_allowed():
    m = GaudinModel(g=1.0, K=2, coupling=[-0.8, 0.6])
    assert_allclose(m.couplings, [-0.8, 0.6])


def test_unnormalised_custom_effective_coupling_reflects_array():
    # custom couplings are used verbatim: effective_coupling is sqrt(sum g_k^2) of the
    # supplied array (here 5.0), NOT necessarily g (=1.0)
    m = GaudinModel(g=1.0, K=2, coupling=[3.0, 4.0])
    assert m.effective_coupling() == pytest.approx(5.0)
    assert m.effective_coupling() != pytest.approx(1.0)
