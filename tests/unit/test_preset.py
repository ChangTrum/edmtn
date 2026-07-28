"""Strategy-preset tests: SolverConfig(preset=...) wiring.

``preset`` fills the recommended compression decomposition (docs/recommended-
config.md).  The trigger is ``compress_decomp == 'exact'`` alone: while it holds,
a preset overwrites ``compress_decomp_q`` too -- an explicitly passed ``q``
included; an explicit ``compress_decomp='rsvd'`` prevents the preset from
changing either compression field.  Default (None) keeps the exact full-SVD
behaviour.  Both presets use quimb rSVD, differing only in power iterations
(balanced = single-pass q=0, robust = cold q=2).
"""

from __future__ import annotations

import numpy as np
import pytest

from edmtn.driver.auto_config import SolverConfig
from edmtn.driver.solver import solve
from edmtn.models import GaudinModel


def test_preset_none_is_exact():
    cfg = SolverConfig(eps=0.1, T=1.0)
    assert cfg.compress_decomp == "exact"


def test_preset_balanced_is_single_pass_rsvd():
    cfg = SolverConfig(eps=0.1, T=1.0, preset="balanced")
    assert cfg.compress_decomp == "rsvd"
    assert cfg.compress_decomp_q == 0


def test_preset_robust_is_cold_rsvd():
    cfg = SolverConfig(eps=0.1, T=1.0, preset="robust")
    assert cfg.compress_decomp == "rsvd"
    assert cfg.compress_decomp_q == 2


def test_explicit_decomp_overrides_preset():
    # an explicit (non-default) compress_decomp is left untouched by the preset
    cfg = SolverConfig(eps=0.1, T=1.0, preset="balanced",
                       compress_decomp="rsvd", compress_decomp_q=2)
    assert cfg.compress_decomp == "rsvd"
    assert cfg.compress_decomp_q == 2          # preset's q=0 did not override


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="unknown preset"):
        SolverConfig(eps=0.1, T=1.0, preset="turbo")


#: the fixed part of the preset comparison; every run below shares it, so a difference can
#: only come from the preset under test
_PRESET_COMMON = dict(T=3.0, eps=0.2, expansion_order=2, cutoff=1e-8, max_bond=400,
                      channel=3)


@pytest.fixture(scope="module")
def exact_reference():
    """The default exact-SVD solve, computed ONCE for every preset.

    It does not depend on the parametrised ``preset``, so building it inside the test body
    ran the identical ``K=12, T=3.0`` Gaudin solve once per parameter set.  Sharing it
    changes no input and no assertion.
    """
    return solve(GaudinModel(g=1.0, K=12), **_PRESET_COMMON)


@pytest.mark.parametrize("preset", ["balanced", "robust"])
def test_preset_end_to_end_matches_exact(preset, exact_reference):
    """solve(preset=...) (rSVD + guard) reproduces the default exact-SVD <S_z(t)>."""
    model = GaudinModel(g=1.0, K=12)
    common = _PRESET_COMMON
    ref = exact_reference                        # default: exact full SVD
    got = solve(model, preset=preset, **common)  # rSVD (guarded)
    n = min(len(ref.polarization), len(got.polarization))
    err = float(np.max(np.abs(np.asarray(ref.polarization[:n])
                              - np.asarray(got.polarization[:n]))))
    assert err < 1e-4


def test_preset_plus_dm_rejected_on_track1():
    """A preset forces compress_decomp='rsvd', which 'dm' cannot execute; the
    combination guard runs after preset resolution and must reject it."""
    with pytest.raises(ValueError, match="dm"):
        SolverConfig(eps=0.1, T=0.2, preset="balanced", compress_method="dm")
