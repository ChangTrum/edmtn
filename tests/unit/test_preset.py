"""Strategy-preset tests (P4): SolverConfig(preset=...) wiring.

`preset` fills the recommended decomposition/canonicalisation (docs/recommended-
config.md) without overriding explicitly-passed strategies; default (None) keeps the
exact StandardSVD + Householder behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from edmtn.decomposition import RandomizedSVD, StandardSVD
from edmtn.driver.auto_config import SolverConfig
from edmtn.driver.solver import solve
from edmtn.models import GaudinModel


def test_preset_none_leaves_strategies_unset():
    cfg = SolverConfig(eps=0.1, T=1.0)
    assert cfg.decomposition is None          # -> StandardSVD at build time
    assert cfg.canonicalization is None        # -> Householder QR


def test_preset_balanced_is_single_pass_rsvd_householder():
    cfg = SolverConfig(eps=0.1, T=1.0, preset="balanced")
    assert isinstance(cfg.decomposition, RandomizedSVD)
    assert cfg.decomposition.n_iter == 0
    assert cfg.canonicalization is None        # Householder (the measured default)


def test_preset_robust_is_cold_rsvd():
    cfg = SolverConfig(eps=0.1, T=1.0, preset="robust")
    assert isinstance(cfg.decomposition, RandomizedSVD)
    assert cfg.decomposition.n_iter == 2


def test_explicit_strategy_overrides_preset():
    explicit = StandardSVD()
    cfg = SolverConfig(eps=0.1, T=1.0, preset="balanced", decomposition=explicit)
    assert cfg.decomposition is explicit       # explicit wins over preset


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="unknown preset"):
        SolverConfig(eps=0.1, T=1.0, preset="turbo")


def test_preset_fresh_instance_per_config():
    a = SolverConfig(eps=0.1, T=1.0, preset="balanced").decomposition
    b = SolverConfig(eps=0.1, T=1.0, preset="balanced").decomposition
    assert a is not b                          # factories, not a shared singleton


@pytest.mark.parametrize("preset", ["balanced", "robust"])
def test_preset_end_to_end_matches_default(preset):
    """solve(preset=...) reproduces the default StandardSVD <S_z(t)> to < xi."""
    model = GaudinModel(g=1.0, K=12)
    common = dict(T=3.0, eps=0.2, expansion_order=2, cutoff=1e-6, max_bond=400, channel=3)
    ref = solve(model, **common)
    got = solve(model, preset=preset, **common)
    n = min(len(ref.polarization), len(got.polarization))
    err = float(np.max(np.abs(np.asarray(ref.polarization[:n])
                              - np.asarray(got.polarization[:n]))))
    assert err < 1e-6
