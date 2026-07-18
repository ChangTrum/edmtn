"""The public docs must describe the code that actually exists (P1-16).

P0/P1 changed several public contracts (cutoff modes, expansion-order inheritance, the time
axis, the result fields, compression semantics, the truncation metric, hardware gating).  This
module is the guard against those docs drifting again: it runs the README's small CPU example
for real, checks the documented fields/axes/defaults against live objects, and fails on a short
list of *specific* stale contract strings.

Deliberately lightweight.  It does NOT re-run the numerical matrices in
``test_truncation_metric.py``, any GPU/HPC hardware body, the paper-scale README parameters, or
a nested pytest -- those live in their own suites and stay the source of truth for behaviour.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from edmtn.driver import solve
from edmtn.driver.auto_config import SolverConfig
from edmtn.evolution.separable_bath import SeparableBathEvolution
from edmtn.evolution.single_bath import SingleBathEvolution
from edmtn.models import GaudinModel, SpinBosonModel

ROOT = pathlib.Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")


# -- the README's small CPU examples must actually run -------------------------------------

def test_readme_spinboson_example_runs():
    sb = SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0)
    res = solve(sb, T=0.3, eps=0.1, expansion_order=2, cutoff=1e-8)
    np.testing.assert_allclose(res.times, [0.1, 0.2, 0.3], atol=1e-12)   # [eps..T]
    assert len(res.polarization) == len(res.times)
    assert res.expansion_order == 2
    assert res.backend.startswith("cpu")


def test_readme_gaudin_example_runs_and_fields_exist():
    g = GaudinModel(g=1.0, K=3)
    res = solve(g, T=0.3, eps=0.1, expansion_order=2, cutoff=1e-8, max_bond=64, channel=3)
    np.testing.assert_allclose(res.times, [0.1, 0.2, 0.3], atol=1e-12)
    # every field the README's result table names must exist ...
    for field in ("times", "polarization", "density_matrices", "sub_bath_counts",
                  "sub_bath_bond_dims", "sub_bath_final_density_matrices", "time_bond_dims",
                  "final_time_bond_dims", "truncation_errors", "sub_baths_used",
                  "expansion_order", "observables", "error_metrics", "backend", "mps",
                  "evolution", "bond_dims", "max_bond"):
        assert hasattr(res, field), field
    # ... with the documented axes
    assert res.density_matrices is None                      # Gaudin Track 1: never a rho(t)
    assert len(res.sub_bath_bond_dims) == len(res.sub_bath_counts)
    assert len(res.truncation_errors) == len(res.sub_bath_counts)
    assert res.sub_baths_used == 3
    assert res.final_time_bond_dims == res.mps.bond_dims
    assert res.bond_dims == res.sub_bath_bond_dims           # documented legacy alias


def test_documented_time_grid_rejects_non_integer_ratio():
    with pytest.raises(ValueError):
        SolverConfig(eps=0.07, T=0.3)          # T/eps not an integer -> never silently rounded


# -- documented defaults must match the real dataclass -------------------------------------

def test_config_defaults_match_the_documented_table():
    cfg = SolverConfig(eps=0.1, T=0.3)
    assert cfg.expansion_order is None          # inherits model.time_step_order
    assert cfg.cutoff == 1e-8
    assert cfg.cutoff_mode == "rel"
    assert cfg.max_bond is None
    assert cfg.record_rho is False
    assert cfg.compress_method == "zipup"
    assert cfg.compress_decomp == "exact"
    assert cfg.compress_decomp_q == 2
    assert cfg.compress_canon == "quimb"
    assert cfg.preset is None
    assert cfg.sub_baths is None
    assert cfg.backend == "cpu"                 # NOT 'auto'
    assert cfg.precision == "f64"
    assert cfg.pathfinder == "cuquantum"
    assert cfg.time_windows is None


def test_expansion_order_inherits_then_overrides_as_documented():
    g = GaudinModel(g=1.0, K=2)                 # time_step_order = 2
    assert solve(g, T=0.2, eps=0.1, channel=3).expansion_order == 2
    assert solve(g, T=0.2, eps=0.1, channel=3, expansion_order=1).expansion_order == 1


def test_time_windows_is_documented_as_unavailable():
    with pytest.raises(NotImplementedError):
        SolverConfig(eps=0.1, T=0.2, backend="hpc", time_windows=2)


# -- the docstrings that carried the fabricated ref_index / rel_ref default ------------------

@pytest.mark.parametrize("obj", [SolverConfig, SingleBathEvolution.run, SeparableBathEvolution.run])
def test_public_docstrings_dropped_ref_index_and_rel_ref_default(obj):
    doc = obj.__doc__ or ""
    assert "ref_index" not in doc
    assert not re.search(r"default[^.]{0,40}rel_ref", doc)


# -- specific stale contract strings must not reappear in the CURRENT-contract docs ---------
# (historical ledgers and validation tests legitimately mention 'auto'/'rel_ref', so they are
#  excluded -- this scan targets exact stale CLAIMS, never a bare token.)

_CURRENT_DOCS = ["README.md", "docs/guides/recommended-config.md",
                 "docs/benchmarks/cpu-vs-gpu-edm.md", "docs/research/coupling-scaling-law.md"]

_STALE = [
    (r"backend='auto'", "backend 'auto' was removed; the default is 'cpu'"),
    (r"examples/studies/", "the directory is examples/research/"),
    (r"round\(T\s*/\s*eps\)", "T/eps must be an exact integer; it is not rounded"),
    (r"ref_index", "there is no ref_index field"),
]


@pytest.mark.parametrize("relpath", _CURRENT_DOCS)
def test_current_docs_have_no_stale_contract_strings(relpath):
    text = (ROOT / relpath).read_text(encoding="utf-8")
    for pattern, why in _STALE:
        assert not re.search(pattern, text), f"{relpath}: stale contract {pattern!r} -- {why}"


def test_readme_states_the_key_current_contracts():
    """The contracts most likely to be misread if they silently drift."""
    # time axis + integer grid
    assert "[eps, 2 eps, …, T]" in README or "[eps, 2 eps, ..., T]" in README
    assert "positive integer" in README
    # compression semantics (compress=False is NOT a zero-cutoff compression)
    assert "skipped entirely" in README
    # truncation metric: the three distinguishable outcomes
    assert "unmeasurable" in README                      # rSVD -> None
    assert "discarded nothing" in README                 # 0.0
    assert "exact-only" in README or "no Track-1 compression" in README   # Track 2 -> []
    # Track 2 scope + hardware gating contract
    assert "Separable/Gaudin only" in README
    assert "--require-multigpu=4" in README
    assert "EDMTN_MULTIGPU_RESULT" in README


def test_readme_does_not_promise_identical_backend_results():
    assert "Same physics, same result" not in README
    assert "agree to round-off." not in README
    # rsvd silently falls back to exact full SVD off NumPy, so the paths are NOT identical
    assert "follow the same algorithm" not in README


def test_readme_does_not_overstate_compressed_accuracy():
    """cutoff is a local per-bond threshold, never an observable/trajectory error bound."""
    assert "compressed paths agree within the configured" not in README
    assert "not** an error bound on the" in README


def test_readme_marks_mixed_precision_experimental():
    """PrecisionPolicy declares f64 decompose, but solve() only casts the contraction."""
    assert "contracts in f32 and decomposes in f64" not in README
    assert "not wired into the solve pipeline" in README


def test_readme_documents_the_strict_channel_contract():
    assert "strict 1-based integer" in README
    assert "`1` only" in README                      # spin-boson
    assert "`S_x`, `S_y`, `S_z`" in README           # Gaudin 1/2/3
    assert "negative indexing" in README             # channel=0 is rejected, not last-operator


def test_readme_documents_timestep_convergence():
    for token in ("timestep_convergence(", "conv.deviation", "conv.converged", "conv.metadata",
                  "dev, ok = conv", "coarse_sub_baths_used"):
        assert token in README, token


def test_multigpu_status_is_not_claimed_as_validated():
    """Single-GPU is verified; the 4-GPU path is currently blocked -- do not call the whole
    recipe 'validated'."""
    for rel in ("README.md", "cluster/cutensornet_mpi.sbatch", "cluster/cutensornet_multigpu.sbatch"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "validated recipe" not in text, rel
        assert "validated recipes" not in text, rel


def test_recommended_config_matches_the_real_preset_behaviour():
    g = (ROOT / "docs/guides/recommended-config.md").read_text(encoding="utf-8")
    assert "balanced default" not in g                  # the API default is preset=None
    assert "accuracy always below" not in g             # a measurement, not a guarantee
    for stale_api in ("RandomizedSVD(", "CholeskyQR(", "canonicalization="):
        assert stale_api not in g, stale_api
    assert "preset=None` is the API default" in g
    # robust is still randomized rSVD -- it must not be sold as bit-reproducible
    assert "still randomized rSVD" in g


def test_public_examples_do_not_silently_round_the_time_grid():
    """T/eps must be an exact positive integer; a public example must not demonstrate
    ``int(round(T / eps))``, which is exactly the pattern SolverConfig now rejects."""
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "examples").rglob("*.py")
        if re.search(r"round\(\s*T\s*/\s*eps\s*\)", path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"examples still round the time grid: {offenders}"


def test_model_docstrings_state_ranges_and_capability_boundaries():
    from edmtn.cumulants import GaussianCumulants, SeparableCorrelation
    from edmtn.models.gaudin import GaudinBathParams

    sb = SpinBosonModel.__doc__
    assert "legal no-coupling baseline" in sb          # J0 = 0 is valid
    assert "NotImplementedError" in sb                 # finite-T is an engine limit, not a ValueError
    gm = GaudinModel.__doc__
    assert "strict non-``bool`` integer" in gm
    assert "generation order" in gm                    # ou is not sorted
    assert "no sorting and no normalisation" in gm     # custom is verbatim
    assert "``g_K == g`` is\n          NOT guaranteed" in gm or "g_K == g" in gm
    # normalisation identity is claimed only for the named profiles
    assert "normalised named profiles" in GaudinBathParams.__doc__
    # copied + read-only containers
    assert "read-only" in GaussianCumulants.__doc__
    assert "read-only" in SeparableCorrelation.__doc__
