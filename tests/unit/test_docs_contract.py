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
                 "docs/benchmarks/cpu-vs-gpu-edm.md", "docs/research/coupling-scaling-law.md",
                 # the Sphinx documentation's current-contract pages (historical
                 # design/research ledgers stay excluded)
                 "docs/index.md", "docs/guide/index.md", "docs/developer/index.md",
                 "docs/getting-started/installation.md", "docs/getting-started/quickstart.md",
                 "docs/guide/concepts.md", "docs/guide/models.md",
                 "docs/guide/solving.md", "docs/guide/results.md",
                 "docs/guide/compression.md", "docs/guide/backends.md",
                 "docs/guide/convergence.md", "docs/guide/performance.md",
                 "docs/guide/cluster.md"]

_STALE = [
    (r"backend='auto'", "backend 'auto' was removed; the default is 'cpu'"),
    (r"examples/studies/", "the directory is examples/research/"),
    (r"round\(T\s*/\s*eps\)", "T/eps must be an exact integer; it is not rounded"),
    (r"ref_index", "there is no ref_index field"),
    (r"left (?:at|those fields at) (?:their )?defaults",
     "presets never checked q; the trigger is compress_decomp == 'exact' alone"),
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


# -- the Sphinx documentation pages (docs/) must match the same contracts ------------------

_QUICKSTART = ROOT / "docs/getting-started/quickstart.md"
_INSTALLATION = ROOT / "docs/getting-started/installation.md"


def test_quickstart_python_blocks_execute():
    """Every ``python`` fence in the quickstart must actually run, in order."""
    text = _QUICKSTART.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert blocks, "quickstart.md has no python fences"
    namespace: dict = {}
    for block in blocks:
        exec(compile(block, str(_QUICKSTART), "exec"), namespace)  # noqa: S102


def test_quickstart_states_the_key_contracts():
    text = _QUICKSTART.read_text(encoding="utf-8")
    assert "positive integer" in text                  # the time grid is never rounded
    assert "[eps, 2 eps, ..., T]" in text or "[eps, 2 eps, …, T]" in text
    assert "strict 1-based" in text                    # the channel contract
    assert "`S_z` (its single coupling channel)" in text   # spin-boson channel 1 is S_z
    assert "`S_x`, `S_y`, `S_z`" in text               # Gaudin 1/2/3


def test_installation_matches_distribution_and_packaging_contract():
    text = _INSTALLATION.read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # packaging metadata: the Python floor and dependency names come from pyproject.toml
    assert 'requires-python = ">=3.11"' in pyproject   # guard the source of truth itself
    assert "Python 3.11 or newer" in text
    for dep in ("NumPy", "SciPy", "quimb", "autoray"):
        assert dep in text, dep
    # distribution policy (not packaging metadata): the project is source-only today
    assert "not published on PyPI" in text
    assert "cupy-cuda12x" in text                      # GPU extra stays optional + explicit
    assert "cuquantum-python-cu12" in text             # hpc extra


def test_concepts_separates_theorems_from_measurements():
    text = (ROOT / "docs/guide/concepts.md").read_text(encoding="utf-8")
    assert "arXiv:2509.00424" in text                  # the paper is cited, with its ID
    assert "at most linearly" in text                  # the theorem bound, stated as a bound
    assert "proved under its assumptions" in text      # theorem scope is explicit
    assert "not** linear in `T`" in text               # runtime is NOT claimed linear
    assert "not** an error" in text                    # cutoff is not an error bound


def test_models_page_states_the_capability_boundaries():
    text = (ROOT / "docs/guide/models.md").read_text(encoding="utf-8")
    assert "NotImplementedError" in text               # both temperature limits are engine limits
    assert "generation order" in text                  # ou is not sorted
    assert "no sorting and no normalisation" in text   # custom arrays are verbatim
    assert "read-only" in text                         # supplied arrays are copied + locked


def test_solving_page_matches_the_config_contract():
    text = (ROOT / "docs/guide/solving.md").read_text(encoding="utf-8")
    assert "There is no `'auto'`" in text              # backend menu has no 'auto'
    for mode in ("abs", "rel", "sum2", "rsum2", "sum1", "rsum1"):
        assert f"`{mode}`" in text, mode               # the full cutoff_mode menu
    assert "not wired into the solve\n  pipeline" in text or \
           "not wired into the solve pipeline" in text  # mixed precision stays experimental
    assert "reserved; must be `None`" in text          # time_windows
    assert "never silently rounded" in text or "never silently clamped" in text


def test_results_page_states_the_truncation_tristate():
    text = (ROOT / "docs/guide/results.md").read_text(encoding="utf-8")
    assert "the metric is defined" in text             # a number
    assert "no discarded weight" in text               # 0.0, incl. skipped compression
    assert "unmeasurable, not zero" in text            # rsvd -> None
    assert "exact-only and performs no truncation" in text   # Track 2 -> []
    assert "not a cumulative or global error" in text  # local per-record quantity
    assert "legacy" in text                            # bond_dims/max_bond are aliases
    # the dm path sums discarded density-matrix EIGENVALUES (lambda = sigma^2) ...
    assert "`Σ lambda_i`" in text
    assert "lambda_i = sigma_i²" in text
    # ... directly -- it must never be described as summing their squares
    assert "**not** `Σ lambda_i²`" in text


def test_preset_overrides_follow_the_real_trigger():
    """The preset trigger is ``compress_decomp == 'exact'`` ALONE: an explicitly
    passed non-default q is still overwritten; an explicit 'rsvd' prevents the
    preset from changing either compression field (docs/guide/solving.md)."""
    c1 = SolverConfig(eps=0.1, T=0.3, preset="balanced", compress_decomp_q=7)
    assert c1.compress_decomp == "rsvd"
    assert c1.compress_decomp_q == 0                   # the explicit q=7 WAS overwritten
    c2 = SolverConfig(eps=0.1, T=0.3, preset="balanced",
                      compress_decomp="rsvd", compress_decomp_q=7)
    assert c2.compress_decomp == "rsvd"
    assert c2.compress_decomp_q == 7                   # explicit 'rsvd': neither field changed


def test_preset_docs_state_the_overwrite_rule():
    """README and the config guide must state that a triggered preset overwrites
    even an explicitly passed q (the stale-claim scan bans the old wording)."""
    assert "overwrites `compress_decomp_q`" in README
    guide = (ROOT / "docs/guides/recommended-config.md").read_text(encoding="utf-8")
    assert "explicitly passed `compress_decomp_q`" in guide.replace("\n> ", " ")


def test_convergence_python_blocks_execute():
    """Every ``python`` fence in the convergence guide must actually run."""
    text = (ROOT / "docs/guide/convergence.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert blocks, "convergence.md has no python fences"
    namespace: dict = {}
    for block in blocks:
        exec(compile(block, "docs/guide/convergence.md", "exec"), namespace)  # noqa: S102
    assert "keeps **every** other field" in text          # replace()-derived fine run
    assert "CuTensorNetContractionError" in text          # the EDMTN Track-2 error is documented
    assert "native runtime exceptions" in text            # ... without claiming it wraps everything
    assert "non-finite/negative truncation metric" in text  # FloatingPointError sources


def test_compression_page_states_the_boundaries():
    raw = (ROOT / "docs/guide/compression.md").read_text(encoding="utf-8")
    text = " ".join(raw.split())                          # collapse hard line wraps
    assert "skipped entirely" in text                     # compress=False
    assert "no-discard recompression" in text             # cutoff=0 AND max_bond=None
    assert "`compress=True, cutoff=0, max_bond=None`" in text
    assert "truncates even at `cutoff=0`" in text         # max_bond alone can discard
    assert "density-matrix **eigendecomposition**" in text  # dm exact is NOT an SVD
    assert "`dm` | `exact` only | `quimb` only" in text   # the compatibility table
    assert "On Track 1, unsupported combinations are rejected" in text
    assert "ignored-field behaviour" in text              # the hpc exception is stated
    assert "silent resolution guard" in text              # rsvd guard exists ...
    assert "not a universal guarantee" in text
    assert "cannot measure what it discarded" in text     # rsvd -> None
    assert "not an error bound" in text                   # cutoff stays local


def test_backends_page_states_the_boundaries():
    raw = (ROOT / "docs/guide/backends.md").read_text(encoding="utf-8")
    text = " ".join(raw.split())                          # collapse hard line wraps
    assert "**not** bit-identical" in text
    assert "falls back to exact full SVD" in text         # rsvd off NumPy
    assert "non-NumPy" in text
    assert "experimental and unvalidated" in text
    assert "does **not** remove the finite-`eps`" in text  # Track-2 'exact' boundary
    assert "spin-boson is not available" in text          # Track 2 scope
    assert "`numpy` and `cupy` are accepted aliases" in text   # the full public menu
    assert "floating-point contraction order" in text     # path/slicing also differ


def test_cluster_page_states_status_and_contracts():
    text = (ROOT / "docs/guide/cluster.md").read_text(encoding="utf-8")
    assert "site-specific" in text                        # launchers are not portable
    assert "legacy/historical" in text                    # the two pmi2 recipes
    assert "blocked" in text                              # 4-GPU is NOT accepted today
    assert "not** constitute current-environment" in text  # old pass != acceptance
    assert "EDMTN_MULTIGPU_RESULT" in text
    assert "not part of the solver's\npublic API" in text or \
           "not part of the solver's public API" in text


def test_performance_page_keeps_measurement_context():
    text = (ROOT / "docs/guide/performance.md").read_text(encoding="utf-8")
    assert "dated measurement" in text                    # figures carry their context
    assert "never a general guarantee" in text
    assert "not linear in `T`" in text                    # runtime claim stays bounded
    assert "Minutes-to-hours" not in text                 # no unsourced runtime estimates
    assert "reproduce_fig4.py" in text and "reproduce_fig6.py" in text
    assert "illustrative" in text                         # non-paper cutoffs are labelled
    assert "unused cached" in text                        # not a constant-memory guarantee
    assert "not a constant-memory guarantee" in text


@pytest.mark.parametrize("obj", [SolverConfig, SingleBathEvolution.run,
                                 SeparableBathEvolution.run])
def test_cutoff_zero_docstrings_carry_the_max_bond_qualifier(obj):
    """The no-discard reading of cutoff=0 holds only with max_bond=None."""
    assert "max_bond=None" in (obj.__doc__ or "")


def test_cutoff_zero_docs_carry_the_max_bond_qualifier():
    for rel in ("README.md", "docs/guide/solving.md", "docs/guide/compression.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "max_bond=None" in text, rel


def test_readme_paper_scale_matches_the_reproduce_scripts():
    """The paper configurations are the reproduce scripts' recorded defaults; other
    parameter sets must be labelled illustrative, and runtime estimates need a source."""
    assert "Minutes-to-hours" not in README
    assert "spin-boson, paper scale" not in README
    assert "reproduce_fig4.py" in README and "reproduce_fig6.py" in README
    assert "`cutoff=1e-5`" in README          # fig4's recorded default
    assert "`cutoff=1e-6`" in README          # fig6's recorded default
    assert "illustrative" in README           # the tighter-cutoff Gaudin block is labelled


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
