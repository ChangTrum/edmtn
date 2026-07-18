# Compression

Track 1 keeps the EDM tractable by compressing the tensor network after
every step (single-bath) or fold (separable). When compression is
enabled, each sweep is dispatched through quimb's
`tensor_network_1d_compress`, executed via autoray on whatever backend
the arrays live on; the contraction, canonicalisation and decomposition
steps depend on `compress_method`, and a sweep discards only what the
settings actually allow. This page covers the mechanics; the knobs
themselves are listed in {doc}`solving`, and what the resulting metric
means in {doc}`results`.

## `compress` vs `cutoff=0` — not the same thing

On the direct evolution API (`SingleBathEvolution.run` /
`SeparableBathEvolution.run`; see {doc}`../api/evolution`):

| setting | what happens |
|---|---|
| `compress=False` | the sweep is **skipped entirely** — exact, with exponentially growing bonds (small-problem reference checks) |
| `compress=True, cutoff=0, max_bond=None` | a **no-discard recompression** with the selected method/decomposition — nothing is discarded |
| `compress=True` with `cutoff>0` and/or a rank-limiting `max_bond` | truncation **may** occur — whether anything was actually discarded is what the metric records (`0.0` vs a positive value) |

Note the `max_bond=None` qualifier: a rank-limiting `max_bond` truncates
even at `cutoff=0` (e.g. `cutoff=0, max_bond=2` produces a positive
discarded weight).

## Methods

`compress_method` selects the sweep algorithm:

- `zipup` (default) — fast and low-memory; full SVD under the exact
  decomposition;
- `direct` — the direct sweep; full SVD under the exact decomposition;
- `dm` — the density-matrix method (fastest but lower precision). Under
  the exact decomposition it uses a density-matrix **eigendecomposition**,
  not an SVD: the split object is `ρ`, whose eigenvalues are `λ = σ²` —
  which is why the dm path measures the discarded weight as
  `Σ λ_discarded` (see {doc}`results`).

## Decompositions and the resolution guard

`compress_decomp` selects the per-bond decomposition:

- `exact` (default) — the deterministic exact decomposition: a full SVD
  under `zipup`/`direct`, a density-matrix eigendecomposition under
  `dm`; the only setting that can report a real truncation metric;
- `rsvd` — a randomized SVD with `compress_decomp_q` power iterations
  (`2` cold, `0` single-pass). A **silent resolution guard** falls back
  to exact full SVD when the randomized result is under-resolved *or the
  backend is not NumPy*. The guard covers the failure modes it detects,
  but rSVD remains a randomized algorithm: measured agreement with full
  SVD is a benchmark result at stated tolerances, not a universal
  guarantee. Its sketch never forms the omitted tail of the spectrum, so
  it cannot measure what it discarded — the truncation metric is `None`.

`compress_canon` selects the canonicalisation QR: `quimb` (default),
`householder` or `cholqr`.

`cutoff` is a **local per-bond** threshold under the selected
`cutoff_mode`; it is not an error bound on the polarization, `rho(t)` or
the trajectory. Per public record the metric aggregates as the *maximum*
over the bonds, sub-steps or folds the record covers.

## Compatibility

Not every combination of the three knobs is executable. On Track 1,
unsupported combinations are rejected with `ValueError` by
`SolverConfig`, by a direct `run()`, and by the low-level
`QuimbEDM.compress()`. `backend='hpc'` does not consume these
compression fields at all, so `SolverConfig` preserves their
ignored-field behaviour there:

| `compress_method` | supported `compress_decomp` | supported `compress_canon` |
|---|---|---|
| `zipup` | `exact`, `rsvd` | `quimb`, `householder`, `cholqr` |
| `direct` | `exact`, `rsvd` | `quimb`, `householder`, `cholqr` |
| `dm` | `exact` only | `quimb` only |

quimb's `dm` path reaches the split with PSD-split keywords and forwards
canonicalisation options into it; only the exact eigendecomposition
driver with the default canonicalisation accepts that call signature. A
preset forces `rsvd`, so on Track 1 `preset` + `dm` is rejected too.

## Presets

`preset='balanced'` (rSVD, `q=0`) and `preset='robust'` (rSVD, `q=2`)
trade the measurable metric for speed; no preset keeps the exact
decomposition. The trigger and override semantics are in {doc}`solving`;
the measurements behind the recommendations are in
{doc}`../guides/recommended-config`.

## Track 2

`backend='hpc'` performs no Track-1 compression at all — none of the
knobs on this page apply there, and its truncation metric is `[]`. See
{doc}`backends`.
