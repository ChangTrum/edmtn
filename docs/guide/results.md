# Results

`solve()` returns a `SolverResult`. Every array field names its own
horizontal axis, so you never need to reach into the internal
`res.evolution` object to know what an index means. Axis-specific
optional fields generally use `None` when the pipeline that ran does not
produce them; collection fields may use `[]` or `{}` where documented
below — absence is explicit, never silently zero-filled.

## Time-axis fields

| field | contract |
|---|---|
| `times` | the physical grid `[eps, 2 eps, ..., T]`, ascending |
| `polarization` | `<S_a(t)>` for the selected channel, aligned with `times` |
| `density_matrices` | `rho(t)` aligned 1:1 with `times`, or `None` when the pipeline produces no time-axis state history. Single-bath: present whenever reduced states were recorded (`record_rho=True`, custom observables, or second order — which needs them anyway). Track 2: always present. **Separable/Gaudin Track 1: always `None`** — its per-`L` states live on the fold axis, below |
| `time_bond_dims` | max bond after each physical step (single-bath Track 1); `None` elsewhere |
| `observables` | custom observable histories (single-bath Track 1 only; separable Track 1 and Track 2 raise `NotImplementedError` when custom observables are requested) |

## Fold-axis fields (separable/Gaudin Track 1)

| field | contract |
|---|---|
| `sub_bath_counts` | the recorded sub-bath counts `L` |
| `sub_bath_bond_dims` | `D_L` after folding in `L` sub-baths, aligned with `sub_bath_counts` |
| `sub_bath_final_density_matrices` | `rho_L(T)` per recorded `L` — a *final-time* state per fold count, **not** a time history; present only with `record_rho=True` |
| `sub_baths_used` | how many sub-baths were actually folded (the resolved `sub_baths`; `K` when `sub_baths=None`); `None` for non-separable models |

## The truncation metric

`truncation_errors` is the real per-record truncation metric, on the
pipeline's own axis: per physical time step for single-bath Track 1
(order 2 takes the max over both sub-steps, staying aligned with
`times`), and per recorded sub-bath count for separable Track 1 (the max
over every fold since the previous record, so `record_every > 1` drops
nothing). Each entry is the largest per-bond **discarded weight** across
the bonds, sub-steps or folds that the record covers. How that weight is
measured depends on the compression path:

- `zipup` / `direct` with the exact decomposition: `Σ sigma_i²` over the
  singular values discarded at a bond;
- `dm`: `Σ lambda_i` over the reduced-density-matrix eigenvalues
  discarded at a bond, where `lambda_i = sigma_i²`. The eigenvalues are
  summed directly — **not** `Σ lambda_i²`.

Both paths measure the same physical discarded weight; note it is the
weight, *not* quimb's discarded 2-norm `sqrt(Σ sigma²)`.

Three outcomes are distinguishable and none may be conflated:

- a **number** — the metric is defined; `0.0` means no discarded weight
  was recorded, including paths where compression was skipped;
- **`None`** — the chosen decomposition cannot measure it exactly
  (`compress_decomp='rsvd'`: the randomized sketch never sees the tail
  of the spectrum it omitted). The metric is unmeasurable, not zero;
- **`[]`** — Track 2, which is exact-only and performs no truncation.

It is a *local, per-record* quantity — not a cumulative or global error
bound for the trajectory.

## Bookkeeping fields

| field | contract |
|---|---|
| `expansion_order` | the Trotter order actually used (`1` or `2`) |
| `backend` | the device/track that **actually** ran, e.g. `'cpu/f64'`, `'gpu/f64'`, `'hpc/exact/cuquantum'` (`.../<n>gpu` when distributed). A requested GPU that was unavailable shows as CPU with a `(fallback: ...)` suffix — the honest record of what executed, not what was asked for |
| `final_time_bond_dims` | the final EDM-MPS's internal bonds along the *time* chain; length `num_sites - 1`, **not** aligned with `times`; `None` on Track 2 |
| `error_metrics` | Track 2 only: reference error metrics (`‖ρ−ρ†‖`, `|Tr ρ−1|`) plus optimizer statistics; `None` on Track 1 |
| `mps`, `evolution` | the final EDM-MPS and the raw Layer-5 output (internal; the fields above are the public contract); both `None` on Track 2 |
| `bond_dims`, `max_bond` | **legacy** pipeline-specific aliases kept for back-compat: `time_bond_dims` on single-bath, `sub_bath_bond_dims` on separable, `[]` on Track 2. Prefer the axis-explicit fields |
