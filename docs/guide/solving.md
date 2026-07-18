# Solving

## Entry points

`edmtn.driver.solve(model, *, T, eps, ...)` is the one-shot convenience
wrapper; it builds an `EDMSolver` from the model and a `SolverConfig`
assembled from the keyword arguments, then solves. Constructing
`SolverConfig` and `EDMSolver` explicitly is equivalent and useful when
the same configuration is reused (for example by
`timestep_convergence()`; see the {doc}`../api/driver` reference).

`SolverConfig` is **frozen and validated at construction**: every field
is checked and normalised in one place, and illegal input — a huge Python
int, `nan`/`inf`, a `bool` posing as an integer, an unknown enum string —
raises `ValueError` at the entry point rather than deep inside quimb. Use
`dataclasses.replace` to derive variants.

## The time grid

`eps` and `T` are finite and positive, and `T / eps` must be a positive
integer (to a small tolerance) — never silently rounded; the integer is
cached as `n_steps`. See the
{doc}`quickstart <../getting-started/quickstart>` for the resulting
`times` axis contract.

## Expansion order

`expansion_order` is the Trotter order of the small-step expansion, `1`
or `2`. The default `None` inherits the model's `time_step_order` (both
bundled models default to `2`); an explicit value overrides it. The order
is resolved once in the driver, so every layer — kernel, expander,
observables, Track-2 assembly and `SolverResult.expansion_order` — uses
the same value.

## Truncation controls (Track 1)

- `cutoff` — the truncation threshold; finite, `>= 0`. `cutoff = 0`
  with `max_bond=None` discards nothing: the sweep still runs, as a
  no-discard recompression with the selected method/decomposition. A
  rank-limiting `max_bond` can truncate even at `cutoff = 0`. (Skipping
  the sweep entirely is a different thing — the Layer-5 engines expose
  it as `compress=False` on their `run()` methods; see
  {doc}`../api/evolution`.)
- `cutoff_mode` — the quimb-native truncation rule: one of `abs`,
  `rel`, `sum2`, `rsum2`, `sum1`, `rsum1`. Default `'rel'`
  (`s_i / s_max <= cutoff`). The paper's custom `rel_ref` rule is
  retired; no such field exists.
- `max_bond` — hard bond-dimension cap; `None` (default) or a positive
  integer.
- `compress_method`, `compress_decomp`, `compress_decomp_q`,
  `compress_canon` — the compression-sweep internals (algorithm,
  decomposition, rSVD power iterations, canonicalisation). Defaults:
  `zipup` / `exact` / `2` / `quimb`. All are Track-1 only: `hpc` is
  exact-only and has no compression sweep. Not every combination is
  executable — see the compatibility table in {doc}`compression`.
- `preset` — `None` (the API default), `'balanced'` or `'robust'`;
  Track 1 only, and an unknown name is rejected on every backend. The
  trigger is `compress_decomp` *alone*: while it is still `'exact'` (its
  default), a preset sets `compress_decomp` to `'rsvd'` **and overwrites
  `compress_decomp_q` with the preset's value — even if you passed a
  `q` explicitly**. Once `compress_decomp` is explicitly `'rsvd'`, a
  preset changes neither field. See
  {doc}`the configuration guide <../guides/recommended-config>` for what
  each preset trades away.

None of these is an accuracy guarantee: `cutoff` is a local per-bond
threshold, not an error bound on observables.

## Other controls

- `record_rho` — store `rho(t)` at every step (strictly a `bool`). Some
  paths record reduced states regardless: second-order spin-boson and
  custom observables need them anyway.
- `precision` — `'f64'` (default) or `'mixed'`. As implemented, `'mixed'`
  casts the tensor contraction to f32; the f64 decomposition stage that
  the backend's `PrecisionPolicy` declares is **not wired into the solve
  pipeline**. Treat `'mixed'` as experimental and unvalidated.
- `sub_baths` — separable models only: fold just the first `L` sub-baths
  in the model's stored coupling order (see
  {doc}`models`). Validated as `None` or a positive integer, then
  re-checked against the model's `K` — never silently clamped.
- `backend` — `'cpu'` (the default), `'numpy'`, `'gpu'`, `'cupy'`
  (Track 1) or `'hpc'` (Track 2). There is no `'auto'`.
- `pathfinder` — `hpc` only: `'cuquantum'` (default) or `'cotengra'`;
  distributed multi-GPU requires `'cuquantum'`.
- `time_windows` — reserved; must be `None`. Manual time-window blocking
  is wired but not implemented, and any non-`None` value raises
  `NotImplementedError` at construction.

## Checking the time step

`EDMSolver.timestep_convergence()` re-solves at `eps/2` and compares the
polarization histories on the common grid. The fine run is derived with
`dataclasses.replace(config, eps=eps/2)`, so it inherits *every* other
resolved field — the two runs are the same physical model and
configuration, differing only in the step. The result carries
`.deviation`, `.converged` (against an optional tolerance) and a
self-describing `.metadata` record including the actually-executed
backends; see {doc}`../api/driver`.
