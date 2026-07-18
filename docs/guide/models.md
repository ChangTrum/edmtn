# Models

Both bundled models are Layer-1 objects with the same validation
philosophy: out-of-range or non-finite constructor parameters raise
`ValueError` immediately at construction, while *legal* parameters whose
bath correlation later overflows float64 raise `FloatingPointError` at
compute time. Integer parameters are strict non-`bool` integers.

## Spin-boson: `SpinBosonModel`

A spin-1/2 with transverse tunnelling, `H_S = mu S_x`, coupled to a
Gaussian bosonic bath through `S_z` — its single coupling channel
(`channel=1`). The bath has a generalised Ohmic spectral density with
exponent `s`.

| parameter | constraint | meaning |
|---|---|---|
| `J0` | finite, `>= 0` | dimensionless coupling strength; `J0 = 0` is a *legal* no-coupling baseline (spectral density and correlation exactly zero), not an invalid value |
| `omega_c` | finite, `> 0` | bath cutoff frequency |
| `mu` | finite, `> 0` | transverse tunnelling strength; sets the time unit |
| `s` | finite, `> 0` | spectral exponent (default `1.0`): `s = 1` Ohmic, `s < 1` sub-Ohmic, `s > 1` super-Ohmic |
| `temperature` | finite, `>= 0` | default `0.0`. The *model* accepts any finite non-negative value, but the Gaussian cumulant engine currently implements the zero-temperature correlation only: a non-zero value raises `NotImplementedError` when the correlation is computed, not at construction |
| `time_step_order` | `1` or `2` | default `2`; the expansion order inherited by the solver when `expansion_order` is not given |

## Gaudin: `GaudinModel`

A central spin-1/2 isotropically coupled to `K` bath spin-1/2 — the
separable-bath model: the bath factorises into `K` sub-baths that the
pipeline folds in one at a time. Coupling channels `1`, `2`, `3` select
`S_x`, `S_y`, `S_z`. The bath temperature is `+inf` — each bath spin
maximally mixed (`I/2`) — and that is the *only* supported case: the
separable correlation engine raises `NotImplementedError` for anything
else.

| parameter | constraint | meaning |
|---|---|---|
| `g` | finite, `> 0` | base coupling constant; sets the time unit `1/g` for the normalised named profiles |
| `K` | integer `>= 1` | number of bath spins (the paper uses `K = 49`) |
| `time_step_order` | `1` or `2` | default `2`, as in the paper |
| `coupling` | name or array | the per-sub-bath profile `g_k`; see below |
| `coupling_params` | dict, optional | extra knobs for a named profile: `beta` for `"exp"`; `seed`/`low`/`high` for `"random"`; `rho`/`seed` for `"ou"`. Ignored for explicit arrays |

### Coupling profiles and their ordering

The `coupling` argument is either a named profile or an explicit
length-`K` array, and the distinctions below matter downstream:

- **Named, sorted** — `"linear"` (the default, as in the paper),
  `"uniform"`, `"exp"`, `"random"`: normalised so that
  `sum_k g_k**2 == g**2` and stored in **descending** order.
- **Named, unsorted** — `"ou"`: normalised the same way but deliberately
  kept in **generation order**; sorting would destroy its sequential
  correlation.
- **Custom array** — used **verbatim** in the order given: any sign
  (negatives are allowed), no sorting and no normalisation are imposed,
  so neither `sum_k g_k**2 == g**2` nor `g_K == g` is guaranteed, and
  `g` no longer sets the scale.

Downstream, `sub_baths=L` always means "the first `L` sub-baths in this
**stored order**" — which is strongest-first only for the sorted named
profiles.

A supplied array is privately copied and marked read-only: mutating your
array afterwards cannot change the model, and `model.couplings` itself is
not writable.
