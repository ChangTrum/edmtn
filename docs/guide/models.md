# Models

All three bundled models are Layer-1 objects with the same validation
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


## Dicke: `DickeModel`

A cavity mode coupled to `K` two-level systems (the inhomogeneous Dicke
model), with optional **local** Lindblad dissipation:

```
H = omega_c a^dag a + sum_k (omega_k / 2) sigma_{k,z}
    + sum_k g_k (a + a^dag) sigma_{k,x}
```

In the interaction picture the coupling factorises into one system
operator and one operator per sub-bath,
`H_I(t) = sum_k S(t) (x) B_k(t)` with
`S(t) = a e^{-i omega_c t} + a^dag e^{i omega_c t}` and
`B_k(t) = g_k [cos(omega_k t) sigma_x - sin(omega_k t) sigma_y]`
(**Pauli** units, not the spin units the Gaudin model uses). So it has a
**single coupling channel** and a **time-dependent separable** bath —
`bath_type = "separable_td"`.

| parameter | constraint | meaning |
|---|---|---|
| `K` | integer `>= 1` | number of bath spins |
| `n_fock` | integer `>= 2` | **dimension** of the truncated cavity Fock space (levels `0 .. n_fock-1`), so `system_dim == n_fock` |
| `coupling` | real scalar, or length-`K` array | a **scalar is the collective coupling `G`**, giving `g_k = G / sqrt(K)`; an array is used **verbatim, in the order given** (any sign, no sorting, no normalisation) |
| `omega_c` | finite, `> 0` | cavity frequency (default `1.0`) |
| `omega` | **finite `> 0`** scalar or length-`K` array | spin splittings `omega_k`; a scalar is the homogeneous case (default `1.0`). Strict positivity is required because `bath_state="ground"` means `r_z = -1`, which is the ground state of `(omega_k/2) sigma_z` only for `omega_k > 0` and is degenerate at `omega_k = 0` |
| `cavity_state` | `"vacuum"` (default), `"coherent"`, `"thermal"`, or an explicit `(n_fock, n_fock)` matrix | named states are built **inside** the truncation and renormalised to unit trace. An **explicit** matrix is checked at construction only for numeric dtype, shape and finiteness; its Hermiticity, unit trace and positive semidefiniteness are checked by `model.validate()`, which `EDMSolver` runs before building any pipeline — so it still fails before any tensor is built, just not inside `__init__` |
| `cavity_params` | dict | `alpha` (complex) for `"coherent"`, `nbar` (finite `>= 0`) for `"thermal"` |
| `bath_state` | `"inf"` (default), `"thermal"`, `"ground"`, or an explicit `(K, 3)` Bloch array | `"inf"` is maximally mixed (`r_k = 0`); an explicit array needs `\|\|r_k\|\| <= 1` |
| `bath_state_params` | dict | `beta` (finite `> 0`, with `k_B = 1`) for bath `"thermal"`, giving `r_{k,z} = -tanh(beta omega_k / 2)` |
| `kappa` | finite, `>= 0` | cavity decay in `kappa D[a]` (default `0`) |
| `pump`, `emission`, `dephasing` | finite `>= 0` scalar or length-`K` array | per-spin `w_k`, `gamma_k`, `gamma_k^phi`; all default `0` |
| `time_step_order` | `1` or `2` | default `2` |

The parameters layer up, and the simplest configuration is the intended
baseline:

1. **closed, homogeneous Dicke Hamiltonian** — scalar `coupling`
   (`g_k = G/sqrt(K)`) and scalar `omega`, every rate `0`, vacuum
   cavity, infinite-temperature spins;
2. **inhomogeneous** — per-spin `coupling` and/or `omega` arrays;
3. **non-equilibrium** — any of `kappa`, `pump`, `emission`,
   `dephasing`.

Keep the Hamiltonian and the *state* apart. Layer 1 is the textbook
Dicke **Hamiltonian**, but the run is a **quench** from the
`vacuum (x) maximally-mixed` product state, not an equilibrium
calculation. The familiar critical coupling
`G_c = sqrt(omega_0 omega_c)/2` belongs to the **zero-temperature,
thermodynamic-limit equilibrium ground-state** transition (normal phase,
superradiant phase, spontaneously broken parity); it does not describe
this default quench from infinite-temperature spins, and the pipeline
neither computes nor verifies it.

The layers are independent switches, which is what makes **controlled
comparison** possible — enable one at a time and compare against the
baseline. With several enabled at once, a difference cannot be
attributed to any single one of them.

**All per-spin arrays share one index `k`** — `coupling`, `omega`, the
three rates and the Bloch vectors are never sorted or renormalised, so
"the `k`-th coupling" and "the `k`-th frequency" always describe the same
spin. (The Gaudin named profiles *are* sorted; mixing that convention
with independent per-spin frequencies would silently pair the wrong
parameters, which is why this model has no named profiles.)

### Capability boundaries

- **Locality is a requirement, not a style.** Every dissipator above acts
  on the cavity or on **one** spin. A collective jump operator (say
  `D[J^-]`) would correlate the sub-baths and invalidate the whole
  separable transfer-tensor construction, so it is not supported.
- **No coupling-channel polarization.** The Eq.-F2/F3 sweep selects a
  coupling-operator arm at one time site and closes the rest. Its time
  mapping is established only for a **time-independent** coupling
  operator: the arm carries `S` at its site's sample time, while the
  environment to its right is the state *before* that step, so with a
  rotating `S(t)` the operator and the state sit at different times.
  Defining and validating that alignment for a time-dependent coupling
  is separate work, not done here — so `res.polarization` is `None` and
  no `channel` may be requested: `channel=1` is a legal index but raises
  `NotImplementedError`, while `0`, `2`, floats, strings and `bool`
  still raise `ValueError`. Read `res.final_density_matrix`, or the
  per-`L` `res.sub_bath_final_density_matrices` with `record_rho=True`.
- **`timestep_convergence()` raises `NotImplementedError`** for the same
  reason; compare `final_density_matrix` from two solves instead.
- **Track 2 (`backend='hpc'`) does not support this bath type** — it is
  Gaudin / `bath_type='separable'` only.

### Choosing `n_fock`

`n_fock` is a *numerical* truncation, not the physical highest occupied
photon number: that number may not even be finite (a coherent or thermal
state has an infinite tail). Check the truncation with the photon-number
distribution itself,

```python
p = model.fock_populations(res.final_density_matrix)   # p[n] = rho_nn
```

and widen `n_fock` until the tail is negligible for your purposes. There
is deliberately no thresholded "maximum occupied level" helper.

Cost scales sharply: the system MPS carries bond dimension `d**2 =
n_fock**2` before any fold, and every folded sub-bath multiplies the
lateral bond by 4, so `n_fock` and the compression knobs (`cutoff`,
`max_bond`) are **two independent convergence checks** — neither
substitutes for the other.

### Time discretisation

The pipeline freezes the generator at the **midpoint** of each physical
step and applies the dissipative channels in Strang half-steps. Under
stated conditions — fixed `n_fock`, a smooth bounded generator, the
interaction-picture-invariant dissipators above, and no discarded weight
(so no compression, reference or round-off floor has been reached) —
`expansion_order=2` is then globally second order; measured on the test
configuration: **1.97** (order 2) and **1.02** (order 1). Outside those
conditions the observed order degrades and the claim does not carry. The
derivation, the exact discretisation maps and the full verification
record are in {doc}`../design/dicke-second-order-discretisation`.
