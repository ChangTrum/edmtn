# Extracting basic observables from the Dicke pipeline

## Why this note exists

In `bath_type='separable_td'` the **system is the cavity** and the **bath is the `K`
two-level systems**. The pipeline therefore hands back the reduced *cavity* state; the
spins are folded into the influence functional and traced out. Cavity observables and
collective-spin observables consequently reach the user by two completely different
routes, and only one of them is post-processing.

This note fixes, at the mathematics-and-physics level, *which* basic observables are
extracted and *how*, then records the implementation contract that follows. It covers
the first round only: readings at the final time `T`, first moments of the collective
spin, and no time axis. Section 8 states what is and is not claimed.

## 1. What the pipeline actually returns

| | |
|---|---|
| system | cavity mode, `system_dim = n_fock` |
| bath | `K` spins, one sub-bath each, lateral (Liouville) dimension 4 |
| coupling channels | one, `d_phys = 3` |
| returned state | `rho_c(T)`, the reduced **cavity** density matrix |
| picture | interaction picture of `H_0 = omega_c a^dag a + sum_k (omega_k/2) sigma_{k,z}` |

`U_0(t)` factorises across the cavity and the spins, and the partial trace over the
spins commutes with the spin factor, so the cavity's picture transformation is the
single-mode rotation

$$\rho^{I}_{c}(t) = e^{\,i\omega_c a^\dagger a\,t}\,\rho^{S}_{c}(t)\,e^{-i\omega_c a^\dagger a\,t}.$$

The Eq.-F2/F3 coupling-channel sweep is **not** available on this pipeline: its time
mapping is established only for a time-independent coupling operator, and `S(t)` here
rotates. This costs nothing for the observables below — Section 2 shows that the operator
that sweep would return, `S(t) = a e^{-i\omega_c t} + a^\dagger e^{i\omega_c t}`, is parity
odd and therefore identically zero in every parity-symmetric configuration.

## 2. The parity constraint

Let `Pi = e^{i pi a^dag a} (x) prod_k sigma_{k,z}`, so `Pi a Pi^dag = -a` and
`Pi sigma_{k,x} Pi^dag = -sigma_{k,x}`. Two separate statements, which must not be
merged (dissipation is not part of the Hamiltonian):

* `[H, Pi] = 0` for **every** parameter choice, including inhomogeneous `g_k` and
  `omega_k`;
* each Lindblad channel used here is parity covariant — `D[a]`, `D[sigma_k^+]`,
  `D[sigma_k^-]` and `D[sigma_{k,z}]` are all invariant under conjugation by `Pi`,
  because the sign incurred by the jump operator is squared away.

Hence a parity-symmetric initial state stays parity symmetric, and the Fock truncation
does not spoil this (parity is diagonal in the number basis). For such a state every
**odd** off-diagonal of `rho_c` vanishes, and

$$\langle a\rangle \equiv 0,\qquad \langle J_x\rangle \equiv 0,\qquad \langle J_y\rangle \equiv 0 .$$

**Breaking parity requires the initial state**, and there are three ways to do it:

1. `cavity_state='coherent'` with `alpha != 0`;
2. an explicit `(K, 3)` Bloch array with non-zero `r_x` or `r_y` — a pure small tilt is
   `r_x = sin(theta)`, `r_z = -cos(theta)`, since `r_x` cannot simply be added on top of
   `r_z = -1` without leaving the Bloch ball;
3. an explicit cavity density matrix carrying coherence between even and odd Fock
   sectors.

Inhomogeneity in `g_k` or `omega_k` does **not** break parity, and neither does any of
the dissipators. A run whose transverse spin moments read zero in a symmetric
configuration is therefore correct, not defective.

## 3. Class 1 — cavity moments

### 3.1 The quantities and their exact matrix elements

All are functionals of `rho_c` alone:

| quantity | expression | part of `rho_c` used |
|---|---|---|
| `<a^dag a>` | `sum_n n p_n` | diagonal |
| `<a^dag a^dag a a>` | `sum_n n(n-1) p_n` | diagonal |
| `Tr rho` | `sum_n p_n` | diagonal |
| `p_n` | `rho_nn` | diagonal (`DickeModel.fock_populations`) |

The second line uses the operator identity `a^dag a^dag a a = n(n-1)`, which is **exact in
the truncated space**: the two annihilations act first and never reach the boundary.

### 3.2 The truncated-space rule

In a Fock space of dimension `d = n_fock`,

$$[a, a^\dagger] = \mathbb 1 - d\,P_{\rm top},\qquad P_{\rm top}=\lvert d-1\rangle\langle d-1\rvert .$$

Rearrangements that silently assume the infinite-dimensional canonical commutator are
therefore wrong at the boundary; for instance the unnormalised quadrature obeys

$$\langle x^2\rangle = 2\langle n\rangle + 1 - d\,p_{\rm top} + 2\,\mathrm{Re}\big[e^{-2i\omega_c t}\langle a^2\rangle_I\big].$$

**Rule: build the operator matrix explicitly in the truncated space and evaluate
`Tr[O rho]`.** Normal-ordered moments happen to be boundary-safe, but that is a
convenience, not the rule.

### 3.3 Picture invariance

`n = a^dag a` commutes with `H_0`, so `<n>` and `<n(n-1)>` are picture independent and
need no rotation. Quantities involving `a^2` or `a` are not: the lab-frame value carries
`e^{-2i omega_c t}` resp. `e^{-i omega_c t}`.

### 3.4 Why `g2` is not a returned quantity

$$g^{(2)}(0) = \frac{\langle a^\dagger a^\dagger a a\rangle}{\langle a^\dagger a\rangle^2}$$

has `<n>^2` in the denominator, and a run started from the vacuum passes through
`<n> = 0`. Choosing a validity threshold is a physics decision belonging to the caller,
so the pipeline returns the **raw moments** `n` and `n_factorial2` and the caller forms
the ratio on the interval it judges valid. No zero-denominator policy is needed because
no ratio is formed inside.

For the same reason `Tr rho` is returned unmodified alongside the moments: a truncation
or step-size trace deviation must be visible to the caller, never silently divided out.

### 3.5 Fock-truncation evidence

`<n(n-1)>` weights the Fock tail by `n^2` and so is far more sensitive to `n_fock` than
`<n>`. The evidence is the **full** photon-number distribution
`DickeModel.fock_populations(rho)`; `p_top` alone is a shortcut indicator, not proof —
the second and third highest levels can already carry significant population. Final
adequacy is settled only by enlarging `n_fock` and comparing the moments, which belongs
to the later capacity work, not here.

## 4. Class 2 — collective spin moments

### 4.1 The problem

The spins are the bath. `rho_c` contains no `<J_z>`. A new extraction is required, and it
is *not* post-processing: it changes what is folded.

### 4.2 The lateral closing

Sub-bath `k`'s transfer tensor is built in the Pauli basis
`sigma = (1, sigma_x, sigma_y, sigma_z)`,

$$A_k(t)[\phi,a,a'] = \tfrac12 \mathrm{Tr}\big[\sigma_a\, B^\phi_k(t)(\sigma_{a'})\big],$$

so the lateral index carries the **Pauli coefficient vector** `x_a = Tr[sigma_a X]` of
whatever operator `X` the chain has accumulated. The two boundaries are: the oldest site's
right index contracted with `r_k = (1, r_x, r_y, r_z)` (from `Omega_k = (1/2)(1 + r_k . sigma)`),
and the newest site's left index fixed to `0`.

Index `0` is `Tr[1 . X] = Tr X`, i.e. *not measuring*. Replacing the newest-end row vector
`e_0^T` by `e_alpha^T` therefore returns `Tr[sigma_alpha X]` — with **no** normalisation
factor, because the coefficient convention above is already dual to the Pauli basis.

### 4.3 The object this defines

Keeping the system arms open and changing only the bath-side closing of sub-bath `k`
gives the `d x d` matrix

$$\tilde\rho^{(\alpha)}_k(T) \;=\; \mathrm{Tr}_B\!\Big[\big(\mathbb 1_c \otimes \sigma^k_\alpha\big)\,\rho^{I}_{CB}(T)\Big],$$

which is Hermitian by the cyclicity of the partial trace with respect to bath-only
operators, and satisfies

$$\mathrm{Tr}_c\,\tilde\rho^{(\alpha)}_k = \langle \sigma^k_\alpha\rangle_I,
\qquad
\mathrm{Tr}_c\big[O\,\tilde\rho^{(\alpha)}_k\big] = \big\langle O \otimes \sigma^k_\alpha\big\rangle_I .$$

`rho_c` is the `alpha = 0` member of this family. The `d x d` matrix — not only its trace —
is what the pipeline retains, because the joint system-bath correlators needed for a
future interaction-energy check are the same object read against a cavity operator.

### 4.4 Units

This model writes its bath operators in **Pauli** units, unlike the Gaudin model's
`J = sigma/2`. Hence

$$J_z = \tfrac12\sum_k \sigma^k_z, \qquad J_+ = \sum_k \sigma^k_+, \qquad j = K/2 .$$

The factor `1/2` lives in one place only (the Dicke closing provider of Section 6).

### 4.5 Picture conversion, and why the transverse pair is one channel

Spin `k`'s picture is generated by `(omega_k/2) sigma_z`, so

$$\sigma^{k,I}_x(t) = \cos(\omega_k t)\,\sigma_x - \sin(\omega_k t)\,\sigma_y,
\qquad
\sigma^{k,I}_+(t) = e^{\,i\omega_k t}\,\sigma_+ .$$

The first coincides with `B_k(t)/g_k`, but is **defined by the trigonometric form**, not by
that quotient: `coupling` may legally contain zeros, and `bath_operator_at` serves only as
a cross-check.

Because `sigma_z` commutes with its own generator, `<J_z>` needs no rotation. The
transverse pair does, at a **per-spin rate**, which is why an inhomogeneous `omega_k`
cannot be handled by one global rotation. Using `sigma_+ = (sigma_x + i sigma_y)/2`, both
transverse components come from a single complex closing:

$$v^{(+)}_k = \tfrac12 e^{\,i\omega_k T}\,(0,\,1,\,i,\,0),
\qquad
v^{(z)}_k = \tfrac12\,(0,\,0,\,0,\,1),$$

$$\langle J_+\rangle_S = \sum_k e^{\,i\omega_k T}\,\langle \sigma^k_+\rangle_I,
\qquad
\langle J_x\rangle_S = \mathrm{Re}\,\langle J_+\rangle_S,
\qquad
\langle J_y\rangle_S = \mathrm{Im}\,\langle J_+\rangle_S .$$

The lateral contraction with `v_k` is **bilinear and must not conjugate**: conjugating
turns `J_+` into `J_-`.

### 4.6 The collective sum: a first-order jet

The influence functional is a product over sub-baths, `F = prod_k F_k`, and a measurement
insertion modifies exactly one factor. Replacing every closing by `e_0 + lambda v_k`,

$$\prod_k F_k(\lambda) \;=\; \prod_k F^{(0)}_k \;+\; \lambda \sum_k F^{(v)}_k \prod_{j\neq k} F^{(0)}_j \;+\; O(\lambda^2),$$

so the coefficient of `lambda` is exactly the collective sum. Carrying the pair
`(M, dM)` through the fold,

```
M_{L+1}  = F^(0)_{L+1} M_L
dM_{L+1} = F^(0)_{L+1} dM_L  +  F^(v)_{L+1} M_L ,        dM_0 = 0
```

is an exact derivative recursion — a **jet**, not a finite difference. The alternative,
one full solve per sub-bath, would cost `K` complete `K`-fold solves and its only extra
product is per-spin resolution, which the first round does not need.

### 4.7 Where the exactness stops

The recursion is an exact derivative **in uncompressed linear tensor algebra only**. Once
`M`, `dM` or their sum are compressed independently, compression is a non-linear map and
`dM` is no longer the exact derivative of the compressed value trajectory. It is then a
jet *approximation*, and the two error sources — the value channel's truncation and the
tangent channel's addition-plus-truncation — must be recorded separately. Reusing the
value channel's truncation record as evidence that the spin readings have converged is
not valid.

### 4.8 Deliberately not extracted

`<J^2>` is a **two-body** quantity, `J^2 = 3K/4 + (1/4) sum_{k != l} (sigma^k_x sigma^l_x +
sigma^k_y sigma^l_y + sigma^k_z sigma^l_z)`: it needs two simultaneous insertions (or the
`lambda^2` coefficient), and under inhomogeneous `omega_k` its Schrödinger-picture form
mixes `xx`, `yy`, `xy` and `yx` correlations rather than summing three co-axial
interaction-picture moments. It is also *not* conserved under local spin dissipation, even
with identical rates. Out of scope here.

Also out of scope: time-resolved spin readings (mathematically the same causal-prefix
argument with a different terminator, but a distinct object), and the separately weighted
channels an interaction-energy check needs under inhomogeneous `g_k` or `omega_k`.

Note that `|<J>|` is the **modulus of the mean collective-spin vector**, not the Casimir;
the two are different quantities, not two names for one.

## 5. Consequences that constrain interpretation

* **`sub_baths = L`.** The jet sums only over the sub-baths actually folded, so the
  returned moments describe the first `L` spins and the bound is
  `|<J>| <= L/2`, with `L = sub_baths_used`.  That bound holds for a *normalised*
  physical state; the pipeline returns the raw moments and the raw trace, so a run whose
  trace has drifted must be judged against its own trace rather than against 1.
* **`|<J>|` is picture invariant only for homogeneous `omega_k`** — a common rotation about
  `z` preserves the modulus; per-spin rotations do not.
* **`<n> + <J_z>` is not conserved.** Excitation-number conservation is a property of the
  rotating-wave (Tavis–Cummings) model; the counter-rotating term `g_k (a + a^dag)
  sigma_{k,x}` is present here. Non-zero rates change it as well. It is a diagnostic of
  non-conserving dynamics, not an identity, and not by itself evidence of "virtual"
  excitations.
* **Energy is not a machine-precision identity.** The midpoint/Strang/EDM map at finite
  `eps` does not conserve it exactly; the closed-system energy is a convergence diagnostic
  in `eps`, truncation and `n_fock`, and is defined only when every rate is zero.
* **The Holstein–Primakoff comparison** — for the homogeneous, closed, `bath_state='ground'`
  baseline the normal-mode frequencies are
  `eps_pm^2 = (1/2)[omega_c^2 + omega^2 +- sqrt((omega_c^2 - omega^2)^2 + 16 G^2 omega_c omega)]`,
  giving `G_c = sqrt(omega_c omega)/2` and `lambda = sqrt(-eps_-^2)` above it; the scalar
  `coupling` argument *is* the standard collective `G` (the `sqrt(K)` cancels). This
  parameter-free comparison applies only in the controlled window after the initial
  transient and before HP depletion or the Fock boundary, and needs the time axis — so it
  is a later test, not part of the `T`-point acceptance here.

## 6. Implementation contract

### 6.1 API

`solve(..., moments=Sequence[str] | None = None)`; `None` (default) computes nothing extra.
Closed vocabulary; an unknown name raises `ValueError` on every model, a legal name on an
unsupported model then raises `NotImplementedError`.

| name | quantity | channel | also returned |
|---|---|---|---|
| `n` | `<a^dag a>` | value-channel post-processing | — |
| `n_factorial2` | `<a^dag a^dag a a>` | value-channel post-processing | — |
| `Jx` / `Jy` | `<J_x>_S` / `<J_y>_S` | `Jplus` | each other |
| `Jz` | `<J_z>_S` | `Jz` | — |
| `Jabs` | `abs(<J>)` | `Jplus` + `Jz` | `Jx`, `Jy`, `Jz` |

By-products of an enabled channel are kept, never discarded; a request for one component
never triggers the other channel and never computes `Jabs`. A non-empty request always
also returns `trace` (raw, un-normalised).

`_resolve_moments()` runs before `_resolve_channel()` and before backend dispatch:
`None` or a non-string sequence; a bare string such as `"Jz"` is rejected rather than
iterated as characters; every item a strict `str`; an empty sequence normalises to "no
request"; duplicates de-duplicated preserving first order. The top-level `solve()` declares
`moments` explicitly, so it cannot fall through `**kwargs` into `SolverConfig`.

Capability is decided by a **Dicke-specific provider**, not by `bath_type == 'separable_td'`
(that is a pipeline class and does not guarantee a Fock cavity or Pauli spins): the model
must expose `collective_spin_closures(t) -> {"Jplus": (K,4), "Jz": (K,4)}`.

### 6.2 Layering

| layer | responsibility |
|---|---|
| `models/dicke.py` | `collective_spin_closures(t)` — the `J = sigma/2` factor, the `e^{i omega_k t}` phase, the `Jplus`/`Jz` names. All Dicke semantics live here and nowhere else. |
| `cumulants/separable_td.py` | unchanged |
| `kernels/separable_td_mpo.py` | `get_kernel_mpo(..., closing=None)`: default keeps the index-0 slice; otherwise contract the newest site's left lateral index with a finite `(4,)` vector, **unconjugated**. Knows nothing of Dicke vocabulary. |
| `evolution/quimb_edm.py` | `QuimbEDM.add_exact(other)` |
| `evolution/separable_bath.py` | the tangent fold loop |
| `driver/solver.py` | `_resolve_moments`, rejected combinations, read-out and packing |

Single-site ordering is fixed: contract `r_k` into the oldest-end right boundary first,
then `v` into the newest-end left boundary, giving `(phi_up, phi_down, 1, 1)`, which
`fold_raw`'s `n == 1` branch consumes correctly.

### 6.3 The fold loop

```
dM[ch] = None                                   # dM_0 = 0; no zero MPS is built
for k in range(n_fold):
    mpo_0     = kernel.for_sub_bath(k).get_kernel_mpo(n_sites)
    mpo_v[ch] = kernel.for_sub_bath(k).get_kernel_mpo(n_sites, closing=v[ch][k])
    M_old = M
    M     = compress(M_old.fold_raw(mpo_0))     # byte-identical to a run without moments
    for ch:
        src    = M_old.fold_raw(mpo_v[ch])      # source term uses M_old, not M
        grown  = src if dM[ch] is None else dM[ch].fold_raw(mpo_0).add_exact(src)
        dM[ch] = compress(grown)                # EVERY fold, the first one included
    release M_old, src, superseded tangent/raw networks and MPO temporaries, then free pool
```

`dM_0 = 0` saves the zero chain's own fold and addition — and nothing else. The
compression the caller configured still applies to `dM_1`: skipping it would make
`cutoff` and `max_bond` silently inert on this channel (at `n_fold = 1`, permanently)
while the truncation record claimed a discarded weight of `0.0`.

`QuimbEDM.add_exact(other)` is a dedicated **lossless** addition, not quimb's `+`: two
`fold_raw` results carry no guarantee of matching internal bond names after
`fuse_multibonds`. Its contract: structural equality of `n`, `d`, `d_phys`, the per-site
physical dimensions, the `OUT`/`RHO0` dimensions and `rho0_vec` semantics, else
`ValueError`; a new object with both inputs unmutated; direct sum of **internal virtual
bonds only** (the physical legs, `OUT` and `RHO0` are shared external legs); element-wise
addition when `n == 1`; no canonicalisation and no compression; allocation on the input's
native backend, with no NumPy round-trip that would pull CuPy data to the host. If
`tensor_network_ag_sum` is used, `site_tags=[f"I{p}" ...]` must be passed explicitly, since
a plain `TensorNetwork` has no default site tags.

Only the final `d x d` reduced matrices are kept; the full tangent chains are released
after the read, so a returned result does not hold one or two extra full-length time
chains.

**Why separate tangent states rather than a two-dimensional jet index on the newest site**:
the index-based variant would let one SVD trade the value channel against the tangent, so
enabling a spin moment would change `<n>`, and the separate tangent-error record that the
audit requires would not exist.

### 6.4 Rejected combinations

Decided before any tensor is built, from the resolved request:

```
needs_tangent = bool(names & {"Jx", "Jy", "Jz", "Jabs"})
```

* public solver: rejected iff `needs_tangent and record_time_reads`. `n` and
  `n_factorial2` coexist freely with `record_time_reads=True`.
* direct `run()`: rejected iff `tangent_closings is not None and record_time_reads and
  compress`; the same combination with `compress=False` is allowed.

The reason is structural: with compression on, `record_time_reads` forces
`compress_method='dm_tracking'`, and `QuimbEDM.compress` requires `dm_tracking` and
`PrefixTerminators` to appear together. The tangent states have no terminators. Letting
the value channel use `dm_tracking` while the tangent silently used `dm` would be two
different numerical paths and would falsify `compression_method_used`.

### 6.5 Result semantics

* `SolverResult.moments: dict | None` — exactly the requested names plus by-products plus
  `trace`; `None` by default.
* `SolverResult.moment_truncation_errors: dict[str, list[float | None]] | None` — keyed by
  **channel** (`Jplus`, `Jz`), so `Jx` and `Jy` share one record rather than appearing as
  independent copies. Only channels that actually ran appear; the field is `None` when only
  `n`/`n_factorial2` were requested. Each list aligns with the public sub-bath-count axis
  and, exactly as the value channel does, holds the maximum discarded weight over every
  fold since the previously recorded `L` — so `record_every > 1` cannot drop un-recorded
  folds. `0.0` under `compress=False`, `None` under `rsvd` or whenever any fold in the
  interval was unmeasurable. It is a **local per-interval record, not a global error bound**.
* Scalar typing: `trace` stays a Python `complex`. `<J_+>` is complex by construction, so
  `Jx = jp.real` and `Jy = jp.imag` with **no** imaginary-part guard. `n`,
  `n_factorial2` and `Jz` are physically real: an absolute-plus-relative imaginary check is
  reduced **on the array's own backend**, `.item()` is taken last, an excess raises
  `ValueError`, and a Python `float` is returned. `Jabs` is formed from the three
  components with a numerically stable `hypot` — three individually finite components
  can still overflow when squared and summed — and the derived value is then checked for
  finiteness in its own right, because the contract covers what is *returned*, not only
  what was read.

### 6.6 Cost

Structurally, per fold: `1 + 2c` complete folds for `c` tangent channels, `c` lossless
additions, and `1 + c` compressions, with the tangent bond after addition approaching the
sum of its two branches and several raw networks alive simultaneously. No multiplier is
claimed; it is measured, not asserted. A source-fold reuse path (recomputing only site 0,
since `mpo_v` and `mpo_0` differ only there) is a possible optimisation but is not part of
this round.

## 7. Verification plan

**Structural / exact**

1. `compress=False`, `K=2`, `n_fock=3`, orders 1 and 2, against an independent dense
   full-Hilbert reference — comparing the **full `d x d` `tangent_density_matrices`** for
   `Jplus` and `Jz` against independent partial traces, not merely the three scalars.
2. `order=1, n_steps=1` single-site coverage of the default `e_0`, of `v_z`, and of the
   complex `v_+` (both boundaries land on the same tensor there).
3. `add_exact`: open-arm additivity `open_arm_tensor(A+B) == open_arm_tensor(A) +
   open_arm_tensor(B)`; inputs with different internal bond dimensions; `n = 1` and
   `n >= 3`; both inputs unchanged after the call; CuPy in, CuPy out under the GPU gate.
4. Value-channel non-pollution: `moments=("Jz",)` versus no request must agree on the main
   MPS tensors, the bond history, `truncation_errors`, `compression_method_used` and
   `final_density_matrix`, array by array on CPU.
5. Vocabulary contract: unknown name, bare string, empty sequence, de-duplication,
   `Jx` returning only `Jx`/`Jy`, `Jabs` returning four keys, and `moments is None` by
   default.
6. Bounds and invariances: `|<J>| <= L/2` **for a normalised physical state**, judged
   against the run's own raw `trace` since nothing is normalised on the way out;
   `|<J>|` picture invariant for homogeneous
   `omega_k` and not for inhomogeneous.

**Compressed paths** (the public solver always compresses, so `compress=False` alone is
not acceptance)

7. `zipup`, `direct` and `dm` end to end on a small system; `cutoff=0` with
   `max_bond=None` numerically equal to the uncompressed reference; under a non-zero
   cutoff, tightening it or raising `max_bond` moves `Jx`/`Jy`/`Jz` toward the uncompressed
   reference; `rsvd` records `None`; `moment_truncation_errors` keys, lengths, interval
   maxima and `None` propagation; one GPU-gated end-to-end tangent run.

**Mutations that must be shown to turn the suite red**

8. `v` conjugated (`J_+` becomes `J_-`); `e^{+i omega_k T}` written as `e^{-i omega_k T}`;
   the `1/2` factor dropped; the tangent source taken from the updated `M` instead of
   `M_old`; `add_exact` direct-summing `OUT` or `RHO0`; the `n_sites == 1` double-boundary
   path; and the symmetric-configuration assertion `Jx = Jy = 0` under a parity-breaking
   initial state.

The conjugation and phase mutations require an **asymmetric** small system with both
`<J_x>` and `<J_y>` non-zero — a parity-breaking initial state together with inhomogeneous
`omega_k` — otherwise the mutated and unmutated results coincide and the test carries no
information.

## 8. What is and is not claimed

* **Claimed.** Replacing the newest-end closing `e_0` by `e_alpha` returns
  `Tr[sigma_alpha . ]` exactly, with no normalisation factor, given the Pauli-coefficient
  convention of Section 4.2. The first-order jet is an exact derivative of the fold
  recursion in uncompressed tensor algebra.
* **Claimed.** In a parity-symmetric configuration the odd sector of `rho_c` and the
  transverse collective moments vanish identically, for any `g_k`, `omega_k` and any of the
  listed dissipators.
* **Not claimed.** Anything about the accuracy of the *compressed* tangent. The tangent is
  a jet approximation there, its error is recorded separately, and the value channel's
  truncation record is not evidence about it.
* **Not claimed.** Any cost multiplier. Section 6.6 gives the structure only.
* **Not claimed.** Machine-precision energy conservation, `<J^2>` behaviour, time-resolved
  spin readings, or `n_fock` adequacy. Each needs work that is deliberately out of this
  round.
