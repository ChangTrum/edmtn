# Time-resolved reads from one contraction: the causal-prefix terminator

## Why this note exists

The separable-bath engine produces an extended density matrix at **one** final time
`T = N eps`. Everything downstream — mean photon number, fluctuations, `g2(0)`,
quadratures — needs `rho(t)` on the whole grid, not just at `T`. The pipeline already
supports that in the crudest possible way: call `solve()` once per target time. That is
`N` complete runs, each restarting from `t = 0` and each folding all `K` sub-baths.

This note records a scheme that gets the whole `rho(t)` history out of a **single**
`K`-sub-bath folding run at `T = N eps`. Without compression that is the one run plus a
single cheap read sweep; with compression the run also has to transport a set of boundary
matrices through every fold and use a compression whose basis changes are known. It states
the identity the scheme rests on, the terminator that has to be carried alongside, the gauge
condition the compression has to satisfy for that terminator to survive, what the scheme
costs, and — this part is not optional — the one accuracy property it does **not** inherit
from the ordinary final-time error control.

The algebra of the EDM itself is untouched. No transfer tensor, no picking tensor, no
Kraus channel changes. What is added is a per-bond boundary matrix and a rule for
transporting it.

## 1. Notation, fixed to the implementation

Two indices must be kept apart, and conflating them is the easiest mistake to make here.

```
q = order              expansion order, 1 or 2
N                      number of PHYSICAL steps;  T = N eps
M = q N                number of SUB-STEP sites
n = 1 .. N             physical step index;  physical time  t_n = n eps
g = 1 .. M             sub-step index, counting from the OLDEST
m = 1 .. M             a CUT, i.e. a bond between sub-steps
```

Sub-step `g` belongs to physical step `n = (g-1)//q + 1`. Chain positions run
`p = 0 .. M-1` **newest first**, so position `p` carries sub-step `g = M - p`. **Cut `m`**
is the left bond of position `p = M - m`; cut `M` is the dangling output leg.

**Only the cuts `m = q n` are physical times, and only those correspond to a legal run.**
A `q n`-site prefix is exactly the chain of an independent run of `n` physical steps,
`T = n eps` — never `m` steps. An odd `m` at `q = 2` is an *algebraic sub-step prefix* and
nothing more: it sits between the two sub-steps of one Strang step, there is no
`n_steps = m/q` the pipeline would accept (`T/eps` would not be an integer), and it is not
the state at any grid time. Writing `Phi(m)` for a noise history of length `m`, the objects
are

```
S_g^phi          system superoperator family at sub-step g          (d^2 x d^2)
A_{k,g}^phi      Eq.-F1 bath transfer tensor of sub-bath k          (D_a x D_a),  D_a = 4
r_k              initial-state boundary  (1, r_x, r_y, r_z)
e_0              (1, 0, 0, 0)  -- component 0 of a Liouville (Pauli-coefficient) index
P[up, mid, down] the picking tensor
```

The bath lateral index has its **output** end on the newer side: the correlation of an
`m`-site chain is

```
C_{k;Phi(m)} = e_0^T A_{k,m}^{phi_m} ... A_{k,1}^{phi_1} r_k .
```

After all `K` sub-baths are folded in, the chain is a plain MPS with one open arm per site,

```
G_p[phi_up, chi_left, chi_right],
```

whose leftmost `chi_left` is the dangling `d^2` output leg `vec(rho)` and whose rightmost
`chi_right` contracts `vec(rho(0))`.

## 2. The identity the scheme rests on

> **Tensor level, any `m`.** The oldest `m` sites of the `T = N eps` chain, with every
> sub-bath's lateral index at cut `m` sliced to `0`, are exactly the sites an `m`-site
> chain is assembled from.
>
> **Run level, `m = q n` only.** At a physical cut, that prefix **is** the chain of an
> independent run of `n` physical steps, `T = n eps`.

The second statement is the one the scheme uses; the first is what makes it true, and it
is worth keeping separate because an odd `m` has no run to be compared with.

Two facts underlie both, and both are properties of the existing code rather than new
assumptions.

**(a) The per-site tensors depend only on `g`.** `A_{k,g}` and `S_g` are functions of `g`,
`eps`, `q` and the rates — never of `N`. The sample time
`t^*_g = ((g-1)//q + 1/2) eps` is likewise a function of `g` alone. So the sub-step-`g`
tensors of a `T = N eps` build and of any shorter build are the *same arrays*, for every
`g` they share.

**(b) The newest site's boundary is a slice at index 0.** The kernel fixes the newest
site's left lateral index to `0` (`sites[0][:, :, 0:1, :]` in
`kernels/separable_td_mpo.py`). Selecting index `0` of a Liouville index is exactly
contracting with `e_0`. Hence the `m`-site kernel is *literally* the `M`-site kernel with
the lateral index at cut `m` sliced to `0`. For `m = q n` that `m`-site kernel is the one a
run of `n` physical steps builds; for an odd `m` it is a well-defined array that no run
constructs.

So the terminator at cut `m` is

```
l_m = I_{d^2} (x) e_0 (x) e_0 (x) ... (x) e_0            (K factors of e_0)
```

— the system index stays **open**, because it is the output leg; only the `K` bath lateral
indices are closed. `l_m` is therefore a `d^2 x chi_m` **matrix**, not a covector, and the
read at physical time `t_n = n eps` is

```
vec(rho(t_n)) = l_{qn} F_{qn},
F_0 = vec(rho(0)),   F_m = G_{M-m}[phi_up = 0] F_{m-1},
```

a `d^2`-vector. Only `phi_up = 0` is ever needed: the reduced state closes every open arm.

Since the transport of section 3 acts on each cut independently, **only the `N` physical
terminators `l_{qn}` need to be maintained** — not all `M`.

### Why `e_0` is the right closure even out of equilibrium

The dissipative case does not move the boundary, because the bath channel is trace
preserving. In the Pauli-coefficient basis that reads `e_0^T D_k(dt) = e_0^T`: the first
row of `D_k` is `(1, 0, 0, 0)` by construction. Since `A^{phi=0} = I`, the same holds for
the assembled sub-step tensors at both orders (`D_h A^0 D_h` and `A^0 D_h` / `D_h A^0`).

### Why the future cannot simply be padded with `phi = 0`

An obvious-looking alternative is to keep the full `M`-site chain and force
`phi_{m+1} = ... = phi_M = 0`. On the **bath** side that is harmless, exactly by
`e_0^T A^0 = e_0^T`. On the **system** side it is not: the `phi = 0` family is not the
identity but

```
S^0 = e^{eps kappa D[a]} . I_S            (order 1; the Strang halves at order 2)
```

which keeps evolving the cavity under its own dissipator. Padding therefore returns

```
( prod_{g > m} S_g^0 ) rho(t_n)   !=   rho(t_n),        m = q n.
```

Termination has to happen **at the bond**.

## 3. Transporting the terminator

The terminator lives on a bond, so anything that changes the bond basis changes it. Two
kinds of operation do.

### 3.1 The fold

Folding sub-bath `k` multiplies every internal bond by `D_a = 4` and fuses it. With the
fusion order `(old bond outer, new lateral inner)` — the order both the hand-rolled
`apply_step` and quimb's `fuse_multibonds` produce, checked explicitly — the update is

```
l_{qn}  <-  l_{qn} (x) e_0^T            i.e.   kron(l_{qn}, e_0^T),     n = 1 .. N-1.
```

Cut `M = qN` is the output leg, whose lateral index the kernel has already sliced, so
`l_M = I_{d^2}` for the whole run — which is why `rho(t_N)` is exactly the ordinary
`reduced_density_matrix()`.

### 3.2 The compression, and why the sweep direction is not free

Write the chain at cut `m` as `L_m R_m`, where `R_m` is the **prefix** (the oldest `m`
sites plus `vec(rho(0))`). The terminator stands in for `L_m`, so under a change of bond
basis `R^new = Y R^old` it transforms as

```
l^new = l^old Y^{-1}.
```

An inverse is unacceptable — `Y` is exactly the object that becomes singular where the
truncation bites. The transport is therefore only usable in a gauge where every
prefix-side factor arrives as a *left* factor to be multiplied, never inverted. That is a
real constraint on the compression, not a formality:

* **Sweep A — canonicalisation.** LQ from the oldest end toward the newest, i.e. write
  `t_p = L Q` with `Q` row-orthonormal and push `L` **away** from the prefix into position
  `p-1`. Then `R^old = L R^new`, so
  ```
  l_m  <-  l_m L .
  ```
  Sweeping the other way (QR from the newest end) pushes the triangular factor *into* the
  prefix and needs `L^{-1}`.

* **Sweep B — truncation.** With every prefix right-isometric, the optimal rank-`chi'`
  truncation at a bond is given by the dominant eigenvectors `V` of the left block's
  reduced density matrix, which obeys the one-line recursion
  ```
  rho_b = sum_phi  t_b[phi]^dag  rho_{b-1}  t_b[phi] ,
  ```
  costing one extra `O(chi^3)` per bond. The prefix picks up exactly `V^dag`, so
  ```
  l_m  <-  l_m V ,        V^dag V = I .
  ```
  This is the standard density-matrix (`dm`) compression. The alternative gauge — SVD
  sweeping with the singular values pushed into the prefix — gives `Y = S V^dag` and needs
  `S^{-1}`.

Both updates are `O(d^2 chi^2)` per maintained cut, so `O(N d^2 chi^2)` per fold. That is
one power of `chi` below the compression sweep's `O(M chi^3)`, and the ratio is

```
N d^2 chi^2 / (q N chi^3)  =  d^2 / (q chi) ,
```

which is **not** automatically small: at `n_fock = 20`, `q = 2` it is 0.2 at `chi = 1000`
but 1.0 at `chi = 200`. Lower order in `chi`, yes; negligible, not in general — the actual
share has to be measured, not assumed.

## 4. The read

One right-to-left sweep over the finished chain: `M` bond matrix-vector products to
traverse the whole chain, and one `d^2 x chi` terminator projection at each of the `N`
physical cuts. Total `O(M chi^2 + N d^2 chi)`. Nothing is duplicated: all reads share the
one chain `G_0 ... G_{M-1}`; only the `N` terminators are extra storage.

**At `q = 2` the odd cuts must never be emitted.** This is not self-diagnosing: the odd
(mid-Strang) cuts still have `|Tr rho - 1| ~ 1e-16`, so a trace check will not catch a
half-step read.

## 5. Cost and storage

| | folding work | reads |
|---|---|---|
| repeated `solve()` | `N` separate `K`-sub-bath runs, of chain lengths `q*1, q*2, ..., q*N` | free |
| causal prefix | **one** `K`-sub-bath run of chain length `M = qN` | `O(M chi^2 + N d^2 chi)` |

The saving is superlinear rather than exactly `N`, because the repeated runs work at
smaller bond dimension at early times — the temporal bond grows with `T` (Theorem 2 /
Corollary 2.1). The honest statement is **one folding run instead of `N` folding runs**.

Terminator storage is `N d^2 chi` complex numbers against the chain's `M d_phys chi^2`, a
ratio `d^2 / (q d_phys chi)`; at `n_fock = 20`, `q = 2`, `d_phys = 3`, `chi = 1000` that is
under 7% of the chain. It does **not** grow as `O(N^2)`.

## 6. The accuracy property this does not inherit

The truncation at each bond is chosen to minimise the error of the object the compression
sees, which is the chain **as a whole** — that is, the EDM at the final time. A bond
direction that is damped away between `t_n` and `T` gets a small weight and is discarded,
even when it carries real weight in `rho(t_n)`. So:

> The ordinary final-time error control does **not** bound the intermediate-time reads.

This was measured, not assumed (`K = 3`, `n_fock = 3`, `N = 6`, `eps = 0.2`, `q = 1` so
cuts and physical steps coincide, `kappa = 0.6`, `emission = 0.5`; error is
`max |rho(t_n) - rho_exact(t_n)|` against the uncompressed chain):

```
                       max_bond = 32                    max_bond = 16
    n     one T=N run   N separate runs      one T=N run   N separate runs
    1       2.2e-15         0.0                3.1e-04         0.0
    2       5.5e-06         2.3e-15            6.5e-03         4.5e-15
    3       2.2e-05         1.5e-14            5.6e-03         5.5e-06
    4       2.2e-05         1.1e-12            5.3e-03         1.4e-04
    5       1.2e-05         5.7e-07            4.3e-03         1.1e-03
    6       5.9e-06         5.9e-06            2.0e-03         2.0e-03
```

Two things to read off. The prefix error is roughly **flat in `n`** at the final-time
level, whereas independent runs get more accurate the earlier the time — so the early-time
reads are orders of magnitude worse than a dedicated run would give. But the excess over
the scheme's *own* final-time error is only a factor of ~3-4, not orders of magnitude. At
`n = N` the two columns agree exactly, as they must (same computation).

The practical consequence is a procedure, not a warning label: converge the truncation as
usual against the final time, then **re-run once with a tighter `cutoff` and/or a larger
`max_bond` and check that every `rho(t_n)` is stable**, not just `rho(T)`. Tightening
`cutoff` alone is not enough — where `max_bond` is the binding constraint it changes
nothing. The check is cheap because it is one extra run.

### An optional multi-time truncation environment

The bond-`b` environment can be made to cover the intermediate reads as well. Bond `b`
terminates the read at cut `m = M-1-b` and is *internal* to every read at `m' > m`; the
extra environments accumulate by another one-line recursion,

```
Psi_b = t_b[0]^dag Psi_{b-1} t_b[0] + l_{M-1-b}^dag l_{M-1-b}       (seeded at maintained cuts)
```

and truncating on `rho_hat + Psi_hat` (each normalised to unit trace) **adds the terminator
directions to the truncation objective and reweights them upward**. It does *not* force
them into the retained subspace: where `max_bond` is below the joint rank of those
directions, retaining all of them is impossible, and the measurements below show the
trade-off directly. Cost: one more `O(chi^3)` per bond.

Measured, it **rebalances** the error across the time axis rather than removing it. At a
converged truncation it improved every time slice (`max_bond = 32`: `2.2e-05 -> 4.2e-06` at
`n = 3`, and `5.9e-06 -> 3.0e-06` at the final time). At an aggressive truncation it buys
early-time accuracy with late-time accuracy (`max_bond = 16`: `6.5e-03 -> 1.9e-03` at
`n = 2`, but `2.0e-03 -> 6.9e-03` at `n = 6`). It is therefore worth having as an option
and is **not** a default.

**It is also explicitly out of scope for the first implementation.** It changes the
truncation objective itself and carries a measured accuracy trade-off, so it belongs in a
separate, later change with its own evidence — not bundled with the base causal-prefix
work, whose correctness argument does not depend on it.

## 7. Implementation plan

Nothing below is built yet. The plan is stated at the level the code would be written at.

**Scope.** Opt-in, and **generic over separable bath types by construction, not
Dicke-specific**. `SeparableBathEvolution` already serves both `bath_type='separable'`
(Gaudin) and `'separable_td'` (Dicke) without branching on the bath type — it reads
whatever per-site tensors the kernel hands it — and the identity of section 2 uses nothing
beyond that. Dicke is only where the question arose, because a time-dependent grid is what
made the missing time axis hurt. The feature must therefore live in the shared engine and
work unchanged for Gaudin today and for the discretised spin-boson bath when it is built;
a Dicke-only implementation would be the wrong shape. The one prerequisite to confirm in
code is that every separable kernel uses the same newest-site `a_left = 0` boundary, since
that is what makes `e_0` the terminator.

Track 2 (`backend='hpc'`) is untouched. **With the flag off nothing changes at all**: the
default remains the final reduced state (plus whatever observables are exposed), the full
compression menu stays available, and no terminator is allocated or transported.

### 7.1 Result contract

The existing axis contract already has the right home for this, and it is **not** a new
field. Per `driver/solver.py`, `SolverResult.density_matrices` is defined as "`rho(t)`
aligned 1:1 with `times`", and the per-`L` states live in
`sub_bath_final_density_matrices`; it is only the *internal*
`SeparableEvolutionResult.density_matrices` that currently means the per-`L` axis. So:

| field | meaning under this feature |
|---|---|
| `SeparableEvolutionResult.time_density_matrices` | **new**; `rho_L(t)` for the final `L`, one per physical step |
| `SeparableEvolutionResult.density_matrices` | unchanged; per-`L` `rho_L(T)` when `record_rho` |
| `SolverResult.density_matrices` | `= ev.time_density_matrices` — the documented `rho(t)` axis, finally populated for separable Track 1 |
| `SolverResult.sub_bath_final_density_matrices` | `= ev.density_matrices`, unchanged |
| `SolverResult.final_density_matrix` | must equal `density_matrices[-1]` |
| `SolverResult.times` | unchanged: `eps * arange(1, N+1)` |
| `SolverResult.compression_method_used` | **new**; the outer compression path entered, not the per-bond decomposition (see §7.2) |

`record_time_reads=True` and `record_rho=True` are **orthogonal and may be combined**: the
first returns the time axis of the final `L`, the second the final-time axis over `L`.
Neither is a request for the full `L x t` grid, which would need one terminator set per
recorded `L` and is out of scope. `sub_baths = L < K` is fine — the reads are then
`rho_L(t)`, which is what that option already means at the final time.

`SolverResult.time_bond_dims` **stays `None`**, and this is a decision, not an open
question. Its documented meaning is the global maximum bond *after each physical time step
of an evolution that grew step by step*. Neither candidate reading reproduces that here:
the local bond at cut `q n`, and the maximum over bonds `1..q n`, are both properties of a
chain that was compressed against the **final-time** environment, which retroactively
changes the early bonds. There is no "history" to report. `final_time_bond_dims` continues
to mean the finished chain's internal bonds. If a per-cut diagnostic is wanted later it
needs a **new name** — `prefix_cut_bond_dims` or similar — and must not reuse
`time_bond_dims`.

### 7.2 Compression contract

`QuimbEDM.compress` delegates to `qtn.tensor_network_1d_compress`, which returns the
compressed network and nothing else — **no per-bond gauge factors**. The transport of
section 3.2 therefore cannot be bolted onto the current compression call. Two options:

* **(A) own gauge-tracking sweep**: sweep A (LQ) + sweep B (`dm` eigen-truncation)
  implemented over the site arrays, returning the per-bond `L` and `V`. Essentially the
  algorithm quimb's `method='dm'` already runs, but with the factors in hand. The
  *terminator transport* is then algebraically exact; the compression itself remains an
  approximation like any other.
* **(B) recover the gauge by overlap** against the pre-compression chain, which `run()`
  transiently holds. Integrates with the existing compression but needs a canonical form
  anyway and is numerically the weaker of the two.

**(A) is the recommendation** — but it replaces the user's compression path, and that must
be an explicit, validated, *dispatched* contract rather than a silent substitution.

**Dispatch.** `compress_method` is currently forwarded verbatim to
`qtn.tensor_network_1d_compress`, and `_COMPRESS_METHODS = ("zipup", "dm", "direct")` is
quimb's menu. A new name is therefore **not** something quimb can be asked to run:
`QuimbEDM.compress` must **intercept `dm_tracking` before the quimb call** and route it to
the in-repo two-sweep implementation. `_COMPRESS_METHODS`, `validate_compression_combination`
and the direct `run()` entry points all have to learn the name in the same change.

**`record_time_reads` is a generic request, not a separable-only switch.** Its meaning is
"populate `density_matrices` with `rho(t)`"; how a pipeline satisfies it is the pipeline's
business. In particular it is **not** true that the other pipelines already fill that field:
single-bath Track 1 sets `need_rho = record_rho or observables or second_order`, so a
first-order run with neither gives `density_matrices = None`. Rejecting it there on the
grounds that "the history already exists" would be wrong on the facts.

| situation | required behaviour |
|---|---|
| separable Track 1, `compress=True` | the causal-prefix scheme; `compress_method='dm_tracking'` required, any other method raises `ValueError` at the entry point, naming the fix |
| separable Track 1, `compress=False` | causal-prefix with **any otherwise-valid compression configuration** — no compression runs, so no basis change occurs and only the fold-time `kron` is needed. "Otherwise-valid" is literal: the existing enum, type and combination checks (e.g. `dm` + `rsvd`) still apply |
| single-bath Track 1 | force `need_rho = True` and use the existing per-step recording. No `dm_tracking`, no new machinery |
| Track 2 (`backend='hpc'`) | accept and satisfy from the `rho(t)` the 2D contraction already produces. No `dm_tracking` |
| `dm_tracking` with `record_time_reads=False` | **rejected at the entry point.** Its extra sweep and gauge bookkeeping exist only to serve the terminator; growing it into a fourth general-purpose compressor is a separate decision with its own test burden, and is not part of this change |

If a later decision narrows the feature to separable Track 1 only, the flag must be
**renamed** (`record_separable_prefix_reads`) rather than keeping a generic name and
refusing the other pipelines.

**Presets.** `preset='balanced'` / `'robust'` set `compress_decomp='rsvd'`, and the
combination check runs *after* preset resolution — which is why `dm` + those presets raises
today. `dm_tracking` gains that protection **only once it is added to the same validator**;
it is not inherited automatically, and the change is not complete until it is.

**Truncation semantics.** `dm_tracking` is specified as **behaviourally identical to the
existing `dm` path**, not as a re-derivation of it. The dm path applies quimb's trimming to
the eigenvalues `lambda = sigma^2` directly, so the per-mode arithmetic of `abs`, `rel`,
`sum1`, `rsum1`, `sum2`, `rsum2` is quimb's and is deliberately **not restated here** —
restating it would be a second source of truth and a place to be wrong. The specification is
the test, mode by mode for all six. Any mode that cannot be matched must be rejected at
entry, never silently remapped.

That test must compare **gauge-invariant** quantities. An eigendecomposition fixes its
eigenvectors only up to a phase, and up to an arbitrary rotation inside any degenerate
subspace, so two decompositions that are physically identical can differ element by element.
The comparison is therefore: a deliberately **non-degenerate** known spectrum, and equality
of

* the retained rank,
* the projector onto the retained subspace, `V V^dag`,
* the discarded weight,
* the final contracted result.

Raw eigenvectors, or un-gauge-fixed tensor elements, must **not** be the contract.

**Truncation reporting.** Each tracking sweep reports `max_b sum_i discarded lambda_i`, the
same discarded-weight definition `_TruncationAccumulator("discarded_weight")` already
collects on the `dm` path; `SeparableBathEvolution` then applies its existing rule
unchanged — the maximum over every fold in the interval between two recorded `L` values.

**Recording which compression path ran.** `SolverResult` has **no** generic metadata carrier
(`metadata` belongs to `TimestepConvergence`, and `mps.meta` is not part of the public
result contract), so this needs a field of its own:

```
compression_method_used : str or None
    The outer 1D-compress path actually entered -- 'zipup', 'direct', 'dm' or
    'dm_tracking'.  None when no compression ran, or on Track 2.
    It does NOT report the per-bond decomposition: 'rsvd' carries a silent
    per-bond guard that falls back to the exact full SVD, so a single run can
    mix the two and one string cannot describe it.  A per-bond fallback ledger,
    if ever wanted, is a separate diagnostic structure.
```

The matching acceptance test proves the `dm_tracking` branch was actually entered — not
that the field summarises "the algorithm", which it cannot.

### 7.3 Touch points

| file | change |
|---|---|
| new `evolution/prefix_reads.py` | terminator container, the `kron` update, the gauge-tracking two-sweep compression, the final read sweep |
| `evolution/separable_bath.py` | `run(..., record_time_reads=False)`; when set, initialise `l_{qn} = I` for `n = 1..N`, apply the `kron` after each `fold_raw`, route compression through the tracking sweep, do the read sweep, fill `time_density_matrices` |
| `evolution/quimb_edm.py` | **intercept `dm_tracking` in `compress()` before the `tensor_network_1d_compress` call** and route it to the in-repo sweep; keep every existing path byte-for-byte |
| `evolution/_validation.py` | extend `validate_compression_combination` for the new method; the `record_time_reads` / method / `compress` compatibility matrix of §7.2 |
| `driver/solver.py`, `driver/auto_config.py` | add `dm_tracking` to `_COMPRESS_METHODS`; plumb the flag; wire `SolverResult.density_matrices = ev.time_density_matrices`; add `compression_method_used`; reject `record_time_reads` on the single-bath and `hpc` pipelines |
| `models/dicke.py` | nothing |

**Further constraints.** The sweep must be array-module agnostic (the `_xp` helper in
`mps_utils.py`), or the option silently breaks the CuPy backend. Reads are emitted only at
`m = q n`, and the returned times are `n * eps`.

### 7.4 Acceptance tests

Each must be able to fail for the right reason:

1. `q = 1` and `q = 2`: `times`, cuts and `rho(t_n)` correspond one-to-one, and `q = 2`
   never emits an odd cut. (A trace check cannot serve as this test — see section 4.)
2. `q = 2` never **creates or maintains** an odd terminator either — asserted on the
   container, not only on the output, so a silently-carried odd cut cannot hide.
3. `SolverResult.density_matrices` has `len == len(times)` and its last entry equals
   `final_density_matrix`.
4. The last time read equals a plain `reduced_density_matrix()` of the finished chain —
   the `l_M = I` boundary of section 3.1, checked independently of `final_density_matrix`.
5. `record_time_reads=True` together with `record_rho=True`: both axes correct and
   independent.
6. `sub_baths = L < K` returns `rho_L(t)`.
7. `record_time_reads=True` with `compress=False` on a direct `run()` works with any
   otherwise-valid compression configuration — no `dm_tracking` requirement where no basis
   change occurs — while an *invalid* one (e.g. `dm` + `rsvd`) still raises.
8. `record_time_reads=True` on **single-bath Track 1** (first order, `record_rho=False`, no
   observables — the case that currently yields `None`) returns a populated
   `density_matrices`; and on **Track 2** it is accepted, not refused.
9. Incompatible compression configurations — including the preset-driven `rsvd` route, and
   `dm_tracking` with `record_time_reads=False` — raise `ValueError` before any tensor is
   constructed.
10. `dm_tracking` matches the existing `dm` path on a deliberately **non-degenerate** known
    spectrum, for each of the six `CUTOFF_MODES`, on the retained rank, the projector
    `V V^dag`, the discarded weight and the final contracted result — **not** on raw
    eigenvectors or un-gauge-fixed tensor elements.
11. Terminator transport verified separately in the no-discard case, the truncated case,
    and on the CuPy backend (GPU-gated).
12. `compression_method_used` shows `dm_tracking` when that branch ran, and the truncation
    metric is the requested-independent measured one.

### 7.5 Consumers

The reads are interaction-picture states with respect to
`H_0 = omega_c a^dag a + sum_k (omega_k/2) sigma_{k,z}`. Cavity observables that commute
with `a^dag a` — mean photon number, the Fock distribution, `g2(0)` — are unaffected.
Quadratures rotate and need the back transformation.

One consumer needs an explicit caveat. The cavity von Neumann entropy equals the
light–matter **entanglement** entropy only when the global state is pure, i.e. closed
dynamics from a pure initial state on *both* sides. The defaults here — infinite-temperature
bath spins, and any non-zero rate — do not satisfy that, so what `rho(t)` yields is a
mixed-state entropy and not an entanglement measure.

## 8. Verification record

Two independent routes. Every check has a companion that **must** fail, so that a pass is
evidence rather than a tautology.

The two levels of section 2 are tested separately, because only one of them can be
compared against a pipeline run:

* the **tensor-level** identity (the per-site arrays depend only on `g`, so any prefix is a
  valid slice) is an algebraic check on the arrays;
* the **run-level** identity is only ever exercised at physical cuts `m = q n`, against a
  production run of `n` physical steps — an odd `m` has no run to compare with.

The numerical scripts used exactly that convention: **every** reference build, for both
levels, was a legal production build `T = n * eps` with `n_sites = q n`. No odd-length
chain was ever assembled by hand, and none could be. The tensor-level rows are still
established for **every** `g`, because a build of `n` physical steps covers `g = 1 .. q n`
and the union over `n = 1 .. N` covers the whole grid; the arbitrary-`m` slice statement is
then a corollary of per-site equality, not a separate measurement. The Mathematica model,
which has one generic family per site and no notion of `q`, exercises the arbitrary-`m`
statement directly.

The `tau`-as-both-indices conflation corrected in an earlier revision of this note was in
the prose only; it does not touch what was measured.

### numpy, against the code as it stands

`scratchpad/v1_structure.py`, `v2_prefix.py`, `v3_compressed.py`, `v4_multitime.py`.

| check | result |
|---|---|
| `e_0^T A_{k,g}^{0} = e_0^T`, dissipative bath, `q = 1, 2`, all `g`, all `k` | `0.0` exactly |
| ... and the same for `phi = 1, 2` (must fail) | deviation `1.0` |
| tensor level: `T = N eps` transfer tensors and sample times vs shorter production builds, on every shared `g` | `0.0` exactly |
| tensor level: `M`-site kernel sites sliced at `a_left = 0` vs the shorter production builds' sites | `0.0` exactly |
| quimb `fuse_multibonds` order | `(v outer, a inner)`; the other order deviates by `8.5` |
| run level: prefix reads at `m = q n` vs `N` independent uncompressed runs of `n` physical steps, `q = 1, 2`, `K = 1, 2, 3`, closed and dissipative | `<= 1.7e-15` |
| lossless recompression + transported terminators vs uncompressed | `8.0e-11` |
| ... same, but resizing the terminator instead of transporting it (must fail) | `1.0` |
| independent numpy fold vs the production engine at `T` | `7.8e-16` |
| truncated case, and the multi-time environment | section 6 |

### Mathematica, generic symbols

`scratchpad/v5_symbolic.wl`. Symbolic tensors throughout (`d^2 = D_a = d_phys = 2`,
`K = 2`, `M = 3`), so a zero is an algebraic identity. Polynomial identities are settled
with `Expand`. The symbolic model has one generic family per site and does not model `q`,
so it establishes the **tensor-level** statement of section 2 at every `m`; the run-level
statement is the numpy table above.

| check | result |
|---|---|
| `e_0^T D_k(dt) = e_0^T` and `e_0^T D_k A^0 D_k = e_0^T`, symbolic rates | `{0,0,0,0}` |
| ... for a generic `4 x 4` (must fail) | `{-1 + q11, q12, q13, q14}` |
| one fused `M`-site MPS terminated at cut `m` vs the `m`-site chain, `m = 1, 2, 3` | `0` |
| ... `rhoPre[1]` vs `rhoRef[2]` (must fail) | non-zero |
| padded read `= (prod_{g>m} S_g^0) rho`, and `!= rho` | both confirmed |
| gauge transport `l.(L R) = (l L).R` | `0` |
| isometry transport `l.R = (l V)(V^dag R)` for `R` in `range(V)`, exact rational `V` | `0` |
| ... for generic `R` (must fail) | non-zero |
| `kron(l, e_0)` in the `(v outer, a inner)` layout | entries match |

The Mathematica reference builds the chain from **nested** picking tensors, one per
sub-bath — bath `k`'s `phi_down` contracts bath `k-1`'s `phi_up`. Giving every sub-bath the
same noise history is a `K = 1` shortcut and is wrong for `K > 1`.

## 9. What is and is not claimed

**Claimed.** Without compression the scheme is exact: the reads are the numbers `N`
independent solves produce, to machine precision. With compression, the terminator's basis
change is algebraically consistent in exact arithmetic and within the retained projection.

**Not claimed.** That the compressed reads carry only truncation error — the deviation from
an uncompressed run also contains floating-point, LQ/eigh and re-normalisation error, which
is what the `8.0e-11` no-discard figure in section 8 measures. That the intermediate reads
inherit the final-time error bound — they do not (section 6). That the multi-time
environment forces every terminator into the retained subspace, or is a free improvement —
it is a reweighting and a rebalancing (section 6). That any of this is implemented — it is
not (section 7).

**Unchanged.** The EDM derivation, the transfer tensors, the picking tensor, the Kraus
channels, the second-order discretisation of
{doc}`dicke-second-order-discretisation`, and the results any existing call returns.
