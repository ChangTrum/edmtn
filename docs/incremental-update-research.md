# Incremental subspace update for separable-bath EDM folds — research record

A complete record of the investigation into accelerating the separable-bath EDM
fold recursion (paper Eq. 21 / Fig. 5c) by reusing the previous step's left
singular subspace, including the compressed-sensing idea that was tested and
**rejected**.  All work lives in `examples/`; the solver pipeline (`src/`) was
**not modified** — these are offline studies whose conclusion is the design of a
future Layer-4 strategy.

- **Status:** validation complete. Recommended design = **two tiers** (pure
  projection + projection/rSVD), routed by a residual-energy estimate.
- **Models exercised:** Gaudin central-spin model (`separable` bath), the
  end-to-end-validated Phase-2 path.
- **Scale:** CPU, `K = 24–28`, `T = 3–4 g⁻¹`, `eps = 0.2 g⁻¹`, second-order
  expansion, `xi = 1e-6`. Smaller than the paper's `K = 49, T = 15` but large
  enough that the qualitative findings (regime dependence, scaling exponent,
  tier speedups) are robust. Exact `L*` and fitted exponents may shift modestly
  at production scale.

---

## 1. Motivation and hypothesis

For a **separable bath** the EDM is built by folding sub-baths in one at a time
(Eq. 21): `rho_{L+1} = C_{L+1} [∏ P] rho_L`. Each fold is one MPO×MPS contraction
along the time axis followed by an SVD recompression — the `O((D·D_a)³)` SVD
sweep is the cost bottleneck (`D_a = 4` is the sub-bath MPO lateral bond).

The **hypothesis** (from the incremental-update + CS proposal) was that folding
sub-bath `L+1` barely rotates the left singular subspace already spanned at step
`L`. If true, at every bond `τ` the bond matrix decomposes as

```
M_τ = M_τ^∥  +  M_τ^⊥
      (in V^(L)_τ)   (residual, claimed rank ≤ D_a = 4)
```

so `M^∥ = U^(L) (U^(L)ᴴ M)` needs only a GEMM (no SVD) and only a small residual
must be recovered. The paper's Fig. 12 (bond dimension collapses onto a universal
curve vs `ḡ_L·t`) motivated the idea but does **not** by itself imply the
subspaces are *nested* — that is exactly what had to be tested.

---

## 2. Methodology — measuring the subspace increment

Two compressed EDM-MPS at steps `L` and `L+1` have the **same** site/leg
structure (same `num_sites`, same `d_phys = 7` open arms, shared `d² = 4`
boundary); only the internal bond dimensions differ. At each internal bond `τ`
the left singular subspace `V^(L)_τ` is the column space of the left-canonical
block, embedded in the shared ambient left space. We compare `V^(L)` and
`V^(L+1)` **without ever forming the ambient block**, via transfer matrices
(`examples/edm_incremental.py`):

- **Cross-overlap** `E_τ = Q_L(τ)ᴴ Q_{L+1}(τ)` (`cross_left_overlaps`): both
  left-canonical isometries; the singular values of `E_τ` are `cos θ_i`, the
  cosines of the principal angles between the two subspaces.
- **Bond density** `ρ_left(τ)` of step `L+1` (`right_bond_density`): the right
  environment, used to weight the residual energy.

Three diagnostics per bond (`analyse_transition`):

1. **Residual energy ratio** `‖M^⊥‖/‖M‖` — the fraction of the new signal's
   weight outside the old subspace, computed exactly and energy-weighted as
   `ratio² = 1 − Tr[E ρ_left Eᴴ]/Tr[ρ_left]`. This equals the **Tier-1
   reconstruction error** (see §4).
2. **New-direction count** `n_new(δ) = #{ new directions with cos θ_i < 1−δ }`
   for `δ ∈ {ξ², ξ, √ξ}`. The hypothesis predicted `n_new(ξ) ≤ D_a = 4`.
3. **Chordal distance** `√(Σ sin²θ_i)/√(D^(L)_τ)` — normalised Grassmann distance
   between old and new left subspaces.

**Estimator self-check:** comparing a step with *itself* gives `n_new = 0`,
residual `~1e-7`, chordal `~1e-7` (numerical floor) — confirming the transfer
math (`validate_subspace.py --self-check`).

---

## 3. Finding 1 — the subspace increment is regime-dependent

`examples/validate_subspace.py` (K=24, T=4, eps=0.2):

| fold `L→L+1` | `ḡ_L` | residual `η` | `n_new(ξ)` (δ=1e-6) | `n_new(√ξ)` (δ=1e-3) | chord/√D |
|---|---|---|---|---|---|
| 3→4  | 0.64 | 7.4e-2 | 82 | 34 | 1.7e-1 |
| 8→9  | 0.86 | 3.4e-2 | 97 | 19 | 8.0e-2 |
| 15→16| 0.98 | 9.0e-3 | 89 | **0** | 1.8e-2 |
| 22→23| 1.00 | **4.4e-4** | **5** | **0** | **8.4e-4** |

The literal `n_new(ξ) ≤ D_a = 4` (with the stringent `δ = ξ = 1e-6`) **fails**
for strong early sub-baths: the subspace genuinely rotates. But every diagnostic
**collapses as `L` grows** — exactly as the proposal's Sec. 3.3 argued
(`Δḡ/ḡ_L ~ g²_{L+1}/(2 ḡ²_L) → 0`). By `L = 22→23` even the stringent
`n_new(ξ) = 5 ≈ D_a`, and with the looser `δ = √ξ` the new-direction count is
**0** already by `L = 15`. The crossover, not a single pass/fail, is the result.

---

## 4. Finding 2 — critical `L*` and a clean scaling law

`examples/critical_L_and_scaling.py` (K=28, T=3, eps=0.2) sweeps every
consecutive fold and fits the diagnostics against `x = g²_{L+1}/ḡ²_L`.

**Critical `L*`** (smallest `L` whose fold satisfies the criterion):

| criterion | `L*` | meaning |
|---|---|---|
| `max dD = 0` (bond stops growing) | **15** | no new bond dimension |
| `n_new(√ξ) = 0` | **15** | no new orthogonal directions at `δ=1e-3` |
| `max_resid < 1e-3` | **24** | pure projection (Tier 1) accurate |
| `n_new(ξ) ≤ D_a=4` | 25 | stringent `δ=1e-6` |
| `max_resid < 1e-4` | 27 | |

**Scaling law** (power-law fit `y ~ C·x^p`, `n = 26` folds):

| diagnostic | fit | `R²` |
|---|---|---|
| max residual ratio | `0.161 · x^0.854` | 0.976 |
| rms residual ratio | `0.0404 · x^0.853` | 0.976 |
| chordal / √D | `0.385 · x^0.830` | 0.945 |

All three share exponent `≈ 0.85` — **close to linear, slightly sublinear** (the
proposal predicted exponent 1: `Δḡ/ḡ_L ~ x/2`). The high `R²` means the
residual magnitude is **predictable from `x`** before a fold runs — the basis for
adaptive routing.

> Note on `r_eff`. The *rank* needed to reach cutoff precision is **not** small
> just because `η` is small: even when `η ~ 1e-3` the residual spectrum has a
> slowly-decaying tail, so `r_eff` (#singular values of `M^⊥` above
> `ξ·s_{d²}`) is moderate. `r_eff` is effectively **bimodal**: ~0 at late folds
> (where Tier 1 already suffices) or tens where it matters. There is no broad
> band of "small `r_eff` but non-negligible residual" — this is what kills the CS
> tier (§7).

---

## 5. Tier 1 — pure projection (kept, core acceleration)

`examples/projection_poc.py`, fold `L = 22→23` (K=24). Replace the per-bond SVD
sweep by a single left-to-right GEMM sweep that projects the uncompressed `L+1`
MPS onto step-`L`'s left isometries (`edm_incremental.tier1_project`), keeping the
bond at `D^(L)`. **Zero SVDs.**

| quantity | value |
|---|---|
| accuracy `max|Δ⟨S_z(t)⟩|` (proj vs full-SVD) | **1.6e-4** |
| reduced-`ρ` Frobenius error | 6.6e-5 |
| full-SVD compress (median of 3) | 2430 ms |
| projection sweep | **94 ms → 26×** |
| incl. one-time left-canonicalisation | 11.8× |

The reconstruction error is exactly the residual energy ratio of §2–3, so Tier 1
is accurate precisely in the regime `L ≳ L* ≈ 24` (and for late folds the bond
does not even grow, `D(proj) = D^(L) = 95`).

---

## 6. Tier 2 — projection + rSVD on the residual (kept, mild)

`examples/projection_poc.py`, fold `L = 8→9` (K=24), per-bond benchmark
(`bond_matrix_and_old_subspace` replicates the sequential right-to-left sweep so
the extracted `M` has the true, already-truncated right bond):

```
M^∥ = U (Uᴴ M)        [GEMM, U carried over]
M^⊥ = M − M^∥
small rSVD of M^⊥ at the residual rank
```

| quantity | value |
|---|---|
| residual rank `r` (to cutoff) | median **48** vs `D_old` median 84 (`r/D_old ≈ 0.6`) |
| reconstruction error | matched to baseline (`~1e-7`) |
| per-bond speedup (full SVD / Tier 2) | median **1.6×**, max 4.5× |

Honest reading: at `L = 8→9` the residual is **not** rank-4 — it is a
*rotation-correction* residual of about half the old rank, so the win is modest.
Some bonds have `r = 0` (pure projection works → up to 11×); others rotate
(`r ~ D_old/2 → ~1.6×`).

---

## 7. Tier 1.5 — compressed sensing (TESTED, REJECTED)

The proposal inserted a CS layer between Tiers 1 and 2: recover the low-rank
`M^⊥` from `p = O(r(m+n) log mn)` cheap rank-one measurements
`y_i = a_iᴴ M b_i − (a_iᴴ U)(Uᴴ M b_i)` (the second term reusing the projection
byproduct `Uᴴ M`), with the picking-tensor `φ=0` null block as a support prior.

Implemented as an examples-only prototype (`examples/cs_recovery.py`: rank-one
measurement operator + Singular Value Projection with a Lipschitz step) and
validated on **real EDM residuals** (`examples/cs_recovery_poc.py`). The
algorithm is **correct** (synthetic self-check recovers a rank-`r` complex matrix
to `2e-10`; recovers real residuals to the cutoff at `p ≈ 2–3·r(m+n)`), but three
independent measurements show it is **not worth integrating for this model**:

1. **The `φ=0` support prior does not hold.** For the post-contraction bond
   residual the `φ=0` column block carries **~95–96%** of `‖M^⊥‖` (not zero) at
   every fold tested — the proposed measurement-reduction is unavailable in this
   form.
2. **No "small-`r`, non-negligible-residual" regime** (the `r_eff` bimodality of
   §4). Fold `L=22→23`: representative bond `r_eff = 13`, info limit `r(m+n) = 5%`
   of `mn`, but Tier 1 alone already gives `η = 1.66e-6 ≈ ξ`, so CS buys almost
   nothing. Fold `L=8→9`: `r_eff` median 48 → info limit **21% of `mn`** — CS
   would need a large fraction of `mn` measurements (Tier-2 territory).
3. **CS recovery is far slower than rSVD.** SVP's measurement operator is dense
   `O(mn·p)` per iteration. On the `L=22→23` bond (`M` = 380×665):

   | method | wall-clock |
   |---|---|
   | one-shot rSVD of `M^⊥` (Tier 2) | **3.35 ms** |
   | full SVD of `M` (baseline) | 35 ms |
   | CS recovery, `p = 11% mn` | **49 s** |
   | CS recovery, `p = 16% mn` | 76 s |

   → CS is **~2×10⁴ slower** than the rSVD it would replace.

**Conclusion.** Drop Tier 1.5 (CS residual recovery) and the picking-tensor
support prior. The dispatcher's *control logic* (predict `r_eff` from the scaling
law, route by `η`) is sound and retained — but as a **two-way** router. CS could
only pay off for a bath whose residual is genuinely low-rank *and* non-negligible,
or if measurements were computed **without forming `M`** (directly from the
tensor factors) — future work, not a reason to touch the pipeline now.

---

## 8. Final framework — two tiers, η-routed

```
Fold L → L+1, bond τ:

  1. GEMM projection  →  M^∥ = U^(L) (U^(L)ᴴ M);  estimate η = ‖M^⊥‖/‖M‖
  2. route by η:
       η <  ξ   →  Tier 1: keep M^∥, no SVD            (L ≳ 24, ~26× end-to-end)
       η ≥  ξ   →  Tier 2: M^∥ + rSVD(M^⊥)             (L < 24, ~1.6× per bond)
```

| component | status | value |
|---|---|---|
| Tier 1 pure projection | **validated**, 26×, `L ≳ 24` | core acceleration |
| Tier 2 projection + rSVD | **validated**, ~1.6× (`L=8`) | mild, strong-coupling band |
| scaling law `η ≈ 0.16 x^0.85` (`R²≈0.97`) | **validated** | routing predictor |
| η-based dispatcher | **logic correct** | two-way (Tier 1 / Tier 2) |
| ~~Tier 1.5 CS recovery~~ | **rejected** | slower than rSVD, no support prior, no niche |
| ~~picking-tensor zero-block prior~~ | **rejected** | `φ=0` block carries ~95% of `‖M^⊥‖` |

The threshold `η < ξ` is the per-fold form; the dispatcher in code uses the
accumulation budget `η < ξ/(T·L)` to keep total error `≤ ξ` over all folds×times.

---

## 9. Reproduce

All scripts are pure CPU / NumPy. Run from the package root with the `src` layout
on the path (matching the other examples):

```bash
PYTHONPATH=src python examples/validate_subspace.py --self-check   # estimator self-check
PYTHONPATH=src python examples/validate_subspace.py                # §3 regime table
PYTHONPATH=src python examples/critical_L_and_scaling.py           # §4 L* + scaling law
PYTHONPATH=src python examples/projection_poc.py                   # §5 Tier 1 + §6 Tier 2
PYTHONPATH=src python examples/cs_recovery.py                      # CS algorithm self-check
PYTHONPATH=src python examples/cs_recovery_poc.py                  # §7 CS validation (rejection)
```

(In the `quimb` conda env, e.g. `/opt/anaconda3/envs/quimb/bin/python`.)

### Files (all under `examples/`, pipeline untouched)

| file | role |
|---|---|
| `edm_incremental.py` | shared: streaming fold loop (snapshots every `L`), transfer-matrix subspace diagnostics, `tier1_project`, randomized SVD, per-bond extraction |
| `validate_subspace.py` | §2–3 per-bond subspace diagnostics + self-check |
| `critical_L_and_scaling.py` | §4 critical `L*` and power-law fit |
| `projection_poc.py` | §5 Tier-1 end-to-end + §6 Tier-2 per-bond benchmark |
| `cs_recovery.py` | Tier-1.5 CS prototype (rank-one measurement SVP) — examples-only |
| `cs_recovery_poc.py` | §7 offline CS validation + dispatcher analysis |
| `data/*.npz`, `pictures/*.png` | saved results + plots |

---

## 10. Caveats and future work

- **Scale.** Validation ran at `K = 24–28`, `T = 3–4 g⁻¹` on CPU. The paper uses
  `K = 49`, `T = 15 g⁻¹`, `D_c = 400`. Qualitative findings are robust; re-fit
  `L*` and the scaling exponent at production scale before hard-coding thresholds.
- **Model.** Only the Gaudin separable bath was studied. The `r_eff` bimodality
  that defeats CS is a property of this bath's residual spectrum; a different bath
  (genuinely low-rank, non-negligible residual) could revive the CS tier.
- **Packaging.** When promoting Tiers 1/2 into `src/edmtn/decomposition/`
  (Strategy D in `edm_technical_plan.md`), the hard part is maintaining a globally
  consistent canonical form while *growing* selected bonds in the streaming
  projection sweep (Tier 2). Tier 1 (no bond growth) is straightforward; the
  `tier1_project` sweep already produces a valid `EDMMPS`.
- **CS, only if.** Revisit CS solely if measurements can be formed without
  materialising `M` (so the cost is `O(p·rank)` not `O(mn·p)`), e.g. directly from
  the MPO×MPS factors during contraction.

---

## 11. Addendum — transition-zone rotation tracking (Procrustes)

`examples/rotation_tracking_poc.py` (K=24, folds L=16/19/22). In the transition
zone the bond is saturated (`dD = 0`, `n_new(√ξ) = 0`) yet still weakly rotates
(`n_new(ξ) > 0`). The change of left subspace is then a **rotation plus a small
tilt**: `U_{L+1} = U_L R + (out-of-span)`, with `R` the optimal Procrustes
rotation (unitary polar factor of `E = U_L^H U_{L+1}`). Measured: the Procrustes
residual `‖U_{L+1} − U_L R‖` equals the chordal distance `√Σsin²θ` to ~1e-5
(identity by construction), confirming the rotation picture.

**Rotation tracking** = project `B = U_L^H M` (the Tier-1 byproduct), then take a
small SVD of the *reduced* `D×n` matrix: `B = R Λ V^H`. This yields the rotation
`R` and the full Schmidt spectrum `Λ` from a `D`-sized decomposition — no random
projection / power iteration.

| measured (transition zone) | value |
|---|---|
| premise `\|procrustes − chordal\|` | ≤ 4.5e-5 (rotation + tilt confirmed) |
| spectrum recovery `‖Λ_track − Λ_true‖` | median **3.7e-10** (near-exact) |
| decomposition speedup | **4.9×** vs full SVD, **1.6×** vs rSVD |
| reconstruction error | `= η` (the dropped out-of-span tilts) |
| bonds with `η ≤ ξ` (track already cutoff-accurate) | ~27% |

**Verdict — partial, useful.** Rotation tracking is the cheap, *deterministic*
way to read a bond's **Schmidt spectrum and rotation** in this zone (≈exact, ~5×
cheaper than full SVD, on a `D×n`/`D×D` matrix instead of `m×n`). It **fully
replaces rSVD where `η ≤ ξ`** (~27% of bonds). Where `η > ξ`, the "weak rotation"
is `r_eff` (15–52) genuine small-angle out-of-span tilts carrying energy `η`; those
are real content above cutoff and still need capturing (rSVD / incremental update)
for strict reconstruction — so rotation tracking is an **auxiliary** to Tier 2,
not a universal replacement, and the headline "O(D²)" is really an `O(D²n)`/`O(D³)`
reduced decomposition (truly `O(D²)` only if `R` is tracked incrementally rather
than recomputed). Pipeline unchanged; this is an examples-only study.

### 11b. Incremental tracking at cutoff vs rSVD

`examples/incremental_rotation_poc.py` pushes the idea to *cutoff* reconstruction:
to reach `xi` the in-span rotation must be augmented by capturing the out-of-span
residual (`r_eff` tilts). Tested at cutoff against cold rSVD across the zone:

| finding | result |
|---|---|
| power iterations needed? | **no** — single-pass rSVD (n_iter=0) reaches cutoff (median residual err 2.2e-7 vs cold 2-iter 1.8e-7) |
| single-pass vs cold 2-iter rSVD | **~2-3× faster** (drops 2 power iterations), same accuracy |
| incremental (in-span SVD + single-pass residual) vs cold rSVD | **0.94× (~parity, slightly slower)** |
| incremental vs single-pass rSVD | **0.37× (incremental ~2.7× slower)** — the in-span SVD is overhead |
| spectrum recovered by incremental | sv_err median **8e-10** (near-exact) |
| rotation per fold (bond τ=13) | max angle 1.7e-2 → 9e-3 → 2e-3 rad over L=16/19/22 (small, decreasing) |

**Verdict.** Incremental rotation tracking **does not beat rSVD for cutoff
compression** — reaching cutoff means capturing the `r_eff` out-of-span tilts
(O(mn·r_eff)), the same dominant cost as rSVD, and the in-span SVD is extra. The
genuine, actionable speedup is orthogonal to rotation tracking: **the residual
spectrum decays fast enough that a single-pass rSVD (no power iterations) reaches
cutoff and is ~2-3× faster than the current cold 2-iter rSVD** — a low-risk Tier-2
tweak. Rotation tracking's distinctive value stays the clean rotation `R` +
Schmidt spectrum it returns for ~free (useful for entanglement diagnostics /
adaptive truncation, not for plain compression). The slowly-evolving per-fold
rotation suggests cross-fold composition of `R` is feasible, but realising it
needs the MPS gauge bookkeeping (the same hard part flagged in §10).

**Net (vs CS, §7).** This line is far more promising than compressed sensing: it
reaches cutoff, recovers the spectrum near-exactly, runs at parity-or-better with
rSVD (not ~10⁴× slower), and yields a concrete pipeline tweak (single-pass
residual rSVD). It is an *enhancement/auxiliary* to the two-tier scheme, not a
replacement for rSVD.

---

## 12. End-to-end adaptive 3-tier algorithm (L=0..K) vs the pipeline

`examples/adaptive_tiers_e2e.py` runs the full fold `L=0..K` with a per-bond
adaptive compressor and compares to the **unmodified pipeline** (`EDMSolver`, the
Fig. 6 algorithm) at identical settings. Per bond, from the projection
`B = U_L^H M`, `M^perp = M - U_L B`, `eta = ‖M^perp‖/‖M‖`:

- **Tier 1** (`eta < xi`): pure projection (in-span SVD of `B`).
- **Tier 1.5** (`eta ≥ xi`, residual "easy"): single-pass rSVD of `M^perp` +
  merge/truncate — accepted iff the computed `n_new(√ξ)=0` and `dD=0`.
- **Tier 2** (otherwise): cold rSVD (2 power iters) + merge/truncate.

Orthogonality is monitored via `|Tr ρ(T) − 1|` with a re-canonicalisation trigger.

**Results (K=24, T=3 g⁻¹, eps=0.2, xi=1e-6, M5 Air / CPU):**

| metric | value |
|---|---|
| `<S_z(t)>` max abs error vs baseline | **2.42e-3** (algorithm correct) |
| tier coverage | T1 14.5% / T1.5 22.7% / T2 62.8% |
| T1.5 bonds with `n_new(√ξ)=0 ∧ dD=0` | **158/158 (100%)** (faithful) |
| re-canonicalisations triggered | 0 (orthogonality held) |
| baseline wall / adaptive wall | 49.4 s / 158.2 s → **0.31× (3.2× slower)** |
| adaptive decomposition time | 47 s (≈ baseline *total*); rest ≈ transport overhead |

**Verdict — accurate & faithful, but NOT faster (naive implementation).** Two
costs sink it:
1. **Per-fold subspace transport.** Building `U_L` in each bond's basis
   (left-canonicalising the *uncompressed* L+1 MPS + the cross-overlap transfer)
   scales like the uncompressed bond (`~4D`) — i.e. like the baseline's own SVD
   cost — and dominates (~110 s of 158 s).
2. **Faithful tier decision is not cheap.** `n_new(√ξ)` (principal-angle
   definition) and `dD` are properties of the *result*; deciding them requires a
   probe decomposition, so the decision costs about as much as just doing the
   compression. (The residual-singular-value shortcut does **not** reproduce the
   principal-angle `n_new` — verified.)

The decomposition FLOPs per bond can be made smaller than a full SVD, but only if
(a) the subspace is **carried across folds with streaming gauge tracking** instead
of re-transported each fold (the hard part flagged in §10), and (b) the tier is
chosen from a **cheap predictor** — e.g. the `η ≈ 0.16 x^0.85` scaling law (§4) as
a function of `L`/`x` — rather than computing `n_new`/`dD` per bond. Until both are
in place, the unmodified pipeline is faster; the adaptive scheme is validated as
*correct and faithful* but its speedup is gated on that streaming machinery.
Examples-only; pipeline unchanged.

### 12b. Follow-up: hard-coded tier, projection-free 1.5/2 -- and a corrected bottleneck

`examples/adaptive_tiers_hardcoded.py` tests the two improvements from §12:
(1) **hard-code the tier** per `(L,tau)` from an offline oracle (the cheap-predictor
assumption -- drops the per-bond decision); (2) **remove the projection from Tier
1.5/2** (rSVD directly on `M`, no `U_L`, no transport); Tier 1 stays pure
projection, with `--t1 project` (transport) vs `--t1 rsvd` to test if it is even
needed. Same K=24/T=3/eps=0.2/CPU setting.

| variant | wall | speedup | max `\|d<Sz>\|` |
|---|---|---|---|
| baseline pipeline | 49.5 s | 1.00× | — |
| §12 adaptive (per-bond decision, projection) | 158.2 s | 0.31× | 2.4e-3 |
| hard-coded, T1=project | 135.9 s | 0.36× | 1.8e-3 |
| hard-coded, T1=rsvd | 128.2 s | 0.39× | 1.8e-3 |

What the experiment establishes:

* **Hard-coding the tier helped** (158→128 s): the per-bond decision/probe cost
  ~30 s. Still 0.39×.
* **Tier-1's projection is unnecessary for accuracy**: `--t1 project` and
  `--t1 rsvd` give *identical* error (1.81e-3); the transport it needs is only
  ~3.3 s (not the ~110 s §12 guessed -- **that attribution was wrong**).
* **Removing the projection did NOT help end-to-end.** It does cut the per-bond
  decomposition (T1.5 52→24 ms, T2 87→56 ms, no probe/merge), but two facts sink
  it: (a) cold rSVD at rank `~D` is *not* cheaper than LAPACK full SVD (≈6 GEMMs of
  `O(mnD)` vs one `O(m²n)`); (b) the decomposition is not the bottleneck anyway.

**Corrected bottleneck (the key result).** A clean component breakdown of the
baseline fold (K=24) is:

| component | time | share |
|---|---|---|
| `fold_uncompressed` (MPO×MPS) | 0.8 s | 2% |
| **`left_canonicalize` (QR sweep on the uncompressed MPS)** | **~32.5 s** | **66%** |
| full-SVD truncation (what the tiers optimise) | ~14 s | 28% |

The **left-canonicalisation QR sweep dominates (~66%), not the SVD (~28%)**.
Folding+canonicalisation is shared by baseline and every adaptive variant, so no
amount of decomposition cleverness (tiers, rSVD, projection, CS) can beat the
baseline by more than ~1.4× — and rSVD-at-rank-`D` is actually slower than the
LAPACK SVD, so the variants lose. **The real optimisation target is the
canonicalisation**: avoid re-canonicalising the full uncompressed MPS every fold
by carrying the canonical form across folds (streaming gauge maintenance) — the
same machinery §10/§12 flagged, now identified as the *dominant* cost rather than
the transport. Until then the unmodified pipeline is the fastest option; the tier
scheme remains validated as correct/faithful and useful for *what it returns*
(spectrum/rotation, §11), not for raw fold speed. Examples-only; pipeline unchanged.

### 12c. Correction: the bond bloat in §12/§12b was a bug (r_cap cap)

> **The §12 and §12b numbers (0.31× / 0.39×, 2.4e-3 error) were produced with a
> bug and are superseded by this section.**

Investigating why the adaptive bonds reached **157 vs the baseline 95** (they
should match — §11 showed `dD=0` in the saturated zone) revealed a cap bug in
`tier_decompose`. Per-fold from the *same* baseline input the adaptive was
faithful (bond ≈ baseline, rho_err ~1e-8), so the bloat had to come from the
adaptive's own trajectory. Direct comparison of the bond sequence pinpointed it:

```
adaptive bonds: [8, 16, 32, 64, 128, 155, 157, ...]   <- doubling, bloats
baseline bonds: [16, 40, 78, 87, 90, 95, 95, ...]
```

**Root cause.** The residual capture used `r_cap = min(D_old, m, n)`. The fold
multiplies the bond by `D_a = 4` (uncompressed `= 4·D_old`), so the residual rank
can be up to `~3·D_old`. Capping it at `D_old` lets the merged basis
`[U_old | W]` hold at most `2·D_old` columns → in **early/strong folds** (where the
true residual rank `>> D_old`) the residual is **under-captured**, the state is
corrupted, and every subsequent fold's residual saturates the cap → the bond
**doubles** until it plateaus (~157). Confirmed by lifting the cap:

```
r_cap = D_old (buggy): [8,16,32,64,128,161,161,...]   bloats
r_cap = min(m,n)     : [16,47,83,90,104,104,104,...]  matches baseline (109)
```

**Reconciliation with "new directions cut in half" (§11).** That `r_eff ~ 0.5·D`
was a **saturated-zone** measurement, where the cap never binds (`r_eff < D_old`)
and the adaptive is faithful. The bloat lived entirely in the **early folds**,
where `r_eff ~ 3·D` and the `D_old` cap mutilated it — a different regime, not a
contradiction.

**Fix.** Probe the residual with a sensible cap `2·D_old + 16`; if even that
saturates (genuinely high-rank residual, early fold), fall back to a full SVD of
`M` (a "Tier 0" = baseline step). Correct everywhere, still cheap where `r_eff` is
small.

**Corrected end-to-end (K=24, T=3, eps=0.2, xi=1e-6, CPU):**

| | buggy §12 | **fixed §12c** |
|---|---|---|
| `<S_z(t)>` max abs error vs baseline | 2.4e-3 | **4.3e-6** (essentially exact) |
| adaptive wall / baseline 49.5 s | 158.2 s → 0.31× | **60.9 s → 0.81×** |
| decomposition time | 47 s (bloated) | 22.7 s |
| tier coverage | — | T1 16.7% / T1.5 35.2% / T2 48.1% / T0 0% |
| T1.5 with `n_new(√ξ)=0 ∧ dD=0` | — | 245/245 (100%) |

So the corrected projection-based adaptive (with the per-bond probe + merge) is
**accurate (4e-6) and close to baseline (0.81×)**. Adding the §12b improvements on
top — **hard-coded tier** (drop the probe/decision) and **projection-free
Tier-1.5/2** (rSVD directly on `M`) — finally turns it positive
(`adaptive_tiers_hardcoded.py`, rebuilt on the fixed oracle):

| variant | wall | speedup | max `\|d<Sz>\|` |
|---|---|---|---|
| baseline pipeline | 49.6 s | 1.00× | — |
| hard-coded, T1=project | 46.5 s | **1.07×** | 5.9e-6 |
| hard-coded, T1=rsvd | 43.7 s | **1.14×** | 6.2e-7 |

**Net (corrected).** With (1) the cap bug fixed, (2) the tier hard-coded (cheap
predictor), and (3) Tier-1.5/2 projection-free, the adaptive fold is **~1.14×
faster than the pipeline at matched accuracy (6e-7)** — and `T1=rsvd` (no
projection/transport at all) is both fastest and most accurate, so Tier-1's
projection/streaming-carry is unnecessary here. The speedup is modest because it
is **capped by the shared left-canonicalisation** (~66% of the fold, §12b): the
decomposition shrank from ~14 s (baseline SVD) to ~9 s, saving ~5 s of ~49 s ≈ the
observed ~1.1×, right at the canon-limited ceiling (~1.4×). To go beyond it the
canonicalisation itself must be carried across folds (streaming gauge), the lever
identified in §12b. Examples-only; pipeline unchanged.

## 13. Collapse of the routing — uniform single-pass rSVD is self-sufficient

The §12 study ended with: Tier-2 = cold rSVD (2 power iterations, the accuracy
guarantor); Tier-1/1.5 = single-pass rSVD; projection removed everywhere. That
leaves one question with a large payoff: **does a *single-pass* rSVD also suffice
on the Tier-2 bonds?** If yes, single-pass rSVD is reliable on its own and we need
**no tier routing, no hard-coded schedule, no probe, and no `η`/`n_new`/residual
computation at all** — and, crucially, no baseline to lean on. The scaling law
(§5) then loses its pre-emptive routing role and becomes a purely information-
theoretic statement, awaiting a new application.

`uniform_rsvd_e2e.py` tests this directly: a **uniform** per-bond decomposition
over the whole `L=0..K` fold with **no oracle and no subspace transport**. The
rank is chosen purely from the rSVD spectrum with a **resolution guard** — grow
the sketch (geometric ×2, warm-started from the previous fold's bond size) until
the smallest *computed* singular value drops below the `rel_ref` cutoff
`ξ·s[d²]`. That guard is what makes it deployable without a reference run: it
guarantees no kept direction can be hiding in an un-computed tail. Three uniform
strategies vs the pipeline:

* `svd` — full SVD per bond (sanity; should reproduce baseline exactly),
* `rsvd0` — single-pass rSVD (`n_iter=0`)  ← the candidate,
* `rsvd2` — cold rSVD (`n_iter=2`)  ← the accuracy reference.

**Results** (Gaudin, `K=24`, `T=3 g⁻¹`, `ε=0.2 g⁻¹`, order 2; CPU):

| ξ | mode | wall | speedup | max `\|d<Sz>\|` | Dmax | sketch tries |
|---|---|---|---|---|---|---|
| 1e-6 | svd | 47.9 s | 1.04× | 0 (exact) | 95 | 1.00 |
| 1e-6 | **rsvd0** | 43.3 s | **1.15×** | **1.55e-7** | **95** | 1.07 |
| 1e-6 | rsvd2 | 54.2 s | 0.92× | 9.7e-12 | 95 | 1.07 |
| 1e-6 | rsvd0, 5 seeds (worst) | 52.9 s | 1.07× | **1.72e-7** | 96 | 1.07 |
| 1e-8 | svd | 210.7 s | 1.10× | 1.2e-14 | 175 | 1.00 |
| 1e-8 | **rsvd0** | 170.7 s | **1.36×** | **8.0e-10** | 187 | 1.11 |
| 1e-8 | rsvd2 | 196.5 s | 1.18× | 2.0e-12 | 175 | 1.11 |

**Verdict: single-pass rSVD is reliable on its own — the routing collapses.**

1. **Accuracy always beats the cutoff.** The single-pass `<S_z(t)>` error sits
   *below ξ itself* at both cutoffs (1.6e-7 < 1e-6; 8e-10 < 1e-8) and is
   **seed-independent** (worst of 5 draws = 1.7e-7). The resolution guard, not the
   luck of the random sketch, sets the accuracy — exactly the property required to
   trust the result with no baseline.
2. **It is the fastest.** `rsvd0` beats both full SVD and cold rSVD at both
   cutoffs (1.15× / 1.36×). Power iterations cost more than they buy here because
   the EDM bond spectrum decays fast.
3. **The one honest caveat is compression, not accuracy.** At the tight ξ=1e-8,
   single-pass slightly **over-retains** the bond (Dmax 187 vs 175, +7%): imperfect
   subspace resolution leaves a few singular values sitting just above the
   threshold, so rank-selection keeps them. This is the *safe* failure mode — you
   pay a little bond dimension, you never lose a direction (the guard forbids it).
   `rsvd2` recovers the exact bond (175) for modest extra cost if compactness
   matters more than speed. At the production ξ=1e-6 there is no inflation at all
   (Dmax 95 = baseline). The mean sketch-tries ≈ 1.1 means the guard rarely even
   has to re-grow.

**Bond-growth trajectory (does it still plateau? is it better than full SVD?).**
`Dmax` after each fold `L` (`--modes svd,rsvd0,rsvd2`, trajectory instrumented in
`run_uniform`):

ξ=1e-6:
```
svd  : 16 40 64 71 75 78 84 90 90 95 95 ... 95     plateau at L=10
rsvd0: 16 40 64 73 76 84 89 92 95 95 95 ... 95     plateau at L= 9
Δ    :  0  0  0  2  1  6  5  2  5  0  0 ...  0
```
ξ=1e-8:
```
svd  : 16 55 96 135 136 147 169 174 174 175 ... 175               plateau at L=10
rsvd0: 16 55 96 135 142 161 173 176 177 178 179 181 ... 187       plateau at L=17
Δ    :  0  0  0   0   6  14   4   2   3   3   4   6  8 10 11 11 12 ... 12
```

Three conclusions:
1. **Growth still plateaus**, but the plateau location is cutoff-dependent, not a
   fixed `L≈15`. At ξ=1e-6 it saturates at L=9 (one fold *earlier* than full SVD)
   at the same ceiling (95). At ξ=1e-8 the plateau is *pushed out* to L=17 and sits
   *higher* (187 vs 175). (The earlier `L*≈15` from §4 is the stricter per-bond
   `dD=0 / n_new=0` criterion, not the whole-chain `Dmax` saturation point.)
2. **Single-pass is never *better* (smaller) than full SVD — only equal or larger.**
   Structurally: the resolution guard forbids dropping a true direction (lower
   bound = exact rank), while imperfect near-threshold resolution can leave a few
   extra above the cutoff (upper bound > exact rank). So its rank lives in
   `[exact, exact+ε]`.
3. **At tight ξ the over-retention *compounds* (it is not just transient).** The
   ξ=1e-8 `Δ` climbs monotonically from L≈8 and locks at +12 (+7%): a ratchet —
   extra directions kept this fold enlarge the next fold's uncompressed MPS, giving
   more room to over-retain, until it self-limits. At ξ=1e-6 the spectral gaps are
   wide enough that over-retention is purely transient in the growth window (Δ≤6)
   and vanishes at the plateau (95=95). **`rsvd2` (2 power iterations) removes the
   ratchet entirely** — its trajectory is byte-identical to full SVD (Δ≡0) at both
   cutoffs, while still beating the baseline wall-clock (1.23×). So there is a clean
   knob: `rsvd0` = fastest, slightly looser bonds at tight ξ; `rsvd2` =
   full-SVD-tight bonds, still faster than the pipeline. Either way accuracy stays
   far below the cutoff.

**Consequences.**
- The 3-tier machinery (oracle, probe, `η`/`n_new`/`dD`, projection, subspace
  transport) is **unnecessary** for this problem. A single uniform routine —
  single-pass rSVD + spectral resolution guard — matches the pipeline at matched
  accuracy and runs faster, with nothing to tune and no reference run.
- The **scaling law η ≈ 0.16·x^0.85 (§5) is demoted** from a routing predictor to
  an information-theoretic characterisation of the fold; it no longer gates any
  decision.
- With compression solved this cheaply, the wall-clock bottleneck has **fully
  moved to (re-)canonicalisation** (~66% of the fold, §12b). That is the next
  target: carrying / streaming the canonical gauge across folds rather than
  recomputing a left-QR sweep each time. (Would touch `src/`; not started.)

Reproduce:

```
PYTHONPATH=src python examples/uniform_rsvd_e2e.py --K 24                         # nominal ξ=1e-6
PYTHONPATH=src python examples/uniform_rsvd_e2e.py --K 24 --modes rsvd0 --seeds 0,1,2,3,4
PYTHONPATH=src python examples/uniform_rsvd_e2e.py --K 24 --cutoff 1e-8           # tight-ξ stress
```

Examples-only; pipeline `src/` unchanged.
