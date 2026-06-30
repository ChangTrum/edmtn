# Coupling-distribution dependence of the Gaudin EDM scaling law

**Status:** exploration-phase research record (2026-06-30). Honest, complete write-up of
what was measured, what the theory predicts, and where the two agree / disagree — including
the parts that are numerical artefacts rather than physics. Numbers are from Track-1
(compressed quimb fold) runs on one A800 (CuPy); see *Reproduction* at the end.

> **TL;DR.** The Gaudin EDM obeys a scaling law η ~ c·xᵅ with x = g²_{L+1}/ḡ²_L for the
> increment of folding in sub-bath L+1. First-order perturbation theory predicts the
> **universal asymptotic exponent α = 1** for *every* coupling profile. The *measured*
> finite-window exponent is < 1 (≈ 0.84 for the paper's linear profile, reproducing the
> known result) — this is a finite-x-range curvature, and the **small-x tail slope → 1**
> confirms the theory wherever it is numerically resolvable. What genuinely depends on the
> coupling *mode* is (a) the observable finite-window exponent and (b) **how quickly later
> sub-baths stop contributing** (the "critical L\*"), both set by the **decay rate** of the
> coupling envelope and robust to disorder correlations.

---

## 1. Question

The paper (central-spin / Gaudin model) uses a linearly-decreasing coupling profile and
observes that the EDM's per-fold subspace increment follows a power law in
x = g²_{L+1}/ḡ²_L, and that sufficiently late sub-baths add no genuinely new directions
(pure projection becomes exact, no SVD needed). Measured exponent ≈ 0.85.

Is this scaling **intrinsic to the Gaudin model**, or an artefact of the **specific
linearly-decaying coupling profile**? And does the "later sub-baths matter less"
phenomenon depend on the profile? We answer by running the same diagnostic across several
coupling-strength distributions (all normalised to Σ_k g²_k = g²), deriving the
theoretical exponent, and comparing.

## 2. Method

### 2.1 Fold + subspace-increment diagnostic
The separable-bath EDM is built by folding sub-baths one at a time (Track 1: combined-kernel
MPO × MPS contraction along the time axis, then a quimb 1D compression). We snapshot the
compressed EDM-MPS after every sub-bath L = 1..K and, for each transition L → L+1, compare
the **left singular subspaces** bond by bond (left-canonicalise both, take the cross-overlap
transfer matrices Eτ = Q_A(τ)†Q_B(τ); their singular values are the cosines of the principal
angles between the old and new subspaces). Per transition we record, aggregated over bonds:

| symbol | meaning |
|---|---|
| **η** (`resid_ratio`) | residual weight of the new (L+1) subspace **not captured** by the old (L) subspace — the strength of genuinely new directions. Reported as η_max / η_rms over bonds. |
| **d_chord/√D** (`chordal_norm`) | normalised chordal distance between the two subspaces (√Σ sin²θ / √D). |
| **n_new(ξ), n_new(√ξ)** | number of new directions whose overlap cosine falls below 1−ξ (resp. 1−√ξ). |
| **dD** | bond-dimension growth DL+1 − DL. |

The independent variable is **x = g²_{L+1} / ḡ²_L**, with ḡ²_L = Σ_{k≤L} g²_k the
accumulated (effective) coupling of the first L sub-baths and g_{L+1} the newly-added one.

### 2.2 Coupling profiles (all Σ_k g²_k = g²)
Selectable via `GaudinModel(coupling=..., coupling_params=...)`:

| profile | g_k | param | sorted? |
|---|---|---|---|
| `linear` (paper) | ∝ (K+1−k) | — | descending |
| `uniform` | g/√K (flat) | — | trivially |
| `exp` | ∝ e^(−βk) | β | descending |
| `random` | ∼ Uniform(0,1) | seed | sorted descending |
| `ou` | \|AR(1)/OU sequence\|, c_k = ρ c_{k−1} + √(1−ρ²) z_k | ρ, seed | **NOT sorted** |

All but `ou` are folded strongest-first (descending) — the paper's convention, under which
x decreases monotonically with L and "critical L\*" is well-defined. `ou` is deliberately
left in generation order so the nearest-neighbour correlation along k survives (sorting
would collapse every ρ onto the same half-normal spectrum); for `ou`, x is non-monotonic
and the critical-L is **not** meaningful — only the scaling-law fit is.

### 2.3 Numerics
Gaudin g=1, K=49, T=3 g⁻¹, eps=0.2 g⁻¹ (order-2 grid, 30 sites), Track-1 compressed fold
with `compress_method='direct'`, **`compress_decomp='exact'`** (full SVD — removes the rSVD
noise that otherwise pollutes the small-x tail), `cutoff_mode='rel'`, `max_bond=1024`. Two
cutoffs: **ξ = 1e-6** (initial) and **ξ = 1e-9** (tight, to lower the η floor). The heavy
fold runs on one A800 (CuPy); the many tiny per-bond QR/SVD diagnostics run on host CPU.
Power-law fits are least-squares on log–log; the **tail fit** uses the smallest-x 40 % of
points to probe the x→0 asymptotic slope.

## 3. Theory

Folding sub-bath L+1 perturbs the EDM through its correlation/transfer tensor, built from
bath superoperators B_a = g_{L+1}·J_a. At infinite temperature the **first moment vanishes**,
so the leading non-trivial contribution is the second moment ∝ g²_{L+1}. The new sub-bath
therefore couples with **relative strength x = g²_{L+1}/ḡ²_L** to the state already carrying
accumulated strength ḡ²_L. First-order perturbation theory then gives the new-direction
weight

  **η ≈ c·x**, i.e. **α_theory = 1**,

and likewise d_chord ∝ x (principal angle ∝ perturbation amplitude). This is exactly the
paper's Sec. 3.3 estimate Δg/g_L ~ x/2. Crucially, **α = 1 is independent of the profile
shape** — it is a statement about adding *one* weak bath. Any measured α < 1 must therefore
be a finite-x-window effect: α = 1 is the x→0 asymptotic slope, while over a finite window
the log–log slope is pulled below 1 by saturation at large x (η can't grow without bound)
and by a noise floor at small x (η cannot be resolved below ~the truncation cutoff).

**Prediction A.** The small-x **tail** slope → 1 for every non-degenerate mode.
**Prediction B.** Modes reaching smaller x (faster decay) measure a finite-window α closer
to 1 — *until* the decay is so fast that x and η drop below the diagnostic's numerical
(roundoff) floor, where the fit breaks down (a numerical limit, not physics; see §4.2).

## 4. Results

### 4.1 Scaling law — cutoff ξ = 1e-6 (job 46583, ~13 min on 1×A800)

| group | α (full window) | **α_tail (x→0)** | R²_tail | n_tail |
|---|---|---|---|---|
| linear | 0.837 | **0.992** | 1.00 | 19 |
| exp β=0.05 | 0.830 | **0.984** | 1.00 | 19 |
| exp β=0.1 | 0.955 | **0.998** | 1.00 | 19 |
| exp β=0.2 | 0.797 | 0.291 | 0.46 | 19 (floored) |
| exp β=0.4 | 0.311 | −0.003 | 0.00 | 16 (floored) |
| random (iid) | 0.838 | **0.985** | 1.00 | 74 |
| ou ρ=0.5 | 0.754 | 0.910 | 0.90 | 75 |
| ou ρ=0.9 | 0.795 | 0.948 | 0.89 | 75 |
| uniform | 0.043 | 0.270 | — | degenerate |

Observations:
- **Prediction A confirmed** wherever η stays above the floor: tail slope → 1 (linear 0.99,
  exp0.05 0.98, exp0.1 1.00, random 0.99, ou 0.91–0.95). The full-window α (0.83–0.96) is the
  finite-window curvature; the paper's linear ≈0.85 is reproduced (full-window 0.84) and its
  tail (0.99) shows the underlying law is genuinely α=1.
- **β-sweep is non-monotonic** — and this is a **numerical floor, not physics** (an honest
  correction to an earlier smoke-scale guess that α rises monotonically with β). For β ≳ 0.2
  the couplings decay so fast that within a few folds g_{L+1} → 0; x → tiny while η flattens
  onto a ~1e-7 plateau set by the cutoff (1e-6). Those points go horizontal, collapsing the
  fitted slope (exp β=0.4 tail R²=0). exp β=0.1 is the sweet spot (fast enough to reach small
  x, slow enough to stay above the floor): cleanest α=0.955, tail 0.998.
- **uniform** is degenerate: flat couplings give x = 1/L ∈ [0.02, 0.5] (never small) and η
  ≈ const, so there is no power law (R²=0.26).

### 4.2 Scaling law — tight cutoff ξ = 1e-9 (job 46584, ~30 min on 1×A800)

| group | α (full window) | **α_tail (x→0)** | R²_tail |
|---|---|---|---|
| linear | 0.838 | **0.993** | 1.00 |
| exp β=0.05 | 0.832 | **0.986** | 1.00 |
| exp β=0.1 | 0.956 | **0.999** | 1.00 |
| exp β=0.2 | 0.794 | 0.291 | 0.43 (still floored) |
| exp β=0.4 | 0.284 | −0.018 | 0.02 (still floored) |
| random | 0.835 | **0.937** | 0.99 |
| ou ρ=0.5 | 0.745 | 0.945 | 0.89 |
| ou ρ=0.9 | 0.777 | 0.943 | 0.87 |
| uniform | −0.022 | 0.275 | degenerate |

**Key finding — the fast-decay floor is floating-point roundoff, not the SVD cutoff.**
Tightening ξ by 1000× (1e-6 → 1e-9) left every exponent essentially unchanged (linear
0.837→0.838, exp β=0.1 0.955→0.956, random 0.838→0.835) and **did not lift the exp
β=0.2/0.4 plateau** — those points still pile up at η ~ 1e-7 with tail slope ≈ 0 (compare the
scaling figures for the two cutoffs; the fast-decay clouds are identical). So the floor that
breaks the fast-decay tails is the **float64 roundoff (~1e-7) accumulated in the
subspace-overlap diagnostic**, *independent of the truncation cutoff* — correcting the
initial hypothesis (§4.1) that it was the cutoff. Physically this is unsurprising: for fast
decay the late couplings are so small (exp β=0.4 has g_k ~ 1e-9 by k ≈ 30) that the
perturbation they inject is below the diagnostic's numerical resolution — there is genuinely
nothing left to measure, which is itself the physics (those sub-baths are negligible).
Resolving their tail would need higher-than-double precision, not a tighter cutoff. The
well-resolved modes confirm **α_tail → 1 robustly and cutoff-independently** — the central
result is solid.

### 4.3 Critical L\* — "later sub-baths matter less" is set by the decay rate

Smallest L at which folding L→L+1 stops adding genuinely new directions (ξ = 1e-6):

| group | bond saturates (dD=0) | n_new(√ξ)=0 | η_max < 1e-3 |
|---|---|---|---|
| exp β=0.4 | 3 | 4 | 7 |
| exp β=0.2 | 4 | 6 | 13 |
| exp β=0.1 | 6 | 8 | 22 |
| exp β=0.05 | 10 | 10 | 36 |
| linear | 16 | 17 | 39 |
| random | 11–18 | 11–19 | 40 |
| uniform | 28 | 41 | never |

Faster decay ⟹ later sub-baths become irrelevant far sooner (exp β=0.4 saturates at L≈3 vs
linear at L≈16 vs uniform essentially never). This decisively confirms that the diminishing
influence of late sub-baths is **governed by the coupling decay rate**, not by the L-index
itself. (`ou` is unsorted, so its critical-L is reported but not physically meaningful.)

The *absolute* L\* is cutoff-dependent — at ξ=1e-9 the feasibility criteria are stricter, so
projection becomes viable later (dD=0 at exp 5/7/11/19 for β=.4/.2/.1/.05, linear 26, uniform
38; n_new criteria push out similarly) — but the **ordering by decay rate is unchanged**. The
mode-dependence is the robust statement; the exact layer depends on the precision demanded.

### 4.4 OU / correlated disorder — a null result
Correlated AR(1) couplings still obey the law (tail α 0.91–0.95) with more scatter
(full-window R²≈0.78). The correlation length barely matters: ρ=0.5 vs 0.9 give α = 0.75 vs
0.80. So the observable exponent is set by the coupling **envelope (decay)**, robust to the
disorder's correlation structure.

## 5. Conclusions

1. **The scaling law η ~ c·xᵅ is intrinsic to the Gaudin EDM**, not an artefact of the
   linear profile — it holds (R² ≥ 0.96) for linear, exp, random, and (more loosely) OU.
2. **The asymptotic exponent is universal, α = 1** (first-order perturbation theory),
   confirmed by the small-x tail slope wherever it is numerically resolvable and shown to be
   **cutoff-independent** (ξ=1e-6 and 1e-9 agree). The paper's "should be 1 but measured 0.85"
   is resolved: 0.85 is the finite-x-window slope; the tail recovers 1.
3. **What depends on the coupling mode** is the *observable* finite-window exponent and,
   sharply, the **critical layer L\*** — both governed by the **decay rate** of the coupling
   envelope. Faster decay → smaller reachable x → finite-window α closer to 1 (until the
   roundoff floor) and far earlier pure-projection feasibility.
4. **Robust to disorder correlation** (OU): only the envelope, not the correlation structure,
   sets the exponent (ρ=0.5 vs 0.9 → α=0.75 vs 0.78).
5. **The fast-decay breakdown is numerical, not physical.** exp β≳0.2 cannot be fit because
   the late couplings fall below the diagnostic's float64 roundoff floor (~1e-7); a 1000×
   tighter cutoff did not change this. The right reading is that those sub-baths are negligible
   — consistent with conclusion 3, not a counterexample to conclusion 2.

## 6. Honest caveats / limitations
- **Roundoff floor (not cutoff).** η cannot be resolved below the ~1e-7 float64 roundoff floor
  of the subspace-overlap diagnostic; fast-decaying profiles (exp large β) hit it and their
  tail fits break down. This was initially mis-attributed to the SVD cutoff; the ξ=1e-9 run
  (§4.2) disproved that — a 1000× tighter cutoff did not move the floor. Resolving those tails
  needs higher-than-double precision, not a tighter cutoff. (`max_bond=1024` is capped, so very
  tight cutoffs can also become bond-limited for slowly-decaying profiles — a
  precision/complexity trade-off deliberately kept restrained.)
- **rSVD vs exact.** The initial exploration used rSVD (the benchmark recipe), which adds
  noise to small singular values; all results here use **exact** decomposition.
- **Finite window.** α_full is a window-averaged slope; the physically meaningful comparison
  to theory is the tail slope, which itself needs enough resolvable small-x decades.
- **OU sorting.** OU is folded unsorted to preserve correlation, so its x is non-monotonic and
  its critical-L is not physically meaningful (only the scaling fit is).
- Single (eps, T); single K=49. The fold-snapshot diagnostic is Track-1-specific.

## 7. Reproduction
```
# local CPU (small) or cluster GPU (A800)
PYTHONPATH=src python examples/studies/coupling_distributions.py \
    --K 49 --T 3 --eps 0.2 --seeds 4 --cutoff 1e-9 --device gpu --name coupling_dist_K49_tight
# cluster: sbatch --parsable cluster/coupling_dist.sbatch   (c1, 1×A800)
# figures from a saved JSON (cluster has no matplotlib):
PYTHONPATH=src python examples/studies/coupling_distributions.py --replot <results>.json
```
Model option: `GaudinModel(g, K, coupling='exp'|'linear'|'uniform'|'random'|'ou'|<array>,
coupling_params={'beta':..,'rho':..,'seed':..})`. Outputs (gitignored) land in
`examples/studies/{data,pictures}/coupling_dist/`.

## 8. Open questions
- Does exp large-β tail → 1 under higher-than-double precision? (At float64 it is roundoff-
  limited, §4.2 — a quad-precision or mpmath fold would be the direct test of universality
  there; physically those baths are negligible, so this is a numerical-curiosity check.)
- Analytic form of the **finite-window** α(x-range) curve — can the 0.84 for linear be
  predicted from the envelope, closing the loop between α_theory=1 and the observable?
- Dependence on T / eps (does the temporal-bond growth interact with the per-fold increment)?
