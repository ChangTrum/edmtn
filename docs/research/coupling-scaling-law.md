# Coupling-distribution dependence of the Gaudin EDM scaling law

A study of how the EDM's per-fold subspace increment scales with the coupling distribution.
Model: Gaudin central spin-1/2, several coupling profiles normalised to Σ_k g²_k = g². Tooling:
`examples/studies/coupling_distributions.py`.

## Summary

The Gaudin EDM obeys a scaling law η ~ c·xᵅ for the increment of folding in sub-bath L+1, with
x = g²_{L+1}/ḡ²_L. First-order perturbation theory gives the **universal asymptotic exponent
α = 1** for every coupling profile; the measured finite-window exponent is < 1 (≈ 0.84 for the
paper's linear profile, reproducing the known value), and the **small-x tail slope → 1** wherever
it is numerically resolvable. What depends on the coupling *mode* is the observable finite-window
exponent and, sharply, **how quickly later sub-baths stop contributing** (the critical layer L\*)
— both set by the **decay rate** of the coupling envelope, and robust to disorder correlations.

## 1. Question

The paper uses a linearly-decreasing coupling profile and observes that the per-fold subspace
increment follows a power law in x = g²_{L+1}/ḡ²_L (exponent ≈ 0.85), with sufficiently late
sub-baths adding no genuinely new directions (pure projection becomes exact). Is the scaling
intrinsic to the Gaudin model or an artefact of the linear profile, and does the
"later-sub-baths-matter-less" phenomenon depend on the profile? The diagnostic is run across
several distributions and compared to the perturbative prediction.

## 2. Method

### 2.1 Fold + subspace-increment diagnostic
The separable-bath EDM is built by folding sub-baths one at a time (combined-kernel MPO × MPS
contraction along the time axis, then a quimb 1D compression). The compressed EDM-MPS is
snapshotted after each sub-bath L = 1..K; for each transition L → L+1 the left singular subspaces
are compared bond by bond (left-canonicalise both, form the cross-overlap transfer matrices
Eτ = Q_A(τ)†Q_B(τ), whose singular values are the cosines of the principal angles). Per transition,
aggregated over bonds:

| symbol | meaning |
|---|---|
| **η** (`resid_ratio`) | residual weight of the new (L+1) subspace not captured by the old (L) subspace — the strength of genuinely new directions (η_max / η_rms over bonds). |
| **d_chord/√D** (`chordal_norm`) | normalised chordal distance √Σ sin²θ / √D. |
| **n_new(ξ), n_new(√ξ)** | new directions with overlap cosine below 1−ξ (resp. 1−√ξ). |
| **dD** | bond-dimension growth D_{L+1} − D_L. |

Independent variable **x = g²_{L+1} / ḡ²_L**, with ḡ²_L = Σ_{k≤L} g²_k.

### 2.2 Coupling profiles (all Σ_k g²_k = g²)
Selectable via `GaudinModel(coupling=..., coupling_params=...)`:

| profile | g_k | param | sorted? |
|---|---|---|---|
| `linear` (paper) | ∝ (K+1−k) | — | descending |
| `uniform` | g/√K (flat) | — | trivially |
| `exp` | ∝ e^(−βk) | β | descending |
| `random` | ∼ Uniform(0,1) | seed | sorted descending |
| `ou` | \|AR(1)/OU sequence\|, c_k = ρ c_{k−1} + √(1−ρ²) z_k | ρ, seed | **not sorted** |

All but `ou` are folded strongest-first (the paper's convention, under which x decreases
monotonically with L and L\* is well-defined). `ou` is left in generation order so the
nearest-neighbour correlation survives (sorting would collapse every ρ to the same half-normal
spectrum); for `ou`, x is non-monotonic and only the scaling-law fit is meaningful, not L\*.

### 2.3 Numerics
Gaudin g=1, K=49, T=3 g⁻¹, eps=0.2 g⁻¹ (order-2 grid), `compress_method='direct'`,
`compress_decomp='exact'`, `cutoff_mode='rel'`, `max_bond=1024`. Two cutoffs ξ = 10⁻⁶ and 10⁻⁹.
The fold runs on one A800 (CuPy); the per-bond QR/SVD diagnostics on host CPU. Power-law fits are
least-squares on log–log; the tail fit uses the smallest-x 40 % of points.

## 3. Theory

Folding sub-bath L+1 perturbs the EDM through its correlation tensor, built from bath
superoperators B_a = g_{L+1}·J_a. At infinite temperature the first moment vanishes, so the
leading contribution is the second moment ∝ g²_{L+1}: the new sub-bath couples with relative
strength x = g²_{L+1}/ḡ²_L to the accumulated state. First-order perturbation theory gives the
new-direction weight

  **η ≈ c·x,  i.e. α = 1**,

and d_chord ∝ x. This is the paper's Sec. 3.3 estimate Δg/g_L ~ x/2, and is independent of the
profile shape — it concerns adding a single weak bath. A measured α < 1 is therefore a
finite-x-window effect: α = 1 is the x→0 asymptotic slope, while over a finite window the log–log
slope is reduced by saturation at large x and by the numerical floor at small x. Two predictions
follow: (A) the small-x tail slope → 1 for every non-degenerate mode; (B) modes reaching smaller x
(faster decay) measure a finite-window α closer to 1, until the decay is so fast that x and η fall
below the diagnostic's numerical floor.

## 4. Results

### 4.1 Scaling law, ξ = 10⁻⁶

| group | α (full window) | α_tail (x→0) | R²_tail |
|---|---|---|---|
| linear | 0.837 | **0.992** | 1.00 |
| exp β=0.05 | 0.830 | **0.984** | 1.00 |
| exp β=0.1 | 0.955 | **0.998** | 1.00 |
| exp β=0.2 | 0.797 | 0.291 | 0.46 (floored) |
| exp β=0.4 | 0.311 | −0.003 | 0.00 (floored) |
| random (iid) | 0.838 | **0.985** | 1.00 |
| ou ρ=0.5 | 0.754 | 0.910 | 0.90 |
| ou ρ=0.9 | 0.795 | 0.948 | 0.89 |
| uniform | 0.043 | 0.270 | degenerate |

The tail slope → 1 wherever η stays above the floor (linear, exp β≤0.1, random, ou), confirming
prediction A; the full-window α (0.83–0.96) is the finite-window curvature. The paper's linear
≈0.85 is reproduced (0.84), and its tail (0.99) shows the underlying law is α = 1. The exp
β-sweep is **non-monotonic** — for β ≳ 0.2 the couplings decay so fast that g_{L+1} → 0 within a
few folds, x → tiny, and η flattens onto a plateau, collapsing the fitted slope; β=0.1 is the
sweet spot. `uniform` is degenerate (flat couplings give x = 1/L, never small, η ≈ const).

### 4.2 Scaling law, ξ = 10⁻⁹ — the fast-decay floor is roundoff, not the cutoff

| group | α (full window) | α_tail (x→0) | R²_tail |
|---|---|---|---|
| linear | 0.838 | **0.993** | 1.00 |
| exp β=0.05 | 0.832 | **0.986** | 1.00 |
| exp β=0.1 | 0.956 | **0.999** | 1.00 |
| exp β=0.2 | 0.794 | 0.291 | 0.43 (floored) |
| exp β=0.4 | 0.284 | −0.018 | 0.02 (floored) |
| random | 0.835 | **0.937** | 0.99 |
| ou ρ=0.5 | 0.745 | 0.945 | 0.89 |
| ou ρ=0.9 | 0.777 | 0.943 | 0.87 |
| uniform | −0.022 | 0.275 | degenerate |

Tightening ξ by 1000× leaves every exponent essentially unchanged and does not lift the exp
β=0.2/0.4 plateau (the fast-decay clouds are identical for the two cutoffs). The floor that breaks
the fast-decay tails is therefore the **float64 roundoff (~10⁻⁷) of the subspace-overlap
diagnostic, not the SVD cutoff**. Physically, for fast decay the late couplings are so small
(exp β=0.4: g_k ~ 10⁻⁹ by k ≈ 30) that the perturbation is below numerical resolution — those
sub-baths are negligible. The well-resolved modes confirm **α_tail → 1, cutoff-independently**.

### 4.3 Critical L\* — "later sub-baths matter less" is set by the decay rate

Smallest L at which folding L→L+1 stops adding genuinely new directions (ξ = 10⁻⁶):

| group | bond saturates (dD=0) | n_new(√ξ)=0 | η_max < 10⁻³ |
|---|---|---|---|
| exp β=0.4 | 3 | 4 | 7 |
| exp β=0.2 | 4 | 6 | 13 |
| exp β=0.1 | 6 | 8 | 22 |
| exp β=0.05 | 10 | 10 | 36 |
| linear | 16 | 17 | 39 |
| random | 11–18 | 11–19 | 40 |
| uniform | 28 | 41 | never |

Faster decay ⟹ later sub-baths become irrelevant far sooner (exp β=0.4 saturates at L≈3 vs
linear at L≈16 vs uniform essentially never). The diminishing influence of late sub-baths is
governed by the coupling decay rate, not the L-index. The absolute L\* is cutoff-dependent (at
ξ=10⁻⁹, dD=0 at exp 5/7/11/19 for β=.4/.2/.1/.05, linear 26, uniform 38), but the ordering by
decay rate is unchanged.

### 4.4 Correlated disorder (OU)

Correlated AR(1) couplings still obey the law (tail α 0.91–0.95) with more scatter (full-window
R² ≈ 0.78), and the correlation length barely matters (ρ=0.5 vs 0.9 → α 0.75 vs 0.80). The
observable exponent is set by the coupling envelope (decay), robust to the disorder's correlation
structure.

## 5. Conclusions

1. The scaling law η ~ c·xᵅ is intrinsic to the Gaudin EDM (R² ≥ 0.96 for linear, exp, random,
   and more loosely OU), not an artefact of the linear profile.
2. The asymptotic exponent is universal, α = 1 (first-order perturbation theory), confirmed by
   the cutoff-independent small-x tail slope. The paper's measured ≈0.85 is the finite-x-window
   slope of the same law.
3. The observable finite-window exponent and the critical layer L\* are set by the **decay rate**
   of the coupling envelope: faster decay → smaller reachable x → finite-window α closer to 1 and
   earlier pure-projection feasibility.
4. The scaling is robust to disorder correlation (OU): only the envelope, not the correlation
   structure, sets the exponent.
5. Fast-decay profiles (exp β ≳ 0.2) cannot be fit because their late couplings fall below the
   diagnostic's float64 roundoff floor — those sub-baths are physically negligible, consistent
   with point 3.

## 6. Caveats

- The fast-decay tail is limited by the ~10⁻⁷ float64 roundoff of the subspace-overlap
  diagnostic, not the SVD cutoff (§4.2); resolving it would need higher-than-double precision.
  With `max_bond=1024` capped, very tight cutoffs can also become bond-limited for slowly-decaying
  profiles.
- α_full is a window-averaged slope; the comparison to theory is the tail slope, which needs
  enough resolvable small-x decades.
- OU is folded unsorted, so its x is non-monotonic and its critical-L is not physically meaningful.
- Single (eps, T), single K=49; the fold-snapshot diagnostic is Track-1-specific.

## 7. Reproduction
```
PYTHONPATH=src python examples/studies/coupling_distributions.py \
    --K 49 --T 3 --eps 0.2 --seeds 4 --cutoff 1e-9 --device gpu --name coupling_dist_K49_tight
# cluster: sbatch --parsable cluster/coupling_dist.sbatch   (c1, 1×A800)
PYTHONPATH=src python examples/studies/coupling_distributions.py --replot <results>.json
```
Model option: `GaudinModel(g, K, coupling='linear'|'uniform'|'exp'|'random'|'ou'|<array>,
coupling_params={'beta':..,'rho':..,'seed':..})`. Outputs (gitignored) in
`examples/studies/{data,pictures}/coupling_dist/`.

## 8. Open directions
- exp large-β tail under higher-than-double precision (a numerical check; those baths are
  negligible).
- Analytic form of the finite-window α as a function of the x-range — predicting the 0.84 of the
  linear profile from the envelope.
- Dependence on T / eps (interaction of temporal-bond growth with the per-fold increment).
