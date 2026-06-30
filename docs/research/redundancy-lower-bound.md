# EDM bond dimension vs the Theorem-1/2 bounds — a careful (self-correcting) check

**Status:** exploration-phase research record (2026-06-30), revised after reading the paper
(arXiv:2509.00424) and an ε-convergence scan. This note **corrects two errors in my first
draft** and reaches a different conclusion than the draft did. Numbers from Track-1
(compressed quimb fold) runs; see *Reproduction*. Honest-uncertainty throughout.

> **TL;DR.** The EDM temporal bond is a **physical, ε-convergent** quantity. At physically
> resolved time step ε it **respects Theorem 2's linear bound D_τ ≤ d²·τ in the bulk**, with
> the gap to the bound *widening* as ε→0 (because the bound is d²·t/ε ∝ 1/ε while the bond
> converges). An apparent "2–3× violation" at coarse ε=0.1 was a **stepping artifact**. The
> only genuine excess is at the first 1–2 time steps, where the open-arm dimension
> d_phys = 2(d²−1)+1 = 7 exceeds d²=4, so the literal per-bond constant d²·τ is locally too
> small. **This numerical test does not substantiate a flaw in the polynomial-complexity
> result** — though it also cannot adjudicate the rigor of the proof (a reviewer's concern).

---

## 1. What the paper actually claims (for reference)
- **Theorem 1 (Eq. 10):** the number of linearly independent EDM vectors over [0,T] is at most
  **N_T ~ d²·T/ε = d²·N**, where N = T/ε is the number of physical time steps and ε is the
  *physical* resolution (the minimal time in which the bath induces a discernible change).
- **Theorem 2 (Eq. 14):** the MPS bond at time-bond τ obeys **D_τ ≤ N_τ ~ d²·τ** (linear in
  time → polynomial complexity). Truncation precision ξ; the Gaudin demo uses **ξ=10⁻⁶,
  ε=0.03 g⁻¹, T=15 g⁻¹ → N=500**, with a hard cap **D_c=400** for cost (Appendix F3: ~10⁻³
  error at L=49). Truncation rule: drop λα with **λα/λ_{d²+1} ≤ ξ**.

## 2. Method
Build the separable Gaudin fold with a tiny cutoff (rel 1e-10) so the **full Schmidt
spectrum** of each bond is exposed (Schmidt σ² = eigenvalues of the right-environment density
matrix of a left-canonical copy). Then per bond τ measure the bond under the **paper's own
criterion** (count λα with λα/λ_{d²+1} > ξ) and, for cross-check, a discarded-weight metric
D_eff (smallest D with √(Σ_{i>D}σ_i²/Σσ_i²) < ξ). Compare against d²·τ and the both-sides
Schmidt bound d²·min(τ, N−τ). Use **order 1** (1 MPS site = 1 physical step) so every bond is
a physical-step bond directly comparable to the theorem (order 2 puts 2 sites/step and muddies
the comparison).

## 3. The key result: ε-convergence (the decisive test)
Fixed physical T=1.5 g⁻¹, K=20, order 1, ξ=10⁻⁶, halving ε:

| ε | N steps | peak bond D (physical) | d²·(N/2) (bound at peak) | peak ratio |
|---|---|---|---|---|
| 0.15 | 10 | 36 | 20 | 1.8 (exceeds) |
| 0.075 | 20 | 39 | 40 | **0.98** |
| 0.0375 | 40 | 33 | 80 | **0.41** |

The **peak (maximum-entanglement) bond converges** to a physical value (~33–39) as ε→0, while
the bound d²·(t/ε) grows ∝1/ε. So the peak-bond ratio falls **1.8 → 0.98 → 0.41**: Theorem 2's
bound holds with widening room to spare once ε is physically fine. The paper's ε=0.03 sits
well inside this safe regime (consistent with its Fig. 6b, where the bond ≈ the bound ≈ the
D_c=400 cap only at the largest times of an N=500 run).

**Correction to my first draft.** An order-1 run at coarse **ε=0.1, T=3, K=49** showed the bond
(paper's criterion) reaching ~117–120 vs d²·min ~ 44–60 — an apparent 2–3× violation. That was
a **coarse-stepping artifact**: at fixed physical time the bond is fixed but d²·t/ε is
artificially small at large ε. With resolved ε the bound holds.

## 4. The one genuine excess: the boundary (small τ) — and a candidate proof-gap
Even at fine ε, the *literal* per-bond claim D_τ ≤ d²·τ fails at τ=1: **D₁ = 7 = 2d²−1 = d_phys
> d² = 4 = N₁** (ε-robust — it's the leg, not a stepping artifact). The EDM's open-arm leg has
dimension **d_phys = 2d²−1 = 7**, so the boundary bond is set by the *arm* (2d²−1), not the
*operator freedom* (d²).

**Candidate locus of the disputed gap** (hypothesis, not certified). The Theorem-2 proof
compresses the history index Φ(τ−1) to N_{τ−1} = d²(τ−1) independent EDM vectors (the d²-per-
slice *operator* freedom, Theorem 1, which is correct — ρ^Φ are Hermitian, verified §5), then
bounds each SVD bond by that column dimension. But the **bond also carries the open-arm
φ-channel (2d²−1 per slice)**; the reduction to the d²-per-slice operator space implicitly
assumes those arms collapse onto the operator freedom. At the boundary they do not — future
correlations resolve the arms, so D₁ keeps all 2d²−1 channels. So the proof appears to carry
Theorem 1's d² constant into the *bond* bound, where the arm factor (2d²−1) belongs. This
changes the **prefactor, not the exponent** — the linear scaling (polynomial complexity) is
untouched; the constant in N_τ ~ d²·τ is the suspect. (Whether this is exactly the reviewer's
objection is a question for the proof, not simulation.)

## 5. A hypothesis I tested and rejected
I initially guessed the bond might track 2d²·τ because the EDM elements ρ^Φ (Φ≠0) are
non-Hermitian (2d² real params vs d²). **Direct test: ρ^Φ are all Hermitian to ~10⁻¹⁷** (the
complex correlation coefficients combine the superoperator action so that each EDM element
stays Hermitian). So the d² per-element count is correct; the non-Hermitian story is wrong and
is discarded. [[preserve-uncertainty]]

## 6. Conclusions
1. **The EDM temporal bond is physical and ε-convergent** — a well-defined entanglement, not a
   discretisation artefact.
2. **Theorem 2's linear bound D_τ ≤ d²·τ holds in the bulk at physically-resolved ε**, with the
   margin growing as ε→0. The polynomial-complexity result is corroborated by this test.
3. **Genuine local excess only at the first 1–2 steps**, from d_phys=7 > d²=4 — a benign
   boundary effect on the constant, not the scaling.
4. **What this does NOT show:** it does not validate the *proof's rigor*. A reviewer's gap may
   be a rigor issue (e.g. the step from "N_τ independent vectors" to "bond ≤ N_τ", or implicit
   assumptions) that need not surface as a bulk numerical violation in one model. Simulation
   can corroborate the bound's truth here; it cannot certify the proof.

## 7. Corrections to my earlier redundancy framing (now retracted)
- **"D_c=400 over-provisions by ~3.4×."** Wrong — D_c=400 is sized for the paper's N=500
  regime where the bond genuinely approaches ~400; I had compared it against short-T runs.
- **"36% carried-bond redundancy."** Overstated — it compared a discarded-weight metric against
  a 1e-10 build; under the paper's own λα/λ_{d²+1} > ξ criterion the bond is larger (closer to
  the carried value), so far less is "recoverable" than I claimed.
- **"Bond saturates the bound / N_actual > N_T."** Artifact of coarse ε (and order-2
  sub-steps); at resolved ε the bond is comfortably under N_T = d²·N.

## 8. Caveats / limitations
- One model (Gaudin, linear couplings), one observable regime, finite ε-range (down to 0.0375).
- D under the build is mildly build-cutoff dependent; the ε-convergence and bound comparisons
  use the paper's ξ criterion, which is the right anchor.
- The bond is measured post-hoc from the exposed spectrum; this matches the paper's
  self-adaptive truncation in spirit (same ξ rule) but is not bit-identical to their pipeline.

## 9. Reproduction
```
# decisive eps-scan (bound convergence)
for e in 0.15 0.075 0.0375; do
  PYTHONPATH=src python examples/studies/redundancy_bound.py \
    --K 20 --T 1.5 --eps $e --order 1 --xi 1e-6 --build-cutoff 1e-10 --build-max-bond 600 --name epsscan_$e
done
# order-1 profile at a single eps
PYTHONPATH=src python examples/studies/redundancy_bound.py --K 49 --T 3 --eps 0.1 --order 1 \
    --xi 1e-6 --build-cutoff 1e-10 --build-max-bond 512 --name redundancy_K49_o1
```
Outputs (gitignored) in `examples/studies/{data,pictures}/redundancy/`. The script prints the
per-bond bound ratios and draws the profile vs d²·τ / d²·min(τ,N−τ).

## 10. Open questions
- Push ε smaller (0.01) and larger T to map the convergence margin precisely; confirm the peak
  ratio keeps falling.
- Whether the small-τ d_phys>d² excess can be folded into a tightened constant in the bound.
- The reviewer's specific objection to the proof — a question for the proof itself, not
  reachable by this simulation.
