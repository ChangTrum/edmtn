# The EDM temporal bond versus the polynomial-complexity bounds

A numerical study of how the bond dimension of the Gaudin EDM compares to the bounds of
Theorem 1 (independent-EDM count) and Theorem 2 (linear bond growth) of arXiv:2509.00424.
Model: central spin-1/2 (d=2), linearly-decreasing couplings, infinite-temperature bath,
truncation precision ξ = 10⁻⁶ (the paper's value). Tooling: `examples/studies/redundancy_bound.py`.

## Summary

The EDM temporal bond is a physical, ε-convergent quantity, and at a physically resolved time
step it obeys the linear bound D_τ ≲ d²·τ throughout the bulk of the chain — so the
polynomial-complexity result holds. The one place the *literal* per-bond constant is wrong is
the boundary: the first bond is **D₁ = 2d²−1 = d_phys**, not d². This reflects a real
structural fact — the bond carries the open-arm index (2d²−1 values per slice), whereas the
independent-EDM count of Theorem 1 is d² (the operator freedom). The bound's exponent (linear)
is correct; its constant near the boundary is governed by 2d²−1, not d².

## 1. The bounds being tested

- **Theorem 1.** Each EDM slice ρ^Φ is a d×d Hermitian operator (d² real parameters), and the
  trajectory has T/ε slices, so the number of independent EDM vectors is N_T = d²·T/ε = d²·N
  (N = T/ε physical steps). The Hermiticity holds because the system superoperators 𝒮 preserve
  it and the Gaudin correlation tensor ℂ is real; it is confirmed numerically (‖ρ^Φ − ρ^Φ†‖ ≈
  10⁻¹⁷ for all Φ in a small exact EDM).
- **Theorem 2.** The MPS bond at time-bond τ satisfies D_τ ≤ N_τ = d²·τ (linear in time).

A useful distinction: per slice there are **2d²−1 = d_phys = 7** open-arm components but only
**d² = 4** independent directions. The component count over a trajectory, (2d²−1)^τ, explodes
exponentially; Theorem 1 states that only ~d²·τ of these are linearly independent. The bond
dimension is the compressed quantity that sits between the two.

## 2. Method

The separable fold is built with a tiny cutoff (relative 10⁻¹⁰) so the full Schmidt spectrum
of each bond is exposed (the squared Schmidt values are the eigenvalues of the right-environment
density matrix of a left-canonical copy). Each bond is then measured under the paper's own
criterion — count the singular values with λα/λ_{d²+1} > ξ — and compared to d²·τ and to the
two-sided form d²·min(τ, N−τ). Order-1 expansion is used so that one MPS site equals one
physical step and every bond is directly comparable to the theorem.

## 3. The bond is physical and ε-convergent

At fixed physical time (T = 1.5, K = 20) the peak bond converges as the step ε is refined,
while the bound d²·(N/2) = d²·T/(2ε) grows as 1/ε:

| ε | N steps | peak bond | d²·(N/2) bound | ratio |
|---|---|---|---|---|
| 0.15 | 10 | 36 | 20 | 1.80 |
| 0.075 | 20 | 39 | 40 | 0.97 |
| 0.0375 | 40 | 33 | 80 | 0.41 |

The peak bond is a fixed physical quantity (~33–39); the bound grows without limit as ε → 0.
For ε ≲ 0.075 the bound holds in the bulk, with margin that widens as the step is refined. The
paper's ε = 0.03 lies well inside this regime, consistent with its Fig. 6b. (Both quantities
are ε-dependent in opposite ways: a bound written as d²·T/ε is only meaningful at the physical
resolution ε, the minimal timescale on which the bath changes the reduced dynamics.)

## 4. The boundary: D₁ = 2d²−1, not d²

The first time-bond is D₁ = 7 = 2d²−1 = d_phys, independent of ε, exceeding N₁ = d² = 4. The
boundary bond is set by the open-arm leg dimension (2d²−1), not by the operator freedom (d²).
This is the one ε-robust place where the literal claim D_τ ≤ d²·τ does not hold.

This points to where the linear bound's constant comes from. Theorem 1 correctly counts d²
independent directions per slice (the operator freedom). The MPS bond, however, transmits the
open-arm φ-channel — 2d²−1 values per slice — and these are not reducible to the d²-dimensional
operator space when later correlations resolve the arms, as they do at the boundary. So the
constant appropriate to the *bond* is the arm dimension 2d²−1, while the d² constant belongs to
the *independent-vector* count. The exponent (linear in τ) is unaffected; only the prefactor in
N_τ ~ d²·τ is at issue. Adjudicating whether the published proof's d² → bond step is rigorous is
a question for the proof itself; the numerics only locate where the constant departs from d².

## 5. Conclusions

1. The EDM temporal bond is a well-defined physical entanglement, convergent as ε → 0.
2. Theorem 2's linear bond growth holds in the bulk at physically resolved ε; the
   polynomial-complexity result is corroborated.
3. The literal per-bond constant is d_phys = 2d²−1 at the boundary, not d². The discrepancy is
   in the constant (operator freedom d² versus open-arm dimension 2d²−1), not in the linear
   scaling.

## 6. Caveats

- One model (Gaudin, linear couplings), ξ = 10⁻⁶, ε down to 0.0375.
- The bond is read post-hoc from the exposed spectrum under the paper's λα/λ_{d²+1} > ξ rule;
  this matches the paper's self-adaptive truncation in spirit but is not bit-identical to its
  pipeline.
- Simulation can show the bound holds (or where its constant departs); it cannot certify the
  rigor of the published proof.

## 7. Reproduction

```
# eps-convergence (bound vs physical bond)
for e in 0.15 0.075 0.0375; do
  PYTHONPATH=src python examples/studies/redundancy_bound.py \
    --K 20 --T 1.5 --eps $e --order 1 --xi 1e-6 --build-cutoff 1e-10 --build-max-bond 600 --name epsscan_$e
done
# single-eps bond profile vs d^2*tau / d^2*min(tau, N-tau)
PYTHONPATH=src python examples/studies/redundancy_bound.py --K 49 --T 3 --eps 0.1 --order 1 \
    --xi 1e-6 --build-cutoff 1e-10 --build-max-bond 512 --name redundancy_K49_o1
```
Outputs (gitignored) in `examples/studies/{data,pictures}/redundancy/`.

## 8. Open directions

- Smaller ε and larger T to map the convergence margin precisely.
- A model with d > 2 (so d_phys/d² differs) to confirm the boundary constant is 2d²−1.
- Re-deriving the bond bound with the (2d²−1) arm factor carried explicitly.
