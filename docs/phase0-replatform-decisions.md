# Phase 0 — re-platform keep/replace decision ledger

The plan (`~/.claude/plans/sharded-wishing-blossom.md`) requires a *per-piece*
keep/replace decision, recorded with evidence, before any cuQuantum code is added.
Decision criterion (from the plan): **bias to the library** (quimb / cotengra /
autoray / cupy) — keep our own wheel only when *both* (a) the change point is
EDM-specific (no universality requirement) *and* (b) ours is *clearly* better, not
merely a small perf edge. All changes are additive and opt-in; the default
(`StandardSVD` + native sweep) path stays byte-for-byte and is the validation
reference. The accuracy bar is the **observable** `⟨S_z(t)⟩`, not the bond dims
(different cutoff rules legitimately give different bonds).

Branch: `quimb-tn-replatform`.

---

## Sub-step 1 — compression: **REPLACE → quimb** (adopted)

The hand-rolled canonicalise + R→L truncation sweep (`left_canonicalize` +
`truncate`, `rel_ref` rule) is replaced, opt-in, by quimb
`tensor_network_1d_compress` (zipup, cotengra path-finding, autoray dispatch) with
a quimb-native cutoff (`rsum2`). The paper's `rel_ref` rule is retired on this path.

- **Evidence:** end-to-end Gaudin `⟨S_z(t)⟩` matches the native `StandardSVD`
  reference to `<1e-4` over the whole trajectory for both `rsum2` (1e-13) and `rel`
  (1e-8) cutoff modes; full suite 325 passed (default path unchanged).
- **Why adopt:** library-maintained, backend-agnostic (CuPy stays on device — the
  natural cuQuantum substrate), and the observable is reproduced. Meets both arms of
  the criterion to *replace* (the cutoff *rule* is EDM-specific, but a quimb-native
  cutoff is behaviour-consistent on the observable, so universality wins).
- **Shipped:** `evolution/quimb_compress.py`, `compression='quimb'` selector through
  `compress()` → evolution engines → `SolverConfig`. Default `'native'`. Commit `206e078`.

---

## Sub-step 2 — fold contraction: **KEEP two-stage** (fused contraction *rejected* on single GPU)

Hypothesis (plan Phase B): fuse the per-fold MPO×MPS apply *and* compression into a
single quimb sweep so the full `D_a·χ` intermediate is never materialised — the
suspected OOM lever. Prototyped (`scratchpad/fused_*.py`) as a two-layer TN (EDM-MPS
site `p` shares its physical leg with the sub-bath kernel-MPO site `p`; trivial MPO
boundary bonds squeezed so the `d²` `vec(ρ)` boundaries survive) fed to
`tensor_network_1d_compress`.

**Correctness (wiring is right):** at `cutoff=0` the fused contraction reproduces the
native `_apply_sub_bath` fold to machine precision across every fold
(`|Δρ| ≤ 2e-15`, bonds identical `16→64→256→1024`, K=4).

**But it loses on the metrics that matter** (vs the validated two-stage = native apply
+ quimb compress, both `rsum2` 1e-13):

| metric | two-stage | fused **zipup** | fused **dm** |
|---|---|---|---|
| steady-state bond (K=12, T=3, fold 12) | **105** | 202 (~2×) | — |
| steady-state bond (K=8, T=2, fold 8) | **60** | ~190 | 16 |
| `⟨ρ⟩` Δ vs two-stage | — | 6e-7 ✓ | **1.7e-4 ✗ (and growing)** |
| fold wall time | **21.2 s** | 37.4 s (~1.8×) | 0.3 s but wrong |

- **fused-zipup** is faithful (≤6e-7) but single-pass truncation is sub-optimal: it
  keeps ~**2× the steady-state bond** at the same cutoff and is ~1.8× slower.
- **fused-dm** is fast but **over-truncates** (pins bond at 16, error climbs past the
  `<1e-4` bar) — unreliable as a self-sufficient path.

**Decisive reason fusion can't help the capacity wall for this problem class.** Gaudin
has a time-independent Hamiltonian → infinite memory time → the **steady-state** bond
grows without bound with `T`, and *that* is what OOMs a single A800 — not the transient
fold peak. fused-zipup's *only* advantage is a lower transient peak (it never forms the
full `a·χ`), but its 2× **steady-state** bond makes the steady-state OOM wall **worse**,
not better. Single-GPU contraction-fusion therefore cannot raise the capacity ceiling.

- **Decision:** **keep the two-stage path** (native `_apply_sub_bath` + quimb
  `compress`, sub-step 1). **No src change** for fused contraction.
- **Where the real capacity lever lives:** **Phase C — cotengra *slicing* +
  multi-GPU** distributes the steady-state bond across the 4×A800 (a different
  mechanism from single-GPU compression quality). That is where the K=24 / T=9 / ξ=1e-8
  single-card-OOM problem gets addressed.

---

## Pending (not yet decided)

- `RandomizedSVD` HMT vs quimb `rsvd`/`svds` (plan 0.1) — wrap + benchmark.
- `EDMMPS`/`KernelMPO` → quimb `MatrixProductState`/`MatrixProductOperator` container
  (plan 0.0 FOUNDATIONAL) — structural; no immediate perf claim, but the substrate for
  autoray/cotengra/cuQuantum throughout.
- `_xp` / `DecompositionBackend` registry → autoray dispatch (thin selectors).
