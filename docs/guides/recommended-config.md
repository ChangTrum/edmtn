# Recommended canonicalisation + compression configuration

> **Scope.** `preset=None` is the API default (`compress_decomp='exact'`, deterministic full
> SVD — and the only setting that can report a real truncation metric). The presets below are
> *recommendations*; each one sets **only** `compress_decomp='rsvd'` plus `compress_decomp_q`
> (`0` for `balanced`, `2` for `robust`) and never touches `compress_canon` or `backend`.
> All numbers here are measurements on specific hardware/parameters, not error guarantees.


Distilled from the incremental-update / bottleneck study (`incremental-update-research.md`,
§13–§16). Two presets are offered as *recommendations* (the API default is `preset=None`):
a **balanced** option and an **accuracy/stability
fallback**. All measurements are end-to-end Gaudin folds (K=24, T=3 g⁻¹, ε=0.2 g⁻¹, order 2)
vs the unmodified pipeline (`EDMSolver`, full-SVD + Householder QR), on CPU.

## Hardware context (why BLAS-3 everywhere)

The target deployment is **GPU-primary**: data-centre **Tesla H200 NVL** and workstation
**RTX 5090**, with CPU only as a fallback. This is the reason every kernel in this study was
pushed toward **BLAS-3 (GEMM-dominated)** form:

- **Householder QR** (cuSOLVER `geqrf`) parallelises poorly on GPU (sequential reflector
  panels) — it is the *slow* path on the target hardware.
- **Cholesky-QR2** is pure BLAS-3 (`syrk` Gram + tiny `potrf` + `trsm` solve) → GPU-fast.
- **Full SVD** (`gesvd`) is hard to parallelise; **randomised SVD** is its GEMM-based
  replacement (sketch `M·Ω`, `Qᴴ·M`, plus one skinny QR) → GPU-fast.

**Consequence:** the CPU numbers below *understate* the GPU advantage of the BLAS-3 choices,
and the one CPU regression we saw (CholQR slower than QR at very tight ξ, §14) is driven by
flop-doubling + shift escalation and **may not survive on GPU** where GEMM throughput
dominates — this must be re-measured on the actual H200/5090 (see open question below).

## Preset 1 — `balanced` (a recommendation, NOT the API default)

| layer | choice |
|---|---|
| canonicalisation | **not set by the preset** — `compress_canon` stays whatever you configure (default `'quimb'`). Householder (`compress_canon='householder'`) measured fastest on GPU/tight ξ; set it explicitly if you want it |
| compression | **single-pass randomised SVD** (`n_iter=0`, oversample 10, spectral resolution guard) |

**Why.** Single-pass rSVD was the dominant win in the measurements below: on that hardware and
those parameters the observed deviation came out at 1.5e-7 (ξ=1e-6) and 8e-10 (ξ=1e-8), with no
tuning and no reference run (the resolution guard grows the sketch until the computed tail drops
below ξ). These are **measured results, not a guarantee** — rSVD stays a randomized algorithm,
and its accuracy is not bounded by ξ in general.

**Canonicalisation = Householder QR (not CholeskyQR2).** Measurement settled this:
CholeskyQR2 only beats Householder on the *canonicalisation step* on **CPU at moderate ξ**
(~1.13× there, §14); it **loses at tight ξ on CPU** (flop-doubling + shift escalation, §14)
and **loses across all ξ on the GPU** (A800 P5b: Householder is fastest, deficit growing to
−14% at χ=371 — cuSOLVER `geqrf` is already efficient and CholQR2's 2-pass GEMM overhead does
not pay). Since the deployment target is GPU-primary, Householder QR is the better choice in
these measurements — but note the **API default is `compress_canon='quimb'`**, and no preset
changes it; set `compress_canon='householder'` explicitly if you want it.
`compress_canon='cholqr'` remains available (Cholesky QR,
machine-precision orthogonality, per-bond Householder fallback) for the narrow CPU-moderate-ξ
niche, but it is **not** the default.

**Numbers (CPU; GPU expected better):**

| ξ | end-to-end vs pipeline | accuracy `\|Δ⟨Sz⟩\|` | bond vs baseline |
|---|---|---|---|
| 1e-6 | ~1.2–1.5× | 1.5e-7 (<ξ) | = baseline (95) |
| 1e-8 | ~0.9–1.0× (auto-degrades to QR-level) | 8e-10 (<ξ) | +7% (single-pass over-retain) |

**Trade-offs accepted:** at very tight ξ, single-pass rSVD over-retains the bond by ~7%
(compounding; §13) — a graceful degradation (accuracy still < ξ), not a failure. Use the
`robust` preset if exact bonds are required.

## Preset 2 — `robust` (fallback, extreme cases)

| layer | choice |
|---|---|
| canonicalisation | **not set by the preset** — `compress_canon` stays as configured (default `'quimb'`); `'householder'` measured best here, set it explicitly |
| compression | **cold randomised SVD** (`compress_decomp='rsvd'`, `compress_decomp_q=2`). NOTE the preset does NOT select full SVD — for that use `compress_decomp='exact'` with **no** preset |

**Why.** Householder QR is immune to conditioning (§16: the raw folded MPS has left-environment
cond ≈ 1/ξ² ≈ 1e12; only an orthogonal transform survives it), so it never degrades regardless
of regime. In these measurements cold rSVD reproduced the baseline bond dimensions (no
over-retention even at tight ξ) at ~1e-12 deviation, while staying GEMM-based. Note this preset
is **still randomized rSVD** (`compress_decomp='rsvd'`, `q=2`) — it does **not** give exact bond
dimensions or bit-reproducibility as a guarantee. For no randomisation at all, use **no preset**
with `compress_decomp='exact'` (the API default), which is also the only setting that can report
a real truncation metric.

**When to switch to this preset:** very tight cutoff; suspected ill-conditioning. If you
actually *require* determinism/bit-reproducibility or a measurable truncation metric, do not use
a preset at all — use `compress_decomp='exact'`.

## Regime guidance

| situation | preset |
|---|---|
| production, moderate cutoff (ξ ≳ 1e-7), GPU | `balanced` |
| very tight cutoff (ξ ≲ 1e-8) | `robust` (or `balanced` — auto-degrades, still correct) |
| need exact bonds / max accuracy / reproducibility / a real truncation metric | **no preset** + `compress_decomp='exact'` |
| CPU-only batch where Householder QR is cheap anyway | either; `balanced` still wins on compression |

## Rejected approaches (do not revisit without new information)

- **skip-QR** (no canonicalisation): unphysical (complex observable), ~25× slower (§14).
- **Newton–Schulz polar** orthogonaliser: ~5× slower — iterates on the tall `m×n` factor (§14).
- **zip-up / fused forward sweep**: over-retention caps it at 1.41× < CholQR's 1.53% (§15).
- **single R→L sweep with carried gauge**: gauge cond ≈ 1e6–1e12 ≫ working range (§16).

## GPU measurement (single A800) — single-pass rSVD validated at scale

Measured on an A800 vs a 256-thread EPYC 9754 (`docs/gpu-scaling-benchmark.md`):
single-pass rSVD (`compress_decomp='rsvd'`, `compress_decomp_q=0` — the `balanced` compression) runs
**7.1× → 10.8× → 15.5× faster than the fully-provisioned CPU full-SVD pipeline**
as the bond grows χ = 95 → 175 → 325, at accuracy below the cutoff and matching
bonds — and the advantage **grows with problem size**. rSVD beats GPU full-SVD by a
steady ~2× (the BLAS-3 payoff). So on GPU, **`balanced` (single-pass rSVD) is the
recommended choice in this measured regime** (the API default remains `preset=None`), and the
GPU is the path for the large-scale regime (small χ does
not need it). Also measured: **MKL gives no benefit over OpenBLAS on Zen 4** (tie
within ~2%) — BLAS choice is immaterial on AMD for this workload.

## Resolved / still-open questions

- **CholQR2-vs-Householder crossover on GPU — RESOLVED (Householder wins).** P2 put
  CholeskyQR in `src/`; P5b measured it on an A800 (`docs/gpu-scaling-benchmark.md`):
  Householder QR is fastest at every ξ, the CholQR2 deficit *growing* with χ (−14% at
  χ=371). The hypothesis that GPU GEMM throughput would keep CholQR2 ahead is refuted —
  cuSOLVER `geqrf` is already efficient. Householder QR measured best here, but is opt-in
  (`compress_canon='householder'`); the API default stays `'quimb'`;
  CholeskyQR2 is kept selectable for the CPU-moderate-ξ niche only.
- **FP64 Tensor Core (DMMA) for complex128 — RESOLVED (engaged; no split needed).** Measured
  on an A800 (`docs/gpu-scaling-benchmark.md`, item 1): both DGEMM **and** ZGEMM sustain
  ~19.4 TFLOP/s = ~2× the FP64 CUDA-core peak (~9.7) ≈ the DMMA peak (~19.5). So the complex128
  rSVD GEMMs already run on the FP64 tensor cores; a real/imag split is unnecessary.
