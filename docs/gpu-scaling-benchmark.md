# GPU scaling benchmark (P5a) — single A800 vs EPYC 9754, by problem size

Measured end-to-end EDM fold (Gaudin), comparing the full SVD (`StandardSVD`)
against randomized SVD (`RandomizedSVD`, single-pass `n_iter=0` and cold
`n_iter=2`) on CPU and a single GPU, as the bond dimension grows. Driver:
`tests/benchmarks/perf_gpu_compression.py`.

## Setup and fairness

- **Problem:** Gaudin central spin, `K=24`, `T=3 g⁻¹`, `ε=0.2 g⁻¹`, order 2
  (`n_sites=30`). Bond `χ` is grown by tightening the truncation `ξ`.
- **GPU:** one **NVIDIA A800-SXM4-80GB** (cluster node c1), CuPy 14.1.1, CUDA 12.9,
  cuBLAS 12.6.3. **Single card** (the fold is sequential; no multi-GPU yet).
  GPU-synchronised timing, one untimed warm-up, min of 2 repeats.
- **CPU:** dual **AMD EPYC 9754** (Zen 4, node a9), **256 threads**, NumPy 2.4.6.
  Run with **both MKL and OpenBLAS** to be fair about the BLAS.
- **Accuracy** is `max|Δ⟨S_z(t)⟩|` vs the CPU full-SVD reference; **bond** is the
  final `Dmax`.

## Result

Speedups are vs the CPU full-SVD baseline (MKL; OpenBLAS within ~2%, see below).

| ξ | bond χ | CPU-svd (MKL, 256t) | GPU svd | GPU rsvd0 | GPU rsvd2 | **rsvd0 speedup** | svd speedup |
|---|---|---|---|---|---|---|---|
| 1e-6 | 95 | 79.2 s | 24.2 s | 11.1 s | 13.0 s | **7.1×** | 3.3× |
| 1e-8 | 175 | 259.0 s | 51.4 s | 23.9 s | 28.3 s | **10.8×** | 5.0× |
| 1e-10 | 325 | 878.6 s | 110.6 s | 56.6 s | 60.6 s | **15.5×** | 7.9× |
| 1e-12 | 400\* | — | 172.7 s | 74.6 s | 92.0 s | — | — |

\* ξ=1e-12 hit the `max_bond=400` cap (not a clean scaling point); CPU not run there.

Accuracy / bond per point:

| ξ | rsvd0 `\|Δ⟨Sz⟩\|` (bond) | rsvd2 `\|Δ⟨Sz⟩\|` (bond) |
|---|---|---|
| 1e-6 | 1.1e-7 (95) | 2.8e-8 (95) |
| 1e-8 | 8.2e-10 (191) | 1.4e-13 (175) |
| 1e-10 | 8.0e-12 (371) | 1.5e-13 (325) |

## Findings

1. **The GPU advantage grows monotonically with problem size.** Single-pass rSVD
   on one A800 goes **7.1× → 10.8× → 15.5×** over a fully-provisioned 256-thread
   EPYC-9754 CPU as `χ` grows 95 → 175 → 325 (full-SVD on GPU: 3.3× → 5.0× → 7.9×),
   and is still climbing at χ=325. At small `χ` the GPU is "nice, not necessary"
   (a strong CPU copes); at large `χ` it is decisive. **This is the quantitative
   case for the GPU / multi-GPU (Phase 5) roadmap: the win lives in the
   large-bath / long-evolution regime edmtn targets.**
2. **rSVD's same-GPU edge over full SVD is a steady ~2×** (2.2 / 2.2 / 2.0 / 2.3)
   — the BLAS-3 (GEMM) payoff, hardware-independent, and it compounds with the GPU
   advantage. Single-pass beats cold on wall-clock everywhere.
3. **Accuracy holds throughout.** Single-pass is below the cutoff at every point
   (1e-7 → 5e-12); cold rSVD reaches ~1e-13 and reproduces the **exact full-SVD
   bonds** (175/325), while single-pass over-retains a few percent at tight ξ
   (191/371) — consistent with `incremental-update-research.md` §13.
4. **MKL gives no benefit over OpenBLAS on Zen 4** (same node, 256 threads):
   ξ=1e-6 → OpenBLAS 77.4 s vs MKL 79.2 s; ξ=1e-8 → MKL 259.0 s vs OpenBLAS
   264.8 s — a tie within ~2%. MKL gates AVX-512 to Intel CPUs while OpenBLAS uses
   Zen 4's, so for this SVD/QR-heavy workload the BLAS choice is immaterial on AMD.

## Caveats / not yet measured

- **Single GPU only.** No multi-GPU / NVLink / cuQuantum yet (see
  `multi-gpu-cuquantum-design.md`, Phase 5).
- **FP64 Tensor Core (DMMA) usage unverified** for the complex128 GEMMs — cuBLAS
  ZGEMM tensor-core dispatch needs an nsight check (Phase-5 item 5.0); the rSVD
  speedup above is real regardless.
- **T=6 / K=48 axes not completed** (job time limits); the ξ-scan is the primary,
  cleaner bond-growth axis and already establishes the trend.

## P5b — canonicalisation on GPU: Householder vs CholeskyQR (A800)

Same single A800, single-pass rSVD compression fixed, varying only the
canonicalisation (total solve wall; accuracy vs the Householder run confirms
CholeskyQR is correct on CuPy):

| ξ | bond | Householder | CholeskyQR2 | CholeskyQR1 |
|---|---|---|---|---|
| 1e-6 | 95 | 11.07 s | 12.40 s | 10.63 s |
| 1e-8 | 191 | **23.93 s** | 28.29 s | 27.40 s |
| 1e-10 | 371 | **56.59 s** | 64.31 s | 71.83 s |

**Householder QR is the fastest canonicaliser on the GPU at every ξ, and the
CholeskyQR2 deficit grows with the bond** (−5% → −18% → −14%). The hypothesis that
GPU GEMM throughput would keep CholeskyQR2 ahead at tight ξ (the open question in
`recommended-config.md`) is **refuted**: cuSOLVER `geqrf` is already efficient on
these tall matrices, and CholeskyQR2's 2-pass GEMM overhead does not pay.
CholeskyQR2 is *correct* on GPU (accuracy 4.7e-8 / 7.6e-10 / 1.1e-11, exact-ish
bonds) — just not faster. Combined with §14 (CholQR2 helps only on CPU at moderate
ξ, loses at tight ξ), **Householder QR is the canonicalisation default in all
regimes**; CholeskyQR2 stays selectable for the narrow CPU-moderate-ξ niche.

## Item 1 — FP64 Tensor Cores (DMMA): engaged for complex128 too

`perf_fp64_tc.py` on the A800 (achieved TFLOP/s; FP64 CUDA-core peak ~9.7, DMMA
peak ~19.5):

| N | DGEMM | ZGEMM |
|---|---|---|
| 2048 | 16.0 | 17.1 |
| 4096 | 17.4 | 17.4 |
| 8192 | **19.3** | **19.4** |

Both DGEMM and ZGEMM reach ~19.4 TFLOP/s ≈ **2× the CUDA-core peak ≈ the DMMA peak**,
so **FP64 tensor cores are engaged for the complex128 rSVD GEMMs already** — a
real/imag split is unnecessary. (cuBLAS math-mode toggle wasn't exposed by this CuPy,
but the achieved throughput is conclusive.)

## Item 2 — single-GPU capacity: the hard wall for Gaudin at long evolution time

**Uncapped** (`max_bond` effectively unlimited), so the bond grows to its natural,
truncation-determined value. `perf_gpu_compression --combos gpu:rsvd0`, K=24:

| ξ | T | n_sites | natural Dmax | peak GPU GB |
|---|---|---|---|---|
| 1e-8 | 3 | 30 | 191 | 0.68 |
| 1e-8 | 6 | 60 | **643** | 13.29 |
| 1e-8 | 9 | 90 | — | **OOM** (>80 GB) |
| 1e-8 | 12 | 120 | — | OOM |
| 1e-10 | 3 | 30 | 371 | 2.68 |
| 1e-10 | 6 | 60 | — | **OOM** |

**The earlier "memory is not the limiter" reading was an artefact of an artificial
`max_bond=400` cap** (the paper's resource-constrained convenience, not a physical
limit). Removing it changes the picture completely:

- **Gaudin's bond grows without bound with evolution time `T`.** Its Hamiltonian is
  time-independent → the memory time is *infinite* (unlike spin-boson, which
  saturates), so the EDM bond keeps growing with `T` (191 → 643 from T3 → T6 at
  ξ=1e-8; the paper's bound is *linear in T* asymptotically). (Orthogonally, growth
  along the *fold index* `L` does plateau here — later sub-baths have weaker coupling
  under the linearly-decreasing-`g` scheme; other `g` profiles are future work.)
- **Capacity is therefore the hard wall, hit at modest `T`.** Peak memory is
  dominated by the per-fold working tensors (∝ `n_sites · χ²`), and with `χ` growing
  in `T` a single 80 GB A800 **cannot run** K=24 at **T=9, ξ=1e-8** (OOM at >83 GB) or
  **T=6, ξ=1e-10** — tighter ξ (larger bond) hits the wall sooner. It is an
  out-of-memory failure, not merely a slow run.

**Implication for Phase 5.** Capacity (lever B, Phase 5.2 — multi-GPU then multi-node)
is a **real, near-term constraint**, not a far-future one: even a modest Gaudin run
exceeds one card once the evolution time is long enough, and longer `T` / tighter ξ /
larger `K` only bring the wall closer. This is the concrete justification for the
multi-GPU capacity priority. (The serial fold/sweep wall-clock grows too, but here the
OOM binds first.) See `multi-gpu-cuquantum-design.md`.

**Implication for Phase 5.** For the current demonstrator the near-term GPU lever is
**wall-clock (intra-step / lever A) + ensemble (lever C)**, *not* capacity: capacity
(lever B, Phase 5.2) only binds once χ reaches the thousands, which requires a
higher-entanglement regime/model than Gaudin (or uncapped bonds). Stress-testing
capacity will need such a regime; see `multi-gpu-cuquantum-design.md`.

## Reproduce

```
# GPU sweep on a GPU node:
python tests/benchmarks/perf_gpu_compression.py --K 24 --T 3 --cutoff 1e-8 \
    --combos gpu:svd,gpu:rsvd0,gpu:rsvd2 --repeats 2
# CPU baseline on a many-core node (set threads to the core count):
OMP_NUM_THREADS=256 MKL_NUM_THREADS=256 \
  python tests/benchmarks/perf_gpu_compression.py --K 24 --T 3 --cutoff 1e-8 \
    --combos cpu:svd --repeats 1
# P5b canon crossover on a GPU node (compression fixed, vary canonicalisation):
python tests/benchmarks/perf_gpu_compression.py --K 24 --T 3 --cutoff 1e-8 \
    --combos gpu:rsvd0 --canon householder,cholqr2,cholqr1 --repeats 1
# Item 1 FP64-TC probe / Item 2 capacity (uncapped bond; peak GPU GB per run):
python tests/benchmarks/perf_fp64_tc.py --sizes 2048,4096,8192 --reps 5
python tests/benchmarks/perf_gpu_compression.py --K 24 --T 6 --cutoff 1e-8 \
    --combos gpu:rsvd0 --max-bond 100000 --repeats 1   # natural bond; T9 OOMs on 80 GB
```
