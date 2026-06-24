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

## Reproduce

```
# GPU sweep on a GPU node:
python tests/benchmarks/perf_gpu_compression.py --K 24 --T 3 --cutoff 1e-8 \
    --combos gpu:svd,gpu:rsvd0,gpu:rsvd2 --repeats 2
# CPU baseline on a many-core node (set threads to the core count):
OMP_NUM_THREADS=256 MKL_NUM_THREADS=256 \
  python tests/benchmarks/perf_gpu_compression.py --K 24 --T 3 --cutoff 1e-8 \
    --combos cpu:svd --repeats 1
```
