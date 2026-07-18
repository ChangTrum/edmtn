# CPU vs GPU for the EDM solver (Phase 1/2 measurements)

> **Measurements below are HISTORICAL** (Phase-1/2 era, specific hardware + parameters). They
> are kept as a record and do NOT describe current capability or current defaults.

## Current status (2026-07)

- `compress_decomp='rsvd'` (Track-1 randomized SVD + guard) and the `balanced`/`robust` presets
  are implemented and available on CPU/GPU; see [../guides/recommended-config.md](../guides/recommended-config.md).
- `precision='mixed'` is **experimental**: only the contraction tensors are cast, the separate
  f64 decomposition re-cast is not wired into `solve()`, and its end-to-end check is skipped.
- GPU/HPC tests are gated by real hardware detection with `--require-gpu` / `--require-cuquantum`
  / `--require-multigpu=N` (P1-14). Track-1 single-GPU parity and Track-2 single-GPU cuTensorNet
  are verified on hardware; the 4-GPU distributed path is currently blocked by a CUDA-aware-MPI
  regression in the test cluster's environment.
- The default backend is `cpu` (there is no `auto`).


## Decision

**Phase 1 (spin-boson) and Phase 2 (Gaudin) run on the CPU by default**
(the default `backend='cpu'`).  The GPU is fully wired, validated, and selectable
(`backend='gpu'`), but it is *not* the faster path at the problem sizes these
phases reach, so it stays opt-in until the Phase-3 decomposition work makes it
worthwhile.  All GPU infrastructure (CuPy backend, `ArrayFactory.auto`,
`MemoryManager`, `PrecisionPolicy`, `OzakiGEMMBackend` seam, the `convert`/
`memory` hooks in the evolution engines) is retained for Phase 3/4.

## Why — the EDM compute profile

The EDM algorithm is **`O(N^2)` in the number of steps**: each step contracts a
kernel MPO over the whole history and recompresses with an SVD sweep.  The work
is therefore **many sequential, small-to-medium dense factorizations (SVD / QR)
with Python orchestration between them**.  The system space is tiny (`d^2 = 4`)
and, until the bond dimension grows large, so are the matrices.

That profile is the *worst case* for a GPU: per-call launch + host↔device
synchronization overhead dominates the negligible compute, and the calls cannot
be batched (each depends on the previous via the recursive construction).  The
CPU, with no launch overhead and good small-matrix LAPACK, wins until the
individual factorizations become large enough to be compute-bound.

## Evidence — spin-boson (Phase 1)

`tests/benchmarks/perf_cpu_gpu.py` times the single-bath evolution on CPU/GPU ×
fp32/fp64.  Conclusion (recorded in Phase 1): the **CPU is fastest** for the
spin-boson regime; the small `d^2 = 4` matrices and bond dimensions ~tens never
amortize the GPU launch overhead.  fp32 roughly halves the work but caps accuracy
near ~1e-6 relative.

## Evidence — Gaudin (Phase 2)

`tests/benchmarks/perf_gaudin.py` times the full separable solve on CPU vs GPU
across bond-dimension caps.  Measured on this machine (RTX 5090; Gaudin `K=12`,
`T=3 g^-1`, `eps=0.1`, order 2, `cutoff=1e-6`):

| `D_c` | `Dmax` | CPU [s] | GPU [s] | GPU/CPU speedup | `max|ΔSz|` (GPU vs CPU) |
|------:|-------:|--------:|--------:|:---------------:|:-----------------------:|
|    50 |     50 |   17.25 |   23.91 |    **0.72×**    |        3.1e-07          |
|   100 |     95 |   49.94 |   75.24 |    **0.66×**    |        2.3e-14          |

So even for Gaudin — whose bond dimension is *much* larger than spin-boson — the
**CPU is still faster at `D_c ≤ 100`**, and the GPU's relative position does not
improve with size in this range (0.72× → 0.66×).  The GPU result is numerically
identical to the CPU (`ΔSz ~ 1e-14`), i.e. the GPU path is correct; it is simply
not faster here.  The same root cause as spin-boson applies: many sequential
medium SVD/QR calls, latency-bound.

## The large-`D` memory bottleneck

The GPU would only overtake once `D_c` is large (the paper uses 400 — a chosen cap,
**not** a physical ceiling: the Gaudin bond grows without bound with evolution time,
see `gpu-scaling-benchmark.md` item 2), making each SVD compute-bound.  But at large `D_c` the *current* compression strategy
(`StandardSVD`, "form the full product, then truncate") hits a memory wall first:
applying a sub-bath MPO multiplies every internal bond by the lateral factor
`D_a = 4`, so the **uncompressed intermediate MPS has bonds up to `4 · D_c`**
(1600 at `D_c = 400`).  A single such site is `7 · 1600 · 1600 · 16 B ≈ 287 MB`,
and the whole transient is tens of GB — it exceeds GPU VRAM (and strains host
RAM) before the GPU's compute advantage can be realized.  Benchmarks at
`D_c = 400` were therefore not completable in this configuration.

## What would make the GPU win — Phase 3

The lever for "GPU-primary" is **not** swapping the backend; it is the Phase-3
decomposition layer (technical plan §6.3–6.4):

- **Randomized SVD** turns the truncation into a few large GEMMs (random
  projection + power iteration) plus one small dense SVD — GEMM-dominated work
  where the GPU (and, on Blackwell, the Ozaki/ADP `OzakiGEMMBackend`) wins.
- **SRC / single-pass compression** compresses *while* contracting, so the full
  `4 · D_c` product is never formed — removing the memory wall and making
  `D_c = 400` feasible on the GPU.

Once those land, the default backend for the separable (and future chain/Kondo)
pipelines should flip to GPU-primary, and these benchmarks rerun to confirm the
crossover.

## Reproduce

```
# spin-boson, single-bath hot path
python tests/benchmarks/perf_cpu_gpu.py
# Gaudin, separable solve, CPU vs GPU across D_c caps
python tests/benchmarks/perf_gaudin.py --quick
python tests/benchmarks/perf_gaudin.py --K 24 --T 12 --max-bonds 100,200,400
```

GPU correctness (vs CPU / vs exact Trotter) is covered by the skipped Phase-3/4
tests `test_gpu_matches_cpu` (tests/unit/test_backend.py) and
`test_gpu_matches_cpu_gaudin` (tests/unit/test_driver_separable.py); the cheap
Layer-0 GPU *interface* tests (CuPy SVD/QR, `ArrayFactory`/`MemoryManager` on
CuPy) remain active and guard the retained interfaces.
