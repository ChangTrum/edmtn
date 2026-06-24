# Phase 5 design study — multi-GPU + cuQuantum (cuTensorNet) scale-out

Status: **research / design only** (no implementation). Drafted while the P5a
single-GPU benchmark runs. Goal: how to push edmtn to ultra-large-scale OQS
(bigger/more-complex baths, longer evolution) using multi-GPU and a cuTensorNet
backend, exploiting NVLink, FP64 Tensor Cores, and InfiniBand RDMA. The CPU
pipeline is treated as done — fallback / small-scale fast path.

This sits *beyond* the current `edm_technical_plan.md` roadmap (Phase 4 =
Ozaki GEMM + mixed precision + new models). It is effectively **Phase 5:
distributed scale-out**.

## 0. Design principles (directives — read before the rest)

**Two cleanly separated tracks. They do not share a distribution layer.**

**Track 1 — Custom pipeline: portable, NON-distributed.**
1. Scope = **pure CPU (including multi-socket within a single node) and single
   GPU**. This is already basically sufficient for it; it does **not** implement
   distribution at all.
2. It is the portable everyday + fallback path: runs on **Windows / macOS /
   Linux**, CPU-only or one GPU. This is why it stays vendor-neutral (no
   cuQuantum dependency) — exactly the platforms that cannot run cuQuantum.
3. All current work lives here (the `RandomizedSVD` from P1, StandardSVD, the
   CPU/single-GPU backends). It remains the maintained reference.

**Track 2 — Distributed pipeline: Linux-HPC, GPU-only, ultra-large-scale.**
4. Distribution is **only** for the Linux HPC ultra-large-scale regime; hardware
   is **GPU** (multi-GPU / multi-node). **CPU-distributed is out of scope** —
   technically possible but the performance gap is too large to bother.
5. Those machines **can run cuQuantum**, so the portability objection that forces
   Track 1 to be custom **does not apply here**. Therefore the distributed track
   is **built on cuQuantum / cuTensorNet** (its distributed contraction + slicing +
   multi-node engine), not a hand-rolled NCCL/MPI MPS layer.
6. Capacity (lever B — exceeding single-card memory) is the **top priority** of
   Track 2; intra-step distributed algebra (lever A) is worth doing on top of it
   despite the Amdahl ceiling, to fully use the hardware.

**Cross-cutting.**
7. Ensemble throughput (lever C) is **trivial** — independent solves dispatched to
   independent devices; a thin utility, works on either track, not a pillar.
8. **Ozaki/ADP (FP64-TC emulation) is parked** — late-stage nice-to-have; native
   A800 DMMA is still exercised for free by the Track-1 rSVD GEMMs.

The earlier draft's "portable identity-distribution layer" is **dropped**: Track 1
never distributes, so there is no need to make a distribution layer degrade to a
no-op on Win/Mac/CPU.

## 1. Where the parallelism is — and isn't (read this first)

A single EDM solve has a **serial critical path** that no amount of GPUs removes:

- the bath is folded sub-bath by sub-bath, `L = 1..K`, and fold `L+1` consumes
  the compressed output of fold `L` → **sequential over L**;
- inside a fold, left-canonicalisation and the truncation sweep walk the time
  chain site by site, each site depending on the previous → **sequential over
  sites**.

So multi-GPU is **not** a "split the loop across GPUs" win. The real leverage is
three-fold, in increasing implementation cost:

| lever | what it parallelises | when it pays | features used |
|---|---|---|---|
| **(C) Ensemble** | independent solves (disorder realisations, parameter/`g` sweeps, trajectories) | always, trivially | IB only for coordination |
| **(A) Intra-step algebra** | the per-site SVD/QR/GEMM of *one* solve | large bond `χ` | NVLink (intra-node), IB+RDMA (multi-node) |
| **(B) Capacity** | hold an MPS / working tensors too big for 80 GB | large `χ` *and/or* long chains | NVLink + IB RDMA |

**Amdahl caveat, stated honestly:** lever (A) speeds up each site's linear
algebra but not the serial site/fold ordering, so a *single* solve's speedup
saturates at the serial fraction. The decisive single-solve win is therefore
bounded; the biggest multi-GPU payoffs are **(B) fitting problems a single GPU
can't hold** and **(C) ensemble throughput**. At today's scale (`χ ~ 100`s)
multi-GPU is *counterproductive* (comm > compute); it only pays in the
large-`χ` / large-memory regime edmtn is built to reach. The P5a scaling curve
(GPU lead growing with `χ`) is the empirical signpost for where that regime
begins.

## 2. Feature mapping

### 2.1 FP64 Tensor Cores
- **A800 (Ampere, cc 8.0):** native FP64 tensor cores (DMMA). cuBLAS dispatches
  FP64 GEMM with `CUBLAS_COMPUTE_64F` to DMMA automatically — **exact FP64, no
  precision loss**. This is available *now* and benefits exactly the rSVD GEMMs
  (`M·Ω`, `Qᴴ·M`), not cuSOLVER SVD/QR. So **the more work the decomposition
  layer shifts from full SVD to randomized SVD (P1), the more FP64-TC is
  exercised** — the two improvements compound.
- **Blackwell (RTX 5090 cc 12.0):** native FP64 throughput is gutted; the path is
  the **Ozaki/ADP emulation** already seamed in `backend/ozaki_gemm.py` (CUDA 13,
  splits FP64 into fixed-point slices on INT8 TCs, ADP picks slice count from the
  condition number, ≥ native FP64 accuracy). Reported up to 13.2× DGEMM on RTX
  PRO 6000 Blackwell. **Hardware-dependent mechanism**: A800 = DMMA, 5090 = Ozaki.
- **Open caveat (from the plan):** **ZGEMM (complex128)** tensor-core support is
  unverified — DMMA/Ozaki docs focus on DGEMM/SGEMM. The EDM matrices are complex.
  Mitigation to test: process real/imag separately (3M or Karatsuba complex GEMM
  built from real DGEMMs that *do* hit TCs), or verify cuBLAS ZGEMM DMMA dispatch
  directly with nsight. **This must be measured, not assumed.**

### 2.2 NVLink (intra-node GPU↔GPU)
- A800 c1 topology (from recon): GPU0-1 NV4, others NV2 — high intra-node BW.
- Used by lever (A): distributed rSVD exchanges partial sketch products via NCCL
  all-reduce over NVLink; and lever (B): halo/boundary-tensor exchange when the
  MPS chain or a large bond matrix is sharded across the 4 GPUs.

### 2.3 InfiniBand RDMA (inter-node)
- Extends (B) to multi-node capacity (MPS chain segments on different nodes,
  RDMA halo exchange during sweeps) and (C) ensemble scale-out across nodes.
- cuTensorNet's distributed API rides MPI + UCX/RDMA for multi-node contraction.

### 2.4 cuTensorNet (cuQuantum) — the engine of the distributed track (Track 2)
Per §0, distribution only ever runs on Linux-HPC GPUs, which have cuQuantum — so
cuTensorNet is the **intended distributed engine**, not an optional add-on.
- What it provides: **contraction path optimisation + slicing** (memory control)
  and **distributed multi-GPU/multi-node contraction** (NVLink intra-node, IB+MPI
  inter-node); recent cuQuantum also ships **MPS / approximate-TN state APIs**
  (apply MPO + SVD-compress — structurally the EDM fold) and `tensorSVD`/`tensorQR`
  primitives, so it can own both the contraction and the truncation at scale.
- It is reached through the existing Layer-0 backend registry, so Track 1 is
  completely unaffected when cuQuantum is absent.
- Validation: its SVD accuracy controls, complex128 support, and EDM-MPS ↔
  cuTensorNet layout interop must all reproduce the Track-1 StandardSVD reference
  to `< ξ` before it is trusted at scale.
- Open question (feasibility, not portability): if cuTensorNet's distributed
  SVD-compression proves insufficient/inaccurate for the EDM fold, a *targeted*
  custom GPU-distributed step (NCCL) may be needed as a supplement — but the
  default intent is to let cuQuantum own distribution.

## 3. Integration architecture — two tracks behind one registry

The codebase already isolates linear algebra behind `DecompositionBackend`
(registry) and arrays behind `ArrayFactory`. That seam is exactly what keeps the
two tracks separate without forking the solver: the fold loop and decomposition
*strategies* stay API-stable; the **backend** decides CPU / single-GPU / (on HPC)
cuTensorNet-distributed.

**Track 1 (portable, non-distributed) — already mostly here.** CPU (incl.
multi-socket: MKL/OpenBLAS threads, as the P5a EPYC run uses) and single GPU
(`RandomizedSVD` P1, CuPy backend). Nothing new architecturally; just keep it
clean and vendor-neutral. This is the reference every Track-2 result is checked
against.

**Track 2 (Linux-HPC, GPU, distributed) — built on cuTensorNet.** The capacity
driver (lever B): what blows past 80 GB first is the *whole time-chain*, not one
bond's SVD matrix (`n_sites·d_phys·χ²·16 B`, e.g. 120×7×2000²×16 ≈ 54 GB, vs a
single `(7χ × 4χ)` working matrix that still fits). cuTensorNet handles the
distributed residency + contraction + slicing across GPUs/nodes; we express the
fold (apply sub-bath MPO, compress) via its MPS / approximate-TN APIs and let its
distributed engine place tensors over NVLink (intra-node) and IB+MPI (inter-node).

New pieces:
1. **`CuTensorNetBackend` (new, Layer 0 — Linux/CUDA only):** contraction +
   `tensorSVD`/`tensorQR`, single- then multi-GPU, then multi-node via cuTensorNet's
   distributed API. Selected through the registry; absent ⇒ Track 1 unaffected.
2. **Fold ↔ cuTensorNet adapter (new):** map the EDM-MPS tensors + sub-bath MPO to
   cuTensorNet network/state objects and back, preserving the gauge/index
   conventions; this is where the interop validation (`< ξ`) lives.
3. **(contingency) targeted custom NCCL step (lever A):** only if cuTensorNet's
   distributed SVD-compression is insufficient — a sharded rSVD on the bond matrix
   reusing P1's resolution-guard numerics. Not built unless measurement forces it.
4. **Ozaki / mixed precision:** parked; native A800 DMMA serves Track-1 rSVD GEMMs.

**Contract:** Track 1 must remain byte-for-byte the current pipeline (no cuQuantum
import on Win/Mac/CPU). Track 2 must reproduce the Track-1 StandardSVD `⟨S_z(t)⟩`
to `< ξ` (the project-wide bar) before it is trusted at scale.

## 4. Proposed rollout (priority-ordered per §0)

Track 1 is essentially done (CPU multi-socket + single-GPU rSVD). The work is
Track 2, on Linux-HPC GPUs via cuTensorNet:

- **5.0 — ensemble utility (lever C, trivial, either track).** A thin SLURM
  job-array / driver for disorder-averaged Gaudin and parameter sweeps (independent
  solves per device). Near-zero difficulty; ship early for throughput. Not the
  architecture.
- **5.1 — cuTensorNet feasibility spike (single GPU).** Stand up
  `CuTensorNetBackend` + the fold↔cuTensorNet adapter on one A800; reproduce the
  Track-1 StandardSVD result to `< ξ`. De-risks the interop + complex128 + SVD
  accuracy questions before any distribution.
- **5.2 — cuTensorNet intra-node multi-GPU for capacity (lever B, *the priority*).**
  Use cuTensorNet's distributed engine over **NVLink** on c1's 4×A800 to hold +
  process an MPS that exceeds one 80 GB card. This is the core deliverable —
  ultra-large OQS starts here.
- **5.3 — multi-node via cuTensorNet (lever B at scale, IB+MPI/RDMA).** Extend to
  several nodes for problems beyond one node's 4×80 GB.
- **5.4 — intra-step utilisation (lever A).** Lean on cuTensorNet's slicing /
  contraction parallelism for the big per-bond work; add the contingency custom
  NCCL rSVD step only if measurement shows cuTensorNet's SVD-compression is the
  limiter.
- **(parked) Ozaki/ADP FP64-TC emulation** — late-stage nice-to-have only.

Each stage gated by the `< ξ` accuracy match against Track 1 + a scaling benchmark
(extend `perf_gpu_compression.py`) showing it only switches on where it wins.

## 5. Risks / unknowns to verify before committing code
- **ZGEMM tensor-core dispatch** (complex128) on DMMA and Ozaki — measure; maybe
  split real/imag.
- **cuSOLVER SVD never uses TCs** → keep favouring rSVD (GEMM) over full SVD; lean
  on cuTensorNet's SVD only after validating its accuracy/complex support.
- **NCCL complex128 collectives** support; distributed SVD (cuSOLVERMp) maturity.
- **Serial-fraction / Amdahl** of one solve — measure the sweep's serial cost vs
  the per-site algebra to bound single-solve multi-GPU gains (sets when 5.2/5.3
  pay). The §16 "streaming gauge" shortcut is dead (conditioning), so the serial
  sweep stays.
- **Distributed rSVD reproducibility** — per-rank sketch seeding / determinism.
- **cuTensorNet ↔ EDM-MPS interop** — tensor layout, gauge conventions, dtype.

## 6. Bottom line
**Two tracks, separated by the existing backend registry.** Track 1 — the custom,
portable, **non-distributed** pipeline (CPU incl. multi-socket, + single GPU) — is
the vendor-neutral default for Windows/macOS/Linux and is essentially done. Track 2
— **distribution for ultra-large-scale** — runs only on Linux-HPC GPUs, where
cuQuantum is available, so it is **built on cuTensorNet** (its distributed
contraction + slicing + multi-node engine) rather than a hand-rolled NCCL/MPI MPS
layer; capacity (fitting an MPS bigger than one card) is its top job. The two never
share a distribution layer. cuTensorNet is reached through the registry, so its
absence leaves Track 1 untouched. Ensemble throughput is a trivial early utility;
Ozaki/ADP is parked. Critical path for Track 2: cuTensorNet feasibility spike
(single GPU, `< ξ`) → intra-node multi-GPU capacity over NVLink → multi-node over
IB/MPI. Single-solve speedup is Amdahl-bounded by the serial sweep, but **capacity
is not — and capacity is the point.**
