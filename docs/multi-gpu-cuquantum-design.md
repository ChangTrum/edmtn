# Track 2 — cuQuantum (cuTensorNet) HPC track: design & status

Status: **design settled (2026-06-28); Phase A DONE.** This is the authoritative
Track 2 record. Track 1 (the portable, validated quimb pipeline on `main`) is the
default + correctness anchor and stays byte-for-byte unchanged and cuQuantum-free.
The full phase plan lives in the approved plan file `sharded-wishing-blossom.md`;
this doc carries the design rationale + per-phase evidence.

Track 2 is the **HPC-only** track: it exists to push **precision** and run **large
heavy** jobs by squeezing NVIDIA hardware. It does **not** reuse Track 1's
sequential fold-then-compress (1D-MPS-in-time). It lays the whole separable-bath
EDM out as a **2D space×time tensor network** (paper Sec. V) and hands it to
cuTensorNet.

## Firm decisions (immutable for Track 2)

1. **Forced binding: NVIDIA GPU + cuQuantum (cuTensorNet) + 2D network.** Not
   configurable; never imported on the Track-1 (CPU/Win/Mac) path.
2. **cotengra excluded; cuTensorNet owns the pipeline** — contraction-path search,
   slicing, hardware scheduling / device & memory management, execution. Rationale:
   one data structure end-to-end + cuTensorNet's optimizer is hardware-aware
   (co-optimizes path + slice count against device memory / NVLink). Escape hatch
   (API-level only): feed a precomputed path to cuTensorNet's *executor* if its
   optimizer underperforms on our 2D net.
3. **2D network, one-shot whole-spacetime preferred.** Slicing + scheduling +
   resource management are cuTensorNet's job — the one-shot whole-spacetime network
   is exactly what it should slice/schedule. A **manual time-window blocking /
   bounding** mode (windowed 2D with a compressed boundary between windows) is
   retained, but it is **user-invoked, never auto-triggered** (see decision 6).
4. **Two levers, separate success criteria.** 2D + cuTensorNet = **precision +
   global-optimization** (less/deferred truncation, globally good order).
   Single-node multi-GPU = **capacity** (slice the big contraction across cards;
   the linear temporal-bond growth is physics — no order avoids it).
5. **Multi-GPU mechanism decided, no spike:** cuTensorNet **distributed slicing** —
   one MPI rank per GPU + automatic slice distribution. MPI-based ⇒ single-node
   4×A800 and multi-node are the same code path, different launch geometry ⇒
   cross-node is a near-free stub.
6. **No silent guard — fail loud, user decides.** Unlike Track 1's silent rSVD
   resolution guard, Track 2 **raises an explicit, descriptive error** on any
   failure of the one-shot path — *including but not limited to* the precision
   target (ξ) being unreachable, slicing failing, or OOM — telling the user they
   may need to **manually construct time-window blocking**, then leaving the
   decision to them. There is **no automatic fallback** from one-shot to windowed,
   and **no silent acceptance** of worse-than-requested precision.
7. **Parameters flow through quimb.** `cutoff` / `cutoff_mode` / `max_bond` and the
   other knobs are passed via **quimb's existing unified API**, which dispatches to
   the cuQuantum/cuTensorNet backend (quimb already supports it). A parameter is
   passed **directly** to cuTensorNet (optionally bypassing quimb) **only** when, as
   with Track 1's CholQR2 `q`, quimb does not expose an otherwise-hidden knob we
   need. Default = through quimb; direct = the documented exception.

## Shared seam with Track 1

Track 2 shares **only the frontend / physics layer**: `models/`, `cumulants/`,
`kernels/` (tensor *construction*) and observable *definitions*. It does **not**
reuse `QuimbEDM` (Track 1's 1D-tagged TN for `tensor_network_1d_compress`); it has
its **own assembler** that lays the same physics tensors out as the 2D network.
That network may be carried as a quimb `TensorNetwork` so the contraction +
truncation + parameter plumbing reuse quimb's API per decision 7.

## Capacity wall (why Track 2 exists)

Measured on the A800 with the bond **uncapped** (`docs/gpu-scaling-benchmark.md`):
capacity is a **real, near-term hard wall**. Gaudin's Hamiltonian is
time-independent ⇒ infinite memory time ⇒ the EDM bond **grows without bound with
`T`** (191 → 643 from T3 → T6 at ξ=1e-8; asymptotically linear in `T`). Peak memory
∝ `n_sites·χ²`, so a single 80 GB A800 **cannot run K=24 at T=9, ξ=1e-8 (OOM)** or
T=6, ξ=1e-10. This is the **capacity milestone** for Phase C: that problem must
complete across 4×A800. The 2D representation does not dodge the linear growth
(it's physics); multi-GPU slicing is the capacity lever, the 2D global contraction
is the precision lever.

## Phase A — DONE (2026-06-28): install/interop de-risk + 2D assembler

- **What was built.** `examples/track2_2d_assembler.py` builds the 2D network from
  the shared physics layer as a backend-agnostic `(operands, integer-modes)` einsum
  description, with NumPy and cuTensorNet backends behind one interface.
  `examples/track2_cutensornet_sanity.py` + `cluster/track2_sanity.sbatch` run it on
  c1.
- **Network geometry (Gaudin / separable).** `(1 system row + K bath rows) × T`
  columns: the system row threads the d²=4 `vec` bond (right end = `vec(ρ0)`, left
  end = the free `vec(ρ(T))` output); each sub-bath row is a uniform chain of
  transfer tensors with lateral bond `D_a`=4 (boundaries fixed to 0); the column
  index (d_phys=7 superoperator leg) threads system → sub-bath 1 → … → sub-bath K
  through the picking tensor (fused into each row's `op`), with the top arm closed
  by δ⁰. Reducing all arms + contracting `vec(ρ0)` gives `vec(ρ(T))`.
- **Install.** `cuquantum-python-cu12` **26.3.2** (cuTensorNet binding **2.12.2** /
  21202) into the `edmtn-gpu` env on c1 (cupy 14.1.1, CUDA 12.9, A800-80GB).
- **Validation.** The cuTensorNet one-shot contraction reproduces Track 1's exact
  fold to **≤ 2.4e-15** across order 1/2, K=1–4, varying `n_steps`, and a sub-baths
  subset (also validated locally with NumPy einsum first — geometry de-risked off
  the cluster). Job 46486 on c1: PASS.
- **cuQuantum 26.x API surface (for B/C/D).** High-level namespace is
  `cuquantum.tensornet`: `contract`, `contract_path`, `einsum`, **`Network`**,
  **`OptimizerOptions` / `PathFinderOptions` / `SlicerOptions` / `ReconfigOptions`**
  (path + slicing control), **`experimental`** (approximate-TN / `contract_decompose`
  for truncation), `tensor.decompose` (SVD/QR), **`get_mpi_comm_pointer`** (MPI
  distributed). Low-level under `cuquantum.bindings.*`. NOTE: `cuquantum.cutensornet`
  and top-level `cutensornet` are gone; `decompose` needs a `QRMethod`/`SVDMethod`
  object, not the `"QR"` string; the **login node cannot `import cuquantum`** (no
  CUDA libs) — test only under sbatch on c1.

## Phase B — single-GPU full 2D contraction into `src/` (next)

The precision / global-optimization win. Promote the assembler into `src/` behind a
Track-2 flag; cuTensorNet owns path + slicing + execution; one-shot whole-spacetime
preferred with the manual time-window mode wired; truncation via cuTensorNet's
approximate contraction with the unified `cutoff`/`cutoff_mode` knobs (decision 7),
validated `<ξ` vs the Track 1 baseline.

**B0 (verify first, on c1).** Settle how the network is driven through quimb so
that **cuTensorNet — not cotengra — owns path-finding** (decision 2 vs 7): confirm
whether quimb's contraction/compression with the cuTensorNet backend lets
cuTensorNet's optimizer own the path, or whether quimb pulls in cotengra for path
order. If the latter, expose cuTensorNet path ownership **directly** for that one
knob (the CholQR-q precedent) and keep everything else through quimb. Also confirm
the `cutoff`/`cutoff_mode` → cuTensorNet truncation mapping (does quimb's
approximate-contraction cutoff convention thread to `contract_decompose`/the
`experimental` MPS cutoff, or must a mode be mapped explicitly).

## Phase C — single-node multi-GPU = cuTensorNet distributed slicing

cuTensorNet distributed (MPI rank/GPU on 4×A800, auto slice distribution over
NVLink); launch via `sbatch` + MPI. **Capacity milestone:** K=24 / T=9 / ξ=1e-8
(1-card OOM) completes across 4 cards at `<ξ`. On failure (slice/OOM/precision),
the explicit error of decision 6 applies.

## Phase D — cross-node interface stub (埋伏笔)

Feature-flagged, detect-only MPI/NCCL seam (`backend/process_group.py` or
equivalent): single-node works; multi-node geometry detects-and-reports-unavailable
("deferred — no test hardware"). Mirrors `OzakiGEMMBackend`. No multi-node
execution.

## Hardware notes (kept)

- **c1:** 2× AMD 7763 (256 threads), 512 GB RAM, **4× A800-SXM-80GB, NVLink**
  (GPU0-1 NV4, others NV2). The Track 2 test node. CPU baselines on **a8/a9** (dual
  EPYC 9754).
- **FP64 Tensor Cores (parked):** A800 has native FP64 DMMA (exact, auto via
  `CUBLAS_COMPUTE_64F`). **ZGEMM (complex128) TC dispatch is unverified** — the EDM
  is complex; measure, don't assume. Ozaki/ADP FP64-TC emulation stays parked.

## Risks / unknowns to verify in Track 2

- **quimb ↔ cuTensorNet path ownership** (B0) — keep cotengra out per decision 2
  while routing params through quimb per decision 7; resolve empirically.
- **cutoff/cutoff_mode → cuTensorNet truncation** mapping for 2D approximate
  contraction (B0).
- **One-shot feasibility vs windowing** at the capacity target — the explicit-error
  path (decision 6) is the contract when one-shot can't fit even after slicing.
- **ZGEMM tensor-core dispatch** (complex128) — measure.
- **cuTensorNet distributed** determinism / complex128 collectives at multi-GPU.

## Bottom line

Track 2 = HPC-only, **2D space×time** EDM contracted **one-shot by cuTensorNet**
(no cotengra; cuTensorNet owns path/slice/schedule/execute), parameters routed
**through quimb**, failures **raised explicitly** (no silent guard; manual windowing
is the user's call). Precision is the 2D lever; capacity is the multi-GPU lever
(cuTensorNet distributed slicing, single-node first, cross-node a cheap stub).
Phase A (install/interop + the 2D assembler) is **done and validated `≤2.4e-15`**
against Track 1. Track 1 stays the untouched, portable, cuQuantum-free reference.
