# edmtn

**Simulate how a small quantum system loses coherence to its environment — fast.**

`edmtn` computes the time evolution of a quantum system (e.g. a single spin)
coupled to a noisy bath, and returns observables like the polarization
`⟨S_z(t)⟩` and the density matrix `ρ(t)`. It implements the polynomial-complexity
method of Chen & Liu, *Polynomial complexity of open quantum system problems*
([arXiv:2509.00424](https://arxiv.org/abs/2509.00424)): the EDM representation has
**polynomial** rather than exponential complexity in the evolution time — in the
regimes studied here the temporal bond grows roughly linearly with `T` — so you can
reach long times and many bath modes on ordinary hardware. (Total run cost is not
simply linear in `T`; see [docs/benchmarks/](docs/benchmarks/).)

You don't need to know tensor networks to use it — pick a model, call `solve(...)`,
read the result. Tuning and internals are there when you want them.

```python
from edmtn.driver import solve
from edmtn.models import GaudinModel

res = solve(GaudinModel(g=1.0, K=3), T=0.3, eps=0.1, channel=3)  # central spin in 3 bath spins
print(res.times, res.polarization)   # t = eps..T,  <S_z(t)>
```

That runs on CPU in a second, anywhere, with nothing extra installed. For the
paper-scale configuration (`K=49`, `T=15`) see
**[Paper-scale runs](#paper-scale-runs)** — same API, much larger compute.

## What it can do

- **Two ready-made models** — *spin-boson* (a spin in a bosonic/Ohmic bath) and
  *Gaudin* (a central spin in `K` bath spins). Both are checked against small
  dense/exact references at the tolerances stated in the tests (uncompressed paths
  agree to ~1e-10 or better). Compressed paths are checked only at the specific
  parameters each test covers, against that test's stated observable/state tolerance —
  `cutoff` is a **local per-bond** truncation threshold, **not** an error bound on the
  polarization, `ρ(t)`, or the trajectory.
- **Three backends, one API** — `cpu` (default, runs everywhere), `gpu` (one NVIDIA
  card), `hpc` (the GPUs/MPI ranks you launch it across, exact contraction,
  separable/Gaudin only). Just change `backend=...`; see
  **[Which backend?](#which-backend)**.
- **Observables out of the box** — `⟨S_z(t)⟩` (or any coupling channel), `ρ(t)`
  (where the pipeline exposes it — see the result table), bond-dimension growth,
  a real truncation metric, and convergence checks.

## Install

```bash
pip install -e .          # from the repo root; or just put src/ on PYTHONPATH
```

CPU works immediately (NumPy). For a GPU add a CuPy wheel matching your CUDA
(`pip install cupy-cuda12x`); for the multi-GPU `hpc` track add
`cuquantum-python-cu12`. See **[Environment](#environment)**.

## First run (copy-paste)

```python
from edmtn.driver import solve
from edmtn.models import SpinBosonModel, GaudinModel

# spin-boson (a spin in a Gaussian bosonic bath)
sb = SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0)
res = solve(sb, T=0.3, eps=0.1, expansion_order=2, cutoff=1e-8)
print(res.times)              # t = eps, 2 eps, ..., T
print(sb.mu * res.times)      # the same grid in dimensionless mu*t, if you want it
print(res.polarization)       # <S_z(t)>

# Gaudin (a central spin in K bath spins); channel 3 = <S_z>
g = GaudinModel(g=1.0, K=3)
res = solve(g, T=0.3, eps=0.1, expansion_order=2, cutoff=1e-8, max_bond=64, channel=3)
print(res.polarization)              # <S_z(t)>
print(res.sub_bath_counts,           # L = 1..K, the sub-bath fold axis ...
      res.sub_bath_bond_dims)        # ... and the bond dimension D_L after each fold
print(res.final_time_bond_dims)      # the final MPS's internal bonds along the time chain
```

**The time grid.** `T / eps` must be a *positive integer* (to a small tolerance) — it is
**not** silently rounded; a non-integer ratio raises `ValueError`. Every public `times`
array is `[eps, 2 eps, ..., T]` on **all** pipelines (spin-boson, Gaudin Track 1, Track 2),
so results are directly comparable.

## What you get back

`solve(...)` returns a `SolverResult`. Each array names its own axis, so you never need to
reach into `res.evolution.*`:

| field | axis | what it is |
|---|---|---|
| `res.times` | time | `[eps, 2 eps, …, T]` |
| `res.polarization` | ∥ `times` | `⟨S_channel(t)⟩` (pick `channel=`) |
| `res.density_matrices` | ∥ `times` | `ρ(t)`, or `None`. Track 2: always present. Spin-boson: present whenever reduced states were recorded — `record_rho=True`, custom observables, **or second order** (which needs them anyway). **Gaudin Track 1: always `None`** — see the next two rows |
| `res.sub_bath_counts` | fold `L` | Gaudin Track 1: the recorded sub-bath counts |
| `res.sub_bath_bond_dims` | ∥ `sub_bath_counts` | `D_L` after folding in `L` sub-baths |
| `res.sub_bath_final_density_matrices` | ∥ `sub_bath_counts` | Gaudin Track 1: `ρ_L(T)` — a *final-time* state per `L`, **not** a time history (needs `record_rho=True`) |
| `res.time_bond_dims` | ∥ `times` | max bond after each physical step (single-bath Track 1) |
| `res.final_time_bond_dims` | MPS bonds | the final MPS's internal bonds along the time chain (length `num_sites-1`); `None` on Track 2 |
| `res.truncation_errors` | pipeline axis | real truncation metric — see **[Truncation metric](#truncation-metric)** |
| `res.sub_baths_used` | — | how many sub-baths were *actually* folded (`None` if N/A) |
| `res.expansion_order` | — | the Trotter order actually used (`1` or `2`) |
| `res.observables` | ∥ `times` | custom observable histories (single-bath only; separable/Track 2 raise `NotImplementedError`) |
| `res.error_metrics` | — | (`hpc` only) `‖ρ−ρ†‖`, `|Tr ρ−1|`, optimizer slice/flop counts |
| `res.backend` | — | the device/track that **actually** ran (a failed GPU request shows as `cpu/... (fallback: …)`) |
| `res.mps` / `res.evolution` | — | the final EDM-MPS / raw Layer-5 output; **both `None` on Track 2** |
| `res.bond_dims`, `res.max_bond` | pipeline axis | *legacy* alias: `time_bond_dims` on single-bath, `sub_bath_bond_dims` on separable, `[]` on Track 2. Prefer the axis-explicit fields above |

## Which backend?

Change one argument, `backend=`. Same model and same time grid — but **not** bit-identical
numbers: the tracks do different algebra.

| `backend` | runs on | use it when |
|---|---|---|
| `cpu` *(default)* | any machine, NumPy | development, small/medium problems — works out of the box, no GPU needed |
| `gpu` | one NVIDIA GPU, CuPy | larger problems where the bond dimension is big |
| `hpc` | the NVIDIA GPUs / MPI ranks you launch it across, cuQuantum | no tensor-network truncation; the largest jobs. **Separable/Gaudin only** |

**Rule of thumb:** start with `cpu`; switch to `gpu` if you have one NVIDIA card
and the run is slow; use `hpc` when you want an untruncated contraction or need to push a
job that's too big for one card.

`cpu` and `gpu` are **Track 1** — they compress the tensor network (truncate small
singular values, controlled by `cutoff`/`max_bond`), which scales to long times and
many bath spins. `hpc` is **Track 2** — it lays the whole problem out as a 2D
space×time network and contracts it with cuQuantum (cuTensorNet) with **no truncation
knobs** and automatic multi-GPU slicing. Track 2 currently supports **only the
separable/Gaudin pipeline**; spin-boson is not available there.

**What "exact" means on Track 2.** It means *no tensor-network truncation* of the
discretised problem. It does **not** remove the finite-`eps` and expansion-order error
(both tracks share those), and it does not imply higher floating-point precision than
Track 1 — it reports `error_metrics` (`‖ρ−ρ†‖`, `|Tr ρ−1|`) precisely because
floating-point error remains. Differences between backends come from truncation settings,
decomposition choice and precision (`precision='mixed'` contracts in f32); CPU/GPU parity
is validated to the tolerances stated in the tests, not bit-for-bit.

### Running on `hpc` (multi-GPU)

`hpc` uses exactly the GPUs/ranks **you launch it across** — cuTensorNet's only
multi-GPU mode is **one MPI rank per physical GPU**, and it needs a **CUDA-aware MPI
runtime** (the distributed contraction runs MPI collectives directly on device buffers).
edmtn does *not* submit jobs, ssh, or call `srun`/`sbatch` for you; that's your workflow.

Requirements for the distributed path:

- one MPI rank per **distinct physical** GPU;
- a CUDA-aware MPI runtime, plus the cuTensorNet MPI wrapper (`CUTENSORNET_COMM_LIB`);
- `pathfinder='cuquantum'` (the default) — **`'cotengra'` is not supported multi-rank**;
- an mpi4py whose MPI ABI matches the launcher you use.

The exact launcher is **site-specific**; use the recipes in [`cluster/`](cluster/) rather
than copying a command from here. [`cluster/test_gpu_hpc.sbatch`](cluster/test_gpu_hpc.sbatch)
is the **current test recipe and status record** — its single-GPU steps are verified on
hardware; its 4-GPU step is not currently passing (see status below).

Your script is unchanged — just `backend='hpc'`:

```python
res = solve(GaudinModel(g=1.0, K=12), T=0.6, eps=0.1, channel=3, backend='hpc')
res.density_matrices   # ρ(t)
res.error_metrics      # {hermiticity, trace_dev, num_slices, flops}
res.backend            # e.g. 'hpc/exact/cuquantum/4gpu'
```

If you run `hpc` on a single GPU (or the problem is small enough to fit one card),
edmtn warns and suggests scaling up or using Track 1 — it still works, it just isn't
where `hpc` pays off.

**Current hardware-validation status** (see
[docs/design/multi-gpu-cuquantum-design.md](docs/design/multi-gpu-cuquantum-design.md)):
Track 1 single-GPU parity and Track 2 single-GPU cuTensorNet are **verified on real
hardware**. The 4-GPU distributed re-run is currently **blocked by a CUDA-aware-MPI
regression in the test cluster's environment** — a site environment issue, not a defect in
the model or the distributed pipeline. An older job passed the 4-GPU path historically;
that is a historical record and does **not** constitute current-environment acceptance.

## Package layout

```
edmtn/
├── pyproject.toml              # package + pytest configuration
├── src/edmtn/                  # the package, organised by layer
│   ├── backend/                # Layer 0: array/linalg backend (NumPy/CuPy via autoray) + GPU compat shims
│   ├── models/                 # Layer 1: physical models (spin-boson, Gaudin)
│   ├── cumulants/              # Layer 2: bath cumulant engines (Gaussian, separable)
│   ├── kernels/                # Layer 3: combined-kernel MPO construction
│   ├── expansion/              # Layer 4: 1st/2nd-order time-step expansion
│   ├── evolution/              # Layer 5: quimb-backed evolution + compression
│   │   ├── quimb_edm.py        #   QuimbEDM: the EDM carried as a quimb TensorNetwork
│   │   ├── quimb_decomp.py     #   rSVD(q)+guard split driver + canonicalisation selector
│   │   ├── separable_bath.py   #   Gaudin outer-loop fold engine
│   │   ├── single_bath.py      #   spin-boson forward-recursion engine
│   │   └── mps_utils.py        #   EDMMPS container, apply_step, dense brute-force reference
│   ├── observables/            # Layer 6: observable extraction + convergence
│   └── driver/                 # Layer 7: orchestration (EDMSolver, SolverConfig, presets)
├── examples/                   # reproduce/ (paper figs), research/ (physics studies), track2/ (cuQuantum), smoke/
├── tests/                      # unit / integration; benchmarks/ holds perf_*.py scripts
└── docs/                       # reference/, design/, benchmarks/, research/, guides/, troubleshooting/
```

## Configuration (tuning knobs)

You only need `T`, `eps`, and maybe `channel` to get going. The rest are optional
knobs — `solve(model, *, T, eps, channel=1, **config)` (and `EDMSolver.from_model`)
accept:

`SolverConfig` is a **frozen dataclass validated at construction**: every knob is checked
and normalised up front (a huge int, `nan`/`inf`, a `bool` posing as an integer, an unknown
enum string → `ValueError` right there, not deep inside quimb), and the instance is
immutable afterwards (use `dataclasses.replace` for a variant).

| knob | values (default) | meaning |
|---|---|---|
| `expansion_order` | `1`, `2`, `None` (**`None`**) | Trotter order. `None` **inherits the model's `time_step_order`**; an explicit value overrides it. Resolved once, then used by every layer and reported as `res.expansion_order` |
| `cutoff` | float ≥ 0 (`1e-8`) | truncation threshold `ξ` |
| `cutoff_mode` | `abs`, `rel`, `sum2`, `rsum2`, `sum1`, `rsum1` (`rel`) | quimb-native truncation rule (`rel` = `s_i/s_max ≤ ξ`). The paper's custom `rel_ref` rule — and the reference-index parameter it needed — are retired |
| `max_bond` | int > 0 or `None` (`None`) | hard bond-dimension cap |
| `record_rho` | bool (`False`) | store `ρ(t)`. Some paths record it anyway (2nd-order spin-boson, custom observables) |
| `compress_method` | `zipup`, `dm`, `direct` (`zipup`) | quimb 1D-compress algorithm. *Track 1 only* |
| `compress_decomp` | `exact`, `rsvd` (`exact`) | per-bond decomposition (`rsvd` = randomized SVD + guard). *Track 1 only* |
| `compress_decomp_q` | int ≥ 0 (`2`) | rSVD power iterations (`2` = cold, `0` = single-pass). *Track 1 only* |
| `compress_canon` | `quimb`, `householder`, `cholqr` (`quimb`) | canonicalisation QR. *Track 1 only* |
| `preset` | `balanced`, `robust`, `None` (`None`) | only fills `compress_decomp`/`compress_decomp_q` when left at defaults; Track 1 only. An unknown name is rejected on every backend |
| `sub_baths` | int > 0 or `None` (`None`) | separable: fold only the first `L` sub-baths **in the model's stored coupling order** (Fig. 6). Re-checked as `1 ≤ L ≤ K` once `K` is known — never silently clamped |
| `backend` | `cpu`, `numpy`, `gpu`, `cupy`, `hpc` (**`cpu`**) | `cpu`/`numpy` and `gpu`/`cupy` = Track 1; `hpc` = Track 2. There is **no `auto`** |
| `precision` | `f64`, `mixed` (`f64`) | **experimental.** `mixed` casts the Track-1 contraction tensors to the f32/complex64 path. A separate f64 re-cast before decomposition is *declared* by `PrecisionPolicy` but is **not wired into the solve pipeline**, and the end-to-end mixed-precision check is still skipped — so treat results as unvalidated |
| `pathfinder` | `cuquantum`, `cotengra` (`cuquantum`) | **`hpc` only** — who finds the contraction path. Distributed multi-GPU requires `cuquantum` |
| `time_windows` | `None` only (`None`) | **`hpc` only, reserved.** Manual window blocking is wired but **not implemented** — any non-`None` value raises `NotImplementedError` at construction |

`solve(...).backend` reports the device/track that actually ran.

### Choosing the channel

`channel` selects which coupling operator's polarization `⟨S_a(t)⟩` is returned. It is a
**strict 1-based integer**:

| model | valid `channel` | operator |
|---|---|---|
| `SpinBosonModel` | `1` only | `S_z` (its single coupling channel) |
| `GaudinModel` | `1`, `2`, `3` | `S_x`, `S_y`, `S_z` |

`0`, negative values, out-of-range values, floats, strings and `bool` all raise `ValueError` —
in particular `channel=0` is rejected rather than silently selecting the last operator by
Python negative indexing. The same contract is enforced by `EDMSolver.solve()`, by a direct
Track-2 call, and by the observable extractor.

### Checking time-step convergence

```python
from edmtn.driver import EDMSolver
from edmtn.models import GaudinModel

solver = EDMSolver.from_model(GaudinModel(g=1.0, K=3), T=0.3, eps=0.1,
                              expansion_order=2, cutoff=1e-8)
conv = solver.timestep_convergence(channel=3, tol=1e-3)

conv.deviation    # max |Δ⟨S_a(t)⟩| between the eps and eps/2 runs
conv.converged    # deviation <= tol, or None if no tol was given
conv.metadata     # full coarse/fine SolverConfig, channel, tolerance,
                  # the ACTUAL executed coarse/fine backend labels, and
                  # coarse_sub_baths_used / fine_sub_baths_used

dev, ok = conv    # still unpacks as the legacy 2-tuple
```

The fine run is derived with `replace(config, eps=eps/2)`, so it keeps **every** other field —
it cannot silently compare a different model.

**Errors you can get, and what they mean:**

| exception | meaning |
|---|---|
| `ValueError` | malformed input — a bad config value, an invalid `channel`, a malformed model, or an illegal argument to a direct `run()` |
| `NotImplementedError` | legal input, capability not implemented — non-zero temperature on the Gaussian engine, `time_windows`, spin-boson on Track 2, custom observables on separable/Track 2 |
| `FloatingPointError` | legal parameters whose correlation overflows float64 |

Note: `compress_method` is a quimb 1D-MPS-compress algorithm and applies to `cpu`/`gpu`
only (the `hpc` 2D contraction has no 1D-compress sweep).

**Compression.** Everything goes through quimb's `tensor_network_1d_compress`
(canonicalise + truncate in one sweep), executed via autoray on whatever backend the
arrays live on. `compress_method='zipup'` (default) is fast and low-memory;
`direct` is the exact SVD sweep; `dm` is the density-matrix method (`eigh`-based,
fastest but lower precision). `compress_decomp='rsvd'` swaps the per-bond full SVD for
a randomized SVD whose power-iteration count is `compress_decomp_q` (`2` cold, `0`
single-pass); a **silent resolution guard** falls back to exact full SVD when the
randomized result is under-resolved or the backend is not NumPy. The guard covers the
failure modes it detects, but rSVD is still a *randomized* algorithm — measured agreement
with full SVD is a benchmark result at stated tolerances, not a universal guarantee — and
it cannot report a truncation metric (see below).

**`compress` vs `cutoff=0` — not the same thing.** On the direct evolution API:

| setting | what happens |
|---|---|
| `compress=False` | compression is **skipped entirely**; exact, with exponentially growing bonds |
| `compress=True, cutoff=0` | an **exact recompression**: canonicalise + full SVD, nothing discarded |
| `compress=True, cutoff>0` (or `max_bond`) | genuinely truncating compression |

### Truncation metric

`res.truncation_errors` reports, per record point, the **largest per-bond discarded
weight**

    w_max = max_b ( Σ_{i discarded at bond b} σ_i² )

Note this is the discarded **weight** (`Σσ²`), not quimb's discarded 2-norm
(`√Σσ²`). For the density-matrix method the object split is `ρ`, whose eigenvalues are
`λ = σ²`, so the same quantity is `Σ λ_discarded` there.

| value | meaning |
|---|---|
| `0.0` | a compression ran and discarded nothing (or none ran, e.g. `compress=False`) |
| `> 0` | that record interval really truncated |
| `None` | **unmeasurable** with the chosen decomposition (`compress_decomp='rsvd'`, whose sketch never forms the omitted tail) — *not* "nothing was discarded" |
| `[]` | Track 2: exact-only, it performs no Track-1 compression |

Axis: one entry per **physical time step** for single-bath (order 2 takes the max over both
sub-steps), and one per **recorded `L`** for Gaudin Track 1 (the max over every fold since
the previous recorded `L`, so `record_every>1` drops nothing). It is a **local** per-record
quantity — not a cumulative trajectory error, and not a bound on observable error.

**Presets** (details in [docs/guides/recommended-config.md](docs/guides/recommended-config.md)):

```python
# preset='balanced' -> rsvd with q=0 (single-pass): the fastest of the two
res = solve(g, T=0.3, eps=0.1, expansion_order=2, cutoff=1e-8, max_bond=64,
            channel=3, preset='balanced')

# preset='robust'   -> rsvd with q=2 (cold): more power iterations, still randomized
res = solve(g, T=0.3, eps=0.1, expansion_order=2, cutoff=1e-8, max_bond=64,
            channel=3, preset='robust')

# no preset (THE DEFAULT) -> compress_decomp='exact': deterministic full SVD,
# and the only setting that can report a real truncation metric.
```

Both presets only set `compress_decomp='rsvd'` plus `compress_decomp_q` (`0` / `2`) —
neither is full SVD, neither touches `compress_canon`, and neither overrides `backend`.
They apply only when you left those fields at their defaults, and only on Track 1. If you
want full SVD, set `compress_decomp='exact'` (the API default) and use no preset. Accuracy
figures quoted in the guide are **measured on specific hardware and parameters**, not
general error guarantees.

The compute is backend-agnostic (autoray), so `cpu` and `gpu` share the same Track-1 physics
pipeline and public API, and agree to the tolerances asserted in the tests. They are not
guaranteed to execute identically: backend kernels and dtypes differ, and
`compress_decomp='rsvd'` **silently falls back to exact full SVD on any non-NumPy backend**, so
the same config can take a different decomposition path on GPU than on CPU. Agreement is stated
per test tolerance — not bit-for-bit, and not across different
`precision`/`compress_decomp` settings. For when to pick which — and how `hpc` differs
(no truncation knobs, multi-GPU, separable-only) — see
**[Which backend?](#which-backend)** above. Under the hood, the `hpc`
track lays the whole problem out as a 2D space×time network and contracts it exactly
with cuQuantum/cuTensorNet; design + status:
[docs/design/multi-gpu-cuquantum-design.md](docs/design/multi-gpu-cuquantum-design.md).

## Performance & design notes

- **Re-platform decision ledger** (what was replaced/retired and why):
  [docs/design/phase0-replatform-decisions.md](docs/design/phase0-replatform-decisions.md).
- **Recommended presets** (balanced vs robust):
  [docs/guides/recommended-config.md](docs/guides/recommended-config.md).
- **GPU scaling** (single A800 vs EPYC-9754):
  [docs/benchmarks/gpu-scaling-benchmark.md](docs/benchmarks/gpu-scaling-benchmark.md).
- **CPU vs GPU** trade-off: [docs/benchmarks/cpu-vs-gpu-edm.md](docs/benchmarks/cpu-vs-gpu-edm.md).
- **Distributed scale-out** (multi-GPU + cuQuantum, two-track design):
  [docs/design/multi-gpu-cuquantum-design.md](docs/design/multi-gpu-cuquantum-design.md).

## Environment

Developed against a `quimb` conda env (Python 3.14, quimb 1.14, autoray, cotengra,
NumPy 2.4). CuPy/GPU is **optional** — the default path is CPU NumPy; CPU-only on
Windows/macOS/Linux needs nothing extra.

On a CUDA machine, add a CuPy wheel matching your CUDA toolkit to enable
`backend='gpu'`, e.g. `pip install cupy-cuda12x` (CUDA 12.x). The GPU path applies
two small compatibility shims for quimb-on-CuPy automatically
(see [docs/troubleshooting/quimb-cupy-namespace-bug.md](docs/troubleshooting/quimb-cupy-namespace-bug.md)).

For `backend='hpc'` (NVIDIA only) also install `cuquantum-python-cu12`. Multi-GPU
needs an MPI launcher (`srun`/`mpirun`) and the cuTensorNet MPI wrapper — the
`cluster/` launch scripts set the required env (`CUTENSORNET_COMM_LIB`,
`LD_PRELOAD`); see [docs/design/multi-gpu-cuquantum-design.md](docs/design/multi-gpu-cuquantum-design.md).
None of this is imported on the CPU / Track-1 path, so CPU-only installs stay clean.

On some Windows quimb envs, set `MKL_THREADING_LAYER=TBB` before NumPy is imported
to avoid an OpenMP-runtime clash (`conda env config vars set MKL_THREADING_LAYER=TBB
-n quimb`); see [docs/troubleshooting/mkl-tbb-threading-layer.md](docs/troubleshooting/mkl-tbb-threading-layer.md).

## Running the tests

```
cd edmtn
PYTHONPATH=src python -m pytest -q       # fast unit suite (integration deselected)
PYTHONPATH=src python -m pytest -m integration   # end-to-end checks (slower)
```

GPU/HPC tests are gated by **real hardware detection**, not unconditional skips. They are
collected everywhere and skip with a specific reason (e.g. `gpu: CuPy not importable`) when
the hardware or stack is absent:

```
pytest -q -m "gpu and not cuquantum and not multigpu" --require-gpu
pytest -q -m "cuquantum and not multigpu"             --require-gpu --require-cuquantum
pytest -q -m multigpu --require-gpu --require-cuquantum --require-multigpu=4
```

The `--require-*` flags make a **missing** stack exit non-zero instead of quietly skipping,
so an all-skipped run can never masquerade as hardware acceptance. `--require-multigpu=N`
additionally requires the distributed worker's result JSON (`EDMTN_MULTIGPU_RESULT`) to
exist and be readable. See [`cluster/test_gpu_hpc.sbatch`](cluster/test_gpu_hpc.sbatch).

`pyproject.toml` puts `src/` on the path via `[tool.pytest.ini_options].pythonpath`,
so no install is required; alternatively `pip install -e .`.

## Paper-scale runs

The examples above are deliberately tiny so they finish instantly. The paper's
configurations are much larger — same API, very different compute:

```python
# Gaudin, paper scale: K=49 bath spins to T=15 (500 steps, bond capped at 400).
# Minutes-to-hours on CPU depending on cutoff; this is where `backend='gpu'` earns its keep.
res = solve(GaudinModel(g=1.0, K=49), T=15.0, eps=0.03, expansion_order=2,
            cutoff=1e-8, max_bond=400, channel=3)

# spin-boson, paper scale
sb = SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0)
res = solve(sb, T=8.0, eps=0.02, expansion_order=2, cutoff=1e-8)
```

Start small, check `res.truncation_errors` and `timestep_convergence()` before scaling up,
and remember `T/eps` must be an exact integer. Ready-to-run reproductions live in
`examples/reproduce/`.

## Examples

```
python examples/reproduce/reproduce_fig4.py --quick   # spin-boson Fig. 4a/4b
python examples/reproduce/reproduce_fig6.py --quick   # Gaudin Fig. 6a/6b
python examples/smoke/retire_gpu_smoke.py             # GPU node: validate the pipeline on CuPy
```

Performance scripts live in `tests/benchmarks/` (named `perf_*.py` so pytest does not
collect them); run them directly.
