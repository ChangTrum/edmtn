# edmtn

Extended Density Matrix (EDM) tensor-network solver for non-Markovian open
quantum systems, built on the **quimb + cotengra + autoray** ecosystem (with
optional **CuPy** GPU execution).

It implements the polynomial-complexity EDM formalism of Chen & Liu,
*Polynomial complexity of open quantum system problems* (arXiv:2509.00424): the
reduced dynamics of a small system coupled to a quantum bath is represented by an
extended-density-matrix tensor network whose bond dimension grows only linearly in
evolution time, propagated by a combined-kernel MPO built from the bath cumulants.

## Status

**Both demonstrator models solved end-to-end and validated** against exact
references (machine precision uncompressed; `⟨S_z(t)⟩` reproduced under
compression):

- **spin-boson** — a spin in a Gaussian bosonic bath (single-bath forward
  recursion).
- **Gaudin** — a central spin in `K` independent bath spins, a *separable*
  non-Gaussian bath (the Eq. 21 outer-loop recursion; `sub_baths` folds the first
  `L` spins, Fig. 6).

**Re-platformed onto quimb (complete).** The compression pipeline is now a single
ecosystem path: the EDM is carried as a quimb `TensorNetwork` and compressed via
`tensor_network_1d_compress` (canonicalisation + truncation), dispatched through
**autoray** so the *same code runs on NumPy and CuPy*. The previous hand-rolled
"native" path (bespoke SVD/QR sweeps, `StandardSVD`/`RandomizedSVD`, the
`s_i/s_{d²+1}` `rel_ref` cutoff) has been **retired** — see
[docs/phase0-replatform-decisions.md](docs/phase0-replatform-decisions.md). Validated
on CPU (full suite) and on a single A800 GPU (CPU↔GPU agreement ~1e-13).

**Next: cuQuantum + single-node multi-GPU** (cuTensorNet decomposition/contraction,
then cotengra slicing across multiple GPUs — the capacity lever for long evolution).
See [docs/multi-gpu-cuquantum-design.md](docs/multi-gpu-cuquantum-design.md).

What's implemented:

- spin-boson model (generalised Ohmic bath, zero temperature) and Gaudin model
  (linearly-decreasing couplings, infinite-temperature spin bath);
- Gaussian second-order cumulants (closed-form Ohmic correlation + numeric
  cross-check) **and** separable bath-correlation transfer tensors (Eq. F1);
- combined-kernel MPO (Gaussian and separable), **first- and second-order**
  time-step expansion (second order on a doubled sub-step grid);
- **quimb-backed compression** — canonicalise + truncate in one call, with a
  choice of method (`zipup`/`dm`/`direct`), per-bond decomposition (exact full SVD
  or randomized SVD with a silent resolution guard), and canonicalisation QR;
- MPS evolution engines: single-bath forward recursion **and** the separable
  outer-loop over sub-baths;
- observable extraction: reduced density matrix, single-time expectations, and the
  all-times coupling-channel polarization (Eq.-F2 / Eq.-F3 environment sweep);
- a driver that wires the pipeline from `bath_type` and selects the compute
  backend, plus convergence helpers.

## Layout

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
├── examples/                   # reproductions / studies (reproduce_fig4.py, reproduce_fig6.py, …)
├── tests/                      # unit / integration; benchmarks/ holds perf_*.py scripts
└── docs/                       # design decisions, benchmarks, environment notes
```

## Quickstart

```python
from edmtn.driver import solve
from edmtn.models import SpinBosonModel, GaudinModel

# spin-boson (Gaussian bath)
sb = SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0)
res = solve(sb, T=8.0, eps=0.02, expansion_order=2, cutoff=1e-8)
res.times, res.polarization, res.bond_dims   # mu*t, <S_z(t)>, bond dim per step

# Gaudin (separable spin bath); channel 3 is <S_z>
g = GaudinModel(g=1.0, K=49)
res = solve(g, T=15.0, eps=0.03, expansion_order=2, cutoff=1e-8,
            max_bond=400, channel=3)
res.times, res.polarization   # t, <S_z(t)>  (res.mps.bond_dims is D_t)
```

## Configuration

`solve(model, *, T, eps, channel=1, **config)` (and `EDMSolver.from_model`) accept:

| knob | values (default) | meaning |
|---|---|---|
| `expansion_order` | `1`, `2` (`1`) | Trotter order (2 = doubled sub-step grid) |
| `cutoff` | float (`1e-8`) | truncation threshold `ξ` |
| `cutoff_mode` | `rel`, `rsum2`, `abs`, … (`rel`) | quimb truncation rule (`rel` = `s_i/s_max < ξ`; the built-in closest to the retired `rel_ref`) |
| `max_bond` | int or `None` (`None`) | hard bond-dimension cap |
| `compress_method` | `zipup`, `dm`, `direct` (`zipup`) | quimb 1D-compress algorithm |
| `compress_decomp` | cpu/gpu: `exact`,`rsvd` (`exact`) | per-bond decomposition (`rsvd` = randomized SVD + guard). *N/A under `hpc`* (Track 2 is exact-only) |
| `compress_decomp_q` | int (`2`) | rSVD power iterations (`2` = cold/robust, `0` = single-pass/fast). *N/A under `hpc`* |
| `compress_canon` | `quimb`, `householder`, `cholqr` (`quimb`) | canonicalisation QR. *N/A under `hpc`* |
| `preset` | `balanced`, `robust`, `None` (`None`) | fills the recommended decomposition (cpu/gpu only) |
| `sub_baths` | int or `None` (`None`) | separable bath: fold/contract only the first `L` sub-baths (Fig. 6) |
| `backend` | `cpu`, `gpu`, `hpc` (`cpu`) | `cpu`/`gpu` = Track 1 (NumPy/CuPy); `hpc` = Track 2, the cuQuantum 2D contraction |
| `pathfinder` | `cuquantum`, `cotengra` (`cuquantum`) | **`hpc` only** — who finds the contraction path (default: cuTensorNet owns it) |
| `time_windows` | int or `None` (`None`) | **`hpc` only** — `None` = one-shot whole-spacetime; int = manual window blocking |

`solve(...).backend` reports the resolved device/track.

Note: `compress_method` is a quimb 1D-MPS-compress algorithm and applies to `cpu`/`gpu`
only (the `hpc` 2D contraction has no 1D-compress sweep).

**Compression.** Everything goes through quimb's `tensor_network_1d_compress`
(canonicalise + truncate in one sweep), executed via autoray on whatever backend the
arrays live on. `compress_method='zipup'` (default) is fast and low-memory;
`direct` is the exact SVD sweep; `dm` is the density-matrix method (`eigh`-based,
fastest but lower precision). `compress_decomp='rsvd'` swaps the per-bond full SVD for
a randomized SVD whose power-iteration count is `compress_decomp_q` (`2` cold, `0`
single-pass); a **silent resolution guard** falls back to exact full SVD when the
randomized result is under-resolved, so it is never less reliable than full SVD.

**Presets** (details in [docs/recommended-config.md](docs/recommended-config.md)):

```python
# preset='balanced' -> single-pass rSVD (q=0): fastest, accuracy < ξ
res = solve(g, T=15.0, eps=0.03, expansion_order=2, cutoff=1e-8, max_bond=400,
            channel=3, backend='gpu', preset='balanced')

# preset='robust'   -> cold rSVD (q=2): exact-baseline accuracy
res = solve(g, T=15.0, eps=0.03, expansion_order=2, cutoff=1e-8, max_bond=400,
            channel=3, preset='robust')

# no preset (default) -> exact full SVD, deterministic.
```

A preset only fills `compress_decomp`/`compress_decomp_q` if you left them at the
default; it never overrides `backend`.

**Backend.** `cpu` (default) runs on NumPy; `gpu` runs on CuPy (needs a matching CuPy
wheel — see *Environment*). GPU pays off once the bond is large; small problems are
CPU-competitive. The compute is backend-agnostic (autoray), so results agree across
devices to round-off.

**HPC track (`backend='hpc'`).** A separate, NVIDIA-GPU-only track for **precision and
multi-GPU capacity**. Instead of Track 1's sequential fold-then-compress, it lays the
whole separable-bath EDM out as a **2D space×time tensor network** (paper Sec. V) and
contracts it **exactly, in one shot, with cuQuantum (cuTensorNet)**, which owns path
search, slicing, hardware scheduling, and execution (cotengra stays a selectable
fallback path-finder). This is the **exact route only** — genuinely no truncation, no
knobs; the 2D framing buys a far larger contraction-order search and **native multi-GPU
slicing** (one MPI rank per GPU) for the exponentially-growing exact contraction. The
**truncated/approximate regime stays in Track 1** (`backend='cpu'`/`'gpu'`), whose
quimb fold already scales to large N/K — cuTensorNet's MPS-method adds nothing there
(single-GPU, and it breaks past ~20 time steps), so the Track-1 truncation knobs
(`compress_decomp`, `cutoff`, `cutoff_mode`, `max_bond`, …) are **N/A under `hpc`**.
The **density operator ρ(t) is returned first-class** (`result.density_matrices`), the
channel polarization is derived only if `channel` is given, and **error metrics** are
reported (`result.error_metrics`: ‖ρ−ρ†‖, |Tr ρ−1|, optimizer slices/flops). Needs
`cuquantum-python-cu12`; never imported on the Track-1 (CPU/Win/Mac) path. Design +
status: [docs/multi-gpu-cuquantum-design.md](docs/multi-gpu-cuquantum-design.md).

```python
# exact 2D contraction; cuTensorNet owns path+slicing; ρ(t) + error metrics returned
res = solve(GaudinModel(g=1.0, K=12), T=0.6, eps=0.1, channel=3, backend='hpc')
res.density_matrices, res.error_metrics      # ρ(t), {hermiticity, trace_dev, num_slices, …}

# multi-GPU (4 ranks, one GPU each); cuTensorNet distributes the slices:
#   srun --mpi=pmi2 --ntasks=4 python your_script.py
```

## Performance & design notes

- **Re-platform decision ledger** (what was replaced/retired and why):
  [docs/phase0-replatform-decisions.md](docs/phase0-replatform-decisions.md).
- **Recommended presets** (balanced vs robust):
  [docs/recommended-config.md](docs/recommended-config.md).
- **GPU scaling** (single A800 vs EPYC-9754):
  [docs/gpu-scaling-benchmark.md](docs/gpu-scaling-benchmark.md).
- **CPU vs GPU** trade-off: [docs/cpu-vs-gpu-edm.md](docs/cpu-vs-gpu-edm.md).
- **Distributed scale-out** (multi-GPU + cuQuantum, two-track design):
  [docs/multi-gpu-cuquantum-design.md](docs/multi-gpu-cuquantum-design.md).

## Environment

Developed against a `quimb` conda env (Python 3.14, quimb 1.14, autoray, cotengra,
NumPy 2.4). CuPy/GPU is **optional** — the default path is CPU NumPy; CPU-only on
Windows/macOS/Linux needs nothing extra.

On a CUDA machine, add a CuPy wheel matching your CUDA toolkit to enable
`backend='gpu'`, e.g. `pip install cupy-cuda12x` (CUDA 12.x). The GPU path applies
two small compatibility shims for quimb-on-CuPy automatically
(see [docs/quimb-cupy-namespace-bug.md](docs/quimb-cupy-namespace-bug.md)).

On some Windows quimb envs, set `MKL_THREADING_LAYER=TBB` before NumPy is imported
to avoid an OpenMP-runtime clash (`conda env config vars set MKL_THREADING_LAYER=TBB
-n quimb`); see [docs/mkl-tbb-threading-layer.md](docs/mkl-tbb-threading-layer.md).

## Running the tests

```
cd edmtn
PYTHONPATH=src python -m pytest -q       # fast unit suite (integration deselected)
PYTHONPATH=src python -m pytest -m integration   # end-to-end checks (slower)
```

`pyproject.toml` puts `src/` on the path via `[tool.pytest.ini_options].pythonpath`,
so no install is required; alternatively `pip install -e .`.

## Examples

```
python examples/reproduce_fig4.py --quick      # spin-boson Fig. 4a/4b
python examples/reproduce_fig6.py --quick      # Gaudin Fig. 6a/6b
python examples/retire_gpu_smoke.py            # GPU node: validate the pipeline on CuPy
```

Performance scripts live in `tests/benchmarks/` (named `perf_*.py` so pytest does not
collect them); run them directly.
