# edmtn

**Simulate how a small quantum system loses coherence to its environment — fast.**

`edmtn` computes the time evolution of a quantum system (e.g. a single spin)
coupled to a noisy bath, and returns observables like the polarization
`⟨S_z(t)⟩` and the density matrix `ρ(t)`. It implements the polynomial-complexity
method of Chen & Liu, *Polynomial complexity of open quantum system problems*
([arXiv:2509.00424](https://arxiv.org/abs/2509.00424)): the cost grows only
**linearly in evolution time** instead of exponentially, so you can reach long
times and many bath modes on ordinary hardware.

You don't need to know tensor networks to use it — pick a model, call `solve(...)`,
read the result. Tuning and internals are there when you want them.

```python
from edmtn.driver import solve
from edmtn.models import GaudinModel

res = solve(GaudinModel(g=1.0, K=49), T=15.0, eps=0.03, channel=3)  # central spin in 49 bath spins
print(res.times, res.polarization)   # t,  <S_z(t)>
```

That runs on CPU, anywhere, with nothing extra installed.

## What it can do

- **Two ready-made models** — *spin-boson* (a spin in a bosonic/Ohmic bath) and
  *Gaudin* (a central spin in `K` bath spins). Both validated to machine precision
  against exact references.
- **Three backends, one API** — `cpu` (default, runs everywhere), `gpu` (one NVIDIA
  card), `hpc` (all GPUs on a node, exact). Just change `backend=...`; see
  **[Which backend?](#which-backend)**.
- **Observables out of the box** — `⟨S_z(t)⟩` (or any coupling channel), the full
  `ρ(t)`, the bond-dimension growth `D_t`, and convergence checks.

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
res = solve(sb, T=8.0, eps=0.02, expansion_order=2, cutoff=1e-8)
print(res.times)          # mu * t
print(res.polarization)   # <S_z(t)>

# Gaudin (a central spin in K bath spins); channel 3 = <S_z>
g = GaudinModel(g=1.0, K=49)
res = solve(g, T=15.0, eps=0.03, expansion_order=2, cutoff=1e-8, max_bond=400, channel=3)
print(res.polarization)   # <S_z(t)>
print(res.mps.bond_dims)  # D_t — how the cost grows (linearly in time)
```

## What you get back

`solve(...)` returns a `SolverResult`:

| field | what it is |
|---|---|
| `res.times` | the time grid |
| `res.polarization` | `⟨S_channel(t)⟩` over time (pick `channel=` in `solve`) |
| `res.bond_dims` / `res.mps.bond_dims` | bond dimension per step / per time `D_t` |
| `res.density_matrices` | the density matrix `ρ(t)` (always first-class on `hpc`; on cpu/gpu set `record_rho=True`) |
| `res.error_metrics` | (`hpc` only) `‖ρ−ρ†‖`, `|Tr ρ−1|`, optimizer slice/flop counts |
| `res.backend` | the resolved device/track that actually ran |

## Which backend?

Change one argument, `backend=`. Same physics, same result (to round-off) — the
choice is about *speed and scale*:

| `backend` | runs on | use it when |
|---|---|---|
| `cpu` *(default)* | any machine, NumPy | development, small/medium problems — works out of the box, no GPU needed |
| `gpu` | one NVIDIA GPU, CuPy | larger problems where the bond dimension is big |
| `hpc` | **all** NVIDIA GPUs on a node, cuQuantum | exact (no truncation) results, and the very largest jobs |

**Rule of thumb:** start with `cpu`; switch to `gpu` if you have one NVIDIA card
and the run is slow; use `hpc` when you want an *exact* answer or need to push a
job that's too big for one card.

`cpu` and `gpu` are **Track 1** — they compress the tensor network (truncate small
singular values, controlled by `cutoff`/`max_bond`), which scales to long times and
many bath spins. `hpc` is **Track 2** — it lays the whole problem out as a 2D
space×time network and contracts it **exactly** with cuQuantum (cuTensorNet), with
**no truncation knobs**; its wins are higher precision and automatic multi-GPU
slicing for the exact contraction. (The truncated regime stays on Track 1, which
already scales there.)

### Running on `hpc` (multi-GPU)

`hpc` uses **every GPU you launch it across** — one process per GPU. edmtn does
*not* submit jobs, ssh, or call `srun`/`sbatch` for you; that's your workflow. To
use 4 GPUs on a SLURM node, launch 4 ranks:

```bash
srun --mpi=pmi2 --ntasks=4 --gres=gpu:4 python your_script.py
```

Your script is unchanged — just `backend='hpc'`:

```python
res = solve(GaudinModel(g=1.0, K=12), T=0.6, eps=0.1, channel=3, backend='hpc')
res.density_matrices   # ρ(t)
res.error_metrics      # {hermiticity, trace_dev, num_slices, flops}
res.backend            # e.g. 'hpc/exact/cuquantum/4gpu'
```

If you run `hpc` on a single GPU (or the problem is small enough to fit one card),
edmtn warns and suggests scaling up or using Track 1 — it still works, it just isn't
where `hpc` pays off. Ready-to-edit launch scripts live in
[`cluster/`](cluster/). Design + status:
[docs/multi-gpu-cuquantum-design.md](docs/multi-gpu-cuquantum-design.md).

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
├── examples/                   # reproductions / studies (reproduce_fig4.py, reproduce_fig6.py, …)
├── tests/                      # unit / integration; benchmarks/ holds perf_*.py scripts
└── docs/                       # design decisions, benchmarks, environment notes
```

## Configuration (tuning knobs)

You only need `T`, `eps`, and maybe `channel` to get going. The rest are optional
knobs — `solve(model, *, T, eps, channel=1, **config)` (and `EDMSolver.from_model`)
accept:

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

The compute is backend-agnostic (autoray), so `cpu` and `gpu` agree to round-off.
For when to pick which — and how `hpc` differs (exact, multi-GPU, no truncation
knobs) — see **[Which backend?](#which-backend)** above. Under the hood, the `hpc`
track lays the whole problem out as a 2D space×time network and contracts it exactly
with cuQuantum/cuTensorNet; design + status:
[docs/multi-gpu-cuquantum-design.md](docs/multi-gpu-cuquantum-design.md).

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

For `backend='hpc'` (NVIDIA only) also install `cuquantum-python-cu12`. Multi-GPU
needs an MPI launcher (`srun`/`mpirun`) and the cuTensorNet MPI wrapper — the
`cluster/` launch scripts set the required env (`CUTENSORNET_COMM_LIB`,
`LD_PRELOAD`); see [docs/multi-gpu-cuquantum-design.md](docs/multi-gpu-cuquantum-design.md).
None of this is imported on the CPU / Track-1 path, so CPU-only installs stay clean.

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
