# edmtn

Extended Density Matrix (EDM) tensor-network solver for non-Markovian open
quantum systems, built on **quimb** + **CuPy**.

It implements the polynomial-complexity EDM formalism of Chen & Liu,
*Polynomial complexity of open quantum system problems* (arXiv:2509.00424): the
reduced dynamics of a small system coupled to a quantum bath is represented by
an extended-density-matrix matrix-product state whose bond dimension grows only
linearly in evolution time, and is propagated by a combined-kernel MPO built
from the bath cumulants.

## Status

**Phase 1 & 2 complete.** Both demonstrator models of the paper are solved
end-to-end and validated against exact references:

- **Phase 1 — spin-boson** (a spin in a Gaussian bosonic bath): reproduces the
  Fig. 4 dynamics.
- **Phase 2 — Gaudin** (a central spin in `K` independent bath spins, a
  *separable* non-Gaussian bath): the outer-loop recursion (Eq. 21) reproduces
  the exact Trotterised reduced dynamics to machine precision and the Fig. 6
  `⟨S_z⟩` / bond-dimension behaviour.

**Phase 3 — faster decomposition (in progress).** A GEMM-based, GPU-friendly
`RandomizedSVD` compression strategy is implemented and validated against the full
`StandardSVD` (`< ξ`, seed-stable) on both CPU and GPU. It is the GPU-fast path:
on a single A800 it runs **7–15× faster than a 256-thread EPYC-9754 CPU**, the lead
growing with bond dimension (see **Performance** below).

Implemented:

- spin-boson model (generalised Ohmic bath, zero temperature) and Gaudin model
  (linearly-decreasing couplings, infinite-temperature spin bath);
- Gaussian second-order cumulants (closed-form Ohmic correlation + numeric
  cross-check) **and** the separable bath-correlation transfer tensors (Eq. F1);
- combined-kernel MPO (Gaussian and separable), **first- and second-order**
  time-step expansion (second order on a doubled sub-step grid);
- truncated-SVD compression with the paper's `s_i / s_{d^2+1}` cutoff rule;
- MPS evolution engines: single-bath forward recursion **and** the separable
  outer-loop over sub-baths (with `sub_baths` to fold the first `L` spins);
- observable extraction: reduced density matrix, single-time expectations, and
  the all-times coupling-channel polarization (Eq.-F2 / Eq.-F3 environment sweep);
- a driver that wires the pipeline from `bath_type` and selects the compute
  backend, plus convergence helpers.

**Compute backend.** The default is **CPU** (`backend='auto'`) — with full-SVD at
small bond dimension the EDM hot path is many sequential medium SVD/QR calls, where
the CPU is competitive. The full CuPy/GPU stack is built, validated and selectable
(`backend='gpu'`); paired with `RandomizedSVD` it is the faster path and its lead
**grows with problem size** (7× → 15× as the bond grows; see **Performance**). See
[docs/cpu-vs-gpu-edm.md](docs/cpu-vs-gpu-edm.md).

## Layout

```
edmtn/
├── pyproject.toml              # package + pytest configuration
├── src/edmtn/                  # the package, organised by layer
│   ├── backend/                # Layer 0: array + linalg backend abstraction (NumPy/CuPy)
│   ├── models/                 # Layer 1: physical models (spin-boson)
│   ├── cumulants/              # Layer 2: bath cumulant engines (Gaussian)
│   ├── kernels/                # Layer 3: combined-kernel MPO construction
│   ├── decomposition/          # Layer 4a: SVD compression strategies
│   ├── expansion/              # Layer 4b: 1st/2nd-order time-step expansion
│   ├── evolution/              # Layer 5: MPS evolution engine
│   ├── observables/            # Layer 6: observable extraction + convergence
│   └── driver/                 # Layer 7: orchestration (EDMSolver)
├── examples/                   # reproductions / studies (reproduce_fig4.py, reproduce_fig6.py)
├── tests/                      # unit / integration; benchmarks/ holds perf_*.py scripts
└── docs/                       # development notes / environment gotchas
```

## Quickstart

```python
from edmtn.driver import solve
from edmtn.models import SpinBosonModel, GaudinModel

# Phase 1 -- spin-boson (Gaussian bath)
sb = SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0)
res = solve(sb, T=8.0, eps=0.02, expansion_order=2, cutoff=1e-6)
res.times, res.polarization, res.bond_dims   # mu*t, <S_z(t)>, bond dim per step

# Phase 2 -- Gaudin (separable spin bath); channel 3 is <S_z>
g = GaudinModel(g=1.0, K=49)
res = solve(g, T=15.0, eps=0.03, expansion_order=2, cutoff=1e-6,
            max_bond=400, channel=3)
res.times, res.polarization   # t, <S_z(t)>  (res.mps.bond_dims is D_t)
```

## Configuration

`solve(model, *, T, eps, channel=1, **config)` (and `EDMSolver.from_model`) accept:

| knob | values (default) | meaning |
|---|---|---|
| `expansion_order` | `1`, `2` (`1`) | Trotter order (2 = doubled sub-step grid) |
| `cutoff` | float (`1e-6`) | SVD truncation `ξ` (rule `s_i / s_{d²+1} ≤ ξ`) |
| `max_bond` | int or `None` (`None`) | hard bond-dimension cap |
| `backend` | `'auto'`, `'cpu'`, `'gpu'` (`'auto'`) | compute device (`auto` → CPU) |
| `decomposition` | strategy (`StandardSVD()`) | **compression** (see below) |
| `canonicalization` | strategy or `None` (`None` → Householder QR) | **canonicalisation** (see below) |
| `preset` | `'balanced'`, `'robust'`, `None` (`None`) | fills recommended strategies (see below); explicit `decomposition`/`canonicalization` win |
| `sub_baths` | int or `None` (`None`) | separable bath: fold only the first `L` sub-baths (Fig. 6) |

`solve(...).backend` reports the resolved device.

**Backend.** `'auto'`/`'cpu'` runs on NumPy; `'gpu'` runs on CuPy (needs a matching
CuPy wheel — see *Environment*; falls back to CPU with a note if no GPU). GPU pays off
once the bond is large; small problems are CPU-competitive.

**Compression (`decomposition=`)** — import from `edmtn.decomposition`:

```python
from edmtn.decomposition import StandardSVD, RandomizedSVD

StandardSVD()             # default: exact full SVD (the paper's baseline)
RandomizedSVD(n_iter=0)   # single-pass rSVD: fastest, accuracy < ξ, GEMM/GPU-friendly
RandomizedSVD(n_iter=2)   # cold rSVD: exact-baseline bonds, ~1e-12 accuracy
```

`RandomizedSVD` finds the rank adaptively (a spectral resolution guard), so it is reliable
with no reference run; single-pass is the GPU-fast default, cold is for exact bonds.

**Canonicalisation (`canonicalization=`)** — import from `edmtn.evolution`:

```python
from edmtn.evolution import CholeskyQR        # HouseholderQR is the default (pass None)

None                      # default: Householder QR — fastest on GPU and at tight ξ
CholeskyQR(passes=2)      # BLAS-3 Cholesky-QR2; only wins on CPU at moderate ξ (niche)
```

Householder QR is conditioning-immune and the measured fastest in every regime except
CPU-moderate-ξ; keep the default unless you are CPU-bound at `ξ ≳ 1e-6`.

**Recommended presets** — the easy way is `preset=` (details + when-to-use in
[docs/recommended-config.md](docs/recommended-config.md)):

```python
# preset='balanced' -> single-pass rSVD + Householder QR (fastest, accuracy < ξ)
res = solve(g, T=15.0, eps=0.03, expansion_order=2, cutoff=1e-6, max_bond=400,
            channel=3, backend='gpu', preset='balanced')

# preset='robust'   -> cold rSVD + Householder QR (exact-baseline bonds, ~1e-12)
res = solve(g, T=15.0, eps=0.03, expansion_order=2, cutoff=1e-6, max_bond=400,
            channel=3, preset='robust')

# no preset (default) -> StandardSVD + Householder QR: exact, deterministic.
```

`preset` only fills strategies you did not set explicitly (`decomposition=` /
`canonicalization=` always win), and never overrides `backend`.

## Performance

- **Recommended presets** (balanced vs robust, when to use which):
  [docs/recommended-config.md](docs/recommended-config.md).
- **GPU scaling** (single A800 vs 256-thread EPYC 9754; single-pass rSVD 7×→15×,
  growing with bond): [docs/gpu-scaling-benchmark.md](docs/gpu-scaling-benchmark.md).
- **Compression / decomposition study** (why single-pass rSVD is reliable, the
  canonicalisation analysis): [docs/incremental-update-research.md](docs/incremental-update-research.md).
- **Distributed scale-out plan** (multi-GPU + cuQuantum, two-track design):
  [docs/multi-gpu-cuquantum-design.md](docs/multi-gpu-cuquantum-design.md).

## Environment

Developed against the `quimb` conda env (Python 3.14, quimb 1.14, CuPy 14.1,
CUDA 13.2, RTX 5090). CuPy/GPU is optional — the default execution path is CPU
NumPy, the faster choice for the Phase-1/2 problem sizes (CPU vs GPU benchmarks
and analysis in [docs/cpu-vs-gpu-edm.md](docs/cpu-vs-gpu-edm.md)).

On a CUDA machine, add a CuPy wheel matching your CUDA toolkit to enable
`backend='gpu'`, e.g. `pip install cupy-cuda12x` (CUDA 12.x) or `cupy-cuda13x`.
CPU-only on Windows/macOS/Linux needs nothing extra.

The env requires `MKL_THREADING_LAYER=TBB` to be set **before NumPy is imported**
(clashing OpenMP runtimes otherwise crash BLAS/LAPACK), configured at env level:

```
conda env config vars set MKL_THREADING_LAYER=TBB -n quimb
```

See `docs/mkl-tbb-threading-layer.md`.

## Running the tests

With the `quimb` env activated (so the threading-layer variable is set):

```
cd edmtn
pytest                       # fast unit suite (integration deselected)
pytest -m integration        # qualitative end-to-end checks (slower, O(N^2))
```

`pyproject.toml` puts `src/` on the path via `[tool.pytest.ini_options].pythonpath`,
so no install is required. For use outside pytest, either activate the env and
add `src/` to `PYTHONPATH`, or `pip install -e .`.

## Examples

```
python examples/reproduce_fig4.py --quick      # spin-boson Fig. 4a/4b (Phase 1)
python examples/reproduce_fig6.py --quick      # Gaudin Fig. 6a/6b   (Phase 2)
```

## Benchmarks

Performance scripts live in `tests/benchmarks/`, named `perf_*.py` so pytest does
not collect them; run them directly:

```
python tests/benchmarks/perf_cpu_gpu.py        # spin-boson: CPU vs GPU, fp32 vs fp64
python tests/benchmarks/perf_gaudin.py --quick # Gaudin: CPU vs GPU across D_c
```
