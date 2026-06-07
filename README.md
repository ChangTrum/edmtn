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

**Compute backend.** Phase 1/2 run on the **CPU** by default — the EDM hot path
is many sequential medium SVD/QR calls, where the CPU beats the GPU at these
bond dimensions. The full CuPy/GPU stack is built, validated and selectable
(`backend='gpu'`), and becomes the faster path with the Phase-3 decomposition
work. See [docs/cpu-vs-gpu-edm.md](docs/cpu-vs-gpu-edm.md).

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

The driver picks the compute backend (`backend='auto'` → CPU for Phase 1/2;
`'gpu'` to force the GPU). `solve(...).backend` reports the choice.

## Environment

Developed against the `quimb` conda env (Python 3.14, quimb 1.14, CuPy 14.1,
CUDA 13.2, RTX 5090). CuPy/GPU is optional — the default execution path is CPU
NumPy, the faster choice for the Phase-1/2 problem sizes (CPU vs GPU benchmarks
and analysis in [docs/cpu-vs-gpu-edm.md](docs/cpu-vs-gpu-edm.md)).

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
