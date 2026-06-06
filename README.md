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

**Phase 1 complete** — the spin-boson model (a spin in a Gaussian bosonic bath)
is solved end-to-end and reproduces the paper's Fig. 4 dynamics. Validated by
177 unit + 4 integration + 2 backend-correctness tests.

Implemented:

- spin-boson model (generalised Ohmic bath, zero temperature);
- Gaussian second-order cumulants (closed-form Ohmic correlation + numeric
  cross-check);
- combined-kernel MPO, **first- and second-order** time-step expansion (the
  second order runs on a doubled sub-step grid with a parity-dependent lag map);
- truncated-SVD compression with the paper's `s_i / s_{d^2+1}` cutoff rule;
- MPS evolution engine (forward recursive construction);
- observable extraction: reduced density matrix, single-time expectations, and
  the all-times coupling-channel polarization via the Eq.-F2 environment sweep;
- a driver that wires the pipeline from `bath_type`, plus convergence helpers.

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
├── examples/                   # reproduce_fig4.py, benchmark_cpu_gpu.py
├── tests/                      # unit / integration / benchmarks
└── docs/                       # development notes / environment gotchas
```

## Quickstart

```python
from edmtn.driver import solve
from edmtn.models import SpinBosonModel

model = SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0)        # Ohmic bath
res = solve(model, T=8.0, eps=0.02, expansion_order=2, cutoff=1e-6)

res.times          # mu*t grid
res.polarization   # <S_z(t)>
res.bond_dims      # EDM bond dimension per step
```

## Environment

Developed against the `quimb` conda env (Python 3.14, quimb 1.14, CuPy 14.1,
CUDA 13.2, RTX 5090). CuPy/GPU is optional — the default execution path is CPU
NumPy, which is the faster choice for the Phase-1 problem sizes (see
`examples/benchmark_cpu_gpu.py`).

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
pytest                       # fast unit suite (integration/benchmark deselected)
pytest -m integration        # Fig. 4 reproduction (slower, O(N^2))
pytest -m benchmark          # backend-correctness checks (needs CuPy for GPU)
```

`pyproject.toml` puts `src/` on the path via `[tool.pytest.ini_options].pythonpath`,
so no install is required. For use outside pytest, either activate the env and
add `src/` to `PYTHONPATH`, or `pip install -e .`.

## Examples

```
python examples/reproduce_fig4.py --quick      # fast preview of Fig. 4a/4b
python examples/benchmark_cpu_gpu.py           # CPU vs GPU, fp32 vs fp64
```
