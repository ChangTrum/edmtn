# Testing

## Layout and invocation

From the repository root (no install needed —
`[tool.pytest.ini_options].pythonpath` puts `src/` on the path):

```bash
PYTHONPATH=src python -m pytest -q                  # default: the unit suite
PYTHONPATH=src python -m pytest -m integration      # opt-in end-to-end checks
```

- `tests/unit/` — the default suite; integration tests are deselected
  by the configured `addopts`.
- `tests/integration/` — slower end-to-end smoke checks, opt-in via
  `-m integration`.
- `tests/benchmarks/` — performance scripts named `perf_*.py`; they are
  **not collected** by pytest and are run directly.

## Markers and hardware gating

| marker | requires |
|---|---|
| `integration` | nothing extra (just opt-in) |
| `gpu` | a real CUDA-capable GPU and CuPy |
| `cuquantum` | cuQuantum/cuTensorNet on a real NVIDIA GPU |
| `multigpu(n)` | an explicitly enabled multi-GPU MPI/SLURM allocation |

Hardware-dependent tests are gated by **real detection** in
`tests/conftest.py`, not unconditional skips: they are collected
everywhere and skip with a specific reason (for example
`gpu: CuPy not importable`) when their stack is absent. Hardware absence
alone never fails the ordinary CPU run.

For *acceptance* runs the contract inverts: `--require-gpu`,
`--require-cuquantum` and `--require-multigpu=N` turn a missing stack
into a non-zero exit, so an all-skipped run cannot masquerade as
hardware validation. `--require-multigpu=N` additionally demands the
distributed worker's result JSON (`EDMTN_MULTIGPU_RESULT`) — see
{doc}`../guide/cluster` for the launch recipes and what that variable is
(and is not).

## The docs-contract guard

`tests/unit/test_docs_contract.py` keeps this documentation honest. It
re-runs the README's two small CPU examples (hand-mirrored, not the
paper-scale or GPU/HPC blocks), executes every python fence of the
quickstart and convergence pages block by block, checks documented
defaults and axes against the live objects, scans the current-contract
pages for a list of specific stale claims, and pins the boundary
statements each guide page must keep. A documentation change that breaks
a contract fails the suite rather than shipping quietly.

## Conventions

- The validated entry points — model construction, `SolverConfig` /
  `EDMSolver`, `solve()`'s outer arguments such as `channel`, both
  direct evolution `run()` methods, and the compression combination in
  `QuimbEDM.compress()` — reject bad input with a clear `ValueError` at
  the entry; tests assert this and, for the direct `run()` paths, use
  spies to prove nothing was constructed before the rejection.
- Numerical assertions state their tolerances; agreement claims in the
  docs are only as strong as the tolerances asserted here.
