# Installation

`edmtn` is not published on PyPI; install it from a checkout of the
repository.

## Requirements

- Python 3.11 or newer.
- NumPy, SciPy, [quimb](https://quimb.readthedocs.io/) and autoray — pulled
  in automatically by `pip`.
- `pytest`, only if you want to run the test suite.

The package is pure Python. CPU-only operation needs nothing beyond the
dependencies above and works on Windows, macOS and Linux. The known-good
development configuration is Python 3.14 with quimb 1.14 and NumPy 2.4.
Other environments satisfying the declared requirements are intended to
work, but the dependency set is not pinned and has not been validated
exhaustively.

## Installing from source

From the repository root:

```bash
pip install -e .
```

Alternatively, skip installation entirely and put `src/` on `PYTHONPATH`.
The test configuration already does the latter
(`[tool.pytest.ini_options].pythonpath` in `pyproject.toml`), so the test
suite runs from a bare checkout with no install step.

A dedicated virtual environment (conda or venv) is convenient but not
required; `edmtn` does not depend on any particular environment name or
layout.

## GPU support (optional)

`backend='gpu'` requires a CuPy wheel matching the installed CUDA toolkit,
for example:

```bash
pip install cupy-cuda12x        # CUDA 12.x
```

The GPU path applies two small quimb-on-CuPy compatibility shims
automatically; see {doc}`../troubleshooting/quimb-cupy-namespace-bug` for
what they work around.

## The `hpc` backend (optional, NVIDIA only)

`backend='hpc'` — the exact cuTensorNet track — additionally requires:

```bash
pip install cuquantum-python-cu12
```

Multi-GPU execution additionally needs a compatible MPI runtime and a
cuTensorNet distributed-interface library selected through
`CUTENSORNET_COMM_LIB`. Launcher commands and any ABI/workaround settings
are site-specific; `cluster/test_gpu_hpc.sbatch` in the repository records
the current test recipe, while the older MPICH/PMI2 scripts there are
historical. See {doc}`../design/multi-gpu-cuquantum-design` for the design
and current status. None of these packages is imported on the CPU/Track-1
path, so CPU-only installations stay clean.

## Verifying the installation

From the repository root:

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m pytest -m integration
```

The first command runs the default unit suite; the second runs the opt-in
end-to-end integration suite. Neither has a fixed runtime — it depends on
the machine.

GPU/HPC tests are gated by real hardware detection: they are collected
everywhere and skip with a specific reason (for example
`gpu: CuPy not importable`) when their required hardware or stack is
absent — hardware absence alone does not fail the ordinary CPU run. For
hardware acceptance runs the `--require-*` flags turn a missing stack into
a non-zero exit instead of a quiet skip; the repository `README.md` lists
the exact invocations.

## Platform notes

On some Windows conda environments, MKL's default OpenMP threading layer
clashes with a second OpenMP runtime shipped in the same environment, and
NumPy BLAS/LAPACK calls then crash the process outright. Setting
`MKL_THREADING_LAYER=TBB` *before NumPy is first imported* avoids the
clash. This is a workaround for that specific MKL/OpenMP configuration,
not a general installation requirement; see
{doc}`../troubleshooting/mkl-tbb-threading-layer` for the diagnosis.
