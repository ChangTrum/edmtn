# edmtn

`edmtn` is an Extended-Density-Matrix (EDM) tensor-network solver for
non-Markovian open-quantum-system dynamics — a quimb + CuPy implementation of
the polynomial-complexity EDM formalism.

Two model families are supported: the spin-boson model (a spin in a Gaussian
bosonic bath) and the Gaudin model (a central spin coupled to `K` bath
spins). Both are solved through a single `solve()` entry point that assembles
the layered pipeline — model → cumulants → kernel MPO → time-step expansion →
MPS evolution → observables. The compressing pipeline runs on CPU (NumPy) or
a single NVIDIA GPU (CuPy) via quimb/autoray dispatch; a separate exact-only
cuTensorNet backend (`backend='hpc'`, separable/Gaudin models only) targets
multi-GPU HPC hardware.

Installation instructions, copy-paste first runs and the full tuning-knob
reference live in the repository
[README](https://github.com/ChangTrum/edmtn/blob/main/README.md).

```{toctree}
:caption: API reference
:maxdepth: 2

api/index
```

```{toctree}
:caption: Guides
:maxdepth: 1

guides/recommended-config
```

```{toctree}
:caption: Benchmarks
:maxdepth: 1

benchmarks/cpu-vs-gpu-edm
benchmarks/gpu-scaling-benchmark
```

```{toctree}
:caption: Design notes
:maxdepth: 1

design/phase0-replatform-decisions
design/multi-gpu-cuquantum-design
```

```{toctree}
:caption: Research records
:maxdepth: 1

research/coupling-scaling-law
research/incremental-update-research
```

```{toctree}
:caption: Troubleshooting
:maxdepth: 1

troubleshooting/mkl-tbb-threading-layer
troubleshooting/quimb-cupy-namespace-bug
```
