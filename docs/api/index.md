# API reference

The public API of `edmtn` is the set of names each layer package exports via
`__all__`; the pages below document exactly those objects. Modules internal
to a layer — anything importable but not re-exported by its package — are
implementation detail and carry no compatibility contract.

The package is organised in layers; lower layers never import higher ones:

| Layer | Package             | Role                                                      |
|------:|---------------------|-----------------------------------------------------------|
| 0     | `edmtn.backend`     | array/linalg backend abstraction (NumPy/CuPy via autoray) |
| 1     | `edmtn.models`      | physical models (spin-boson, Gaudin)                      |
| 2     | `edmtn.cumulants`   | bath cumulant / correlation engines                       |
| 3     | `edmtn.kernels`     | combined kernel-tensor (MPO) construction                 |
| 4     | `edmtn.expansion`   | first/second-order time-step expansion                    |
| 5     | `edmtn.evolution`   | quimb-backed MPS evolution + compression                  |
| 6     | `edmtn.observables` | observable extraction + convergence diagnostics           |
| 7     | `edmtn.driver`      | orchestration (`solve`, `EDMSolver`, `SolverConfig`)      |

```{toctree}
:maxdepth: 1

backend
models
cumulants
kernels
expansion
evolution
observables
driver
```
