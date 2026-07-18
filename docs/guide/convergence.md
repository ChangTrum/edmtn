# Convergence checking

The time step is an approximation knob like any other; check it before
trusting a production run. `EDMSolver.timestep_convergence()` re-solves
the same configuration at `eps/2` and compares the polarization
histories on the common grid:

```python
from edmtn.driver import EDMSolver
from edmtn.models import GaudinModel

solver = EDMSolver.from_model(GaudinModel(g=1.0, K=3), T=0.3, eps=0.1,
                              expansion_order=2, cutoff=1e-8)
conv = solver.timestep_convergence(channel=3, tol=1e-3)

print(conv.deviation)   # max |delta <S_a(t)>| between the eps and eps/2 runs
print(conv.converged)   # deviation <= tol, or None if no tol was given
print(conv.metadata["coarse_backend"], conv.metadata["fine_backend"])

dev, ok = conv          # still unpacks as the legacy 2-tuple
```

The fine run is derived with `dataclasses.replace(config, eps=eps/2)`,
so it keeps **every** other field — `sub_baths`, `backend`, `precision`,
compression settings and any future knob. Coarse and fine are the same
physical model and configuration, differing only in the time step; the
comparison cannot silently drift onto a different model.

`metadata` is a self-describing record: the full coarse/fine
`SolverConfig`, the normalised channel, the tolerance, the *actually
executed* backend labels (revealing e.g. a GPU→CPU fallback), and
`coarse_sub_baths_used` / `fine_sub_baths_used` read back from the
results rather than the request.

## The main exceptions from the entry guards and execution layers

| exception | meaning |
|---|---|
| `ValueError` | malformed input — a bad config value, an invalid `channel`, a malformed model, an unsupported parameter combination, or an illegal argument to a direct `run()` |
| `NotImplementedError` | *legal* input, capability not implemented — non-zero temperature on the Gaussian engine, `time_windows`, spin-boson on Track 2, custom observables on separable/Track 2 |
| `FloatingPointError` | the computation produced a non-finite number from legal parameters — a bath correlation overflowing float64, or a non-finite/negative truncation metric |
| `CuTensorNetContractionError` (a `RuntimeError`) | EDMTN-detected Track-2 setup or dispatch failures — e.g. a missing distributed MPI wrapper, or an unsupported multi-rank pathfinder. CuPy, cuQuantum and MPI calls may also raise their native runtime exceptions |

Model, config and direct-`run()` arguments are validated at their entry
points, before any tensor is built; runtime execution failures — like
the Track-2 contraction errors above — can by nature only surface during
execution.
