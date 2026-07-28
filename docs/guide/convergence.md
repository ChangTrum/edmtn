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

## Models without a polarization history

`timestep_convergence()` compares polarization histories, so it raises
`NotImplementedError` on a pipeline that publishes none — currently
`bath_type='separable_td'` (`DickeModel`). Compare final states directly
instead:

```python
import numpy as np
from edmtn.driver import solve
from edmtn.models import DickeModel

model = DickeModel(K=3, n_fock=6, coupling=0.5, kappa=0.1)
coarse = solve(model, T=0.6, eps=0.05, expansion_order=2, cutoff=1e-8)
fine = solve(model, T=0.6, eps=0.025, expansion_order=2, cutoff=1e-8)
deviation = np.max(np.abs(coarse.final_density_matrix - fine.final_density_matrix))
```

At `expansion_order=2` that deviation falls by about a factor of 4 per
halving of `eps`, and at `expansion_order=1` by about 2 — **provided**
the time-discretisation error dominates: `n_fock` is held fixed, the
compression error (`cutoff`, `max_bond`) is well below it, and neither
has reached a floor. Once truncation or round-off dominates, the ratio
degrades and no longer measures the scheme's order.

## The main exceptions from the entry guards and execution layers

| exception | meaning |
|---|---|
| `ValueError` | malformed input — a bad config value, an invalid `channel`, a malformed model, an unsupported parameter combination, or an illegal argument to a direct `run()` — **and** an extracted quantity that is physically real coming back with a non-negligible imaginary part (the coupling polarization, and the `n` / `n_factorial2` / `Jz` moments): a legal computation whose result is not trustworthy, refused rather than returned |
| `NotImplementedError` | *legal* input, capability not implemented — non-zero temperature on the Gaussian engine, `time_windows`, spin-boson on Track 2, custom observables on separable/Track 2, a `channel` or `timestep_convergence()` on `separable_td`, `moments` on a model that supplies no Dicke closings |
| `FloatingPointError` | the computation produced a non-finite number from legal parameters — a bath correlation overflowing float64, a non-finite/negative truncation metric, or a non-finite final observable: any returned moment, the raw `trace`, `<J_+>`, or the derived `|<J>|` (checked again after it is formed, since finite components can still combine to overflow) |
| `CuTensorNetContractionError` (a `RuntimeError`) | EDMTN-detected Track-2 setup or dispatch failures — e.g. a missing distributed MPI wrapper, or an unsupported multi-rank pathfinder. CuPy, cuQuantum and MPI calls may also raise their native runtime exceptions |

Model, config and direct-`run()` arguments are validated at their entry
points, before any tensor is built; runtime execution failures — like
the Track-2 contraction errors above — can by nature only surface during
execution.
