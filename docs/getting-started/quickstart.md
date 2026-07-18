# Quickstart

This walk-through runs both bundled model families on CPU; each example
finishes in seconds. It assumes `edmtn` is importable — see
{doc}`installation`.

## A spin in a bosonic bath (spin-boson)

```python
from edmtn.driver import solve
from edmtn.models import SpinBosonModel

sb = SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0)
res = solve(sb, T=0.3, eps=0.1, expansion_order=2, cutoff=1e-8)

print(res.times)          # [0.1, 0.2, 0.3] -- the grid is eps, 2*eps, ..., T
print(res.polarization)   # <S_z(t)> on that grid
print(res.backend)        # what actually executed, e.g. 'cpu/f64'
```

One `solve()` call assembles the whole layered pipeline: model → bath
cumulants → kernel MPO → time-step expansion → MPS evolution → observable
extraction. `expansion_order` may be omitted, in which case it inherits the
model's default time-step order; `res.expansion_order` records what was
actually used.

## A central spin in `K` bath spins (Gaudin)

```python
from edmtn.driver import solve
from edmtn.models import GaudinModel

g = GaudinModel(g=1.0, K=3)
res = solve(g, T=0.3, eps=0.1, expansion_order=2, cutoff=1e-8,
            max_bond=64, channel=3)

print(res.polarization)          # <S_z(t)>
print(res.sub_bath_counts)       # L = 1..K: the sub-bath fold axis ...
print(res.sub_bath_bond_dims)    # ... and the bond dimension D_L after each fold
print(res.final_time_bond_dims)  # the final MPS's bonds along the time chain
```

The separable-bath pipeline folds the `K` sub-baths into the extended
density matrix one at a time; the per-fold diagnostics above expose that
process alongside the polarization history itself.

## The time grid

`T / eps` must be a *positive integer* (to a small tolerance). It is never
silently rounded: a non-integer ratio raises `ValueError` at configuration
time, before anything is computed. Every public `times` array is
`[eps, 2 eps, ..., T]` on all pipelines, so the time axes of different
backends and tracks align index for index. Numerical agreement still
depends on the backend, truncation, decomposition, precision and expansion
settings.

## Choosing the channel

`channel` selects which coupling operator's polarization history is
returned. It is a strict 1-based integer (default `1`):

| model            | valid `channel` | operator                            |
|------------------|-----------------|-------------------------------------|
| `SpinBosonModel` | `1` only        | `S_z` (its single coupling channel) |
| `GaudinModel`    | `1`, `2`, `3`   | `S_x`, `S_y`, `S_z`                 |

`0`, negative values, out-of-range values, floats, strings and `bool` all
raise `ValueError` — in particular `channel=0` is rejected rather than
silently selecting the last operator through Python negative indexing.

## Where next

- {doc}`../guide/index` — configuring and tuning the solver.
- {doc}`../api/index` — the public API, layer by layer.
- The repository `README.md` — backend selection and paper-scale runs.
