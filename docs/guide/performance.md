# Performance

## Scaling up

The quickstart examples are deliberately tiny. The paper's actual
configurations live as ready-to-run reproductions in
`examples/reproduce/` in the repository:

- `reproduce_fig4.py` — spin-boson (Ohmic, `omega_c = 5 mu`):
  `T = 17 mu^-1`, `eps = 0.01`, `cutoff = 1e-5`, second order;
- `reproduce_fig6.py` — Gaudin: `K = 49`, `T = 15 g^-1`, `eps = 0.03`,
  `cutoff = 1e-6`, `max_bond = 400`, second order.

A larger illustrative run — the paper-sized Gaudin `K`/`T` grid with a
tighter cutoff than the paper's — looks like:

```python
res = solve(GaudinModel(g=1.0, K=49), T=15.0, eps=0.03, expansion_order=2,
            cutoff=1e-8, max_bond=400, channel=3)
```

Before scaling up: start small, watch `res.truncation_errors`, and run
{doc}`timestep_convergence() <convergence>`; remember `T/eps` must be an
exact integer.

Runtime is dominated by bond dimension: the cost of a step grows with
the bonds it has to move, so total runtime is not linear in `T` even
though the formalism's equation count is (see {doc}`concepts`).

## Choosing the speed/accuracy point

- The **default** (`preset=None`, the exact decomposition) is
  deterministic and the only setting that reports a real truncation
  metric — start there.
- `preset='balanced'` / `'robust'` switch to randomized rSVD for speed;
  the measurements behind them, with their hardware and parameters, are
  in {doc}`../guides/recommended-config`.
- `max_bond` caps memory growth at the price of truncation; watch the
  metric.
- `backend='gpu'` helps once bond dimensions are large; see
  {doc}`backends` for the parity caveats.

## GPU memory

On GPU, after each separable fold the implementation asks CuPy to
release **unused cached** device and pinned-memory blocks
(`MemoryManager`; a no-op on CPU). Live tensors in the evolving MPS
remain allocated, so this is not a constant-memory guarantee.

## Benchmarks

Every performance figure in this documentation is a **dated measurement
on specific hardware with stated parameters, backend, precision and
truncation settings** — never a general guarantee. The records live
under {doc}`../developer/index`: CPU-vs-GPU trade-off and GPU scaling,
each with its configuration spelled out.
