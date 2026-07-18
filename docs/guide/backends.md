# Backends

One argument selects where and how the contraction runs: `backend=`.
Same model, same time grid — but **not** bit-identical numbers: the
tracks do different algebra.

| `backend` | runs on | use it when |
|---|---|---|
| `cpu` / `numpy` *(default: `cpu`)* | any machine, NumPy (Track 1) | development, small/medium problems |
| `gpu` / `cupy` | one NVIDIA GPU, CuPy (Track 1) | larger problems with big bond dimensions |
| `hpc` | the NVIDIA GPUs / MPI ranks you launch it across, cuQuantum (Track 2) | untruncated contraction; the largest jobs. **Separable/Gaudin only** |

`numpy` and `cupy` are accepted aliases for `cpu` and `gpu`.

Rule of thumb: start with `cpu`; switch to `gpu` when you have one
NVIDIA card and the run is slow; use `hpc` when you want an untruncated
contraction or need to push past one card. A requested GPU that is
unavailable falls back to CPU and says so: `res.backend` reports the
device that *actually* ran, with a `(fallback: ...)` suffix.

## Track 1: `cpu` and `gpu`

Both compress the tensor network (see {doc}`compression`) and share the
same physics pipeline and public API through autoray dispatch. They are
**not** guaranteed to execute identically: backend kernels and dtypes
differ, and `compress_decomp='rsvd'` **falls back to exact full SVD on
any non-NumPy backend** — the same configuration can take a different
decomposition path on GPU than on CPU. CPU/GPU parity is validated to
the tolerances stated in the tests, not bit-for-bit, and not across
different `precision`/`compress_decomp` settings.

`precision='mixed'` casts the Track-1 contraction tensors to the
f32/complex64 path; the declared f64 decomposition recast is not wired
into the solve pipeline. Mixed precision remains **experimental and
unvalidated** — prefer the default `'f64'`.

## Track 2: `hpc`

`hpc` lays the whole space×time problem out as a 2D tensor network and
contracts it with cuQuantum/cuTensorNet — no truncation knobs, automatic
multi-GPU slicing. It currently supports **only the separable/Gaudin
pipeline**; spin-boson is not available there and raises
`NotImplementedError`.

**What "exact" means on Track 2.** It means *no tensor-network
truncation* of the discretised problem. It does **not** remove the
finite-`eps` and expansion-order error (both tracks share those), and it
does not imply higher floating-point precision than Track 1 — it reports
`error_metrics` (`‖ρ−ρ†‖`, `|Tr ρ−1|`) precisely because floating-point
error remains.

Differences between backends therefore come from truncation settings,
decomposition choice, precision, and floating-point contraction order —
the contraction path and slicing differ between backends — not from
different physics.

Multi-GPU launch requirements and the current hardware-validation status
are on {doc}`cluster`.
