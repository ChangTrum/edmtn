# Architecture

## The layering rule

The package is organised in the layers listed in {doc}`../api/index`,
and one rule holds throughout: **lower layers never import higher
ones**. Data flows upward — a model (L1) knows nothing about cumulant
engines (L2); a kernel engine (L3) consumes cumulant data without
knowing which driver assembled it; the driver (L7) is the only layer
that sees the whole pipeline. The public API of each layer is its
package `__all__`; anything else is implementation detail.

The two execution tracks separate at Layer 5: Track 1 is the
quimb-backed compressing pipeline, Track 2 the exact cuTensorNet
contraction. The Track-1 path is deliberately free of any `cuquantum`
import, and every GPU dependency is imported lazily — the whole package
imports cleanly on a CPU-only machine.

## Preferred extension points: options and registries

Algorithm and capability extensions enter as first-class options — a
constructor argument, an entry in a registry, or a hook the underlying
library provides:

- **Models** register with `ModelRegistry` (importing `edmtn.models`
  registers the bundled two).
- **Pipelines** are keyed by the model's `bath_type` in the driver's
  registry: `register_pipeline` / `build_pipeline` /
  `available_pipelines`.
- **Decomposition backends** register with the Layer-0 registry
  (`edmtn.backend.register` / `create` / `available`); importing the
  package registers `'numpy'`, `'cupy'` and `'quimb'`.
- **quimb extensions go through quimb's public hooks.** The two
  registered split drivers are the precedent: `edm_rsvd` wraps quimb's
  own `rand_linalg.rsvd` to expose the power-iteration knob, and
  `edm_eigh_metric` mirrors quimb's `eigh` driver to add the truncation
  metric. Neither bypasses quimb's compression machinery; reaching an
  otherwise-hidden knob means registering through the hook, not going
  around the library.

One deliberately isolated exception exists: `apply_quimb_cupy_compat`
(Layer 0) patches over current quimb/autoray/CuPy interface
incompatibilities — a CuPy `cholesky` signature shim, an autoray
dispatch override, and a namespace-cache replacement — so the GPU path
works at all. It is a compatibility shim, not a recommended extension
mechanism, and new functionality must not imitate it.

## Feature-flagged seams

Two Layer-0 modules reserve interfaces without implementing the
capability, so a future feature lands as an activation rather than a
redesign. Their behaviours differ and are worth stating precisely:

- `OzakiGEMMBackend` — the seam for cuBLAS Ozaki/ADP FP64 GEMM
  emulation. `enabled` reports hardware *capability* (a new enough GPU
  and CUDA), **not** that acceleration is active: `gemm()` currently
  always executes the plain backend matmul regardless of `enabled`, and
  the internal accelerated path raises `NotImplementedError`.
- `ProcessGroup` — the cross-node distributed seam for the `hpc` track.
  cuTensorNet's distributed contraction is node-count-agnostic, so
  multi-node is the same code path with a larger rank geometry; since no
  multi-node hardware has validated it, `require_supported()` **raises**
  `NotImplementedError` on a multi-node layout by default, and
  `EDMTN_ALLOW_MULTINODE=1` opts in with a warning instead. In its own
  words: a reservation, not an implementation.

Both modules import cleanly with no GPU or MPI present.

## Where decisions live

The dated design ledgers below this page record *why* the architecture
looks like this — the re-platform decision ledger for what was replaced
and retired, and the multi-GPU design document for the two-track split
and the current hardware-validation status.
