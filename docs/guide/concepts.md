# Concepts

This page explains what the solver computes and how the pieces fit
together, at the level a user needs to read the rest of the guide. The
formalism itself is developed in the paper cited in
[References](#references); equation and figure numbers used across this
documentation and the source code (Eq. F2, Fig. 3d, …) refer to it.

## The extended density matrix

For a non-Markovian open quantum system, the reduced density matrix at a
given time depends on the whole history of the system–bath interaction,
not only on the state one step earlier. The EDM formalism makes that
history explicit: the density matrix is extended with auxiliary degrees of
freedom that carry the interaction history, and the resulting object is
organised as a tensor network with a *temporal* dimension.

Bath influence enters through correlation data, not through an explicit
bath Hilbert space: a Gaussian two-time correlation function for the
bosonic bath of the spin-boson model, and per-sub-bath correlation data
for the separable Gaudin bath. That data is compiled into a
*combined-kernel* matrix product operator (MPO), which is what the
evolution actually applies.

Two construction patterns are implemented:

- **Single-bath (spin-boson).** The EDM is built forward in time: each
  physical step applies the combined-kernel MPO and the system
  superoperators, then recompresses the temporal MPS.
- **Separable bath (Gaudin).** The bath factorises into `K` independent
  sub-baths. The pipeline folds them into the EDM one at a time — an
  outer loop over `L = 1..K` — compressing after each fold. Diagnostics
  are therefore per fold: bond dimension `D_L` and, on request, the
  final-time state `rho_L(T)` after `L` sub-baths.

## The layered pipeline

A `solve()` call assembles a fixed pipeline of layers; lower layers never
import higher ones (see the {doc}`../api/index` for the per-layer
contracts):

1. the **model** (L1) supplies the system Hamiltonian, coupling
   operators, initial state and bath parameters;
2. the **cumulant engine** (L2) computes the bath correlation data;
3. the **kernel engine** (L3) compiles it into the combined-kernel MPO;
4. the **expansion layer** (L4) produces first- or second-order
   small-step superoperators;
5. the **evolution engine** (L5) applies the step and compresses;
6. the **observable extractor** (L6) reads polarization histories and
   reduced states off the final EDM;
7. the **driver** (L7) validates every input at the entry point and wires
   the layers together, with the array backend (L0) underneath.

## Two execution tracks

- **Track 1** (`backend='cpu'` or `'gpu'`) carries the EDM as a quimb
  tensor network and compresses it after every step or fold. It is the
  default and the reference implementation; the approximation is
  controlled by the truncation knobs (`cutoff`, `max_bond`, …).
- **Track 2** (`backend='hpc'`, separable/Gaudin only) lays the whole
  space×time EDM out as a two-dimensional tensor network and contracts it
  with cuTensorNet, *exactly*: no tensor-network truncation is performed,
  and there are no compression knobs. Exact-only refers to the
  contraction alone — the finite time step, the expansion order and
  floating-point arithmetic still apply.

## Complexity: what is proven and what is measured

The complexity statements behind the formalism are theorems of the paper,
proved under its assumptions and for the tensor-contraction procedure it
prescribes: the number of equations grows **at most linearly** with
evolution time and polynomially with bath size, and the tensor-network
bond dimension grows at most linearly with time.

What the implementation adds are *measurements*, not stronger claims:

- Uncompressed runs are validated against small dense references at the
  tolerances stated in the tests (agreement ~1e-10 or better at the
  tested sizes); compressed runs are checked only at the specific
  parameters each test covers.
- Total runtime is **not** linear in `T`: the cost of a step grows with
  the bond dimension it has to move.
- Observed bond growth under compression depends on the model, the
  coupling distribution and the truncation settings, and is characterised
  only over the tested ranges — see the dated records under
  {doc}`../developer/index` and the research records for specifics.
- `cutoff` is a local per-bond truncation threshold, **not** an error
  bound on the polarization, `rho(t)` or the trajectory.

## References

Chong Chen and Ren-Bao Liu, *Polynomial complexity of open quantum system
problems*, arXiv:2509.00424 \[quant-ph\] (2025).
<https://doi.org/10.48550/arXiv.2509.00424>
