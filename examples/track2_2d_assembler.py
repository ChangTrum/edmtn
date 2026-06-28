"""Track 2 (HPC) — 2D space×time EDM network assembler prototype (Phase A1).

Track 2 abandons Track 1's sequential fold-then-compress (1D-MPS-in-time) and
instead lays the *whole* separable-bath EDM out as a single **2D tensor network**
(paper Sec. V), to be contracted in one shot by cuTensorNet (which owns path
search, slicing, hardware scheduling, and execution — no cotengra). This script
builds that network from the **shared physics layer** (the same `models` /
`cumulants` / `kernels` Track 1 uses) and contracts it with a pluggable backend.

The network *description* — a list of operands plus integer mode labels — is
backend-agnostic, so the geometry is validated **locally with NumPy einsum**
against Track 1's exact fold before any GPU/cuTensorNet run. On c1, pass
``--backend cutensornet`` to contract the identical description with cuQuantum.

Geometry (Gaudin / separable bath), for ``n_sites`` time columns and ``K`` rows:

    output(d²) ─ S(t_n) ── S ── … ── S(t_1) ─ vec(ρ0)        (system row)
                  │          │          │
                 op_1 ──── op_1 ──── op_1                     (sub-bath 1, D_a)
                  │          │          │
                  …          …          …
                  │          │          │
                 op_K ──── op_K ──── op_K                     (sub-bath K, D_a)
                  ●          ●          ●                     (top arms closed δ⁰)

* horizontal: the system row threads the d²=4 ``vec`` bond; each sub-bath row k
  is a uniform chain of transfer tensors with lateral bond ``D_a``=4, both
  boundaries fixed to 0 (σ₀=I / Ωₖ=I/2);
* vertical (d_phys=7 superoperator legs): the column index threads system →
  sub-bath 1 → … → sub-bath K through the picking tensor (already fused into the
  per-row ``op``); the top arm above sub-bath K is closed with δ⁰ to give ρ(T).

Reducing all open arms (δ⁰) and contracting onto ``vec(ρ0)`` yields ``vec(ρ(T))``
→ reshape to the (d×d) reduced density matrix.

Phase A1 contracts the network **exactly** (no truncation) on a small case to
prove the assembly reproduces Track 1. Truncation (ξ / cutoff_mode) enters in
Phase B via cuTensorNet's approximate contraction (``contract_decompose``).

Run (local, NumPy geometry check):

    PYTHONPATH=src python examples/track2_2d_assembler.py --K 2 --n-steps 2 --order 1
    PYTHONPATH=src python examples/track2_2d_assembler.py --K 3 --n-steps 3 --order 2

Run (c1, cuTensorNet):

    PYTHONPATH=src python examples/track2_2d_assembler.py --backend cutensornet --K 3 --n-steps 4
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np


# --------------------------------------------------------------------------
# network assembly (backend-agnostic description)
# --------------------------------------------------------------------------

def build_2d_network(model, expander, eps: float, n_steps: int, sub_baths=None):
    """Build the 2D space×time EDM network for a separable-bath model.

    Returns ``(operands, modes, out_modes, meta)``: ``operands[i]`` is contracted
    with integer mode labels ``modes[i]``; ``out_modes`` are the open output
    labels (the d² ``vec(ρ(T))`` leg). This is the canonical einsum-interleaved
    description consumed by every backend.
    """
    from edmtn.kernels.separable_mpo import SeparableKernelEngine

    order = expander.order
    n_sites = order * n_steps
    d = model.system_dim

    kernel_engine = SeparableKernelEngine.from_model(model, n_steps * eps, eps)
    K = kernel_engine.K
    n_fold = K if sub_baths is None else min(int(sub_baths), K)
    if n_fold < 1:
        raise ValueError(f"sub_baths must be >= 1, got {sub_baths}")
    d_phys = kernel_engine.d_phys

    # -- system families per column (newest-first), mirroring
    #    SeparableBathEvolution._build_system_mps exactly
    fam_cache: dict[int, list] = {}
    sys_tensors = []
    for p in range(n_sites):
        g = n_sites - p                 # sub-step 1..n_sites (oldest = 1)
        n = (g - 1) // order + 1        # physical step
        sub = (g - 1) % order           # 0 -> S_1, 1 -> S_2
        if n not in fam_cache:
            fam_cache[n] = expander.build_at(model, n * eps, eps).families
        sys_tensors.append(np.asarray(fam_cache[n][sub], dtype=np.complex128))

    # -- bath site tensors per row (newest-first), boundaries pre-sliced to 0
    bath_sites = [
        [np.asarray(s, dtype=np.complex128)
         for s in kernel_engine.for_sub_bath(k).get_kernel_mpo(n_sites).site_tensors]
        for k in range(n_fold)
    ]

    # -- mode (bond) labels
    ids = itertools.count()
    o = [next(ids) for _ in range(n_sites + 1)]                        # system d² bonds; o[0]=output
    lat = [[next(ids) for _ in range(n_sites + 1)] for _ in range(n_fold)]   # lateral D_a bonds
    v = [[next(ids) for _ in range(n_fold + 1)] for _ in range(n_sites)]     # vertical phi legs

    operands: list = []
    modes: list = []

    # system row: S[phi=v[p][0], i=o[p], j=o[p+1]]  (matrix S_newest @ … @ S_oldest @ vec)
    for p in range(n_sites):
        operands.append(sys_tensors[p])
        modes.append([v[p][0], o[p], o[p + 1]])

    # bath rows: op[u=v[p][k+1], dn=v[p][k], l=lat[k][p], r=lat[k][p+1]];
    # drop size-1 boundary lateral axes so no dangling indices reach the backend
    for k in range(n_fold):
        for p in range(n_sites):
            site = bath_sites[k][p]                 # (u, dn, l, r)
            u_, dn_, l_, r_ = site.shape
            md = [v[p][k + 1], v[p][k]]
            shape = [u_, dn_]
            if l_ > 1:
                md.append(lat[k][p]); shape.append(l_)
            if r_ > 1:
                md.append(lat[k][p + 1]); shape.append(r_)
            operands.append(site.reshape(shape))
            modes.append(md)

    # top arms closed with δ⁰ (select phi_up = 0 above the last sub-bath)
    e0 = np.zeros(d_phys, dtype=np.complex128)
    e0[0] = 1.0
    for p in range(n_sites):
        operands.append(e0)
        modes.append([v[p][n_fold]])

    # right boundary: vec(ρ(0)) on the oldest system bond
    rho0 = model.initial_system_state().reshape(-1).astype(np.complex128)
    operands.append(rho0)
    modes.append([o[n_sites]])

    out_modes = [o[0]]
    meta = dict(d=d, d_phys=d_phys, n_sites=n_sites, K=K, n_fold=n_fold,
                n_operands=len(operands))
    return operands, modes, out_modes, meta


# --------------------------------------------------------------------------
# contraction backends (same description, different engine)
# --------------------------------------------------------------------------

def contract_numpy(operands, modes, out_modes):
    """Contract the network with NumPy's interleaved einsum (local geometry check)."""
    args: list = []
    for op, md in zip(operands, modes):
        args += [op, md]
    args.append(out_modes)
    return np.einsum(*args, optimize=True)


def contract_cutensornet(operands, modes, out_modes):
    """Contract the identical description with cuQuantum cuTensorNet (c1 only).

    cuTensorNet owns path search + slicing + execution. Operands are moved to GPU
    via CuPy; the interleaved (operand, modes, …, out_modes) form is passed through.
    """
    import cupy as cp
    try:
        from cuquantum.tensornet import contract as cutn_contract
    except ImportError:  # older cuquantum-python layout
        from cuquantum import contract as cutn_contract

    gpu_ops = [cp.asarray(op) for op in operands]
    args: list = []
    for op, md in zip(gpu_ops, modes):
        args += [op, md]
    args.append(out_modes)
    res = cutn_contract(*args)
    return cp.asnumpy(res)


_BACKENDS = {"numpy": contract_numpy, "cutensornet": contract_cutensornet}


def reduced_density_matrix_2d(model, expander, eps, n_steps, sub_baths=None, backend="numpy"):
    """Assemble + contract the 2D network → ρ(T) (d×d)."""
    operands, modes, out_modes, meta = build_2d_network(
        model, expander, eps, n_steps, sub_baths=sub_baths)
    vec = _BACKENDS[backend](operands, modes, out_modes)
    d = meta["d"]
    return np.asarray(vec).reshape(d, d), meta


# --------------------------------------------------------------------------
# Track 1 reference (exact fold, no compression)
# --------------------------------------------------------------------------

def reduced_density_matrix_track1(model, expander, eps, n_steps, sub_baths=None):
    """Track 1 exact (uncompressed) fold ρ(T) — the correctness anchor."""
    from edmtn.evolution.separable_bath import SeparableBathEvolution
    from edmtn.kernels.separable_mpo import SeparableKernelEngine

    kernel_engine = SeparableKernelEngine.from_model(model, n_steps * eps, eps)
    evo = SeparableBathEvolution(expander)
    result = evo.run(model, kernel_engine, eps, n_steps,
                     compress=False, sub_baths=sub_baths)
    return result.mps.reduced_density_matrix()


def _expectation(rho, op):
    return complex(np.trace(op @ rho))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _make_expander(order: int):
    if order == 1:
        from edmtn.expansion.first_order import FirstOrderExpander
        return FirstOrderExpander()
    if order == 2:
        from edmtn.expansion.second_order import SecondOrderExpander
        return SecondOrderExpander()
    raise ValueError("order must be 1 or 2")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=sorted(_BACKENDS), default="numpy")
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=2, help="number of bath spins")
    ap.add_argument("--n-steps", type=int, default=2)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--order", type=int, choices=(1, 2), default=1)
    ap.add_argument("--sub-baths", type=int, default=None,
                    help="fold only the first L sub-baths (default: all K)")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the Track 1 cross-check (large cases blow up exactly)")
    args = ap.parse_args()

    from edmtn.models import GaudinModel

    model = GaudinModel(g=args.g, K=args.K, time_step_order=args.order)
    expander = _make_expander(args.order)
    Sz = model.system_operators()["Sz"]

    rho2d, meta = reduced_density_matrix_2d(
        model, expander, args.eps, args.n_steps,
        sub_baths=args.sub_baths, backend=args.backend)
    sz2d = _expectation(rho2d, Sz).real

    print(f"[2D / {args.backend}] K={args.K} n_steps={args.n_steps} order={args.order} "
          f"-> {meta['n_operands']} operands, n_sites={meta['n_sites']}, n_fold={meta['n_fold']}")
    print(f"  Tr(rho)   = {np.trace(rho2d).real:+.12f}")
    print(f"  <S_z(T)>  = {sz2d:+.12f}")

    if not args.no_check:
        rho1 = reduced_density_matrix_track1(
            model, expander, args.eps, args.n_steps, sub_baths=args.sub_baths)
        sz1 = _expectation(rho1, Sz).real
        err = float(np.max(np.abs(rho2d - rho1)))
        print(f"[Track 1 exact] <S_z(T)> = {sz1:+.12f}")
        print(f"  max|rho_2d - rho_track1| = {err:.3e}  "
              f"{'PASS' if err < 1e-10 else 'FAIL'}")


if __name__ == "__main__":
    main()
