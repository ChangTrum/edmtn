"""Track 2 Phase B0 probe — settle the exact/approximate contraction surface (c1).

Runs the 2D EDM network through several routes to answer, empirically on an A800:

  P1. **exact one-shot, cuTensorNet-owned path** — `cuquantum.tensornet.contract`
      (its optimizer owns path + slicing). Report `exact`-mode error metrics
      (hermiticity ‖ρ−ρ†‖, |Tr ρ−1|, vs Track 1 baseline) + optimizer info.
  P2. **routing through quimb / cotengra** — quimb `.contract(backend="cuquantum")`,
      and cotengra `array_contract_expression(implementation="cuquantum")` (the
      conduit that hands the whole contraction to cuTensorNet's Network).
  P3. **approximate with cutoff** — (a) through quimb: `tn.contract_compressed(
      optimize, max_bond, cutoff, ...)` on cupy; (b) cuTensorNet native truncation
      surface: the dataclass fields of `SVDMethod`/`MPSConfig`/`ContractDecomposeAlgorithm`
      (so we learn the exact cutoff param names) + a `contract_decompose` smoke.

Defensive per-section (prints what works / the exception) so one job maps the
surface. Run on c1 via cluster/track2_b0.sbatch.
"""

from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import track2_2d_assembler as t2  # noqa: E402


def _section(t):
    print(f"\n=== {t} ===", flush=True)


def _interleaved(operands, modes, out_modes):
    args = []
    for op, md in zip(operands, modes):
        args += [op, list(md)]
    args.append(list(out_modes))
    return args


def _metrics(rho, ref, tag):
    herm = float(np.max(np.abs(rho - rho.conj().T)))
    trace = complex(np.trace(rho))
    err = float(np.max(np.abs(rho - ref)))
    print(f"  [{tag}] vs Track1 max|Δ|={err:.2e}  ‖ρ−ρ†‖={herm:.2e}  "
          f"Tr ρ={trace.real:+.10f}{trace.imag:+.1e}i")
    return err


def to_quimb_tn(operands, modes):
    import quimb.tensor as qtn
    return qtn.TensorNetwork([qtn.Tensor(op, inds=tuple(f"i{m}" for m in md))
                              for op, md in zip(operands, modes)])


def _fields(cls):
    try:
        return [f.name for f in dataclasses.fields(cls)]
    except Exception:
        return [a for a in dir(cls) if not a.startswith("_")]


def main() -> int:
    from edmtn.models import GaudinModel

    K, n_steps, order = 3, 5, 1
    model = GaudinModel(g=1.0, K=K, time_step_order=order)
    expander = t2._make_expander(order)
    operands, modes, out_modes, meta = t2.build_2d_network(model, expander, 0.1, n_steps)
    d = meta["d"]
    out = out_modes[0]
    out_ix = f"i{out}"
    print(f"case K={K} n_steps={n_steps} order={order}: {len(operands)} operands")
    ref = t2.reduced_density_matrix_track1(model, expander, 0.1, n_steps)

    import cupy as cp
    gpu_ops = [cp.asarray(o) for o in operands]

    # -- P1: exact one-shot, cuTensorNet owns path -----------------------------
    _section("P1 exact one-shot via cuquantum.tensornet.contract")
    from cuquantum.tensornet import contract
    res, info = None, None
    try:
        res, info = contract(*_interleaved(gpu_ops, modes, out_modes), return_info=True)
    except TypeError:
        res = contract(*_interleaved(gpu_ops, modes, out_modes))
    rho = cp.asnumpy(res).reshape(d, d)
    _metrics(rho, ref, "exact/contract")
    if info is not None:
        oi = info[1] if isinstance(info, tuple) else info
        print(f"  optimizer_info: num_slices={getattr(oi, 'num_slices', '?')} "
              f"opt_cost={getattr(oi, 'opt_cost', getattr(oi, 'flop_count', '?'))} "
              f"path_len={len(getattr(oi, 'path', []) or [])}")

    # -- P2: through quimb / cotengra -----------------------------------------
    _section("P2 routing through quimb / cotengra")
    tn = to_quimb_tn(gpu_ops, modes)
    for kw in (dict(optimize="auto"), dict(optimize="auto", backend="cuquantum")):
        try:
            r = tn.contract(output_inds=(out_ix,), **kw)
            arr = cp.asnumpy(r.data if hasattr(r, "data") else r).reshape(d, d)
            _metrics(arr, ref, f"quimb.contract({kw})")
        except Exception as e:  # noqa: BLE001
            print(f"  quimb.contract({kw}) -> {type(e).__name__}: {e}")
    try:
        import cotengra as ctg
        expr = ctg.array_contract_expression(
            [list(m) for m in modes], list(out_modes),
            shapes=[o.shape for o in operands],
            optimize="auto", implementation="cuquantum")
        r = expr(*gpu_ops)
        _metrics(cp.asnumpy(r).reshape(d, d), ref, "cotengra expr impl=cuquantum")
    except Exception as e:  # noqa: BLE001
        print(f"  cotengra array_contract_expression(impl=cuquantum) -> {type(e).__name__}: {e}")

    # -- P3a: approximate through quimb (contract_compressed) ------------------
    _section("P3a approximate via quimb contract_compressed")
    for cutoff in (1e-2, 1e-4):
        try:
            tn3 = to_quimb_tn([cp.asarray(o) for o in operands], modes)
            r = tn3.contract_compressed(
                "auto", output_inds=(out_ix,), max_bond=64, cutoff=cutoff)
            arr = cp.asnumpy(r.data if hasattr(r, "data") else r).reshape(d, d)
            print(f"  contract_compressed(cutoff={cutoff}) max|Δ vs exact|="
                  f"{float(np.max(np.abs(arr - ref))):.2e}")
        except Exception as e:  # noqa: BLE001
            print(f"  contract_compressed(cutoff={cutoff}) -> {type(e).__name__}: {e}")

    # -- P3b: cuTensorNet native truncation surface ---------------------------
    _section("P3b cuTensorNet native truncation surface")
    from cuquantum.tensornet.tensor import SVDMethod, QRMethod  # noqa: F401
    print(f"  SVDMethod fields = {_fields(SVDMethod)}")
    try:
        from cuquantum.tensornet.experimental import (
            ContractDecomposeAlgorithm, MPSConfig, TNConfig, contract_decompose)
        print(f"  ContractDecomposeAlgorithm fields = {_fields(ContractDecomposeAlgorithm)}")
        print(f"  MPSConfig fields = {_fields(MPSConfig)}")
        print(f"  TNConfig fields = {_fields(TNConfig)}")
        # contract_decompose smoke: contract a,b then SVD-truncate the new bond
        a = cp.asarray((np.random.randn(12, 8) + 1j * np.random.randn(12, 8)))
        b = cp.asarray((np.random.randn(8, 12) + 1j * np.random.randn(8, 12)))
        for kwargs in ({"rel_cutoff": 1e-2}, {"max_extent": 4}):
            try:
                alg = ContractDecomposeAlgorithm(svd_method=SVDMethod(**kwargs))
                u, s, v = contract_decompose("ij,jk->il,lk", a, b, algorithm=alg)
                print(f"  contract_decompose(SVDMethod({kwargs})) -> "
                      f"u{tuple(u.shape)} s{tuple(s.shape)} v{tuple(v.shape)}")
            except Exception as e:  # noqa: BLE001
                print(f"  contract_decompose(SVDMethod({kwargs})) -> {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"  experimental import failed: {type(e).__name__}: {e}")

    _section("RESULT")
    print("B0 probe complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
