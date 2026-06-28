"""Track 2 (HPC) — cuTensorNet install/interop sanity + 2D assembler on GPU (Phase A0/A1).

Run on c1 (NVIDIA A800, env ``edmtn-gpu`` with ``cuquantum-python-cu12`` installed):

    PYTHONPATH=src python examples/track2_cutensornet_sanity.py

It (1) reports the cuQuantum / cuTensorNet / CuPy / CUDA versions **and dumps the
high-level API surface** (the module layout moved across cuquantum-python
releases, so this makes any path mismatch self-diagnosing), (2) smoke-tests the
``contract`` primitive the assembler relies on, plus a **best-effort**
``tensor.decompose`` SVD/QR (used later in Phase B truncation — non-fatal here),
and (3) contracts the **2D space×time EDM network** (``track2_2d_assembler``) with
the cuTensorNet backend for several small Gaudin cases, cross-checking each
against Track 1's exact CPU fold (the real Phase-A1 gate). A non-zero exit means
the contract smoke or the assembler gate failed.
"""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np

# make the sibling example importable (examples/ is not a package)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import track2_2d_assembler as t2  # noqa: E402


def _section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def _first_import(paths):
    """Return the first importable module from ``paths`` (or None)."""
    for p in paths:
        try:
            return importlib.import_module(p)
        except Exception:
            continue
    return None


def report_versions():
    """Print versions + the cuquantum high-level API surface (self-diagnosing)."""
    _section("versions")
    import cupy as cp
    import cuquantum

    print(f"cuquantum   : {getattr(cuquantum, '__version__', '?')}")
    cutn_mod = _first_import(
        ["cuquantum.bindings.cutensornet", "cuquantum.cutensornet", "cutensornet"])
    if cutn_mod is not None:
        gv = getattr(cutn_mod, "get_version", None)
        ver = gv() if callable(gv) else getattr(cutn_mod, "__version__", "?")
        print(f"cutensornet : {cutn_mod.__name__} -> {ver}")
    else:
        print("cutensornet : (binding not found at known paths)")
    print(f"cupy        : {cp.__version__}")
    rt = cp.cuda.runtime.runtimeGetVersion()
    print(f"CUDA runtime: {rt // 1000}.{(rt % 1000) // 10}")
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    name = name.decode() if isinstance(name, bytes) else name
    _, total = cp.cuda.Device(0).mem_info
    print(f"device 0    : {name}, {total / 1e9:.0f} GB")

    # API discovery — print once so any remaining path can be fixed precisely
    print(f"dir(cuquantum) = {[a for a in dir(cuquantum) if not a.startswith('_')]}")
    tn = _first_import(["cuquantum.tensornet"])
    if tn is not None:
        print(f"dir(cuquantum.tensornet) = {[a for a in dir(tn) if not a.startswith('_')]}")
    else:
        print("cuquantum.tensornet : (not importable)")
    return cp


def smoke_primitives(cp) -> None:
    """Contract smoke (required) + best-effort SVD/QR decompose (Phase-B prep)."""
    _section("raw cuTensorNet primitives")
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((32, 24)) + 1j * rng.standard_normal((32, 24))).astype(np.complex128)
    a_g = cp.asarray(a)

    # contract (high-level) — the primitive the 2D assembler relies on
    tn = _first_import(["cuquantum.tensornet", "cuquantum"])
    contract = getattr(tn, "contract", None) if tn is not None else None
    if contract is None:
        raise RuntimeError("no cuTensorNet `contract` entry point found")
    b = cp.asarray(rng.standard_normal((24, 16)) + 1j * rng.standard_normal((24, 16)))
    c = contract("ij,jk->ik", a_g, b)
    c_err = float(cp.max(cp.abs(c - a_g @ b)))
    print(f"contract matmul   max|Δ| = {c_err:.2e}")
    assert c_err < 1e-10, "contract smoke failed"

    # decompose (SVD/QR) — best-effort; matters for Phase B, not the A1 gate
    tmod = _first_import(["cuquantum.tensornet.tensor", "cuquantum.cutensornet.tensor"])
    if tmod is None or not hasattr(tmod, "decompose"):
        print("decompose : no entry point at known paths — skipped (Phase B will wire it)")
        return
    try:
        decompose = tmod.decompose
        SVDMethod = getattr(tmod, "SVDMethod", None)
        method = SVDMethod() if SVDMethod is not None else "SVD"
        u, s, v = decompose("ij->ik,kj", a_g, method=method)
        print(f"SVD reconstruction max|Δ| = {float(cp.max(cp.abs((u * s) @ v - a_g))):.2e}")
        q, r = decompose("ij->ik,kj", a_g, method="QR")
        print(f"QR  reconstruction max|Δ| = {float(cp.max(cp.abs(q @ r - a_g))):.2e}")
    except Exception as e:  # noqa: BLE001 - best-effort, surface but don't fail
        print(f"decompose smoke skipped ({type(e).__name__}: {e})")


def assembler_on_gpu() -> None:
    """Contract the 2D EDM network on GPU (cuTensorNet) vs Track 1 exact (CPU)."""
    _section("2D assembler on GPU vs Track 1 exact")
    from edmtn.models import GaudinModel

    cases = [
        dict(K=2, n_steps=2, order=1),
        dict(K=3, n_steps=3, order=1),
        dict(K=2, n_steps=2, order=2),
        dict(K=4, n_steps=2, order=1),
        dict(K=3, n_steps=2, order=2, sub_baths=2),
    ]
    worst = 0.0
    for c in cases:
        order = c["order"]
        model = GaudinModel(g=1.0, K=c["K"], time_step_order=order)
        expander = t2._make_expander(order)
        Sz = model.system_operators()["Sz"]
        rho_gpu, _ = t2.reduced_density_matrix_2d(
            model, expander, 0.1, c["n_steps"],
            sub_baths=c.get("sub_baths"), backend="cutensornet")
        rho_ref = t2.reduced_density_matrix_track1(
            model, expander, 0.1, c["n_steps"], sub_baths=c.get("sub_baths"))
        err = float(np.max(np.abs(rho_gpu - rho_ref)))
        worst = max(worst, err)
        sz = float(np.trace(Sz @ rho_gpu).real)
        tag = "PASS" if err < 1e-10 else "FAIL"
        print(f"  {c}  <S_z>={sz:+.10f}  max|Δ vs Track1|={err:.2e}  {tag}")
    assert worst < 1e-10, f"assembler-on-GPU mismatch {worst:.2e}"
    print(f"all cases PASS (worst {worst:.2e})")


def main() -> int:
    cp = report_versions()
    smoke_primitives(cp)
    assembler_on_gpu()
    _section("RESULT")
    print("Phase A0/A1 GPU sanity: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
