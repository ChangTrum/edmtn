"""QuimbEDM container tests (re-platform structural sub-step).

The EDM carried as a quimb TensorNetwork must (a) round-trip to/from EDMMPS
preserving the reduced state, (b) compute the same reduced density matrix as the
native container, and (c) drive the separable solver to the same physics as the
native StandardSVD path -- the observable is the invariant.
"""

from __future__ import annotations

import numpy as np
import pytest

from edmtn.evolution.mps_utils import EDMMPS
from edmtn.evolution.quimb_edm import QuimbEDM
from edmtn.driver.solver import solve
from edmtn.models import GaudinModel, SpinBosonModel


def _random_edmmps(n, d, d_phys, chi, rng):
    d2 = d * d
    tensors, left = [], d2
    for p in range(n):
        right = d2 if p == n - 1 else chi
        t = (rng.standard_normal((d_phys, left, right))
             + 1j * rng.standard_normal((d_phys, left, right))).astype(np.complex128)
        tensors.append(t)
        left = right
    return EDMMPS(tensors=tensors, d=d, d_phys=d_phys, rho0_vec=np.ones(d2, np.complex128))


def test_container_roundtrip_and_reduced_dm():
    rng = np.random.default_rng(0)
    mps = _random_edmmps(6, 2, 7, 10, rng)
    q = QuimbEDM.from_edmmps(mps)
    # structure mirrors EDMMPS
    assert q.num_sites == mps.num_sites
    assert q.bond_dims == mps.bond_dims
    # reduced density matrix matches the native closure
    assert np.max(np.abs(q.reduced_density_matrix() - mps.reduced_density_matrix())) < 1e-10
    # round-trip back to EDMMPS preserves the reduced state
    back = q.to_edmmps()
    assert np.max(np.abs(back.reduced_density_matrix() - mps.reduced_density_matrix())) < 1e-10


def test_container_fold_matches_native():
    """One quimb container fold == native _apply_sub_bath + quimb compress."""
    from edmtn.kernels.separable_mpo import SeparableKernelEngine
    from edmtn.expansion.second_order import SecondOrderExpander
    from edmtn.evolution.separable_bath import SeparableBathEvolution
    from edmtn.evolution.mps_utils import compress

    model = GaudinModel(g=1.0, K=4)
    eps, T, order = 0.25, 1.0, 2
    ke = SeparableKernelEngine.from_model(model, T=T, eps=eps)
    ev = SeparableBathEvolution(expander=SecondOrderExpander())
    d, d_phys = model.system_dim, ke.d_phys
    n_steps = int(round(T / eps))
    n = order * n_steps
    rho0 = model.initial_system_state().reshape(-1).astype(np.complex128)
    base = ev._build_system_mps(model, eps, n_steps, order, d, d_phys, rho0, lambda a: a)

    mpo = list(ke.for_sub_bath(0).get_kernel_mpo(n).site_tensors)
    cut, mode = 1e-13, "rsum2"
    # native two-stage
    nat = ev._apply_sub_bath(base.copy(), mpo, d, d_phys, rho0)
    nat, _ = compress(nat, engine="quimb", compress_cutoff=cut,
                      compress_cutoff_mode=mode, compress_method="zipup", max_bond=None)
    # container fold
    con = QuimbEDM.from_edmmps(base.copy()).fold(
        mpo, cutoff=cut, cutoff_mode=mode, method="zipup", max_bond=None)
    assert con.max_bond == nat.max_bond
    assert np.max(np.abs(con.reduced_density_matrix() - nat.reduced_density_matrix())) < 1e-8


@pytest.mark.parametrize("mode,cutoff", [("rsum2", 1e-13), ("rel", 1e-8)])
def test_container_solver_matches_physics(mode, cutoff):
    """Separable Gaudin <S_z(t)> via the quimb container matches the native solve."""
    model = GaudinModel(g=1.0, K=12)
    common = dict(T=3.0, eps=0.2, expansion_order=2, cutoff=1e-6, max_bond=400, channel=3)
    ref = solve(model, **common)
    got = solve(model, compression="quimb", compress_cutoff_mode=mode,
                compress_cutoff=cutoff, **common)
    n = min(len(ref.polarization), len(got.polarization))
    err = float(np.max(np.abs(np.asarray(ref.polarization[:n])
                              - np.asarray(got.polarization[:n]))))
    assert err < 1e-4


@pytest.mark.skipif(True, reason="no CuPy GPU available")
def test_container_reduced_dm_stays_on_gpu():  # pragma: no cover - GPU node only
    """reduced_density_matrix must keep CuPy-backed results on device (regression:
    np.asarray forced an implicit CuPy->NumPy conversion and crashed the GPU path)."""
    import cupy as cp  # noqa: PLC0415

    rng = cp.random.default_rng(0)
    d, d_phys, chi, n, d2 = 2, 7, 10, 5, 4
    tensors, left = [], d2
    for p in range(n):
        right = d2 if p == n - 1 else chi
        tensors.append((rng.standard_normal((d_phys, left, right))
                        + 1j * rng.standard_normal((d_phys, left, right))).astype(cp.complex128))
        left = right
    mps = EDMMPS(tensors=tensors, d=d, d_phys=d_phys, rho0_vec=cp.ones(d2, cp.complex128))
    rho = QuimbEDM.from_edmmps(mps).reduced_density_matrix()
    assert rho.__class__.__module__.split(".")[0] == "cupy"  # stayed on device, no crash


def test_decomp_canon_knobs_helpers():
    """The decomposition/canonicalisation selectors map to the right quimb opts."""
    from edmtn.evolution.quimb_decomp import compress_opts_for, canonize_opts_for

    assert compress_opts_for("exact", 2) == {}
    assert compress_opts_for("rsvd", 2) == {"method": "edm_rsvd"}  # registers the driver
    import quimb.tensor.decomp as d
    assert "edm_rsvd" in d._SPLIT_FNS
    assert canonize_opts_for("quimb") == {}
    assert canonize_opts_for("householder") == {"method": "qr"}
    assert canonize_opts_for("cholqr") == {"method": "qr:cholesky"}
    with pytest.raises(ValueError):
        compress_opts_for("bogus", 2)
    with pytest.raises(ValueError):
        canonize_opts_for("bogus")


@pytest.mark.parametrize("decomp,q,canon", [
    ("rsvd", 2, "quimb"), ("rsvd", 0, "quimb"),
    ("exact", 2, "householder"), ("exact", 2, "cholqr"),
])
def test_decomp_canon_knobs_match_exact(decomp, q, canon):
    """rSVD (q=2/0, silent guard) and the canon options reproduce the native solve."""
    model = GaudinModel(g=1.0, K=6)
    common = dict(T=1.0, eps=0.25, expansion_order=2, cutoff=1e-6, channel=3)
    ref = solve(model, **common)
    got = solve(model, compression="quimb", compress_method="direct",
                compress_cutoff_mode="rel", compress_cutoff=1e-8,
                compress_decomp=decomp, compress_decomp_q=q, compress_canon=canon, **common)
    n = min(len(ref.polarization), len(got.polarization))
    err = float(np.max(np.abs(np.asarray(ref.polarization[:n])
                              - np.asarray(got.polarization[:n]))))
    assert err < 1e-4


@pytest.mark.parametrize("order", [1, 2])
def test_container_single_bath_matches_physics(order):
    """Single-bath (spin-boson) <S_z(t)> via the quimb container (step + compress)
    matches the native solve -- the chain-growing engine, not the separable fold."""
    model = SpinBosonModel(J0=0.6, omega_c=5.0, mu=1.0)
    common = dict(T=2.0, eps=0.1, expansion_order=order, cutoff=1e-6, channel=1)
    ref = solve(model, **common)
    got = solve(model, compression="quimb", compress_cutoff_mode="rsum2",
                compress_cutoff=1e-13, **common)
    n = min(len(ref.polarization), len(got.polarization))
    err = float(np.max(np.abs(np.asarray(ref.polarization[:n])
                              - np.asarray(got.polarization[:n]))))
    assert err < 1e-4
