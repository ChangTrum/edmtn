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


def test_container_fold_method_consistency():
    """A container fold is consistent across compression methods at a tight cutoff."""
    from edmtn.kernels.separable_mpo import SeparableKernelEngine
    from edmtn.expansion.second_order import SecondOrderExpander
    from edmtn.evolution.separable_bath import SeparableBathEvolution

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
    a = QuimbEDM.from_edmmps(base.copy()).fold(
        mpo, cutoff=1e-12, cutoff_mode="rel", method="zipup", max_bond=None)
    b = QuimbEDM.from_edmmps(base.copy()).fold(
        mpo, cutoff=1e-12, cutoff_mode="rel", method="direct", max_bond=None)
    assert np.max(np.abs(a.reduced_density_matrix() - b.reduced_density_matrix())) < 1e-9


#: The fixed part of the convergence check below; the reference and every cutoff-mode run
#: share it, so a difference can only come from the cutoff rule under test.
#:
#: These numbers are chosen from measurement, not habit, and each one is load-bearing:
#:
#: * ``max_bond = 64`` **binds**.  Without a cap this configuration reaches bond 260, so
#:   the rank-limited path is genuinely exercised -- which is the point of having a cap at
#:   all.  A far larger cap on a far larger problem (the historical ``K=12, T=3.0,
#:   max_bond=400``) exercises the same path for ~80x the wall time.
#: * The cap must not be pushed further down.  At ``max_bond = 32`` the reference and both
#:   cutoff modes agree to **exactly** ``0.0``: the cap alone then decides the state and the
#:   cutoff rule under test has no effect at all, so the assertion would hold no matter what
#:   the cutoff modes did.  Measured, and the reason 32 is not used.
#: * The assertion still fails when it should: at ``cutoff_mode='rel'`` with ``cutoff=1e-4``
#:   the error is 2.8e-4, above the 1e-4 threshold; by ``3e-3`` the pipeline's own imaginary
#:   part guard rejects the run first.  The two cutoffs actually tested land at 1.7e-7 and
#:   3.7e-9.
_CONVERGENCE_COMMON = dict(T=2.0, eps=0.2, expansion_order=2, max_bond=64, channel=3)
_CONVERGENCE_K = 6


@pytest.fixture(scope="module")
def gaudin_tight_reference():
    """The tight-cutoff reference solve, computed ONCE for every cutoff mode.

    It does not depend on the parametrised ``(mode, cutoff)``, so building it inside the
    test body ran the identical solve once per parameter set.  Sharing it changes no input
    and no assertion; it only stops the same numbers being computed twice.
    """
    return solve(GaudinModel(g=1.0, K=_CONVERGENCE_K), cutoff=1e-12, cutoff_mode="rel",
                 **_CONVERGENCE_COMMON)


def _polarization_deviation(ref, got) -> float:
    """``max |<S_z(t)>_ref - <S_z(t)>_got|`` on the common grid."""
    n = min(len(ref.polarization), len(got.polarization))
    return float(np.max(np.abs(np.asarray(ref.polarization[:n])
                               - np.asarray(got.polarization[:n]))))


@pytest.mark.parametrize("mode,cutoff", [("rsum2", 1e-13), ("rel", 1e-8)])
def test_container_solver_physics_converges(mode, cutoff, gaudin_tight_reference):
    """Separable Gaudin <S_z(t)> at a working cutoff matches a tight reference."""
    model = GaudinModel(g=1.0, K=_CONVERGENCE_K)
    ref = gaudin_tight_reference
    got = solve(model, cutoff=cutoff, cutoff_mode=mode, **_CONVERGENCE_COMMON)
    # The cap is a HARD upper limit, so equality is the whole statement: `>=` would also
    # accept a bond the cap failed to enforce.  Measured: reference and both working
    # cutoffs all sit exactly on 64, i.e. the rank-limited path is what runs.
    cap = _CONVERGENCE_COMMON["max_bond"]
    assert max(ref.bond_dims) == cap
    assert max(got.bond_dims) == cap
    assert _polarization_deviation(ref, got) < 1e-4


def test_the_convergence_threshold_actually_discriminates(gaudin_tight_reference):
    """The negative control: a cutoff coarse enough to matter must FAIL the 1e-4 threshold.

    Without this, a future parameter change could land in a regime where the cap alone
    decides the state and the cutoff rule has no effect -- measured at ``max_bond = 32``,
    where the reference and both cutoff modes agree to exactly ``0.0`` -- and the test
    above would stay green while checking nothing.  Shares the same reference, so it costs
    one extra small solve.

    ``rel`` at ``1e-4`` is coarse enough to be caught (measured deviation 2.8e-4, and the
    bond drops to 20 because the cutoff, not the cap, then decides).  Coarser still and the
    pipeline's own imaginary-part guard rejects the run first; that is also a failure, so
    it is accepted here as one.
    """
    model = GaudinModel(g=1.0, K=_CONVERGENCE_K)
    try:
        got = solve(model, cutoff=1e-4, cutoff_mode="rel", **_CONVERGENCE_COMMON)
    except ValueError as exc:                      # the polarization imaginary-part guard
        assert "imaginary part" in str(exc)
        return
    assert _polarization_deviation(gaudin_tight_reference, got) >= 1e-4


@pytest.mark.gpu
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
    common = dict(T=1.0, eps=0.25, expansion_order=2, cutoff=1e-8, cutoff_mode="rel", channel=3)
    ref = solve(model, compress_method="direct", **common)  # default decomp=exact
    got = solve(model, compress_method="direct",
                compress_decomp=decomp, compress_decomp_q=q, compress_canon=canon, **common)
    n = min(len(ref.polarization), len(got.polarization))
    err = float(np.max(np.abs(np.asarray(ref.polarization[:n])
                              - np.asarray(got.polarization[:n]))))
    assert err < 1e-4


@pytest.mark.parametrize("order", [1, 2])
def test_container_single_bath_physics_converges(order):
    """Single-bath (spin-boson) <S_z(t)> at a working cutoff matches a tight reference
    -- the chain-growing engine (step + compress), not the separable fold."""
    model = SpinBosonModel(J0=0.6, omega_c=5.0, mu=1.0)
    common = dict(T=2.0, eps=0.1, expansion_order=order, channel=1)
    ref = solve(model, cutoff=1e-12, cutoff_mode="rel", **common)
    got = solve(model, cutoff=1e-8, cutoff_mode="rel", **common)
    n = min(len(ref.polarization), len(got.polarization))
    err = float(np.max(np.abs(np.asarray(ref.polarization[:n])
                              - np.asarray(got.polarization[:n]))))
    assert err < 1e-4
