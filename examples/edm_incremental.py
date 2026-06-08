"""Shared machinery for the EDM incremental-update / compressed-sensing study.

This module backs three example scripts:

* ``validate_subspace.py``      -- per-bond subspace diagnostics (假设1).
* ``critical_L_and_scaling.py`` -- find the critical L* and the scaling law.
* ``projection_poc.py``         -- Tier-1 / Tier-2 proof-of-concept + speedups.

It exposes (a) a *streaming* separable-bath fold loop that snapshots the
compressed EDM-MPS after every sub-bath (so all ``L`` are available from one
pass), (b) transfer-matrix routines that compare the left singular subspaces of
two compressed EDM-MPS bond by bond, and (c) the Tier-1 pure-projection update
and a randomized SVD for Tier-2.

Everything is pure CPU / NumPy.  The fold loop mirrors
``SeparableBathEvolution.run`` exactly (same ``_apply_sub_bath`` +
``mps_utils.compress`` with ``cutoff_mode='rel_ref', ref_index=d**2``), so the
snapshots are byte-identical to what the solver produces.
"""

from __future__ import annotations

import numpy as np

from edmtn.decomposition.standard_svd import StandardSVD
from edmtn.evolution import mps_utils
from edmtn.evolution.mps_utils import EDMMPS
from edmtn.evolution.separable_bath import SeparableBathEvolution
from edmtn.expansion.first_order import FirstOrderExpander
from edmtn.expansion.second_order import SecondOrderExpander
from edmtn.kernels.separable_mpo import SeparableKernelEngine


# --------------------------------------------------------------------------
# streaming fold loop (snapshots every L)
# --------------------------------------------------------------------------

def _expander(order):
    return FirstOrderExpander() if order == 1 else SecondOrderExpander()


def make_context(model, *, T, eps, order, cutoff, max_bond):
    """Build the reusable fold context (kernel engine, system MPS, params)."""
    ke = SeparableKernelEngine.from_model(model, T=T, eps=eps)
    ev = SeparableBathEvolution(expander=_expander(order), decomposition=StandardSVD())
    d = model.system_dim
    n_steps = int(round(T / eps))
    n_sites = order * n_steps
    rho0 = model.initial_system_state().reshape(-1).astype(np.complex128)
    mps0 = ev._build_system_mps(model, eps, n_steps, order, d, ke.d_phys, rho0,
                                convert=lambda a: a)
    return {
        "model": model, "ke": ke, "ev": ev, "d": d, "d_phys": ke.d_phys,
        "n_sites": n_sites, "rho0": rho0, "ref_index": d * d,
        "cutoff": cutoff, "max_bond": max_bond, "eps": eps, "order": order,
        "mps0": mps0, "D_a": ke.corr.bond_dim,
    }


def fold_uncompressed(ctx, mps_L, k):
    """Fold sub-bath index ``k`` (0-based) onto compressed ``mps_L`` (no compress)."""
    mpo_sites = ctx["ke"].for_sub_bath(k).get_kernel_mpo(ctx["n_sites"]).site_tensors
    return ctx["ev"]._apply_sub_bath(mps_L, mpo_sites, ctx["d"], ctx["d_phys"], ctx["rho0"])


def compress_full(ctx, mps_unc):
    """Full-SVD compression (the production baseline); returns ``(mps, max_err)``."""
    mps, infos = mps_utils.compress(
        mps_unc, strategy=ctx["ev"].decomposition, max_bond=ctx["max_bond"],
        cutoff=ctx["cutoff"], cutoff_mode="rel_ref", ref_index=ctx["ref_index"],
    )
    return mps, (max((i["error"] for i in infos), default=0.0))


def fold_all_L(ctx, K=None, *, verbose=False):
    """Stream the separable fold; return ``{L: compressed EDMMPS}`` for ``L=1..K``."""
    K = ctx["ke"].K if K is None else min(int(K), ctx["ke"].K)
    mps = ctx["mps0"]
    out = {}
    for k in range(K):
        unc = fold_uncompressed(ctx, mps, k)
        mps = unc if unc.num_sites <= 1 else compress_full(ctx, unc)[0]
        out[k + 1] = mps.copy()
        if verbose:
            print(f"  folded L={k + 1:>3}: sites={mps.num_sites}, Dmax={mps.max_bond}")
    return out


# --------------------------------------------------------------------------
# transfer-matrix subspace comparison (same conventions as validate_subspace)
# --------------------------------------------------------------------------

def left_canonical_copy(mps):
    """Left-canonical copy (sites 0..n-2 isometric, centre at n-1)."""
    cm = mps.copy()
    mps_utils.left_canonicalize(cm)
    return cm


def cross_left_overlaps(A, B):
    """``E_tau = Q_A(tau)^H Q_B(tau)`` at every internal bond (``Es[tau-1]``).

    ``A``, ``B`` are *left-canonical* EDM-MPS with identical leg structure.  The
    singular values of ``E_tau`` are the cosines of the principal angles between
    the two left subspaces.
    """
    n = A.num_sites
    la0, lb0 = A.tensors[0].shape[1], B.tensors[0].shape[1]
    if la0 != lb0:
        raise ValueError(f"left-boundary mismatch: {la0} vs {lb0}")
    E = np.eye(la0, dtype=np.complex128)
    Es = []
    for p in range(n):
        Ap, Bp = A.tensors[p], B.tensors[p]
        T1 = np.einsum("pla,lm->pam", np.conj(Ap), E, optimize=True)
        E = np.einsum("pam,pmc->ac", T1, Bp, optimize=True)
        if p < n - 1:
            Es.append(E)
    return Es


def right_bond_density(B):
    """Bond density matrices ``rho_left(tau)`` of left-canonical ``B`` (``{tau: rho}``)."""
    n = B.num_sites
    R = np.eye(B.tensors[-1].shape[2], dtype=np.complex128)
    out = {}
    for p in range(n - 1, -1, -1):
        Bp = B.tensors[p]
        T = np.einsum("plr,rs->pls", Bp, R, optimize=True)
        R = np.einsum("pls,pms->lm", T, np.conj(Bp), optimize=True)
        if p >= 1:
            out[p] = R
    return out


def analyse_transition(mps_L, mps_L1, xi):
    """Per-bond subspace diagnostics for the fold ``L -> L+1`` (arrays, bond tau=1..n-1)."""
    if mps_L.num_sites != mps_L1.num_sites:
        raise ValueError("L and L+1 MPS have different site counts")
    A = left_canonical_copy(mps_L)
    B = left_canonical_copy(mps_L1)
    Es = cross_left_overlaps(A, B)
    rhos = right_bond_density(B)
    deltas = {"xi^2": xi * xi, "xi": xi, "sqrt(xi)": np.sqrt(xi)}

    rec = {"tau": [], "DL": [], "DL1": [], "dD": [], "resid_ratio": [],
           "chordal_norm": [], **{f"n_new[{k}]": [] for k in deltas}}

    for tau in range(1, A.num_sites):
        E = Es[tau - 1]
        R = rhos[tau]
        DL, DL1 = E.shape
        cos = np.clip(np.linalg.svd(E, compute_uv=False), 0.0, 1.0)
        total = float(np.trace(R).real)
        captured = float(np.einsum("ab,bc,ac->", E, R, np.conj(E), optimize=True).real)
        resid = np.sqrt(max(0.0, 1.0 - captured / total)) if total > 0 else 0.0
        sin2 = np.clip(1.0 - cos**2, 0.0, None)
        rec["tau"].append(tau)
        rec["DL"].append(DL)
        rec["DL1"].append(DL1)
        rec["dD"].append(DL1 - DL)
        rec["resid_ratio"].append(resid)
        rec["chordal_norm"].append(float(np.sqrt(sin2.sum())) / np.sqrt(DL))
        for name, delta in deltas.items():
            rec[f"n_new[{name}]"].append(DL1 - int(np.count_nonzero(cos >= 1.0 - delta)))
    return {k: np.asarray(v) for k, v in rec.items()}


# --------------------------------------------------------------------------
# Tier 1: pure projection onto the known (step-L) left subspace -- no SVD
# --------------------------------------------------------------------------

def tier1_project(mps_unc_L1, mps_compressed_L, old_isometry=None):
    """Project the uncompressed ``L+1`` MPS onto step-``L``'s left subspaces.

    Returns a bond-``D^(L)`` EDM-MPS ``= Pi^(L) psi_{L+1}`` built with one
    left-to-right GEMM sweep -- no SVD.  ``old_isometry`` may be a pre-canonical
    copy of step ``L`` (the "carried-over" subspace); otherwise it is computed.
    """
    A = old_isometry if old_isometry is not None else left_canonical_copy(mps_compressed_L)
    Ts = mps_unc_L1.tensors
    n = len(Ts)
    if A.num_sites != n:
        raise ValueError("site count mismatch")
    C = np.eye(A.tensors[0].shape[1], dtype=np.complex128)  # (old a_l, unc x_l)
    new = []
    for p in range(n):
        # M[phi, a_l, x_r] = sum_{x_l} C[a_l, x_l] T[phi, x_l, x_r]
        M = np.einsum("ax,pxr->par", C, Ts[p], optimize=True)
        if p < n - 1:
            Ap = A.tensors[p]
            new.append(Ap)
            C = np.einsum("pla,plr->ar", np.conj(Ap), M, optimize=True)
        else:
            new.append(M)
    return EDMMPS(tensors=new, d=mps_unc_L1.d, d_phys=mps_unc_L1.d_phys,
                  rho0_vec=mps_unc_L1.rho0_vec)


# --------------------------------------------------------------------------
# Tier 2 ingredients: randomized SVD (HMT) + per-bond projection benchmark
# --------------------------------------------------------------------------

def randomized_svd(M, rank, *, n_oversamples=10, n_iter=2, rng=None):
    """Halko-Martinsson-Tropp randomized SVD; returns ``(U, s, Vh)`` of rank ``<= rank``."""
    if rng is None:
        rng = np.random.default_rng(0)
    m, n = M.shape
    r = min(int(rank) + n_oversamples, m, n)
    if r <= 0:
        return (np.zeros((m, 0), M.dtype), np.zeros(0), np.zeros((0, n), M.dtype))
    Omega = rng.standard_normal((n, r)) + 1j * rng.standard_normal((n, r))
    Q, _ = np.linalg.qr(M @ Omega)
    for _ in range(n_iter):
        Q, _ = np.linalg.qr(M.conj().T @ Q)
        Q, _ = np.linalg.qr(M @ Q)
    B = Q.conj().T @ M
    Ub, s, Vh = np.linalg.svd(B, full_matrices=False)
    keep = min(int(rank), s.shape[0])
    return (Q @ Ub)[:, :keep], s[:keep], Vh[:keep]


def _truncate_right_of(ctx, mps, tau):
    """Sequentially truncate bonds ``n-1 .. tau+1`` (right-to-left), in place.

    Replicates ``mps_utils.truncate`` down to -- but not including -- bond ``tau``,
    so that site ``tau`` ends up with the *already-compressed* right bond the real
    sweep would present to it.  Sites ``0 .. tau-1`` are left untouched.
    """
    strat = ctx["ev"].decomposition
    for p in range(mps.num_sites - 1, tau, -1):
        G = mps.tensors[p]
        dp, chil, chir = G.shape
        mat = G.transpose(1, 0, 2).reshape(chil, dp * chir)
        res = strat.compress(mat, max_bond=ctx["max_bond"], cutoff=ctx["cutoff"],
                             cutoff_mode="rel_ref", ref_index=ctx["ref_index"], absorb="left")
        kk = res.bond
        mps.tensors[p] = res.right.reshape(kk, dp, chir).transpose(1, 0, 2)
        mps.tensors[p - 1] = np.tensordot(mps.tensors[p - 1], res.left, axes=([2], [0]))
    return mps


def bond_matrix_and_old_subspace(ctx, mps_compressed_L, k, tau):
    """Extract the step-``L+1`` bond matrix ``M`` at bond ``tau`` and the embedded
    step-``L`` orthonormal left subspace ``U`` (same row basis).

    ``M`` is exactly what the production truncation SVD factorises at bond ``tau``:
    site ``tau`` grouped ``(chi_left)|(phi,chi_right)`` *after* the bonds to its
    right have already been compressed by the sequential sweep (so ``chi_right`` is
    the real, truncated dimension -- not the uncompressed one).  ``U``
    (``chi_left x D^(L)``) is the old subspace embedded in that row basis via the
    cross-overlap transfer -- the subspace a streaming solver carries for free.
    """
    unc = fold_uncompressed(ctx, mps_compressed_L, k)
    B = left_canonical_copy(unc)               # uncompressed L+1, left-canonical
    A = left_canonical_copy(mps_compressed_L)   # step-L isometries (old subspace)
    # U overlap depends only on left sites 0..tau-1 (untouched by the right sweep)
    E = cross_left_overlaps(A, B)[tau - 1]      # (D^(L), chi_left)
    U, _ = np.linalg.qr(E.conj().T)             # (chi_left, D^(L)), orthonormal

    _truncate_right_of(ctx, B, tau)             # compress bonds right of tau
    G = B.tensors[tau]                          # (phi, chi_left, chi_right_truncated)
    dp, chil, chir = G.shape
    M = G.transpose(1, 0, 2).reshape(chil, dp * chir)
    return M, U
