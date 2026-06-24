"""MPS containers and helpers for the EDM evolution engine (Layer 5).

The extended density matrix is carried as an *operator-valued* matrix-product
state in the system-folded form of Appendix E (Eq. E3 / Fig. 9a).  After folding
the system superoperators into the sites, the EDM is a plain complex MPS:

    sites[0] (newest, leftmost)  ...  sites[-1] (oldest, rightmost)

with one physical open-arm leg ``phi_up`` (dimension ``d_phys``) per site and a
fused virtual bond between neighbours.  The fused bond carries *both* the kernel
MPO bond and the system index that threads the superoperator product, so no
special operator machinery is needed -- the chain contracts like a standard MPS.

Boundaries:

* the leftmost site's left bond has dimension ``d**2`` -- this is the dangling
  *output operator* leg ``vec(rho)`` of the EDM;
* the rightmost site's right bond has dimension ``d**2`` and is contracted with
  ``vec(rho(0))``.

The reduced density matrix ``rho(t) = delta^0_{Phi} rho^{Phi}`` is obtained by
selecting ``phi_up = 0`` on every site (the closing tensor) and contracting the
chain onto ``vec(rho(0))``.

Index conventions for a site tensor ``G``::

    G[phi_up, chi_left, chi_right]

with ``chi`` fused as ``(old_mps_bond outer, kernel_bond inner)`` at every site
except the newest, whose right bond is ``(system_in outer, kernel_bond inner)``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from ..decomposition.standard_svd import StandardSVD


def _xp(a):
    """Return the array module (``numpy`` or ``cupy``) backing ``a``."""
    if type(a).__module__.split(".")[0] == "cupy":
        import cupy  # noqa: PLC0415

        return cupy
    return np


@dataclass
class EDMMPS:
    """Operator-valued EDM as a system-folded complex MPS.

    Attributes
    ----------
    tensors : list[ndarray]
        Site tensors ``G[phi_up, chi_left, chi_right]`` ordered newest
        (leftmost, index 0) to oldest (rightmost, index -1).
    d : int
        System Hilbert-space dimension (``vec`` length ``d**2``).
    d_phys : int
        Open-arm (superoperator) index dimension.
    rho0_vec : ndarray
        Row-major ``vec(rho(0))`` of length ``d**2`` (the right boundary).
    """

    tensors: list
    d: int
    d_phys: int
    rho0_vec: object
    meta: dict = field(default_factory=dict)

    # -- structure ---------------------------------------------------------

    @property
    def num_sites(self) -> int:
        return len(self.tensors)

    @property
    def bond_dims(self) -> list[int]:
        """Internal bond dimensions (between adjacent sites), left to right."""
        return [self.tensors[p].shape[2] for p in range(self.num_sites - 1)]

    @property
    def max_bond(self) -> int:
        bd = self.bond_dims
        return max(bd) if bd else 1

    def copy(self) -> "EDMMPS":
        return EDMMPS(
            tensors=[t.copy() for t in self.tensors],
            d=self.d,
            d_phys=self.d_phys,
            rho0_vec=self.rho0_vec.copy(),
            meta=dict(self.meta),
        )

    # -- extraction --------------------------------------------------------

    def reduced_density_matrix(self):
        """Return ``rho(t)`` (``d x d``) by closing every open arm with ``delta^0``."""
        xp = _xp(self.tensors[0])
        # M = product of the phi_up = 0 slices: (d**2 output) x (d**2 rho0)
        acc = self.tensors[0][0]  # (chi_left = d**2, chi_right)
        for t in self.tensors[1:]:
            acc = acc @ t[0]
        vec = acc @ self.rho0_vec  # (d**2,)
        return vec.reshape(self.d, self.d)

    def open_arm_tensor(self):
        """Dense EDM ``rho^{Phi}`` with all open arms left free (small ``t`` only).

        Returns an array of shape ``(d_phys,) * num_sites + (d, d)``.
        """
        xp = _xp(self.tensors[0])
        n = self.num_sites
        # contract bonds keeping every phi_up leg open
        acc = self.tensors[0]  # (phi0, d**2, chi)
        # acc axes: [phi0, out(d**2), chi]
        for t in self.tensors[1:]:
            # contract running right bond with next left bond
            acc = xp.tensordot(acc, t, axes=([acc.ndim - 1], [1]))
            # acc: [phi0, out, ..., phi_p, chi]
        # acc: [phi0, out, phi1, ..., phi_{n-1}, chi_right=d**2]
        acc = xp.tensordot(acc, self.rho0_vec, axes=([acc.ndim - 1], [0]))
        # acc: [phi0, out, phi1, ..., phi_{n-1}]
        # move the output leg (axis 1) to the end and split into (d, d)
        phi_axes = [0] + list(range(2, n + 1))
        acc = xp.transpose(acc, phi_axes + [1])
        return acc.reshape((self.d_phys,) * n + (self.d, self.d))


# --------------------------------------------------------------------------
# per-step kernel application (system-folded operator-valued MPS)
# --------------------------------------------------------------------------

def apply_step(mps, kernel_sites, sfamily, d, rho0_vec):
    """Advance the EDM-MPS by one step (Eq. 8), returning the enlarged MPS.

    Parameters
    ----------
    mps : EDMMPS or None
        EDM at step ``n - 1`` (``None`` for the first step).
    kernel_sites : list[ndarray]
        Combined-kernel MPO site tensors ``K[phi_up, phi_down, b_left, b_right]``
        for step ``n``, ordered newest-first (matches MPS site positions).
    sfamily : ndarray
        System superoperators ``S[phi, d**2, d**2]`` at the new step's time.
    d : int
        System dimension.
    rho0_vec : ndarray
        ``vec(rho(0))`` carried as the right boundary.

    Returns
    -------
    EDMMPS
        EDM at step ``n`` (un-compressed; bonds enlarged by the kernel bond and
        the freshly inserted system index).
    """
    d2 = d * d
    d_phys = sfamily.shape[0]
    n = len(kernel_sites)
    xp = _xp(sfamily)

    # -- newest site: fold the new superoperator into the leftmost site -----
    k0 = kernel_sites[0]  # (phi_up, phi_down, 1, Br)
    k0_sq = k0[:, :, 0, :]  # (phi_up, phi_down, Br)
    # tmp[phi_up, b, i, j] = sum_phi_down k0[phi_up, phi_down, b] S[phi_down, i, j]
    tmp = xp.tensordot(k0_sq, sfamily, axes=([1], [0]))  # (u, b, i, j)
    Br = k0.shape[3]
    # G0[phi_up, chi_left = i (d**2 output), chi_right = (j outer, b inner)]
    G0 = xp.transpose(tmp, (0, 2, 3, 1)).reshape(d_phys, d2, d2 * Br)

    new_tensors = [G0]

    if mps is not None:
        for p in range(1, n):
            ksite = kernel_sites[p]            # (phi_up, phi_down, bl, br)
            g = mps.tensors[p - 1]             # (phi_down, al, ar)
            # out[phi_up, al, bl, ar, br] (contract phi_down, then reorder)
            out = xp.tensordot(ksite, g, axes=([1], [0]))  # (u, x, y, a, c)
            out = xp.transpose(out, (0, 3, 1, 4, 2))        # (u, a, x, c, y)
            al, bl = g.shape[1], ksite.shape[2]
            ar, br = g.shape[2], ksite.shape[3]
            new_tensors.append(out.reshape(d_phys, al * bl, ar * br))

    return EDMMPS(tensors=new_tensors, d=d, d_phys=d_phys, rho0_vec=rho0_vec)


# --------------------------------------------------------------------------
# compression (canonicalisation + truncation sweep)
# --------------------------------------------------------------------------

def left_canonicalize(mps, canon=None):
    """Left-canonicalise sites ``0 .. n-2``, leaving site ``n-1`` as centre.

    ``canon`` selects a :class:`~edmtn.evolution.canonicalize.CanonicalizationStrategy`
    (e.g. ``CholeskyQR``); ``None`` (default) is the historical Householder QR sweep,
    kept byte-for-byte.
    """
    if canon is not None:
        return canon.left_canonicalize(mps)
    xp = _xp(mps.tensors[0])
    n = mps.num_sites
    for p in range(n - 1):
        G = mps.tensors[p]
        dp, chil, chir = G.shape
        Q, R = xp.linalg.qr(G.reshape(dp * chil, chir))
        k = Q.shape[1]
        mps.tensors[p] = Q.reshape(dp, chil, k)
        nxt = mps.tensors[p + 1]  # (dp, chil = chir, chir2)
        # R[r,c] . nxt[p,c,x] -> [p,r,x]
        mps.tensors[p + 1] = xp.transpose(xp.tensordot(R, nxt, axes=([1], [1])), (1, 0, 2))
    return mps


def truncate(mps, strategy=None, *, max_bond=None, cutoff=0.0,
             cutoff_mode="rel_ref", ref_index=None):
    """Right-to-left truncation sweep; returns ``(mps, info_per_bond)``.

    Assumes the MPS is left-canonical (call :func:`left_canonicalize` first).
    Each internal bond ``p-1 | p`` is truncated by an SVD of site ``p`` grouped
    as ``(chi_left) | (phi_up, chi_right)``, keeping site ``p`` right-canonical.
    """
    if strategy is None:
        strategy = StandardSVD()
    if ref_index is None:
        ref_index = mps.d * mps.d
    xp = _xp(mps.tensors[0])
    infos = []
    for p in range(mps.num_sites - 1, 0, -1):
        G = mps.tensors[p]
        dp, chil, chir = G.shape
        mat = G.transpose(1, 0, 2).reshape(chil, dp * chir)
        res = strategy.compress(
            mat,
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            ref_index=ref_index,
            absorb="left",
        )
        US, Vh = res.left, res.right  # US: (chil, k), Vh: (k, dp*chir)
        k = res.bond
        mps.tensors[p] = Vh.reshape(k, dp, chir).transpose(1, 0, 2)
        prev = mps.tensors[p - 1]  # (dp', l, r = chil)
        # prev[p,l,r] . US[r,k] -> [p,l,k]
        mps.tensors[p - 1] = xp.tensordot(prev, US, axes=([2], [0]))
        infos.append(res.info)
    infos.reverse()
    return mps, infos


def compress(mps, strategy=None, *, canon=None, **trunc):
    """Canonicalise then truncate in place; returns ``(mps, info_per_bond)``.

    ``canon`` chooses the canonicalisation strategy (default Householder QR);
    ``strategy`` chooses the truncation/decomposition strategy.
    """
    left_canonicalize(mps, canon=canon)
    return truncate(mps, strategy=strategy, **trunc)


# --------------------------------------------------------------------------
# brute-force reference (no compression) for validation
# --------------------------------------------------------------------------

def _kernel_dense_advance(kernel_engine, t, C_prev):
    """Dense advance ``C^{(t)} = K_t . C^{(t-1)}`` (mirrors the Layer-3 test)."""
    K = kernel_engine.get_kernel_mpo(t).to_dense()
    if t == 1:
        return K * C_prev
    letters = "abcdefghijklmnopqrstuvwxyz"
    ups = list(letters[:t])
    d_new = letters[t]
    mids = list(letters[t + 1 : t + 1 + (t - 1)])
    es = list(letters[t + 1 + (t - 1) : t + 1 + 2 * (t - 1)])
    k_sub = "".join(ups) + d_new + "".join(mids)
    c_sub = "".join(mids) + "".join(es)
    out_sub = "".join(ups) + d_new + "".join(es)
    return np.einsum(f"{k_sub},{c_sub}->{out_sub}", K, C_prev, optimize=True)


def dense_open_armed_correlation(kernel_engine, n):
    """Dense ``C^{(n)}`` with axes ``[phi_up(n..1), phi_down(n..1)]``."""
    C = np.array(1.0 + 0j)
    for t in range(1, n + 1):
        C = _kernel_dense_advance(kernel_engine, t, C)
    return C


def dense_reduced_density_matrix(kernel_engine, sfamilies, rho0_vec, n, d):
    """Brute-force ``rho(n)`` independent of the MPS engine.

    ``sfamilies[k]`` is the system superoperator family ``S[phi, d**2, d**2]`` at
    step ``k + 1`` (``k = 0 .. n-1``), with ``sfamilies[n-1]`` the newest.
    """
    C = dense_open_armed_correlation(kernel_engine, n)
    d_phys = sfamilies[0].shape[0]
    # close all open arms (phi_up = 0): select axes 0..n-1 at index 0
    closed = C[(0,) * n]  # shape (d_phys,) * n indexed by [phi_down(n..1)]
    out = np.zeros(d * d, dtype=np.complex128)
    for downs in itertools.product(range(d_phys), repeat=n):
        coeff = closed[downs]
        if coeff == 0:
            continue
        # downs = (phi_down(n), ..., phi_down(1)); newest acts last (leftmost)
        v = rho0_vec
        for k in range(n):  # apply oldest first: phi_down(1) .. phi_down(n)
            phi = downs[n - 1 - k]
            v = sfamilies[k][phi] @ v
        out += coeff * v
    return out.reshape(d, d)
