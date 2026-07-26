"""Causal-prefix time-resolved reads for the separable-bath engine (Layer 5).

The separable engine folds every sub-bath into one temporal chain built for a single final
time ``T = N eps``.  This module reads ``rho(t_n)`` at **every** physical step off that one
chain, instead of running the whole fold once per target time.

The identity it rests on (``docs/design/causal-prefix-time-reads.md``): the oldest ``q n``
sites of the ``T = N eps`` chain, with every sub-bath's lateral index sliced to ``0`` at the
cut, are exactly the chain an independent run of ``n`` physical steps builds.  Slicing a
Liouville index to ``0`` is contracting with ``e_0``, i.e. **tracing the sub-bath out**, and
that agrees with never having evolved it past ``t_n`` because the bath channel is trace
preserving (``e_0^T D_k = e_0^T``).

Two things make this less trivial than "keep a pointer into the chain":

* The **terminator is a matrix**, not a covector.  The bond at a cut carries the vectorised
  system state *tensored with* the sub-baths' open lateral indices; only the latter are
  closed, so ``l_m`` has shape ``(d**2, chi_m)`` and the read is a ``d**2`` vector.
* The terminator lives in the **bond basis**, so every basis change has to be transported
  into it.  Under ``R_new = Y R_old`` a terminator transforms as ``l -> l Y**-1``, and ``Y``
  is exactly what goes singular where a truncation bites.  The compression therefore has to
  run in a gauge where every prefix-side factor arrives as a left multiplication:
  :func:`tracking_compress` canonicalises with LQ **from the oldest end** (pushing the
  triangular factor away from the prefix) and truncates with the density-matrix isometry
  (which the prefix meets as ``V^dag``).  Sweeping the other way would need ``S**-1``.

The truncation *rule* is not re-derived here.  The retained rank comes from quimb's own
``_trim_and_renorm_svd_result``, fed the eigenvalues exactly as the ``dm`` path does, so
``cutoff`` / ``cutoff_mode`` / ``max_bond`` mean the same thing they always did.  (The same
private helper is already used by ``edm_eigh_metric`` in :mod:`edmtn.evolution.quimb_decomp`.)

Everything here is opt-in.  With ``record_time_reads=False`` nothing in this module runs, no
terminator is allocated, and the compression menu is untouched.
"""

from __future__ import annotations

import math
import numbers

from .mps_utils import _xp


def _scalar(value) -> float:
    """Backend-safe 0-d array -> Python float, without an implicit device->host cast.

    Mirrors ``_max_scalar`` in :mod:`edmtn.evolution.quimb_edm`: the reduction happens on
    the value's own backend and only ``.item()`` crosses the device boundary, so a CuPy
    array never goes through ``float()``/``np.asarray``.
    """
    item = getattr(value, "item", None)
    return float(item()) if item is not None else float(value)


def physical_cuts(n_steps: int, order: int) -> list[int]:
    """Sub-step cuts that are physical times: ``m = order * n`` for ``n = 1 .. n_steps``.

    At ``order = 2`` the odd cuts sit *between* the two algebraic sub-steps of one Strang
    step and are not the state at any grid time.  They are never created, never maintained
    and never read -- and the omission is deliberate rather than filtered at the end,
    because a half-step read is not self-diagnosing: its trace is still ``1`` to machine
    precision, so no downstream sanity check would catch it.
    """
    if isinstance(n_steps, bool) or not isinstance(n_steps, numbers.Integral) or n_steps < 1:
        raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    return [order * n for n in range(1, int(n_steps) + 1)]


class PrefixTerminators:
    """The per-cut boundary matrices ``l_m`` and their transport rules.

    One ``(d**2, chi_m)`` matrix per **physical** cut.  ``l_M`` (the newest cut) is the
    dangling output leg itself, whose lateral index the kernel has already sliced, so it
    stays ``I`` for the whole run -- which is why the last time read is exactly the ordinary
    :meth:`~edmtn.evolution.mps_utils.EDMMPS.reduced_density_matrix`.

    Parameters
    ----------
    d2 : int
        ``d**2``, the vectorised system dimension (the open output index).
    n_steps, order : int
        The physical time grid; the maintained cuts are ``order * n``.
    like : ndarray
        Any array from the run, used only to pick the array module and dtype so the
        terminators live on the same backend (CPU/GPU) as the chain.
    """

    def __init__(self, d2: int, n_steps: int, order: int, like):
        xp = _xp(like)
        self.d2 = int(d2)
        self.order = int(order)
        self.n_sites = int(order) * int(n_steps)
        self.cuts = physical_cuts(n_steps, order)
        self._xp = xp
        self._dtype = like.dtype
        self.terms = {m: xp.eye(self.d2, dtype=like.dtype) for m in self.cuts}

    # -- transport ---------------------------------------------------------

    def fold(self, mpo_sites) -> None:
        """Absorb one sub-bath fold: ``l_m <- kron(l_m, e_0)`` on every cut but the newest.

        The fold multiplies each internal bond by that sub-bath's lateral dimension and
        fuses it as ``(old bond outer, new lateral inner)`` -- the layout both the
        hand-rolled ``apply_step`` and quimb's ``fuse_multibonds`` produce.  ``D_a`` is read
        off the kernel site rather than assumed, so a sub-bath with a different Liouville
        dimension needs no change here.
        """
        xp = self._xp
        for m in self.cuts:
            if m == self.n_sites:
                continue                       # the output leg: kernel already sliced it
            p = self.n_sites - m               # cut m is the LEFT bond of position p
            d_a = int(mpo_sites[p].shape[2])
            e0 = xp.zeros((1, d_a), dtype=self._dtype)
            e0[0, 0] = 1.0
            self.terms[m] = xp.kron(self.terms[m], e0)

    def transport(self, m: int, factor) -> None:
        """Carry cut ``m``'s terminator through one bond-basis change: ``l_m <- l_m @ X``.

        ``X`` is whatever factor the compression pushes **away** from the prefix -- the LQ
        triangular factor on the canonicalisation sweep, the retained-subspace isometry on
        the truncation sweep.  Never an inverse; see the module docstring.
        """
        if m in self.terms:
            self.terms[m] = self.terms[m] @ factor

    # -- extraction --------------------------------------------------------

    def read(self, tensors, rho0_vec, d: int) -> list:
        """One right-to-left sweep: ``rho(t_n)`` for every physical step, oldest first.

        Costs ``n_sites`` bond matrix-vector products to traverse the chain plus one
        ``(d**2, chi)`` projection per physical cut.  All reads share the one chain; nothing
        is recontracted.
        """
        out, F = {}, rho0_vec
        wanted = set(self.cuts)
        for m in range(1, self.n_sites + 1):
            F = tensors[self.n_sites - m][0] @ F     # phi_up = 0: the reduced state closes every arm
            if m in wanted:
                out[m] = (self.terms[m] @ F).reshape(d, d)
        return [out[m] for m in self.cuts]


# --------------------------------------------------------------------------
# gauge-tracking compression
# --------------------------------------------------------------------------

def _lq(mat, xp):
    """``mat = L Q`` with ``Q`` row-orthonormal, via the QR of ``mat^dag``.

    The direction matters: this pushes the triangular factor towards the *newer* side, so
    the prefix meets only ``L`` as a left multiplication and never its inverse.
    """
    q, r = xp.linalg.qr(xp.conj(mat).T)
    return xp.conj(r).T, xp.conj(q).T


def _retained_rank(lam, U, cutoff, cutoff_mode, max_bond, xp) -> int:
    """Rank quimb's own trimming would keep for the spectrum ``lam``.

    Delegates to ``quimb.tensor.decomp._trim_and_renorm_svd_result`` with the ``dm`` path's
    conventions (``positive=1``, ``absorb=None``), so ``cutoff`` / ``cutoff_mode`` /
    ``max_bond`` keep exactly the meaning they have on every other compression path.  The
    rule is deliberately NOT restated here -- a second copy of it would be a second place to
    be wrong.
    """
    from quimb.tensor import decomp  # noqa: PLC0415

    vh = xp.conj(xp.swapaxes(U, -2, -1))
    kept = decomp._trim_and_renorm_svd_result(
        U, lam, vh,
        cutoff=cutoff,
        cutoff_mode=decomp._CUTOFF_MODE_MAP[cutoff_mode],
        max_bond=-1 if max_bond is None else int(max_bond),
        absorb=None, renorm=0, use_abs=False, xp=xp)
    return int(kept[0].shape[-1])


def tracking_compress(tensors, terminators, *, cutoff, cutoff_mode, max_bond):
    """Density-matrix compression that also transports the prefix terminators.

    Two sweeps, in the one gauge where the terminator transport needs no inverse:

    * **A -- canonicalise.**  LQ from the oldest end towards the newest, pushing the
      triangular factor away from the prefix.  Every prefix ends right-isometric and each
      terminator picks up ``L``.
    * **B -- truncate.**  With the prefix canonical, the optimal truncation at a bond is the
      dominant eigenvectors of the left block's reduced density matrix, which obeys
      ``rho_b = sum_phi t_b[phi]^dag rho_{b-1} t_b[phi]``.  The prefix meets ``V^dag``, so
      each terminator picks up the isometry ``V``.

    Returns
    -------
    (list[ndarray], float)
        The compressed site tensors, and the largest per-bond discarded weight
        ``max_b sum_i discarded lambda_i`` -- the same quantity the ``dm`` path reports
        through ``_TruncationAccumulator("discarded_weight")``, so the number stays
        comparable with the other methods.
    """
    xp = _xp(tensors[0])
    n = len(tensors)
    t = list(tensors)
    if n <= 1:
        return t, 0.0

    # -- sweep A: LQ from the oldest end; every prefix becomes right-isometric --
    for p in range(n - 1, 0, -1):
        d_phys, left, right = t[p].shape
        mat = xp.reshape(xp.transpose(t[p], (1, 0, 2)), (left, d_phys * right))
        lo, q = _lq(mat, xp)
        t[p] = xp.transpose(xp.reshape(q, (-1, d_phys, right)), (1, 0, 2))
        t[p - 1] = xp.tensordot(t[p - 1], lo, axes=([2], [0]))
        terminators.transport(n - p, lo)

    # -- sweep B: density-matrix truncation, newest end first --
    worst = 0.0
    rho = None
    for b in range(n - 1):
        d_phys = t[b].shape[0]
        if rho is None:
            block = xp.reshape(xp.transpose(t[b], (1, 0, 2)), (-1, t[b].shape[2]))
            rho = xp.conj(block).T @ block
        else:
            rho = sum(xp.conj(t[b][x]).T @ rho @ t[b][x] for x in range(d_phys))
        rho = 0.5 * (rho + xp.conj(rho).T)          # kill the accumulated asymmetry

        lam, vecs = xp.linalg.eigh(rho)
        lam = xp.clip(xp.flip(lam, axis=-1).real, 0.0, None)   # dm path: positive=1
        vecs = xp.flip(vecs, axis=-1)
        keep = _retained_rank(lam, vecs, cutoff, cutoff_mode, max_bond, xp)
        # reduce on the array's OWN backend, then bring the single scalar across with
        # .item() -- never an implicit CuPy device->host conversion (the `_max_scalar`
        # convention in quimb_edm.py)
        weight = _scalar(lam[keep:].sum()) if keep < lam.shape[0] else 0.0
        # Same public contract as the ordinary compression paths, whose
        # ``_TruncationAccumulator`` raises on a non-finite or negative metric.  Python's
        # ``max`` would NOT propagate a NaN here -- ``max(0.0, nan)`` is ``0.0`` -- so a
        # broken sweep would be reported as a perfectly healthy zero discarded weight.
        if not math.isfinite(weight) or weight < 0.0:
            raise FloatingPointError(
                f"non-finite/negative truncation metric from the tracking sweep "
                f"(discarded weight {weight!r} at bond {b})")
        worst = max(worst, weight)

        v = vecs[:, :keep]
        t[b] = xp.tensordot(t[b], v, axes=([2], [0]))
        t[b + 1] = xp.tensordot(xp.conj(v).T, t[b + 1], axes=([1], [1])).transpose(1, 0, 2)
        rho = xp.conj(v).T @ rho @ v
        terminators.transport(n - 1 - b, v)

    return t, worst
