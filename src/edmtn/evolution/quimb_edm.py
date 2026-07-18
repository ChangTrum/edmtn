"""EDM carried as a quimb ``TensorNetwork`` (Layer 5, ecosystem container).

The structural re-platform (plan Phase 0.0): instead of the bespoke
:class:`~edmtn.evolution.mps_utils.EDMMPS` + hand-rolled ``tensordot`` fold, carry
the extended density matrix as a generic 1D quimb tensor network across the whole
sub-bath fold loop.  Every linear-algebra step -- the MPO x MPS fold, the
canonicalise + truncation compression, and the reduced-density-matrix contraction
-- is then the maintained **quimb + cotengra + autoray** stack, and the whole path
is backend-agnostic (NumPy / CuPy / ... via autoray) and the natural substrate for
cuQuantom (cuTensorNet) execution.

Representation (the validated Phase-0.0 mapping):

* site ``p`` is a ``Tensor`` with a physical index ``k{p}`` (the open arm
  ``phi_up``, dim ``d_phys``) and virtual bonds ``v{p}`` between neighbours;
* the operator-valued boundaries -- the dangling ``d**2`` output leg ``OUT`` and
  the ``rho0`` contraction leg ``RHO0`` -- are ordinary dangling indices.

Both evolution engines are covered: the separable bath grows every bond by a
sub-bath MPO fold (:meth:`QuimbEDM.fold_raw`, a lossless index fusion), the single
(Gaussian) bath grows the chain by one new time-site per step (:meth:`QuimbEDM.step`);
both then share :meth:`QuimbEDM.compress`, which the evolution loop applies
*conditionally* so ``compress=False`` genuinely skips compression rather than doing a
zero-cutoff recompression.  :meth:`QuimbEDM.fold` is the backward-compatible
fold-then-compress combo.

The fold/step reproduce the **two-stage** path (the per-site contraction that
``_apply_sub_bath`` / ``apply_step`` do with ``tensordot``, then a quimb
compression) -- *not* a fused single-pass apply, which the Phase-0 ledger showed
keeps ~2x the bond and is slower (``docs/design/phase0-replatform-decisions.md``).  So the
kernel is contracted into the EDM exactly (forming the fused ``a*chi`` bond),
parallel bonds are fused, and only then is the chain compressed (zipup, a
quimb-native ``rel`` cutoff by default).  This keeps the observable ``<S_z(t)>``
identical to the native path while removing the custom container.
"""

from __future__ import annotations

import math

from ._validation import validate_compression_combination
from .mps_utils import EDMMPS


def _max_scalar(value) -> float:
    """Backend-safe max of a scalar / 0-d / batched array as a Python float.

    Never routes a CuPy array through ``np.asarray`` (implicit device->host is forbidden):
    the batch max is taken on the value's OWN backend, then ``.item()`` brings the single
    scalar across.
    """
    if getattr(value, "shape", ()):          # a real array -> reduce on its own backend
        value = value.max()
    item = getattr(value, "item", None)
    return float(item()) if item is not None else float(value)


class _TruncationAccumulator(dict):
    """Per-``compress()`` accumulator of the largest per-bond discarded weight (P1-15).

    quimb writes its truncation metric into the SAME ``info`` dict once per bond split,
    overwriting the key each time, so the only way to see every bond is to intercept
    ``__setitem__``.  Two keys are understood:

    * ``"error"``            -- quimb's exact-SVD discarded 2-norm ``sqrt(sum sigma**2)``,
      so the discarded WEIGHT is ``error**2``;
    * ``"discarded_weight"`` -- our ``edm_eigh_metric`` adapter's ``sum(clip(lambda, 0, inf))``,
      which is already a weight (the dm path's eigenvalues are ``lambda = sigma**2``).

    A fresh instance is created per compress call -- there is no module-level or shared
    state, so two accumulators can never contaminate each other.  ``info`` must be
    pre-seeded with the key, because quimb only computes opt-in extras
    (``parse_info_extras``).
    """

    _KEYS = ("error", "discarded_weight")

    def __init__(self, key: str = "error"):
        super().__init__()
        dict.__setitem__(self, key, None)  # pre-seed: opt in to quimb computing it
        self.max_weight = 0.0
        self.n_splits = 0

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        if value is None or key not in self._KEYS:
            return
        v = _max_scalar(value)
        weight = v * v if key == "error" else v   # error is a 2-norm; the other is a weight
        if not math.isfinite(weight) or weight < 0.0:
            raise FloatingPointError(
                f"non-finite/negative truncation metric from quimb ({key}={value!r})")
        if weight > self.max_weight:
            self.max_weight = weight
        self.n_splits += 1


class QuimbEDM:
    """Operator-valued EDM carried as a generic 1D quimb ``TensorNetwork``.

    Mirrors the parts of :class:`EDMMPS` the evolution loop and observables touch
    (``num_sites`` / ``max_bond`` / ``bond_dims`` / ``reduced_density_matrix``),
    plus :meth:`fold_raw` (the per-sub-bath MPO x MPS contraction, no compression) and
    :meth:`fold` (fold_raw + :meth:`compress`, kept for back-compat).
    """

    def __init__(self, tn, n, d, d_phys, rho0_vec, meta=None, max_discarded_weight=0.0):
        self.tn = tn
        self.n = n
        self.d = d
        self.d_phys = d_phys
        self.rho0_vec = rho0_vec
        self.meta = dict(meta or {})
        #: Largest per-bond discarded weight of the SINGLE compression sweep that produced
        #: this object -- ``max_b sum sigma_i**2`` on the ``zipup``/``direct`` exact paths,
        #: ``max_b sum lambda_i`` of the discarded density-matrix eigenvalues
        #: (``lambda_i = sigma_i**2``) on the ``dm`` path.  NOT a cumulative or global
        #: error bound for the whole evolution.  ``0.0`` when no compression ran or nothing
        #: was discarded; ``None`` when the chosen decomposition cannot measure it exactly
        #: (``rsvd``).
        self.max_discarded_weight: float | None = max_discarded_weight

    # -- construction ------------------------------------------------------

    @classmethod
    def from_edmmps(cls, mps: EDMMPS) -> "QuimbEDM":
        """Wrap an :class:`EDMMPS` (e.g. the freshly built system MPS ``rho_0``)."""
        import quimb.tensor as qtn  # noqa: PLC0415

        n = mps.num_sites
        ts = []
        for p in range(n):
            left = "OUT" if p == 0 else f"v{p - 1}"
            right = "RHO0" if p == n - 1 else f"v{p}"
            ts.append(qtn.Tensor(mps.tensors[p], inds=(f"k{p}", left, right), tags={f"I{p}"}))
        return cls(qtn.TensorNetwork(ts), n, mps.d, mps.d_phys, mps.rho0_vec,
                   meta=getattr(mps, "meta", None))

    @classmethod
    def empty(cls, rho0_vec, d, d_phys) -> "QuimbEDM":
        """An empty EDM (no sites yet) -- the single-bath start before step 1."""
        import quimb.tensor as qtn  # noqa: PLC0415

        return cls(qtn.TensorNetwork([]), 0, d, d_phys, rho0_vec)

    def to_edmmps(self) -> EDMMPS:
        """Extract back into an :class:`EDMMPS` (per-site ``(phi, chi_l, chi_r)``)."""
        import quimb.tensor as qtn  # noqa: PLC0415

        tensors = []
        for p in range(self.n):
            t = self.tn[f"I{p}"]
            left = "OUT" if p == 0 else list(qtn.bonds(self.tn[f"I{p - 1}"], t))[0]
            right = "RHO0" if p == self.n - 1 else list(qtn.bonds(t, self.tn[f"I{p + 1}"]))[0]
            tensors.append(t.transpose(f"k{p}", left, right).data)
        return EDMMPS(tensors=tensors, d=self.d, d_phys=self.d_phys,
                      rho0_vec=self.rho0_vec, meta=dict(self.meta))

    # -- structure ---------------------------------------------------------

    @property
    def num_sites(self) -> int:
        return self.n

    @property
    def bond_dims(self) -> list[int]:
        import quimb.tensor as qtn  # noqa: PLC0415

        out = []
        for p in range(self.n - 1):
            t, nxt = self.tn[f"I{p}"], self.tn[f"I{p + 1}"]
            b = list(qtn.bonds(t, nxt))[0]
            out.append(int(t.ind_size(b)))
        return out

    @property
    def max_bond(self) -> int:
        bd = self.bond_dims
        return max(bd) if bd else 1

    # -- extraction --------------------------------------------------------

    def reduced_density_matrix(self):
        """``rho(t)`` (``d x d``): close every open arm with ``delta^0`` and
        contract onto ``vec(rho(0))``, the same closure as :meth:`EDMMPS`."""
        import quimb.tensor as qtn  # noqa: PLC0415

        sel = self.tn.isel({f"k{p}": 0 for p in range(self.n)})
        net = sel | qtn.Tensor(self.rho0_vec, inds=("RHO0",))
        vec = net.contract(output_inds=("OUT",)).data
        # keep the result on its native backend (NumPy stays NumPy, CuPy stays CuPy);
        # forcing np.asarray here breaks the GPU path (CuPy forbids implicit conversion)
        return vec.reshape(self.d, self.d)

    # -- compression -------------------------------------------------------

    def compress(self, *, cutoff, cutoff_mode, method, max_bond,
                 decomp="exact", decomp_q=2, canon="quimb"):
        """Canonicalise + truncate the chain via quimb (cotengra/autoray).

        ``decomp`` selects the per-bond decomposition (``'exact'`` full SVD, or
        ``'rsvd'`` randomized with power iterations ``decomp_q`` + silent guard);
        ``canon`` selects the canonicalisation QR (``'quimb'`` default, ``'householder'``,
        ``'cholqr'``).  See :mod:`edmtn.evolution.quimb_decomp`.
        """
        # illegal combinations are rejected regardless of chain length -- before the
        # n <= 1 early return, so a direct low-level call can never leak a TypeError
        validate_compression_combination(method, decomp, canon)
        import quimb.tensor as qtn  # noqa: PLC0415

        if self.n <= 1:  # nothing to compress -> a genuine zero, not a stale inherited value
            return QuimbEDM(self.tn, self.n, self.d, self.d_phys, self.rho0_vec,
                            meta=self.meta, max_discarded_weight=0.0)
        from ..backend.quimb_linalg import apply_quimb_cupy_compat  # noqa: PLC0415
        from .quimb_decomp import (  # noqa: PLC0415
            canonize_opts_for, compress_opts_for, register_eigh_metric_driver)

        apply_quimb_cupy_compat()  # make quimb/autoray safe on CuPy-backed tensors
        # only forward the opts when non-default: an empty dict is still passed
        # through to the per-method split, and 'dm' (eigh) rejects canonize_opts
        opts = {}
        copts = dict(compress_opts_for(decomp, decomp_q))
        canopts = canonize_opts_for(canon)
        if canopts:
            opts["canonize_opts"] = canopts

        # -- truncation metric: each 1D-compress method reaches tensor_split by a DIFFERENT
        #    route, so the accumulator has to be injected three different ways (verified
        #    against quimb 1.14; do NOT merge these branches):
        #      * zipup  -> calls `C.split(**compress_opts)` directly      => TOP-LEVEL info
        #      * direct -> goes via compress_between/tensor_compress_bond, which consumes its
        #                  own top-level `info` (singular_values only) and forwards only the
        #                  INNER compress_opts to the split  => NESTED compress_opts={"info":...}
        #      * dm     -> calls `rhoi.split(**compress_opts)` directly, but quimb's built-in
        #                  eigh driver takes no `info`, so it needs our adapter => TOP-LEVEL info
        #    rSVD is deliberately NOT measured: rand_linalg.rsvd never sees the tail of the
        #    spectrum it omitted, so any "error" it could report would silently under-count.
        acc = None
        if decomp == "exact":
            if method == "dm":
                copts["method"] = register_eigh_metric_driver()
                acc = _TruncationAccumulator("discarded_weight")
                copts["info"] = acc
            elif method == "zipup":
                acc = _TruncationAccumulator("error")
                copts["info"] = acc
            elif method == "direct":
                acc = _TruncationAccumulator("error")
                copts["compress_opts"] = {"info": acc}
        if copts:
            opts["compress_opts"] = copts

        cq = qtn.tensor_network_1d_compress(
            self.tn, max_bond=max_bond, cutoff=cutoff, method=method,
            site_tags=[f"I{p}" for p in range(self.n)], permute_arrays=False,
            cutoff_mode=cutoff_mode, optimize="auto", **opts)
        return QuimbEDM(cq, self.n, self.d, self.d_phys, self.rho0_vec, meta=self.meta,
                        max_discarded_weight=(acc.max_weight if acc is not None else None))

    # -- single-bath step (one new time-site, Eq. 8) -----------------------

    def step(self, kernel_sites, sfamily, d):
        """Advance the EDM by one time-step (single bath), growing the chain by one
        site (uncompressed; the caller then :meth:`compress`).

        The per-site fold (the new superoperator into the newest site, the kernel
        sites into the existing ones) is the exact array contraction of
        :func:`~edmtn.evolution.mps_utils.apply_step`; carried back into a quimb TN
        so the state stays in the ecosystem container.
        """
        from .mps_utils import apply_step  # noqa: PLC0415

        prev = self.to_edmmps() if self.n > 0 else None
        enlarged = apply_step(prev, kernel_sites, sfamily, d, self.rho0_vec)
        return QuimbEDM.from_edmmps(enlarged)

    # -- separable fold (MPO x MPS contraction + compression) --------------

    def fold_raw(self, mpo_sites) -> "QuimbEDM":
        """Fold one sub-bath's combined-kernel MPO into the EDM, WITHOUT compression.

        ``new[phi_up, (a_l, chi_l), (a_r, chi_r)] = sum_{phi_down}
        T[phi_up, phi_down, a_l, a_r] G[phi_down, chi_l, chi_r]`` per site (exact, the
        two-stage apply), then each parallel ``(v, a)`` bond fused into one.  ``fuse_multibonds``
        is a *lossless* index fusion -- there is NO canonicalisation / SVD / truncation here.  The
        caller compresses separately (:meth:`compress`), so ``compress=False`` genuinely skips
        compression.  Returns a new (uncompressed) :class:`QuimbEDM`; ``self`` is not mutated.
        """
        import quimb.tensor as qtn  # noqa: PLC0415

        n = self.n
        folded = []
        for p in range(n):
            G = self.tn[f"I{p}"]                    # inds (k{p}, left, right)
            T = mpo_sites[p]                        # (phi_up, phi_down, a_l, a_r)
            inds = [f"u{p}", f"k{p}"]               # u: new phys (phi_up); k: down (shared with G)
            if n == 1:
                # single site is both first AND last: drop BOTH trivial MPO boundaries so no
                # lateral a-index dangles (else to_edmmps sees an unhandled a0 -> ValueError)
                T = T[:, :, 0, 0]
            elif p == 0:
                T = T[:, :, 0, :]                   # trivial left MPO bond -> OUT stays d**2
                inds += [f"a{p}"]
            elif p == n - 1:
                T = T[:, :, :, 0]                   # trivial right MPO bond -> RHO0 stays d**2
                inds += [f"a{p - 1}"]
            else:
                inds += [f"a{p - 1}", f"a{p}"]
            site = (G & qtn.Tensor(T, inds=tuple(inds))).contract()  # contract shared k{p}
            site.reindex({f"u{p}": f"k{p}"}, inplace=True)
            site.add_tag(f"I{p}")
            folded.append(site)
        tn = qtn.TensorNetwork(folded)
        tn.fuse_multibonds(inplace=True)            # (v{p}, a{p}) -> single fused bond (lossless)
        return QuimbEDM(tn, n, self.d, self.d_phys, self.rho0_vec, meta=self.meta)

    def fold(self, mpo_sites, *, cutoff, cutoff_mode, method, max_bond,
             decomp="exact", decomp_q=2, canon="quimb"):
        """Fold one sub-bath's MPO into the EDM, then compress (fold_raw + compress combo).

        Backward-compatible convenience wrapper preserving the original ``fold + compress``
        semantics for direct callers (tests, ``examples/``).  When ``compress=False`` must
        genuinely skip compression, call :meth:`fold_raw` and then :meth:`compress`
        conditionally instead (as :meth:`SeparableBathEvolution.run` does).
        """
        return self.fold_raw(mpo_sites).compress(
            cutoff=cutoff, cutoff_mode=cutoff_mode, method=method, max_bond=max_bond,
            decomp=decomp, decomp_q=decomp_q, canon=canon)
