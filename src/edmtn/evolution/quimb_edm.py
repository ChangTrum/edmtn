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

import numpy as np

from ._validation import validate_compression_combination
from .mps_utils import EDMMPS, _xp


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
                 decomp="exact", decomp_q=2, canon="quimb", terminators=None):
        """Canonicalise + truncate the chain via quimb (cotengra/autoray).

        ``decomp`` selects the per-bond decomposition (``'exact'`` full SVD, or
        ``'rsvd'`` randomized with power iterations ``decomp_q`` + silent guard);
        ``canon`` selects the canonicalisation QR (``'quimb'`` default, ``'householder'``,
        ``'cholqr'``).  See :mod:`edmtn.evolution.quimb_decomp`.

        ``method='dm_tracking'`` is the one method quimb does **not** implement: it is the
        in-repo two-sweep density-matrix compression of
        :mod:`edmtn.evolution.prefix_reads`, whose per-bond basis changes are handed back so
        the causal-prefix ``terminators`` can be transported through them.  It is
        intercepted here, before the quimb call -- ``method`` is otherwise forwarded
        verbatim, so without this branch the name would leak out as a bare ``KeyError``
        from quimb's method registry.  The two arguments are required together in both
        directions.
        """
        # illegal combinations are rejected regardless of chain length -- before the
        # n <= 1 early return, so a direct low-level call can never leak a TypeError
        validate_compression_combination(method, decomp, canon)
        if (method == "dm_tracking") != (terminators is not None):
            raise ValueError(
                "compress_method='dm_tracking' and `terminators` go together: got "
                f"method={method!r} with terminators="
                f"{'a PrefixTerminators' if terminators is not None else None}.  The "
                "tracking sweep exists only to transport the prefix terminators, and the "
                "terminators can only be transported by it.")
        if method == "dm_tracking":
            from .prefix_reads import tracking_compress  # noqa: PLC0415

            edm = self.to_edmmps()
            edm.tensors, weight = tracking_compress(
                edm.tensors, terminators,
                cutoff=cutoff, cutoff_mode=cutoff_mode, max_bond=max_bond)
            out = QuimbEDM.from_edmmps(edm)
            out.max_discarded_weight = weight
            return out
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
        #
        # -- when nothing CAN be discarded, do not ask for the metric at all ------------
        # ``cutoff = 0`` with ``max_bond = None`` gives quimb ``truncation = False``
        # (``parse_split_opts``), and an ``absorb`` of left/right then resolves
        # ``method='auto'`` to **qr** rather than svd (``parse_method_absorb``).  A QR takes
        # no ``info``, so injecting one is both meaningless and, on some backends, fatal:
        # ``qr_stabilized`` forwards ``**kwargs`` straight into ``xp.linalg.qr``, which
        # raises ``TypeError: qr() got an unexpected keyword argument 'info'`` on CuPy.  It
        # survives on NumPy only by accident -- ``qr_stabilized_numpy`` drops ``**kwargs``
        # for 2-d input on its way to the numba kernel -- so this was a real CPU/GPU
        # divergence, not a GPU-only quirk.
        #
        # The discarded weight in this regime is not merely small, it is exactly ``0.0``:
        # no cutoff and no rank limit means nothing is dropped.  So report that constant
        # and skip the accumulator.  ``max_bond`` being set is NOT covered -- a rank limit
        # truncates even at ``cutoff = 0``, and there the real metric must still be
        # collected.  ``rsvd`` is excluded too, keeping its documented ``None``.
        known_lossless = decomp == "exact" and cutoff == 0.0 and max_bond is None
        acc = None
        if decomp == "exact" and not known_lossless:
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
        elif known_lossless and method == "dm":
            # keep the same eigh driver so the numerical path is untouched; just no info
            copts["method"] = register_eigh_metric_driver()
        if copts:
            opts["compress_opts"] = copts

        cq = qtn.tensor_network_1d_compress(
            self.tn, max_bond=max_bond, cutoff=cutoff, method=method,
            site_tags=[f"I{p}" for p in range(self.n)], permute_arrays=False,
            cutoff_mode=cutoff_mode, optimize="auto", **opts)
        if acc is not None:
            weight = acc.max_weight
        elif known_lossless:
            weight = 0.0          # nothing could be discarded; not "unmeasurable"
        else:
            weight = None         # rsvd: the omitted tail was never seen, so no honest value
        return QuimbEDM(cq, self.n, self.d, self.d_phys, self.rho0_vec, meta=self.meta,
                        max_discarded_weight=weight)

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

    # -- exact addition (the tangent channel's source term) ----------------

    def add_exact(self, other: "QuimbEDM") -> "QuimbEDM":
        """Return the **lossless** sum ``self + other`` as a new EDM.

        The two chains are combined by taking the direct sum of the **internal virtual
        bonds only**: the open arms ``k{p}``, the output leg ``OUT`` and the ``RHO0`` leg
        are *shared external* legs of both summands, so direct-summing them would build a
        different object entirely rather than a sum.  Every contraction of the result with
        a closing therefore equals the sum of the two contractions, exactly -- there is no
        canonicalisation, no SVD and no truncation here, and the bond of the result is the
        sum of the two input bonds.  The caller compresses afterwards if it wants to.

        This is deliberately *not* quimb's ``+``.  Two networks produced by independent
        :meth:`fold_raw` calls carry no guarantee of matching internal bond *names* after
        ``fuse_multibonds``, and a name-based addition would either fail or silently
        contract the wrong pair of legs.  The index conventions are re-derived here from
        the site order, exactly as :meth:`to_edmmps` does.

        Both inputs are left unmutated and the result shares no buffer with either.
        Arrays are allocated on the inputs' own backend, so a CuPy chain stays on the
        device; mixing backends is rejected rather than silently transferred.

        Raises
        ------
        ValueError
            if the two EDMs are not structurally addable -- differing ``n``, ``d``,
            ``d_phys``, per-site open-arm dimensions, ``OUT``/``RHO0`` dimensions, array
            backends, or a different ``rho0_vec`` (the sum of two chains closing onto
            different initial states is not the sum of what they represent).
        """
        if not isinstance(other, QuimbEDM):
            raise TypeError(
                f"add_exact expects a QuimbEDM, got {type(other).__name__}")
        if (self.n, self.d, self.d_phys) != (other.n, other.d, other.d_phys):
            raise ValueError(
                f"add_exact needs structurally identical EDMs: (n, d, d_phys) = "
                f"{(self.n, self.d, self.d_phys)} vs {(other.n, other.d, other.d_phys)}")
        if self.n == 0:
            raise ValueError(
                "add_exact needs at least one site; an empty EDM carries no tensors and is "
                "not a summand")

        left = self.to_edmmps().tensors
        right = other.to_edmmps().tensors
        if _xp(left[0]) is not _xp(right[0]):
            raise ValueError(
                "add_exact needs both EDMs on the same array backend; adding a CuPy chain "
                "to a NumPy one would move data across the device boundary implicitly")
        for p in range(self.n):
            if left[p].shape[0] != right[p].shape[0]:
                raise ValueError(
                    f"add_exact: site {p} has open-arm dimension {left[p].shape[0]} on one "
                    f"side and {right[p].shape[0]} on the other")
        if left[0].shape[1] != right[0].shape[1]:
            raise ValueError(
                f"add_exact: the OUT legs differ, {left[0].shape[1]} vs {right[0].shape[1]}")
        if left[-1].shape[2] != right[-1].shape[2]:
            raise ValueError(
                f"add_exact: the RHO0 legs differ, {left[-1].shape[2]} vs "
                f"{right[-1].shape[2]}")
        r_self, r_other = self.rho0_vec, other.rho0_vec
        if getattr(r_self, "shape", None) != getattr(r_other, "shape", None):
            raise ValueError(
                f"add_exact: rho0_vec shapes differ, {getattr(r_self, 'shape', None)} vs "
                f"{getattr(r_other, 'shape', None)}")
        # reduce on the arrays' own backend and bring ONE scalar across with .item(), so a
        # CuPy comparison does not rely on an implicit synchronising bool() conversion
        if _xp(r_self) is not _xp(r_other) or not bool((r_self == r_other).all().item()):
            raise ValueError(
                "add_exact: the two EDMs close onto different rho0_vec values, so their "
                "sum would not represent the sum of the two evolutions")

        xp = _xp(left[0])
        n = self.n
        summed = []
        for p in range(n):
            a, b = left[p], right[p]
            shared_left = p == 0                    # OUT
            shared_right = p == n - 1               # RHO0
            if shared_left and shared_right:        # single site: both legs external
                summed.append(a + b)
                continue
            n_left = a.shape[1] if shared_left else a.shape[1] + b.shape[1]
            n_right = a.shape[2] if shared_right else a.shape[2] + b.shape[2]
            block = xp.zeros((a.shape[0], n_left, n_right),
                             dtype=np.promote_types(a.dtype, b.dtype))
            la = slice(None) if shared_left else slice(0, a.shape[1])
            lb = slice(None) if shared_left else slice(a.shape[1], n_left)
            ra = slice(None) if shared_right else slice(0, a.shape[2])
            rb = slice(None) if shared_right else slice(a.shape[2], n_right)
            block[:, la, ra] = a
            block[:, lb, rb] = b
            summed.append(block)
        return QuimbEDM.from_edmmps(
            # copied, not shared: the result must own every buffer it carries, or a caller
            # writing into it would reach back into an input (a backend-native copy, so a
            # CuPy vector stays on the device)
            EDMMPS(tensors=summed, d=self.d, d_phys=self.d_phys,
                   rho0_vec=self.rho0_vec.copy(), meta=dict(self.meta)))
