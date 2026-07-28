"""Separable-bath EDM evolution engine (Layer 5).

Implements the outer-loop recursion of Eq. 21 / Fig. 5c.  A separable bath is a
set of ``K`` independent sub-baths; the EDM is built by folding them in one at a
time:

    rho_0^{Phi}   = S^{Phi} rho(0)                      (pure system evolution)
    rho_{L}^{Phi} = C_{L;Phi'} [prod_t P] rho_{L-1}^{Phi''}   (Eq. 21)

The picking tensors ``P`` are already baked into the Layer-3 combined-kernel MPO
``C_L`` (one operatorised, time-uniform site per time slice).  So each sub-bath
step is a **matrix-product-operator * matrix-product-state contraction along the
time axis**, growing every internal bond by the lateral factor ``D_a`` and then
recompressing -- unlike the single-bath engine, which grows the chain by one new
*time* site per step.

The reduced density matrix ``rho_L(T) = delta^0_Phi rho_L^{Phi}`` is recovered by
closing every open arm (the standard :meth:`EDMMPS.reduced_density_matrix`); the
linearly increasing bond-dimension theorem holds for every ``rho_L`` (Theorem 2),
so the cost stays polynomial.

First- and second-order time-step expansions are supported.  Second order runs on
the doubled sub-step grid (``2 N`` sites): the system MPS ``rho_0`` alternates the
``S_1`` / ``S_2`` families, and the sub-bath MPO follows the same sub-step map.

This engine serves **both** separable bath types.  For ``bath_type='separable'``
(Gaudin) the sub-bath MPO is time-uniform, because those bath spins have no
self-Hamiltonian.  For ``bath_type='separable_td'`` (Dicke) it is not: every time site
carries its own tensor, the system operators rotate too, and local Lindblad channels are
folded in.  The engine itself does not branch on this -- it reads whatever per-site
tensors the kernel provides -- but it does sample the system operators at the **midpoint**
of each physical step (see :meth:`SeparableBathEvolution._build_system_mps`) and offers an
optional ``check_grid`` hook so a time-specific kernel can reject a grid it was not built
for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..expansion.second_order import SecondOrderExpander
from ._validation import (
    validate_bool,
    validate_compression_combination,
    validate_cutoff_mode,
    validate_expansion_order,
    validate_final_time,
    validate_nonnegative_finite_float,
    validate_optional_positive_int,
    validate_positive_finite_float,
    validate_positive_int,
    validate_separable_bath_kernel,
    validate_tangent_closings,
    validate_tangent_time_read_combination,
    validate_time_read_combination,
)
from .mps_utils import EDMMPS
from .prefix_reads import PrefixTerminators


@dataclass
class SeparableEvolutionResult:
    """Output of :meth:`SeparableBathEvolution.run`.

    Attributes
    ----------
    mps : EDMMPS
        The final EDM-MPS (all ``K`` sub-baths folded in).
    n_sub_baths : int
        Number of sub-baths ``K``.
    recorded_L : list[int]
        The sub-bath counts ``L`` at which results were recorded.
    bond_dims : list[int]
        Maximum internal bond dimension after folding in sub-bath ``L``.
    density_matrices : list[ndarray] or None
        ``rho_L(T)`` at each recorded ``L`` (if ``record_rho``).
    truncation_errors : list[float | None]
        One entry per **recorded sub-bath count** ``L`` (so ``len == len(recorded_L)``): the
        largest per-bond **discarded weight** of each recorded fold interval --
        ``max_b sum_{i discarded at bond b} sigma_i**2`` on the ``zipup``/``direct`` exact
        paths, ``max_b sum_{i discarded at bond b} lambda_i`` of the discarded density-matrix
        eigenvalues (``lambda_i = sigma_i**2``, NOT ``lambda_i**2``) on the ``dm`` path --
        over every fold since the PREVIOUS recorded ``L`` up to this one -- so a
        ``record_every > 1`` never silently drops the un-recorded folds' truncation.  This is
        the discarded WEIGHT, not quimb's discarded 2-norm (``error``), and it is a local
        per-interval quantity, NOT a cumulative error bound.  ``0.0`` means compression ran
        and discarded nothing (or ``compress=False``); ``None`` means the chosen
        decomposition cannot measure it exactly (``compress_decomp='rsvd'``).
    """

    mps: object
    n_sub_baths: int
    recorded_L: list = field(default_factory=list)
    bond_dims: list = field(default_factory=list)
    density_matrices: list | None = None
    truncation_errors: list[float | None] = field(default_factory=list)
    #: ``rho_L(t_n)`` for the FINAL ``L``, one entry per physical step (axis = TIME), or
    #: ``None`` unless ``record_time_reads``.  A different axis from
    #: :attr:`density_matrices`, which is ``rho_L(T)`` per sub-bath count -- the two are
    #: orthogonal and may be recorded together.  See
    #: :mod:`edmtn.evolution.prefix_reads`.
    time_density_matrices: list | None = None
    #: outer compression path actually entered, or ``None`` if none ran
    compression_method_used: str | None = None
    #: ``{channel: (d, d) array}`` -- the tangent channels' reduced matrices
    #: ``tilde_rho^(alpha) = Tr_B[(1 (x) sigma_alpha) rho_CB(T)]`` at the FINAL sub-bath
    #: count, or ``None`` unless ``tangent_closings`` was given.  The full tangent chains
    #: are released after this read; only these ``d x d`` matrices survive.
    tangent_density_matrices: dict | None = None
    #: ``{channel: list[float | None]}`` aligned with :attr:`recorded_L`, with exactly the
    #: same per-interval semantics as :attr:`truncation_errors` -- and recorded
    #: **separately**, because once the chains are compressed the tangent is a jet
    #: approximation whose error the value channel's record says nothing about.
    tangent_truncation_errors: dict | None = None


class SeparableBathEvolution:
    """Outer-loop EDM evolution for a separable bath (Eq. 21).

    Parameters
    ----------
    expander : TimeStepExpander, optional
        Time-step expansion (default :class:`SecondOrderExpander`, matching the
        paper's Gaudin calculation).
    compress_method, compress_decomp, compress_decomp_q, compress_canon :
        quimb compression controls (see :mod:`edmtn.evolution.quimb_decomp`).
    """

    def __init__(self, expander=None, *, compress_method="zipup",
                 compress_decomp="exact", compress_decomp_q=2, compress_canon="quimb"):
        self.expander = expander if expander is not None else SecondOrderExpander()
        if self.expander.order not in (1, 2):
            raise NotImplementedError(f"unsupported expansion order {self.expander.order}")
        self.compress_method = compress_method         # 'zipup'|'dm'|'direct'|'dm_tracking'
        self.compress_decomp = compress_decomp         # 'exact' | 'rsvd'
        self.compress_decomp_q = compress_decomp_q     # rsvd power iterations
        self.compress_canon = compress_canon           # 'quimb' | 'householder' | 'cholqr'

    def run(
        self,
        model,
        kernel_engine,
        eps: float,
        n_steps: int,
        *,
        max_bond: int | None = None,
        cutoff: float = 0.0,
        cutoff_mode: str = "rel",
        record_rho: bool = False,
        record_every: int = 1,
        record_time_reads: bool = False,
        compress: bool = True,
        convert=None,
        sub_baths: int | None = None,
        memory=None,
        tangent_closings=None,
    ) -> SeparableEvolutionResult:
        """Fold the ``K`` sub-baths into the EDM one at a time.

        All arguments are validated at the entry point (before any tensor is built or the
        kernel is read), so a direct call bypassing the driver still fails loudly with a
        clear ``ValueError`` -- see :mod:`edmtn.evolution._validation`.

        Parameters
        ----------
        model : AbstractOQSModel
            Separable-bath model (supplies the initial state and the system
            superoperators via the expander).
        kernel_engine : SeparableKernelEngine
            Per-sub-bath combined-kernel provider (Layer 3).
        eps, n_steps : float, int
            Time step and number of *physical* steps (the grid has
            ``order * n_steps`` sub-steps).
        max_bond, cutoff, cutoff_mode :
            Truncation controls for the per-sub-bath compression sweep.  ``cutoff_mode``
            defaults to the quimb-native ``'rel'`` (``s_i / s_max <= cutoff``).  The paper's
            custom ``rel_ref`` rule (``lambda_a / lambda_{d**2+1} <= xi``) -- and the
            reference-index parameter it needed -- are retired; no such argument exists.
        record_rho : bool
            Record ``rho_L(T)`` after the recorded sub-baths.
        record_every : int
            Record every ``record_every``-th sub-bath (and always the last).
        record_time_reads : bool
            Also produce ``rho_L(t_n)`` at **every physical step** for the final ``L``, from
            this one fold, via the causal-prefix terminators of
            :mod:`edmtn.evolution.prefix_reads` -- instead of re-running the whole fold once
            per target time.  Lands in :attr:`SeparableEvolutionResult.time_density_matrices`.
            Orthogonal to ``record_rho`` (a different axis); both may be set.  When
            ``compress`` is also true the compression must be ``'dm_tracking'``, because the
            terminators have to be carried through every bond-basis change and no other
            method exposes them; with ``compress=False`` no basis change happens and any
            otherwise-valid compression configuration is accepted.  Bath-type agnostic: this
            works for ``'separable'`` (Gaudin) and ``'separable_td'`` (Dicke) alike, since
            both kernels use the same newest-site ``a_left = 0`` boundary.
        compress : bool
            If ``False``, genuinely SKIP compression after each fold -- exact, with
            exponentially growing bonds (small-``K`` reference checks).  ``True`` compresses
            every fold: with ``cutoff=0`` and ``max_bond=None`` a no-discard recompression,
            with ``cutoff>0`` and/or a rank-limiting ``max_bond`` a potentially truncating
            one.  (Previously ``False`` silently ran a zero-cutoff recompression rather
            than skipping.)
        convert : callable, optional
            Backend/precision cast applied to every array fed into the MPS
            (initial state, system superoperators, kernel sites).  Defaults to
            identity (CPU, complex128).
        sub_baths : int, optional
            Fold in only the first ``sub_baths`` sub-baths in the model's stored coupling
            order, instead of all ``K`` -- the paper's "first L spins" curves (Fig. 6).
            ``None`` (default) folds all ``K``.  An out-of-range / non-integer value raises
            (no silent clamp; see :func:`~edmtn.models.base.validate_sub_baths`).
        memory : MemoryManager, optional
            GPU memory manager; its pool blocks are freed after each sub-bath so
            the O(K) outer loop does not accumulate VRAM (Sec. 8.4).  No-op on CPU.
        tangent_closings : mapping, optional
            ``{channel: (K, lateral) array}`` of newest-site lateral closing vectors, row
            ``k`` for sub-bath ``k``.  Requesting a channel folds a **second** chain per
            channel, the first-order jet of the sub-bath product: with every closing
            replaced by ``e_0 + lambda v_k``, the coefficient of ``lambda`` is exactly the
            SUM over sub-baths of the one-insertion chains, so one extra fold per channel
            gives the collective quantity instead of ``K`` separate solves::

                M_{L+1}  = F^(0)_{L+1} M_L
                dM_{L+1} = F^(0)_{L+1} dM_L + F^(v)_{L+1} M_L,      dM_0 = 0

            with the source term built from ``M_L`` (the state *before* this fold), not
            from the updated one.  This is an exact derivative recursion in **uncompressed**
            tensor algebra only; once the chains are compressed independently the tangent is
            a jet approximation, which is why its truncation is recorded separately in
            :attr:`SeparableEvolutionResult.tangent_truncation_errors` and why the value
            channel's record is not evidence about it.  The value channel itself is
            untouched: it folds and compresses exactly as it would without this argument.
            Needs a kernel engine whose providers accept a lateral ``closing``, and is
            rejected together with ``record_time_reads`` under compression (see
            :func:`~edmtn.evolution._validation.validate_tangent_time_read_combination`).
        """
        from ..models.base import validate_sub_baths  # noqa: PLC0415

        # -- entry validation (before convert / _build_system_mps / kernel read / QuimbEDM) --
        eps = validate_positive_finite_float("eps", eps)
        n_steps = validate_positive_int("n_steps", n_steps)
        validate_final_time(eps, n_steps)
        cutoff = validate_nonnegative_finite_float("cutoff", cutoff)
        max_bond = validate_optional_positive_int("max_bond", max_bond)
        record_rho = validate_bool("record_rho", record_rho)
        record_time_reads = validate_bool("record_time_reads", record_time_reads)
        compress = validate_bool("compress", compress)
        record_every = validate_positive_int("record_every", record_every)
        cutoff_mode = validate_cutoff_mode("cutoff_mode", cutoff_mode)
        validate_compression_combination(
            self.compress_method, self.compress_decomp, self.compress_canon)
        # dm_tracking exists only to carry the prefix terminators, and the terminators can
        # only be carried by it -- so the two are tied together, in both directions, before
        # any tensor is built.  compress=False is exempt: nothing changes the bond basis.
        validate_time_read_combination(
            record_time_reads=record_time_reads, compress=compress,
            compress_method=self.compress_method)
        order = validate_expansion_order("evolution order", self.expander.order)
        # structural model/kernel check: d_phys, matching K, for_sub_bath interface
        K = validate_separable_bath_kernel(model, kernel_engine)
        # sub_baths only after model/kernel K agree; None -> K; K+1 / 2.9 / True -> ValueError
        n_fold = validate_sub_baths(sub_baths, K)
        tangent_closings = validate_tangent_closings("tangent_closings", tangent_closings, K)
        validate_tangent_time_read_combination(
            tangent_closings=tangent_closings, record_time_reads=record_time_reads,
            compress=compress)
        # capability, not bath-type string: a kernel whose providers do not take a lateral
        # closing would otherwise fail deep in the fold loop as a bare TypeError
        if tangent_closings is not None and not getattr(kernel_engine, "supports_closings", False):
            raise ValueError(
                "tangent_closings needs a kernel engine whose sub-bath providers accept a "
                f"lateral `closing` (it must set supports_closings); "
                f"{type(kernel_engine).__name__} does not, so its folds cannot carry a "
                "bath-side measurement insertion")
        # Optional grid hook: a kernel whose site tensors are TIME-SPECIFIC (the
        # separable_td engine) exposes check_grid and rejects a grid it was not built for.
        # The site count alone cannot catch this -- (eps, n_steps, order) = (0.1, 4, 1) and
        # (0.1, 2, 2) give the same number of sites -- and a mismatch is not a crash but a
        # silently wrong sampling of the bath.  Called BEFORE any tensor is built.  Kernels
        # with time-uniform sites (Gaudin) have no such attribute and are unaffected.
        check_grid = getattr(kernel_engine, "check_grid", None)
        if check_grid is not None:
            check_grid(eps=eps, n_steps=n_steps, order=order)

        d = model.system_dim
        if convert is None:
            convert = lambda a: a  # noqa: E731
        n_sites = order * n_steps
        d_phys = kernel_engine.d_phys

        rho0_vec = convert(model.initial_system_state().reshape(-1).astype(np.complex128))

        # rho_0 = S^Phi rho(0): pure system evolution MPS (bond dim d**2), carried
        # as a quimb TensorNetwork through the fold loop.
        from .quimb_edm import QuimbEDM  # noqa: PLC0415

        mps = QuimbEDM.from_edmmps(
            self._build_system_mps(model, eps, n_steps, order, d, d_phys, rho0_vec, convert))

        result = SeparableEvolutionResult(mps=None, n_sub_baths=n_fold)
        if record_rho:
            result.density_matrices = []
        # one (d**2, chi) boundary matrix per PHYSICAL cut; the odd (mid-Strang) cuts are
        # never created, so nothing downstream can read a half-step by accident
        terminators = (PrefixTerminators(d * d, n_steps, order, rho0_vec)
                       if record_time_reads else None)

        # -- tangent (jet) channels: one extra chain per channel, dM_0 = 0.  `None` stands
        #    for the zero MPS -- a chain of zeros would have to be built and added for no
        #    effect, so the first fold's source term simply becomes dM_1.  It is still
        #    compressed like any other fold (see the loop below).
        tangent = {ch: None for ch in tangent_closings} if tangent_closings else {}
        tangent_interval: dict[str, float | None] = dict.fromkeys(tangent, 0.0)
        if tangent:
            result.tangent_truncation_errors = {ch: [] for ch in tangent}

        interval_weight: float | None = 0.0  # max discarded weight since the last recorded L
        for k in range(n_fold):
            provider = kernel_engine.for_sub_bath(k)   # built once, used by every channel
            mpo_sites = [convert(s) for s in provider.get_kernel_mpo(n_sites).site_tensors]
            # the value channel's own input to this fold; the tangent source term needs the
            # state BEFORE the fold, so it is held until the channel loop below
            mps_before = mps if tangent else None
            mps = mps.fold_raw(mpo_sites)              # lossless MPO x MPS growth
            if terminators is not None:
                terminators.fold(mpo_sites)            # l_m <- kron(l_m, e_0), same fused layout
            if compress:                               # compress=False genuinely skips compression
                # ONE call: 'dm_tracking' dispatch lives in QuimbEDM.compress, so the
                # method-to-implementation map is in a single place rather than duplicated
                # here.  terminators is None on every ordinary run.
                mps = mps.compress(
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    method=self.compress_method,
                    max_bond=max_bond,
                    decomp=self.compress_decomp,
                    decomp_q=self.compress_decomp_q,
                    canon=self.compress_canon,
                    terminators=terminators,
                )
                w = mps.max_discarded_weight
                result.compression_method_used = self.compress_method
                # accumulate across the WHOLE interval since the last recorded L, so a
                # record_every > 1 cannot silently drop the un-recorded folds' truncation
                if w is None:
                    interval_weight = None
                elif interval_weight is not None:
                    interval_weight = max(interval_weight, w)

            for ch, rows in (tangent_closings or {}).items():
                src_sites = [convert(s) for s in
                             provider.get_kernel_mpo(n_sites, rows[k]).site_tensors]
                src = mps_before.fold_raw(src_sites)   # F^(v)_{L+1} M_L
                # dM_0 = 0 lets the FIRST fold skip the zero chain's own fold and addition
                # -- and nothing else.  The compression the caller configured still applies
                # to dM_1, exactly as it does to the value channel's first fold; skipping it
                # would silently ignore `cutoff` / `max_bond` on this channel (and at
                # n_fold = 1 they would never take effect at all) while recording 0.0.
                grown = (src if tangent[ch] is None
                         else tangent[ch].fold_raw(mpo_sites).add_exact(src))
                if compress:
                    grown = grown.compress(
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        method=self.compress_method,
                        max_bond=max_bond,
                        decomp=self.compress_decomp,
                        decomp_q=self.compress_decomp_q,
                        canon=self.compress_canon,
                        terminators=None,              # tangents carry no prefix terminators
                    )
                    w = grown.max_discarded_weight
                    if w is None:
                        tangent_interval[ch] = None
                    elif tangent_interval[ch] is not None:
                        tangent_interval[ch] = max(tangent_interval[ch], w)
                tangent[ch] = grown
                del grown, src, src_sites
            # drop the pre-fold value chain and the MPO temporaries before the pool is
            # freed, so the release actually reclaims them
            del mps_before, mpo_sites, provider

            # release the previous sub-bath's GPU intermediates (no-op on CPU)
            if memory is not None:
                memory.free_all_blocks()

            L = k + 1
            if L == n_fold or (L % record_every == 0):
                result.recorded_L.append(L)
                result.bond_dims.append(mps.max_bond)
                result.truncation_errors.append(interval_weight)
                interval_weight = 0.0  # start a fresh interval for the next recorded L
                for ch in tangent:
                    result.tangent_truncation_errors[ch].append(tangent_interval[ch])
                    tangent_interval[ch] = 0.0
                if record_rho:
                    result.density_matrices.append(mps.reduced_density_matrix())

        if tangent:
            # only the d x d reads survive: a returned result must not hold one full-length
            # time chain per tangent channel on top of the value chain
            result.tangent_density_matrices = {
                ch: chain.reduced_density_matrix() for ch, chain in tangent.items()}
            tangent.clear()
            if memory is not None:
                memory.free_all_blocks()

        # hand back a plain EDMMPS so observable extraction reads per-site tensors
        result.mps = mps.to_edmmps()
        if terminators is not None:
            # one right-to-left sweep over the finished chain; the last entry is the very
            # rho_L(T) that reduced_density_matrix() returns, because l_M is the identity
            result.time_density_matrices = terminators.read(result.mps.tensors, rho0_vec, d)
        return result

    # -- construction ------------------------------------------------------

    def _build_system_mps(self, model, eps, n_steps, order, d, d_phys, rho0_vec, convert) -> EDMMPS:
        """Build ``rho_0 = S^{Phi} rho(0)`` as a system-folded MPS (bond dim ``d**2``).

        Site ``p`` (newest first) carries the system superoperator family of
        sub-step ``g = n_sites - p``; for order 2 the family alternates
        ``S_1`` (odd ``g``) / ``S_2`` (even ``g``).

        The interaction-picture operators are sampled at the **midpoint** of the physical
        step, ``t_n^* = (n - 1/2) eps``, with both algebraic sub-steps of a step sharing
        that one time.  Midpoint sampling is what makes ``order = 2`` globally second
        order for a time-dependent coupling (an endpoint reproduces the first Magnus term
        only to ``O(eps^2)`` and caps the scheme at first order); see
        ``docs/design/dicke-second-order-discretisation.md``.  For a model whose
        interaction-picture operators are constant -- Gaudin, where ``H_S = 0`` -- the
        sampling point is irrelevant and the tensors are unchanged.  The bath side must
        use the identical map, which is what the ``check_grid`` hook in :meth:`run`
        guards.
        """
        n_sites = order * n_steps
        fam_cache: dict[int, list] = {}
        tensors = []
        for p in range(n_sites):
            g = n_sites - p                 # sub-step index 1..n_sites (oldest = 1)
            n = (g - 1) // order + 1        # physical step 1..n_steps
            sub = (g - 1) % order           # 0 -> S_1, 1 -> S_2
            if n not in fam_cache:
                fam_cache[n] = self.expander.build_at(model, (n - 0.5) * eps, eps).families
            S = fam_cache[n][sub]           # (d_phys, d**2, d**2)
            tensors.append(convert(np.asarray(S, dtype=np.complex128)))
        return EDMMPS(tensors=tensors, d=d, d_phys=d_phys, rho0_vec=rho0_vec)
