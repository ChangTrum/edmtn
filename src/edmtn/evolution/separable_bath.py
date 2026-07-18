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
the doubled sub-step grid (``2 N`` sites): the only difference for a separable
bath is that the system MPS ``rho_0`` alternates the ``S_1`` / ``S_2`` families
(the bath sub-bath MPO is time-uniform either way, the Gaudin bath being
time-independent).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..expansion.second_order import SecondOrderExpander
from ._validation import (
    validate_bool,
    validate_cutoff_mode,
    validate_expansion_order,
    validate_final_time,
    validate_nonnegative_finite_float,
    validate_optional_positive_int,
    validate_positive_finite_float,
    validate_positive_int,
    validate_separable_bath_kernel,
)
from .mps_utils import EDMMPS


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
        largest per-bond **discarded weight** ``max_b sum_{i discarded at bond b} sigma_i**2``
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
        self.compress_method = compress_method         # quimb 1D-compress: 'zipup'|'dm'|'direct'
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
        compress: bool = True,
        convert=None,
        sub_baths: int | None = None,
        memory=None,
    ) -> SeparableEvolutionResult:
        """Fold the ``K`` sub-baths into the EDM one at a time.

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
        max_bond, cutoff, cutoff_mode, ref_index :
            Truncation controls for the per-sub-bath compression sweep.  The
            default ``cutoff_mode='rel_ref'`` with ``ref_index = d**2`` is the
            paper's ``lambda_a / lambda_{d**2+1} <= xi`` rule.
        record_rho : bool
            Record ``rho_L(T)`` after the recorded sub-baths.
        record_every : int
            Record every ``record_every``-th sub-bath (and always the last).
        compress : bool
            If ``False``, genuinely SKIP compression after each fold -- exact, with
            exponentially growing bonds (small-``K`` reference checks).  ``True`` compresses
            every fold: with ``cutoff=0`` an exact canonicalise + full-SVD recompression,
            with ``cutoff>0`` a truncating one.  (Previously ``False`` silently ran a
            zero-cutoff recompression rather than skipping.)
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

        All arguments are validated at the entry point (before any tensor is built or the
        kernel is read), so a direct call bypassing the driver still fails loudly with a
        clear ``ValueError`` -- see :mod:`edmtn.evolution._validation`.
        """
        from ..models.base import validate_sub_baths  # noqa: PLC0415

        # -- entry validation (before convert / _build_system_mps / kernel read / QuimbEDM) --
        eps = validate_positive_finite_float("eps", eps)
        n_steps = validate_positive_int("n_steps", n_steps)
        validate_final_time(eps, n_steps)
        cutoff = validate_nonnegative_finite_float("cutoff", cutoff)
        max_bond = validate_optional_positive_int("max_bond", max_bond)
        record_rho = validate_bool("record_rho", record_rho)
        compress = validate_bool("compress", compress)
        record_every = validate_positive_int("record_every", record_every)
        cutoff_mode = validate_cutoff_mode("cutoff_mode", cutoff_mode)
        order = validate_expansion_order("evolution order", self.expander.order)
        # structural model/kernel check: d_phys, matching K, for_sub_bath interface
        K = validate_separable_bath_kernel(model, kernel_engine)
        # sub_baths only after model/kernel K agree; None -> K; K+1 / 2.9 / True -> ValueError
        n_fold = validate_sub_baths(sub_baths, K)

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

        interval_weight: float | None = 0.0  # max discarded weight since the last recorded L
        for k in range(n_fold):
            mpo_sites = [
                convert(s) for s in kernel_engine.for_sub_bath(k).get_kernel_mpo(n_sites).site_tensors
            ]
            mps = mps.fold_raw(mpo_sites)              # lossless MPO x MPS growth
            if compress:                               # compress=False genuinely skips compression
                mps = mps.compress(
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    method=self.compress_method,
                    max_bond=max_bond,
                    decomp=self.compress_decomp,
                    decomp_q=self.compress_decomp_q,
                    canon=self.compress_canon,
                )
                # accumulate across the WHOLE interval since the last recorded L, so a
                # record_every > 1 cannot silently drop the un-recorded folds' truncation
                w = mps.max_discarded_weight
                if w is None:
                    interval_weight = None
                elif interval_weight is not None:
                    interval_weight = max(interval_weight, w)

            # release the previous sub-bath's GPU intermediates (no-op on CPU)
            if memory is not None:
                memory.free_all_blocks()

            L = k + 1
            if L == n_fold or (L % record_every == 0):
                result.recorded_L.append(L)
                result.bond_dims.append(mps.max_bond)
                result.truncation_errors.append(interval_weight)
                interval_weight = 0.0  # start a fresh interval for the next recorded L
                if record_rho:
                    result.density_matrices.append(mps.reduced_density_matrix())

        # hand back a plain EDMMPS so observable extraction reads per-site tensors
        result.mps = mps.to_edmmps()
        return result

    # -- construction ------------------------------------------------------

    def _build_system_mps(self, model, eps, n_steps, order, d, d_phys, rho0_vec, convert) -> EDMMPS:
        """Build ``rho_0 = S^{Phi} rho(0)`` as a system-folded MPS (bond dim ``d**2``).

        Site ``p`` (newest first) carries the system superoperator family of
        sub-step ``g = n_sites - p``; for order 2 the family alternates
        ``S_1`` (odd ``g``) / ``S_2`` (even ``g``).
        """
        n_sites = order * n_steps
        fam_cache: dict[int, list] = {}
        tensors = []
        for p in range(n_sites):
            g = n_sites - p                 # sub-step index 1..n_sites (oldest = 1)
            n = (g - 1) // order + 1        # physical step 1..n_steps
            sub = (g - 1) % order           # 0 -> S_1, 1 -> S_2
            if n not in fam_cache:
                fam_cache[n] = self.expander.build_at(model, n * eps, eps).families
            S = fam_cache[n][sub]           # (d_phys, d**2, d**2)
            tensors.append(convert(np.asarray(S, dtype=np.complex128)))
        return EDMMPS(tensors=tensors, d=d, d_phys=d_phys, rho0_vec=rho0_vec)
