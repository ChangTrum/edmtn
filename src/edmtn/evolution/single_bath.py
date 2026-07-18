"""Single-bath EDM evolution engine (Layer 5).

Implements the forward recursive construction of the EDM tensor network
(Fig. 3d / Fig. 9b) for a single (Gaussian) bath.  Each physical step:

1. fetch the combined-kernel MPO ``K_t`` (Layer 3);
2. build the system superoperators ``S^{phi}(t)`` (Layer 4b);
3. apply the kernel + new superoperator, growing the EDM-MPS by one site
   (:func:`~edmtn.evolution.mps_utils.apply_step`, Eq. 8);
4. recompress the enlarged bonds by a quimb compression sweep (Layer 4a) under the
   configured ``cutoff`` / ``cutoff_mode`` (default the quimb-native ``'rel'``:
   discard ``s_i / s_max <= cutoff``).  The paper's custom ``rel_ref`` rule
   (``s_i / s_{d**2+1}``) is retired.  Step 4 is skipped entirely when
   ``compress=False``.

The reduced density matrix ``rho(t)`` is recovered after each complete physical
step by closing the open arms with ``delta^0``.

First- and second-order time-step expansions are supported.  First order inserts
one superoperator per physical step.  Second order runs on a doubled sub-step
grid: each physical step is two ``apply_step`` calls (``S_1`` then ``S_2``),
driven by a second-order kernel whose lag map accounts for the sub-step parity;
``rho(t)`` is recorded only after the second sub-step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..expansion.first_order import FirstOrderExpander
from ._validation import (
    validate_bool,
    validate_cutoff_mode,
    validate_expansion_order,
    validate_final_time,
    validate_nonnegative_finite_float,
    validate_optional_positive_int,
    validate_positive_finite_float,
    validate_positive_int,
    validate_single_bath_kernel,
)


@dataclass
class EvolutionResult:
    """Output of :meth:`SingleBathEvolution.run`.

    Attributes
    ----------
    mps : EDMMPS
        The final EDM-MPS (step ``n_steps``).
    times : list[float]
        Physical time of each recorded step.
    bond_dims : list[int]
        Maximum internal bond dimension after compression at each step.
    density_matrices : list[ndarray] or None
        ``rho(t)`` at each step if ``record_rho`` was set.
    truncation_errors : list[float | None]
        One entry per **physical time step** (so ``len == len(times)``, also for order 2,
        where it is the max over BOTH sub-steps): the largest per-bond **discarded weight**
        ``max_b sum_{i discarded at bond b} sigma_i**2`` of the compressions run in that
        step.  This is the discarded WEIGHT, not quimb's discarded 2-norm (``error``), and
        it is a per-step local quantity, NOT a cumulative error bound.  ``0.0`` means a
        compression ran and discarded nothing (or none ran at all); ``None`` means the
        chosen decomposition cannot measure it exactly (``compress_decomp='rsvd'``, whose
        randomized sketch never sees the omitted tail of the spectrum).
    """

    mps: object
    times: list = field(default_factory=list)
    bond_dims: list = field(default_factory=list)
    density_matrices: list | None = None
    truncation_errors: list[float | None] = field(default_factory=list)


class SingleBathEvolution:
    """Forward EDM-MPS evolution for a single bath.

    Parameters
    ----------
    expander : TimeStepExpander, optional
        Small-step expansion (default :class:`FirstOrderExpander`).
    compress_method, compress_decomp, compress_decomp_q, compress_canon :
        quimb compression controls (see :mod:`edmtn.evolution.quimb_decomp`).
    """

    def __init__(self, expander=None, *, compress_method="zipup",
                 compress_decomp="exact", compress_decomp_q=2, compress_canon="quimb"):
        self.expander = expander if expander is not None else FirstOrderExpander()
        if self.expander.order not in (1, 2):
            raise NotImplementedError(
                f"unsupported expansion order {self.expander.order}"
            )
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
        compress: bool = True,
        convert=None,
    ) -> EvolutionResult:
        """Evolve the EDM for ``n_steps`` steps.

        Parameters
        ----------
        model : AbstractOQSModel
            Supplies the initial state and interaction-picture coupling
            operators.
        kernel_engine : KernelProvider
            Combined-kernel MPO provider (Layer 3).
        eps : float
            Time step.
        n_steps : int
            Number of physical steps to evolve.
        max_bond, cutoff, cutoff_mode :
            Truncation controls passed to the compression sweep.  ``cutoff_mode``
            defaults to the quimb-native ``'rel'`` (``s_i / s_max <= cutoff``).  The paper's
            custom ``rel_ref`` rule -- and the reference-index parameter it needed --
            are retired; no such argument exists.
            ``cutoff = 0`` with ``compress=True`` keeps every singular value (an exact
            canonicalise + full-SVD recompression), which is NOT the same as
            ``compress=False`` (no compression at all).
        record_rho : bool
            Store ``rho(t)`` at every step.
        compress : bool
            If ``False``, genuinely SKIP the compression sweep -- exact, with
            exponentially growing bonds (small-``t`` reference checks).  ``True``
            compresses each step: with ``cutoff=0`` an exact canonicalise + full-SVD
            recompression, with ``cutoff>0`` (or a ``max_bond``) a truncating one.
        convert : callable, optional
            Applied to every array fed into the MPS (initial state, kernel sites,
            system superoperators).  Use it to move the computation onto another
            backend or precision, e.g. ``lambda a: cupy.asarray(a, cupy.complex64)``
            for single-precision GPU.  Defaults to identity (CPU, complex128).

        All arguments are validated at the entry point (before any tensor is built or the
        kernel is read), so a direct call bypassing the driver still fails loudly with a
        clear ``ValueError`` -- see :mod:`edmtn.evolution._validation`.
        """
        # -- entry validation (before convert / initial_system_state / kernel read / QuimbEDM) --
        eps = validate_positive_finite_float("eps", eps)
        n_steps = validate_positive_int("n_steps", n_steps)
        validate_final_time(eps, n_steps)
        cutoff = validate_nonnegative_finite_float("cutoff", cutoff)
        max_bond = validate_optional_positive_int("max_bond", max_bond)
        record_rho = validate_bool("record_rho", record_rho)
        compress = validate_bool("compress", compress)
        cutoff_mode = validate_cutoff_mode("cutoff_mode", cutoff_mode)
        order = validate_expansion_order("evolution order", self.expander.order)
        # structural model/kernel check: d_phys, get_kernel_mpo, and matching order (both ways)
        validate_single_bath_kernel(model, kernel_engine, order)

        d = model.system_dim
        if convert is None:
            convert = lambda a: a  # noqa: E731
        rho0_vec = convert(model.initial_system_state().reshape(-1).astype(np.complex128))

        result = EvolutionResult(mps=None)
        if record_rho:
            result.density_matrices = []

        # the EDM is carried as a quimb TensorNetwork through the step loop
        from .quimb_edm import QuimbEDM  # noqa: PLC0415

        mps = QuimbEDM.empty(rho0_vec, d, kernel_engine.d_phys)
        g = 0  # global sub-step index (1-based)
        for n in range(1, n_steps + 1):
            t_phys = n * eps
            families = [convert(f) for f in self.expander.build_at(model, t_phys, eps).families]
            # largest per-bond discarded weight over THIS physical step; for order 2 that
            # spans both sub-steps, so the axis stays one entry per physical time step.
            # None (unmeasurable, e.g. rsvd) from any executed compression wins.
            step_weight: float | None = 0.0
            for sub in range(order):  # S_1 then S_2 for order 2; single for order 1
                g += 1
                ksites = [convert(k) for k in kernel_engine.get_kernel_mpo(g).site_tensors]
                mps = mps.step(ksites, families[sub], d)
                if compress and mps.num_sites > 1:
                    # compress=False skips this entirely (exact, exponentially growing
                    # bonds); it is NOT a zero-cutoff compression
                    mps = mps.compress(
                        cutoff=cutoff if compress else 0.0,
                        cutoff_mode=cutoff_mode,
                        method=self.compress_method,
                        max_bond=max_bond if compress else None,
                        decomp=self.compress_decomp,
                        decomp_q=self.compress_decomp_q,
                        canon=self.compress_canon,
                    )
                    w = mps.max_discarded_weight
                    if w is None:
                        step_weight = None
                    elif step_weight is not None:
                        step_weight = max(step_weight, w)

            # record after the complete physical step
            result.times.append(t_phys)
            result.bond_dims.append(mps.max_bond)
            result.truncation_errors.append(step_weight)
            if record_rho:
                result.density_matrices.append(mps.reduced_density_matrix())

        # hand back a plain EDMMPS so observable extraction reads per-site tensors
        result.mps = mps.to_edmmps()
        return result
