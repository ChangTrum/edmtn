"""Single-bath EDM evolution engine (Layer 5).

Implements the forward recursive construction of the EDM tensor network
(Fig. 3d / Fig. 9b) for a single (Gaussian) bath.  Each physical step:

1. fetch the combined-kernel MPO ``K_t`` (Layer 3);
2. build the system superoperators ``S^{phi}(t)`` (Layer 4b);
3. apply the kernel + new superoperator, growing the EDM-MPS by one site
   (:func:`~edmtn.evolution.mps_utils.apply_step`, Eq. 8);
4. recompress the enlarged bonds by an SVD sweep (Layer 4a), using the EDM
   paper's truncation rule ``discard s_i / s_{d**2+1} <= xi``.

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

from ..decomposition.standard_svd import StandardSVD
from ..expansion.first_order import FirstOrderExpander
from . import mps_utils


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
    truncation_errors : list[float]
        Largest per-bond discarded Frobenius weight at each step.
    """

    mps: object
    times: list = field(default_factory=list)
    bond_dims: list = field(default_factory=list)
    density_matrices: list | None = None
    truncation_errors: list = field(default_factory=list)


class SingleBathEvolution:
    """Forward EDM-MPS evolution for a single bath.

    Parameters
    ----------
    expander : TimeStepExpander, optional
        Small-step expansion (default :class:`FirstOrderExpander`).  Must be
        first order in Phase 1.
    decomposition : DecompositionStrategy, optional
        Compression strategy (default :class:`StandardSVD`).
    """

    def __init__(self, expander=None, decomposition=None):
        self.expander = expander if expander is not None else FirstOrderExpander()
        if self.expander.order not in (1, 2):
            raise NotImplementedError(
                f"unsupported expansion order {self.expander.order}"
            )
        self.decomposition = decomposition if decomposition is not None else StandardSVD()

    def run(
        self,
        model,
        kernel_engine,
        eps: float,
        n_steps: int,
        *,
        max_bond: int | None = None,
        cutoff: float = 0.0,
        cutoff_mode: str = "rel_ref",
        ref_index: int | None = None,
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
        max_bond, cutoff, cutoff_mode, ref_index :
            Truncation controls passed to the compression strategy.  The default
            ``cutoff_mode='rel_ref'`` with ``ref_index = d**2`` reproduces the
            paper's rule.  ``cutoff = 0`` keeps every singular value (exact).
        record_rho : bool
            Store ``rho(t)`` at every step.
        compress : bool
            If ``False``, skip the SVD sweep (exact, exponential bonds; for
            small-``t`` reference checks).
        convert : callable, optional
            Applied to every array fed into the MPS (initial state, kernel sites,
            system superoperators).  Use it to move the computation onto another
            backend or precision, e.g. ``lambda a: cupy.asarray(a, cupy.complex64)``
            for single-precision GPU.  Defaults to identity (CPU, complex128).
        """
        d = model.system_dim
        if ref_index is None:
            ref_index = d * d
        if convert is None:
            convert = lambda a: a  # noqa: E731
        rho0_vec = convert(model.initial_system_state().reshape(-1).astype(np.complex128))
        order = self.expander.order
        if order == 2 and getattr(kernel_engine, "order", 1) != 2:
            raise ValueError(
                "second-order evolution needs a second-order kernel engine "
                "(GaussianKernelEngine(..., order=2))"
            )

        result = EvolutionResult(mps=None)
        if record_rho:
            result.density_matrices = []

        mps = None
        g = 0  # global sub-step index (1-based)
        for n in range(1, n_steps + 1):
            t_phys = n * eps
            families = [convert(f) for f in self.expander.build_at(model, t_phys, eps).families]
            err = 0.0
            for sub in range(order):  # S_1 then S_2 for order 2; single for order 1
                g += 1
                ksites = [convert(k) for k in kernel_engine.get_kernel_mpo(g).site_tensors]
                mps = mps_utils.apply_step(mps, ksites, families[sub], d, rho0_vec)
                if compress and mps.num_sites > 1:
                    mps, infos = mps_utils.compress(
                        mps,
                        strategy=self.decomposition,
                        max_bond=max_bond,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        ref_index=ref_index,
                    )
                    if infos:
                        err = max(err, max(info["error"] for info in infos))

            # record after the complete physical step
            result.times.append(t_phys)
            result.bond_dims.append(mps.max_bond)
            result.truncation_errors.append(float(err))
            if record_rho:
                result.density_matrices.append(mps.reduced_density_matrix())

        result.mps = mps
        return result
