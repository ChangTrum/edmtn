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

from ..decomposition.standard_svd import StandardSVD
from ..expansion.second_order import SecondOrderExpander
from . import mps_utils
from .mps_utils import EDMMPS, _xp


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
    truncation_errors : list[float]
        Largest per-bond discarded weight when folding in sub-bath ``L``.
    """

    mps: object
    n_sub_baths: int
    recorded_L: list = field(default_factory=list)
    bond_dims: list = field(default_factory=list)
    density_matrices: list | None = None
    truncation_errors: list = field(default_factory=list)


class SeparableBathEvolution:
    """Outer-loop EDM evolution for a separable bath (Eq. 21).

    Parameters
    ----------
    expander : TimeStepExpander, optional
        Time-step expansion (default :class:`SecondOrderExpander`, matching the
        paper's Gaudin calculation).
    decomposition : DecompositionStrategy, optional
        Compression strategy (default :class:`StandardSVD`).
    """

    def __init__(self, expander=None, decomposition=None, canonicalization=None,
                 compression="native", compress_cutoff=1e-12,
                 compress_cutoff_mode="rel", compress_method="zipup"):
        self.expander = expander if expander is not None else SecondOrderExpander()
        if self.expander.order not in (1, 2):
            raise NotImplementedError(f"unsupported expansion order {self.expander.order}")
        self.decomposition = decomposition if decomposition is not None else StandardSVD()
        self.canonicalization = canonicalization  # None -> Householder QR
        self.compression = compression  # 'native' | 'quimb'
        self.compress_cutoff = compress_cutoff
        self.compress_cutoff_mode = compress_cutoff_mode
        self.compress_method = compress_method

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
            If ``False``, skip compression (exact, exponential bonds; small ``K``
            reference checks).
        convert : callable, optional
            Backend/precision cast applied to every array fed into the MPS
            (initial state, system superoperators, kernel sites).  Defaults to
            identity (CPU, complex128).
        sub_baths : int, optional
            Fold in only the first ``sub_baths`` (strongest) sub-baths instead of
            all ``K`` -- the paper's "first L spins" curves (Fig. 6).  Defaults to
            all sub-baths.
        memory : MemoryManager, optional
            GPU memory manager; its pool blocks are freed after each sub-bath so
            the O(K) outer loop does not accumulate VRAM (Sec. 8.4).  No-op on CPU.
        """
        d = model.system_dim
        if ref_index is None:
            ref_index = d * d
        if convert is None:
            convert = lambda a: a  # noqa: E731
        order = self.expander.order
        n_sites = order * n_steps
        K = kernel_engine.K
        n_fold = K if sub_baths is None else min(int(sub_baths), K)
        if n_fold < 1:
            raise ValueError(f"sub_baths must be >= 1, got {sub_baths}")
        d_phys = kernel_engine.d_phys

        rho0_vec = convert(model.initial_system_state().reshape(-1).astype(np.complex128))

        # rho_0 = S^Phi rho(0): pure system evolution MPS (bond dim d**2)
        mps = self._build_system_mps(model, eps, n_steps, order, d, d_phys, rho0_vec, convert)

        # 'quimb' carries the EDM as a quimb TensorNetwork through the fold loop
        # (the structural re-platform); 'native' keeps the bespoke EDMMPS path.
        use_quimb = self.compression == "quimb"
        if use_quimb:
            from .quimb_edm import QuimbEDM  # noqa: PLC0415

            mps = QuimbEDM.from_edmmps(mps)

        result = SeparableEvolutionResult(mps=None, n_sub_baths=n_fold)
        if record_rho:
            result.density_matrices = []

        for k in range(n_fold):
            mpo_sites = [
                convert(s) for s in kernel_engine.for_sub_bath(k).get_kernel_mpo(n_sites).site_tensors
            ]
            err = 0.0
            if use_quimb:
                # exact fold (compress=False) is reproduced by a zero cutoff
                mps = mps.fold(
                    mpo_sites,
                    cutoff=self.compress_cutoff if compress else 0.0,
                    cutoff_mode=self.compress_cutoff_mode,
                    method=self.compress_method,
                    max_bond=max_bond if compress else None,
                )
            else:
                mps = self._apply_sub_bath(mps, mpo_sites, d, d_phys, rho0_vec)
                if compress and mps.num_sites > 1:
                    mps, infos = mps_utils.compress(
                        mps,
                        strategy=self.decomposition,
                        canon=self.canonicalization,
                        engine=self.compression,
                        compress_cutoff=self.compress_cutoff,
                        compress_cutoff_mode=self.compress_cutoff_mode,
                        compress_method=self.compress_method,
                        max_bond=max_bond,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        ref_index=ref_index,
                    )
                    if infos:
                        err = max(info["error"] for info in infos)

            # release the previous sub-bath's GPU intermediates (no-op on CPU)
            if memory is not None:
                memory.free_all_blocks()

            L = k + 1
            if L == n_fold or (L % record_every == 0):
                result.recorded_L.append(L)
                result.bond_dims.append(mps.max_bond)
                result.truncation_errors.append(float(err))
                if record_rho:
                    result.density_matrices.append(mps.reduced_density_matrix())

        # hand back a plain EDMMPS so observable extraction (which reads per-site
        # tensors) is identical regardless of the carried container
        result.mps = mps.to_edmmps() if use_quimb else mps
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

    @staticmethod
    def _apply_sub_bath(mps, mpo_sites, d, d_phys, rho0_vec) -> EDMMPS:
        """Contract a sub-bath's combined-kernel MPO with the EDM along time.

        ``new[phi_up, (a_l, chi_l), (a_r, chi_r)] = sum_{phi_down}
        T[phi_up, phi_down, a_l, a_r] G[phi_down, chi_l, chi_r]``, fusing the
        lateral kernel bond (outer) with the existing MPS bond (inner).
        """
        xp = _xp(mpo_sites[0])
        new_tensors = []
        for p in range(mps.num_sites):
            T = mpo_sites[p]            # (phi_up, phi_down, a_l, a_r)
            G = mps.tensors[p]          # (phi_down, chi_l, chi_r)
            out = xp.tensordot(T, G, axes=([1], [0]))  # (phi_up, a_l, a_r, chi_l, chi_r)
            out = xp.transpose(out, (0, 1, 3, 2, 4))   # (phi_up, a_l, chi_l, a_r, chi_r)
            u, al, cl, ar, cr = out.shape
            new_tensors.append(out.reshape(u, al * cl, ar * cr))
        return EDMMPS(tensors=new_tensors, d=d, d_phys=d_phys, rho0_vec=rho0_vec)
