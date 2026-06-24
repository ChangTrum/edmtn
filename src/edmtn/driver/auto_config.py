"""Pipeline auto-configuration (Layer 7).

Selects and constructs the engine stack (cumulant -> kernel -> expansion ->
decomposition -> evolution) from a model's ``bath_type``.  Pipelines are kept in
a small registry so future bath types (``separable``, ``chain``) slot in without
touching the driver.

Phase 1 ships the ``gaussian`` pipeline only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..decomposition.randomized_svd import RandomizedSVD
from ..decomposition.standard_svd import StandardSVD
from ..expansion.first_order import FirstOrderExpander
from ..expansion.second_order import SecondOrderExpander
from ..kernels.gaussian_mpo import GaussianKernelEngine
from ..kernels.separable_mpo import SeparableKernelEngine
from ..evolution.separable_bath import SeparableBathEvolution
from ..evolution.single_bath import SingleBathEvolution


@dataclass
class SolverConfig:
    """Configuration for :class:`~edmtn.driver.solver.EDMSolver`.

    Attributes
    ----------
    eps : float
        Time step.
    T : float
        Total evolution time; ``n_steps = round(T / eps)``.
    cutoff : float
        SVD truncation precision (``0`` keeps every singular value -- exact but
        not scalable).
    cutoff_mode : str
        Truncation rule (default ``'rel_ref'``, the paper's ``s_i / s_{d^2+1}``).
    max_bond : int, optional
        Hard bond-dimension cap.
    ref_index : int, optional
        Reference index for ``'rel_ref'`` (defaults to ``d**2``).
    expansion_order : int
        Trotter order.  Phase 1 supports ``1``; ``2`` is accepted here but the
        single-bath engine currently rejects it (doubled sub-step grid pending).
    record_rho : bool
        Store ``rho(t)`` at every step (needed for custom observables).
    decomposition : DecompositionStrategy, optional
        Compression strategy (default :class:`StandardSVD`).
    canonicalization : CanonicalizationStrategy, optional
        Canonicalisation strategy (default Householder QR; e.g. ``CholeskyQR()``).
    """

    eps: float
    T: float
    cutoff: float = 1e-6
    cutoff_mode: str = "rel_ref"
    max_bond: int | None = None
    ref_index: int | None = None
    expansion_order: int = 1
    record_rho: bool = False
    decomposition: object | None = None
    canonicalization: object | None = None  # None -> Householder QR; e.g. CholeskyQR()
    preset: str | None = None  # None | 'balanced' | 'robust' (see docs/recommended-config.md)
    sub_baths: int | None = None  # separable: fold only the first L sub-baths (Fig. 6)
    backend: str = "auto"  # 'auto' | 'cpu' | 'gpu' (auto -> CPU for Phase 1/2; see docs/cpu-vs-gpu-edm.md)
    precision: str = "f64"  # 'f64' | 'mixed' (mixed: f32 contraction, f64 decompose -- Phase 3/4)

    def __post_init__(self):
        # Resolve a strategy preset, but never override explicitly-passed strategies.
        # Default (preset=None): StandardSVD + Householder (exact, deterministic).
        if self.preset is None:
            return
        if self.preset not in _PRESETS:
            raise ValueError(
                f"unknown preset {self.preset!r}; choose from {sorted(_PRESETS)} or None"
            )
        spec = _PRESETS[self.preset]
        if self.decomposition is None and "decomposition" in spec:
            self.decomposition = spec["decomposition"]()
        if self.canonicalization is None and "canonicalization" in spec:
            self.canonicalization = spec["canonicalization"]()

    @property
    def n_steps(self) -> int:
        return int(round(self.T / self.eps))


# Recommended strategy presets (docs/recommended-config.md).  Canonicalisation is
# Householder QR in both (the measured default everywhere), so only the
# decomposition differs; factories so each solver gets a fresh instance.
_PRESETS: dict = {
    # balanced: fastest, GPU-friendly, accuracy < cutoff
    "balanced": {"decomposition": lambda: RandomizedSVD(n_iter=0)},
    # robust: exact-baseline bonds, ~1e-12 accuracy (cold rSVD)
    "robust": {"decomposition": lambda: RandomizedSVD(n_iter=2)},
}


def _make_expander(order: int):
    if order == 1:
        return FirstOrderExpander()
    if order == 2:
        return SecondOrderExpander()
    raise ValueError(f"unsupported expansion_order {order!r}")


# -- pipeline registry -----------------------------------------------------

_PIPELINES: dict = {}


def register_pipeline(bath_type: str, builder) -> None:
    """Register a ``builder(model, config) -> (kernel_engine, evolution)``."""
    _PIPELINES[bath_type] = builder


def available_pipelines() -> tuple:
    return tuple(sorted(_PIPELINES))


def build_pipeline(model, config: SolverConfig):
    """Construct ``(kernel_engine, evolution_engine)`` for ``model``."""
    bt = model.bath_type
    if bt not in _PIPELINES:
        raise NotImplementedError(
            f"no EDM pipeline registered for bath_type={bt!r}; "
            f"available: {available_pipelines()}"
        )
    return _PIPELINES[bt](model, config)


def _build_gaussian(model, config: SolverConfig):
    kernel_engine = GaussianKernelEngine.from_model(
        model, T=config.T, eps=config.eps, order=config.expansion_order
    )
    decomposition = config.decomposition or StandardSVD()
    evolution = SingleBathEvolution(
        expander=_make_expander(config.expansion_order),
        decomposition=decomposition,
        canonicalization=config.canonicalization,
    )
    return kernel_engine, evolution


def _build_separable(model, config: SolverConfig):
    kernel_engine = SeparableKernelEngine.from_model(model, T=config.T, eps=config.eps)
    decomposition = config.decomposition or StandardSVD()
    evolution = SeparableBathEvolution(
        expander=_make_expander(config.expansion_order),
        decomposition=decomposition,
        canonicalization=config.canonicalization,
    )
    return kernel_engine, evolution


register_pipeline("gaussian", _build_gaussian)
register_pipeline("separable", _build_separable)
