"""Pipeline auto-configuration (Layer 7).

Selects and constructs the engine stack (cumulant -> kernel -> expansion ->
decomposition -> evolution) from a model's ``bath_type``.  Pipelines are kept in
a small registry so future bath types (``separable``, ``chain``) slot in without
touching the driver.

Phase 1 ships the ``gaussian`` pipeline only.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field, replace

from ..expansion.first_order import FirstOrderExpander
from ..expansion.second_order import SecondOrderExpander
from ..kernels.gaussian_mpo import GaussianKernelEngine
from ..kernels.separable_mpo import SeparableKernelEngine
from ..evolution._validation import CUTOFF_MODES as _CUTOFF_MODES
from ..evolution._validation import validate_compression_combination
from ..evolution.separable_bath import SeparableBathEvolution
from ..evolution.single_bath import SingleBathEvolution

# -- config validation -----------------------------------------------------
# Every SolverConfig knob is validated at construction (in __post_init__) so a
# bad value fails loudly and immediately, at the config entry point, rather than
# surfacing as a divide-by-zero / int-cast / silent-round deep inside a pipeline
# -- and so BOTH tracks (cpu/gpu Track 1 and the hpc Track 2) reject an illegal
# time grid with the SAME error instead of Track 2 silently rounding T/eps down.

_NSTEPS_RTOL = 1e-9  # T/eps must equal an integer within this relative tolerance

# Public allowed enum values -- EDMTN's supported/tested config contract, read
# from the code that consumes each knob (a backend/library may accept more; these
# are the values EDMTN exposes and tests):
_BACKENDS = ("cpu", "numpy", "gpu", "cupy", "hpc")   # solver._resolve_backend + hpc path
_PRECISIONS = ("f64", "mixed")                        # solver._resolve_backend
# _CUTOFF_MODES is imported from evolution._validation (single source of truth shared with
# the direct evolution run() entry points, so the driver/direct contracts cannot drift).
_COMPRESS_METHODS = ("zipup", "dm", "direct")         # EDMTN's tested subset of quimb 1D-compress
_COMPRESS_DECOMPS = ("exact", "rsvd")                 # quimb_decomp.compress_opts_for
_COMPRESS_CANONS = ("quimb", "householder", "cholqr")  # quimb_decomp._CANON_METHOD
_PATHFINDERS = ("cuquantum", "cotengra")              # cutensornet path-finder select


def _is_int(value) -> bool:
    """True for a genuine integer (Python ``int`` or NumPy integer), excluding ``bool``."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_real(value) -> bool:
    """True for a real number usable as a float, excluding ``bool``."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _to_float(name: str, value) -> float:
    """``float(value)`` with ``TypeError`` / ``ValueError`` / ``OverflowError`` -> ``ValueError``.

    A huge Python ``int`` (e.g. ``10**400``) is a real number but overflows ``float()``; the
    honest response is a project ``ValueError``, not a leaked ``OverflowError`` (cf. P0-2).
    """
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite real number, got {value!r}") from None


def _positive_finite_float(name: str, value) -> float:
    if not _is_real(value):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    v = _to_float(name, value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return v


def _nonnegative_finite_float(name: str, value) -> float:
    if not _is_real(value):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    v = _to_float(name, value)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return v


def _nonnegative_int(name: str, value) -> int:
    if not _is_int(value):
        raise ValueError(f"{name} must be an integer (not bool), got {value!r}")
    v = int(value)
    if v < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return v


def _optional_positive_int(name: str, value):
    if value is None:
        return None
    if not _is_int(value):
        raise ValueError(f"{name} must be a positive integer or None (not bool), got {value!r}")
    v = int(value)
    if v < 1:
        raise ValueError(f"{name} must be >= 1 or None, got {value!r}")
    return v


def _boolean(name: str, value) -> bool:
    if not isinstance(value, bool):  # strict: reject 1 / "yes" / np.bool_
        raise ValueError(f"{name} must be a bool, got {value!r}")
    return value


def _choice(name: str, value, allowed):
    if value not in allowed:
        raise ValueError(f"{name} must be one of {tuple(allowed)}, got {value!r}")
    return value


def _validated_n_steps(T: float, eps: float) -> int:
    """Number of steps ``T / eps``, requiring it to be a positive integer within
    ``_NSTEPS_RTOL`` so the grid lands exactly on ``T`` (no silent round-down)."""
    ratio = T / eps
    if not math.isfinite(ratio):  # e.g. huge/tiny inputs overflow or underflow the quotient
        raise ValueError(f"T/eps must be finite; got T={T!r}, eps={eps!r} -> T/eps={ratio!r}")
    nearest = round(ratio)
    if nearest < 1 or abs(ratio - nearest) > _NSTEPS_RTOL * max(1.0, abs(ratio)):
        raise ValueError(
            f"T/eps must be a positive integer within relative tol {_NSTEPS_RTOL:g} "
            f"(so the time grid lands exactly on T); got T={T!r}, eps={eps!r} -> "
            f"T/eps={ratio!r}")
    return int(nearest)


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for :class:`~edmtn.driver.solver.EDMSolver`.

    **Frozen and validated at construction.** Every field is checked and normalised in
    ``__post_init__``; the instance is immutable afterwards (use
    ``dataclasses.replace`` to derive a variant).  Illegal input -- a huge Python int,
    ``nan``/``inf``, a ``bool`` posing as an integer, an unknown enum string -- raises
    ``ValueError`` here, at the entry point, rather than deep inside quimb.

    Attributes
    ----------
    eps : float
        Time step; finite and > 0.
    T : float
        Total evolution time; finite and > 0.  ``T / eps`` must be a **positive integer**
        (to a small tolerance) -- it is NOT silently rounded -- so the grid lands exactly
        on ``T``.  ``n_steps`` caches that integer.
    cutoff : float
        Truncation threshold (finite, >= 0).  ``0`` keeps every singular value: with
        ``compress=True`` that is an exact canonicalise + full-SVD recompression, which is
        NOT the same as skipping compression.
    cutoff_mode : str
        quimb-native truncation rule; one of :data:`_CUTOFF_MODES`
        (``abs``, ``rel``, ``sum2``, ``rsum2``, ``sum1``, ``rsum1``).  Default ``'rel'``
        (``s_i / s_max <= cutoff``).  The paper's custom ``rel_ref`` rule -- and the
        reference-index parameter it needed -- are retired; no such field exists.
    max_bond : int, optional
        Hard bond-dimension cap; ``None`` (default) or a positive integer.
    expansion_order : int, optional
        Trotter order (``1`` or ``2``).  ``None`` (**the default**) inherits the model's
        ``time_step_order``; an explicit value overrides it.  Resolved once in the driver
        (see :func:`resolve_config_for_model`) so every layer -- kernel, expander,
        observables, Track-2 assembly and ``SolverResult.expansion_order`` -- uses the
        same value.
    record_rho : bool
        Store ``rho(t)`` at every step.  Strictly a ``bool``.  Note some paths record
        ``rho(t)`` regardless (second-order spin-boson, custom observables).
    precision : str
        ``'f64'`` (default) or ``'mixed'``.  ``'mixed'`` currently casts the Track-1
        contraction tensors to the f32/complex64 path; the declared f64 decomposition
        recast is NOT wired into the solve pipeline -- mixed precision remains
        experimental and unvalidated.
    compress_method, compress_decomp, compress_decomp_q, compress_canon :
        quimb compression controls (see :mod:`edmtn.evolution.quimb_decomp`); all are
        Track-1 only -- ``hpc`` is exact-only and has no 1D-compress sweep.
    preset : str, optional
        ``None`` (default), ``'balanced'`` or ``'robust'``; Track 1 only.  The trigger is
        ``compress_decomp`` alone: while it is still ``'exact'`` (the default), the preset
        sets ``compress_decomp = 'rsvd'`` AND overwrites ``compress_decomp_q`` with its own
        value -- an explicitly passed ``q`` included.  An explicit ``compress_decomp='rsvd'``
        prevents the preset from changing either compression field.  An unknown name is
        rejected on every backend.
    sub_baths : int, optional
        Separable only: fold just the first ``L`` sub-baths **in the model's stored
        coupling order** (strongest-first only for the sorted named profiles).  Validated
        here as ``None`` or a positive integer, then re-checked against the model's ``K``
        (``1 <= L <= K``) once ``K`` is known -- never silently clamped or truncated.
    backend : str
        ``'cpu'`` (**default**), ``'numpy'``, ``'gpu'``, ``'cupy'`` (Track 1) or ``'hpc'``
        (Track 2, cuQuantum 2D contraction).  There is no ``'auto'``.
    pathfinder : str
        ``hpc`` only: ``'cuquantum'`` (default) or ``'cotengra'``.  Distributed multi-GPU
        requires ``'cuquantum'``.
    time_windows : int, optional
        **Reserved; must be ``None``.** Manual time-window blocking is wired but not
        implemented -- any non-``None`` value raises ``NotImplementedError`` here.
    """

    eps: float
    T: float
    cutoff: float = 1e-8
    cutoff_mode: str = "rel"       # quimb-native cutoff (rel: faithful to the retired rel_ref)
    max_bond: int | None = None
    expansion_order: int | None = None  # None -> inherit the model's time_step_order
    record_rho: bool = False
    compress_method: str = "zipup"        # 'zipup'|'dm'|'direct' (quimb 1D-compress; N/A under backend='hpc')
    compress_decomp: str = "exact"        # cpu/gpu: 'exact'|'rsvd' (N/A under 'hpc': Track 2 is exact-only, no truncation)
    compress_decomp_q: int = 2            # rsvd power iterations (2=cold, 0=single-pass; N/A under 'hpc')
    compress_canon: str = "quimb"         # 'quimb'|'householder'|'cholqr' (canon QR; N/A under 'hpc')
    preset: str | None = None  # None|'balanced'|'robust' (cpu/gpu only; see docs/guides/recommended-config.md)
    sub_baths: int | None = None  # separable only: fold the first L sub-baths in the model's stored coupling order (Fig. 6); None = all K
    backend: str = "cpu"   # 'cpu'|'gpu' -> numpy/cupy (Track 1); 'hpc' -> cuQuantum 2D contraction
    precision: str = "f64"  # 'f64' | 'mixed' (f32 contraction only; f64 decompose recast NOT wired in -- experimental)
    # -- backend='hpc' only; ignored otherwise --
    pathfinder: str = "cuquantum"   # 'cuquantum' (default, cuTensorNet owns path) | 'cotengra'
    time_windows: int | None = None  # None = one-shot whole-spacetime; int = manual window blocking
    # NB: hpc has no GPU-count knob -- it uses exactly the GPUs/ranks it was LAUNCHED
    # across (cuTensorNet is one MPI rank per physical GPU, and needs a CUDA-aware MPI
    # runtime). Launch it with your own workflow; see cluster/ for the current test recipes
    # and their status (single-GPU verified; 4-GPU currently blocked by a site MPI issue).
    # edmtn itself does not submit/srun/ssh -- that is the user's job.

    # derived + cached at construction (frozen: set via object.__setattr__ below)
    _n_steps: int = field(init=False, repr=False, compare=False, default=0)

    def __post_init__(self):
        # frozen dataclass: bypass immutability to store validated / normalised values
        def _set(name, value):
            object.__setattr__(self, name, value)

        # -- time grid (validated identically for both tracks) --
        _set("eps", _positive_finite_float("eps", self.eps))
        _set("T", _positive_finite_float("T", self.T))
        _set("_n_steps", _validated_n_steps(self.T, self.eps))

        # -- truncation / compression knobs --
        _set("cutoff", _nonnegative_finite_float("cutoff", self.cutoff))
        _set("cutoff_mode", _choice("cutoff_mode", self.cutoff_mode, _CUTOFF_MODES))
        _set("max_bond", _optional_positive_int("max_bond", self.max_bond))
        if self.expansion_order is not None:  # None -> inherit model.time_step_order at solve time
            if not _is_int(self.expansion_order) or int(self.expansion_order) not in (1, 2):
                raise ValueError(
                    f"expansion_order must be None or the integer 1 or 2, got {self.expansion_order!r}")
            _set("expansion_order", int(self.expansion_order))
        _set("record_rho", _boolean("record_rho", self.record_rho))
        _set("compress_method", _choice("compress_method", self.compress_method, _COMPRESS_METHODS))
        _set("compress_decomp", _choice("compress_decomp", self.compress_decomp, _COMPRESS_DECOMPS))
        _set("compress_decomp_q", _nonnegative_int("compress_decomp_q", self.compress_decomp_q))
        _set("compress_canon", _choice("compress_canon", self.compress_canon, _COMPRESS_CANONS))
        _set("sub_baths", _optional_positive_int("sub_baths", self.sub_baths))

        # -- backend / precision / hpc-only knobs --
        _set("backend", _choice("backend", self.backend, _BACKENDS))
        _set("precision", _choice("precision", self.precision, _PRECISIONS))
        _set("pathfinder", _choice("pathfinder", self.pathfinder, _PATHFINDERS))
        if self.time_windows is not None:  # concept is valid but not implemented -> NotImplementedError
            raise NotImplementedError(
                "manual time-window blocking (time_windows) is wired but not yet "
                "implemented; the hpc track ships one-shot whole-spacetime only. "
                "Use time_windows=None.")

        # -- presets: validate legality for ALL backends; APPLY only on Track 1 --
        # (Track-1 rSVD recipes don't apply to the hpc track, but an unknown preset
        #  name is still rejected there rather than silently ignored.)
        if self.preset is not None and self.preset not in _PRESETS:
            raise ValueError(
                f"unknown preset {self.preset!r}; choose from {sorted(_PRESETS)} or None")
        if self.preset is not None and self.backend != "hpc":
            spec = _PRESETS[self.preset]
            if self.compress_decomp == "exact":  # only fill if left at the default
                _set("compress_decomp", spec["compress_decomp"])
                _set("compress_decomp_q", spec["compress_decomp_q"])

        # After preset resolution (a preset can move decomp onto 'rsvd'), Track 1 only:
        # 'hpc' does not consume the compression fields at all, and this guard must not
        # silently tighten that contract.
        if self.backend != "hpc":
            validate_compression_combination(
                self.compress_method, self.compress_decomp, self.compress_canon)

    @property
    def n_steps(self) -> int:
        return self._n_steps


# Recommended presets (docs/guides/recommended-config.md): both use quimb rSVD, differing
# only in power iterations -- balanced = single-pass (q=0, fastest), robust = cold
# (q=2, exact-baseline accuracy).  The silent guard falls back to full SVD either way.
_PRESETS: dict = {
    "balanced": {"compress_decomp": "rsvd", "compress_decomp_q": 0},
    "robust": {"compress_decomp": "rsvd", "compress_decomp_q": 2},
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


def resolve_expansion_order(model, config: SolverConfig) -> int:
    """The effective Trotter order used by every layer: the model's ``time_step_order``
    unless the config overrides it via an explicit ``expansion_order``.  Validated to be
    the integer ``1`` or ``2`` here (a model with a bad ``time_step_order`` fails loudly,
    rather than being silently treated as first order downstream)."""
    value = model.time_step_order if config.expansion_order is None else config.expansion_order
    if not _is_int(value) or int(value) not in (1, 2):
        raise ValueError(
            f"resolved expansion order must be the integer 1 or 2, got {value!r} "
            f"(config.expansion_order={config.expansion_order!r}, "
            f"model.time_step_order={getattr(model, 'time_step_order', None)!r})")
    return int(value)


def resolve_config_for_model(model, config: SolverConfig) -> SolverConfig:
    """Return a config whose ``expansion_order`` is the concrete resolved order, so every
    layer (driver, kernel, expander, extractor) reads a single field.  The original frozen
    config is left untouched; an explicit ``expansion_order`` is returned unchanged.  Both
    :class:`~edmtn.driver.solver.EDMSolver` and :func:`build_pipeline` funnel through this."""
    if config.expansion_order is not None:
        return config
    return replace(config, expansion_order=resolve_expansion_order(model, config))


def build_pipeline(model, config: SolverConfig):
    """Construct ``(kernel_engine, evolution_engine)`` for ``model``.

    Resolves the effective order first (so a default ``expansion_order=None`` inherits the
    model's ``time_step_order``) -- this public entry point works standalone, not only via
    :class:`~edmtn.driver.solver.EDMSolver`.
    """
    config = resolve_config_for_model(model, config)
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
    evolution = SingleBathEvolution(
        expander=_make_expander(config.expansion_order),
        compress_method=config.compress_method,
        compress_decomp=config.compress_decomp,
        compress_decomp_q=config.compress_decomp_q,
        compress_canon=config.compress_canon,
    )
    return kernel_engine, evolution


def _build_separable(model, config: SolverConfig):
    kernel_engine = SeparableKernelEngine.from_model(model, T=config.T, eps=config.eps)
    evolution = SeparableBathEvolution(
        expander=_make_expander(config.expansion_order),
        compress_method=config.compress_method,
        compress_decomp=config.compress_decomp,
        compress_decomp_q=config.compress_decomp_q,
        compress_canon=config.compress_canon,
    )
    return kernel_engine, evolution


register_pipeline("gaussian", _build_gaussian)
register_pipeline("separable", _build_separable)
