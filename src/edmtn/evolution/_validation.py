"""Direct-call parameter validation for the Layer-5 evolution engines (internal).

The normal entry to an EDM evolution is the driver (Layer 7), whose ``SolverConfig``
already validates every knob.  But :meth:`SingleBathEvolution.run` and
:meth:`SeparableBathEvolution.run` are *also* callable directly -- reference checks,
tests, notebooks -- bypassing the driver, so they must guard their own arguments.  An
unguarded illegal value otherwise surfaces as a ``ZeroDivisionError`` (``record_every=0``),
a ``range()`` ``TypeError`` (``n_steps=2.5``), an all-``t=0`` trajectory (``eps=0``), a
non-finite time grid (``eps=1e308``) or a deep quimb error (bad ``cutoff_mode``) -- never a
clear ``ValueError`` at the entry point.

These validators mirror the driver's ``auto_config`` semantics (``numbers.Integral`` /
``numbers.Real`` excluding ``bool``; ``float()`` overflow -> ``ValueError``; NumPy scalars
normalised to Python ``int``/``float``) so a direct call and a driver call reject the same
bad value identically.  They are an *internal* evolution concern -- ``eps`` / ``cutoff`` /
``max_bond`` are evolution knobs, not part of the model API -- and are deliberately NOT
exported from :mod:`edmtn.models`.

Signatures follow the driver's ``(name, value)`` order so the project has one convention,
not two opposite ones.  :data:`CUTOFF_MODES` is the single source of truth for the allowed
``cutoff_mode`` strings; the driver imports it here (see
:mod:`edmtn.driver.auto_config`) so the two contracts cannot drift.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping

import numpy as np

#: allowed ``cutoff_mode`` strings (quimb-native cutoff modes); the driver's
#: ``SolverConfig`` validation imports this so both entry points share one set.
CUTOFF_MODES = ("abs", "rel", "sum2", "rsum2", "sum1", "rsum1")


def _is_int(value) -> bool:
    """True for a genuine integer (Python ``int`` or NumPy integer), excluding ``bool``."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_real(value) -> bool:
    """True for a real number usable as a float, excluding ``bool``."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _as_float(name: str, value) -> float:
    """``float(value)`` with ``TypeError`` / ``ValueError`` / ``OverflowError`` -> ``ValueError``.

    A huge Python ``int`` such as ``10**400`` is a real number but overflows ``float()``; the
    honest response is a project ``ValueError``, not a leaked ``OverflowError`` (cf. P0-2).
    """
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite real number, got {value!r}") from None


def validate_positive_finite_float(name: str, value) -> float:
    """Return ``value`` as a finite Python ``float`` > 0; raise ``ValueError`` otherwise."""
    if not _is_real(value):
        raise ValueError(f"{name} must be a real number > 0, got {value!r}")
    v = _as_float(name, value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return v


def validate_nonnegative_finite_float(name: str, value) -> float:
    """Return ``value`` as a finite Python ``float`` >= 0; raise ``ValueError`` otherwise."""
    if not _is_real(value):
        raise ValueError(f"{name} must be a real number >= 0, got {value!r}")
    v = _as_float(name, value)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return v


def validate_positive_int(name: str, value) -> int:
    """Return ``value`` as a Python ``int`` >= 1 (rejecting ``bool``); raise ``ValueError``."""
    if not _is_int(value):
        raise ValueError(f"{name} must be an integer (not bool), got {value!r}")
    v = int(value)
    if v < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}")
    return v


def validate_optional_positive_int(name: str, value):
    """Return ``None`` or a Python ``int`` >= 1 (rejecting ``bool``); raise ``ValueError``."""
    if value is None:
        return None
    if not _is_int(value):
        raise ValueError(f"{name} must be a positive integer or None (not bool), got {value!r}")
    v = int(value)
    if v < 1:
        raise ValueError(f"{name} must be >= 1 or None, got {value!r}")
    return v


def validate_bool(name: str, value) -> bool:
    """Return ``value`` if it is a genuine ``bool`` (rejecting ``1`` / ``'yes'`` / ``np.bool_``)."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {value!r}")
    return value


def validate_cutoff_mode(name: str, value) -> str:
    """Return ``value`` if it is one of :data:`CUTOFF_MODES`; raise ``ValueError`` otherwise."""
    if value not in CUTOFF_MODES:
        raise ValueError(f"{name} must be one of {CUTOFF_MODES}, got {value!r}")
    return value


def validate_expansion_order(name: str, value) -> int:
    """Return the expansion order as a Python ``int`` in ``{1, 2}``; raise ``ValueError``.

    Because ``True == 1`` and ``1.0 == 1``, a bool or float order otherwise slips through a
    ``not in (1, 2)`` membership or an ``==`` comparison and only surfaces much later as a
    deep ``range()`` / lag-map error.  Enforce a genuine non-bool ``numbers.Integral`` in
    ``{1, 2}`` -- the same contract the driver's ``expansion_order`` uses.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or int(value) not in (1, 2)
    ):
        raise ValueError(f"{name} must be the integer 1 or 2 (not bool), got {value!r}")
    return int(value)


def validate_final_time(eps: float, n_steps: int) -> float:
    """Require the final grid time ``n_steps * eps`` to be finite; return it.

    ``eps`` finite-and-positive plus ``n_steps`` a positive integer is not enough: e.g.
    ``eps=1e308, n_steps=2`` gives a non-finite ``n * eps`` that would then poison every
    interaction-picture time and cumulant.  Call this *after* ``eps`` and ``n_steps`` are
    validated/normalised.  No arbitrary numerical ceiling is imposed -- only that the time
    the evolution actually uses is a finite number.
    """
    try:
        t_final = n_steps * eps
    except OverflowError:
        raise ValueError(
            f"final time grid n_steps*eps overflowed (eps={eps!r}, n_steps={n_steps!r})"
        ) from None
    if not math.isfinite(t_final):
        raise ValueError(
            f"final time grid must be finite: eps={eps!r} * n_steps={n_steps!r} = {t_final!r}"
        )
    return t_final


# -- structural model/kernel compatibility (NOT full provenance) -----------------------------


def _validate_kernel_dims(model, kernel_engine) -> int:
    """Common check: ``kernel_engine.d_phys == 2 * n_channels + 1`` for the model's channels.

    Returns the number of coupling channels.  This proves *structural* ``d_phys``
    compatibility ONLY -- it does NOT prove the kernel was built from THIS model's
    parameters (a kernel with the same ``d_phys`` built from different couplings still
    passes).  Callers must not present it as full provenance verification.
    """
    n_ch = len(model.coupling_operators())
    expected = 2 * n_ch + 1
    d_phys = getattr(kernel_engine, "d_phys", None)
    if d_phys != expected:
        raise ValueError(
            f"kernel/model mismatch: kernel d_phys={d_phys!r} but the model has {n_ch} "
            f"coupling channel(s) (expected d_phys = 2*{n_ch}+1 = {expected})"
        )
    return n_ch


def validate_single_bath_kernel(model, kernel_engine, evolution_order: int) -> None:
    """Structural single-bath kernel checks (dimension + interface + order).

    Proves structural compatibility only, NOT that the kernel encodes THIS model's
    parameters.  ``evolution_order`` must match the kernel's order in BOTH directions: the
    second-order Gaussian kernel's lag map differs from first order, so neither a
    second-order evolution on a first-order kernel nor a first-order evolution on a
    second-order kernel is valid.
    """
    _validate_kernel_dims(model, kernel_engine)
    if not callable(getattr(kernel_engine, "get_kernel_mpo", None)):
        raise ValueError("single-bath kernel_engine must provide a callable get_kernel_mpo()")
    # strict on the kernel's own order too: True / 1.0 must not pass as 1 (the lag map differs)
    kernel_order = validate_expansion_order("kernel order", getattr(kernel_engine, "order", None))
    if kernel_order != evolution_order:
        raise ValueError(
            f"evolution order {evolution_order} needs a matching order-{evolution_order} "
            f"kernel engine, got kernel order {kernel_order!r} (the lag map differs by order)"
        )


def validate_separable_bath_kernel(model, kernel_engine) -> int:
    """Structural separable-bath kernel checks (dimension + sub-bath count + interface).

    Returns the model's sub-bath count ``K``.  Proves structural compatibility only, NOT
    that the kernel encodes THIS model's couplings: it catches a kernel built from a
    *different* ``K`` (which would silently fold the wrong number of sub-baths) and a
    ``d_phys`` mismatch, but two same-``K`` / same-``d_phys`` models are not distinguished.
    """
    _validate_kernel_dims(model, kernel_engine)
    model_K = model.bath_params().K
    kernel_K = getattr(kernel_engine, "K", None)
    if kernel_K != model_K:
        raise ValueError(
            f"kernel/model mismatch: kernel K={kernel_K!r} but the model bath has "
            f"K={model_K!r} sub-baths"
        )
    if not callable(getattr(kernel_engine, "for_sub_bath", None)):
        raise ValueError("separable kernel_engine must provide a callable for_sub_bath()")
    return model_K


def validate_compression_combination(method, decomp, canon) -> None:
    """Reject compression-knob combinations quimb's method path cannot execute.

    quimb's ``dm`` (density-matrix) 1D-compress reaches ``tensor_split`` with PSD-split
    keywords and forwards ``canonize_opts`` into the split driver, so only the exact
    eigh driver with the default quimb canonicalisation accepts its call signature; any
    other combination leaks a deep quimb ``TypeError``.  Shared by ``SolverConfig``
    (Track 1), both direct ``run()`` entry points and ``QuimbEDM.compress()``: the
    driver and direct evolution entry points reject the combination before tensor
    construction; ``QuimbEDM.compress()`` rejects it before any compression work or
    the ``n <= 1`` early return.
    """
    if method in ("dm", "dm_tracking") and (decomp != "exact" or canon != "quimb"):
        raise ValueError(
            f"compress_method={method!r} supports only compress_decomp='exact' with "
            f"compress_canon='quimb'; got compress_decomp={decomp!r}, "
            f"compress_canon={canon!r}")


def validate_time_read_combination(*, record_time_reads, compress, compress_method,
                                   requires_tracking: bool = True,
                                   supports_tracking: bool | None = None) -> None:
    """Tie ``record_time_reads`` and ``compress_method='dm_tracking'`` together, both ways.

    ``dm_tracking`` is not a quimb method: it is the in-repo two-sweep compression whose
    per-bond basis changes are exposed so the causal-prefix terminators can be transported
    through them (:mod:`edmtn.evolution.prefix_reads`).  That gives two obligations, and
    both are checked here -- at the entry point, before any tensor is built:

    * time reads with compression **need** it, because no other method hands the basis
      changes back and silently substituting one would change the user's compression;
    * it is **only** useful for that, so requesting it on its own is refused rather than
      quietly grown into a fourth general-purpose compressor with no test coverage of its
      own.

    ``compress=False`` is exempt in the first direction: no compression runs, so no bond
    basis changes and only the fold-time ``kron`` is needed.  The other configuration
    checks still apply there -- "any method" means any *otherwise valid* one.

    ``requires_tracking`` selects which obligations apply, and exists because the two are
    **not** equally universal.  ``record_time_reads`` is a generic "populate ``rho(t)``"
    request that every pipeline honours its own way -- single-bath Track 1 just turns on its
    per-step recorder, Track 2 already produces the history -- and only the separable engines
    need the prefix terminators.  So a config-level check that does not know the bath type
    must pass ``requires_tracking=False`` and enforce only the second obligation, which is
    universal; the separable engine passes ``True`` and enforces both.  Enforcing the first
    everywhere would refuse a perfectly valid single-bath request.
    """
    tracking = compress_method == "dm_tracking"
    if tracking and supports_tracking is False:
        raise ValueError(
            "compress_method='dm_tracking' is only implemented by the separable-bath "
            "engine, which is the one that carries prefix terminators; this pipeline "
            "satisfies record_time_reads with its own per-step recorder.  Use "
            "'zipup', 'dm' or 'direct'.")
    if requires_tracking and record_time_reads and compress and not tracking:
        raise ValueError(
            "record_time_reads=True with compress=True needs "
            f"compress_method='dm_tracking', got {compress_method!r}: the prefix "
            "terminators have to be carried through every bond-basis change, and no other "
            "compression method reports them.  Either select 'dm_tracking' or run with "
            "compress=False.")
    if tracking and not record_time_reads:
        raise ValueError(
            "compress_method='dm_tracking' is only for record_time_reads=True; it exists to "
            "transport the causal-prefix terminators and is not a general-purpose "
            "compressor.  Use 'dm' for an ordinary density-matrix compression.")


def validate_tangent_closings(name: str, value, K: int):
    """Normalise the per-channel bath-side closing vectors, or return ``None``.

    ``value`` maps a channel name to a ``(K, lateral)`` array of newest-site lateral
    closing vectors, one row per sub-bath -- row ``k`` belongs to sub-bath ``k`` in the
    model's own stored order, so a shorter or longer first axis is a mismatch, never a
    silent broadcast.  The row length is *not* checked here: the lateral dimension belongs
    to the kernel, which validates it when the vector is contracted.

    An empty mapping raises rather than normalising to "no channels": it would otherwise be
    an argument that silently does nothing.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{name} must be a mapping of channel name -> ({K}, lateral) closing array or "
            f"None, got {type(value).__name__}")
    if not value:
        raise ValueError(
            f"{name} is empty; pass None to request no tangent channel rather than a "
            f"mapping that computes nothing")
    out = {}
    for key, rows in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings, got {key!r}")
        try:
            arr = np.asarray(rows, dtype=np.complex128)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name}[{key!r}] must be a numeric ({K}, lateral) array") from exc
        if arr.ndim != 2 or arr.shape[0] != K:
            raise ValueError(
                f"{name}[{key!r}] must have shape ({K}, lateral) -- one closing vector per "
                f"sub-bath -- got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name}[{key!r}] must be finite")
        out[key] = arr
    return out


def validate_tangent_time_read_combination(*, tangent_closings, record_time_reads,
                                           compress) -> None:
    """Reject compressed tangent channels together with ``record_time_reads``.

    The reason is structural, not a policy: with compression on, ``record_time_reads``
    forces ``compress_method='dm_tracking'``, and ``QuimbEDM.compress`` requires
    ``dm_tracking`` and the ``PrefixTerminators`` to appear together in both directions.
    The tangent chains carry no terminators, so they cannot take that path; letting them
    compress with ``'dm'`` while the value channel used ``'dm_tracking'`` would run two
    different numerical paths under one reported ``compression_method_used``.

    With ``compress=False`` nothing compresses, so the combination is allowed.
    """
    if tangent_closings is not None and record_time_reads and compress:
        raise ValueError(
            "tangent_closings with record_time_reads=True and compress=True is not "
            "supported: time reads force compress_method='dm_tracking', which exists only "
            "to transport the causal-prefix terminators, and the tangent chains have none. "
            "Run the time reads and the collective-spin moments as two solves, or set "
            "compress=False.")
