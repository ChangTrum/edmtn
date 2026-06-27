"""Guarded randomized-SVD split driver + canonicalisation selector for the quimb
compression path (re-platform step toward retiring the hand-rolled ``native`` path).

quimb's 1D-compress methods (``direct``/``zipup``) take their per-bond decomposition
from ``compress_opts['method']`` and their canonicalisation QR from
``canonize_opts['method']``.  This module exposes two EDM-facing knobs over that:

* **decomposition** -- ``'exact'`` (LAPACK/cuSOLVER full SVD, the default) or
  ``'rsvd'`` (randomized SVD).  rSVD avoids the full SVD where it dominates cost
  (the ``direct`` sweep / large GPU bonds).  The power-iteration count ``q`` is the
  knob: ``q=2`` is *cold* rSVD (the robust default, == quimb's own), ``q=0`` is
  *single-pass*.  A **silent resolution guard** runs automatically: rSVD uses quimb's
  adaptive (``adapt+block``) mode so the rank grows until the cutoff resolves inside
  the computed spectrum, and any failure / under-resolution / non-NumPy backend falls
  back to the exact full SVD -- so the result is never less reliable than full SVD.
* **canonicalisation** -- ``'quimb'`` (quimb's default QR), ``'householder'``
  (``'qr'``), or ``'cholqr'`` (``'qr:cholesky'``, single-pass Cholesky QR).

The randomized engine is quimb's own (``quimb.linalg.rand_linalg.rsvd``), so this is
the hybrid the plan's item 0.1 anticipated: our rank/guard policy, quimb's sketch+SVD.
"""

from __future__ import annotations

_RSVD_STATE = {"q": 2}  # power iterations for the 'edm_rsvd' driver (set per solve)
_REGISTERED = {"done": False}

_CANON_METHOD = {"quimb": None, "householder": "qr", "cholqr": "qr:cholesky"}


def set_rsvd_q(q: int) -> None:
    """Set the power-iteration count used by the ``'edm_rsvd'`` split driver."""
    _RSVD_STATE["q"] = int(q)


def _register_edm_rsvd() -> None:
    """Register the guarded randomized-SVD driver under the name ``'edm_rsvd'``."""
    if _REGISTERED["done"]:
        return
    import numpy as np  # noqa: PLC0415
    from quimb.tensor import decomp  # noqa: PLC0415
    from quimb.linalg import rand_linalg  # noqa: PLC0415

    exact = decomp._SPLIT_FNS["svd"]  # full-SVD driver, the guard's fallback

    @decomp.register_split_driver("edm_rsvd", sparse=True)
    def edm_rsvd(x, cutoff=0.0, cutoff_mode=decomp.cutoff_mode_rsum2,
                 max_bond=-1, absorb=decomp.get_Usq_sqVH, renorm=0):
        # Only NumPy gets q-controlled randomized SVD; any other backend (CuPy) or
        # anomaly silently uses the exact full SVD -- guaranteed reliable.
        if not isinstance(x, np.ndarray):
            return exact(x, cutoff, cutoff_mode, max_bond, absorb, renorm)
        q = _RSVD_STATE["q"]
        try:
            if max_bond > 0:
                if cutoff > 0.0:
                    U, s, VH = rand_linalg.rsvd(x, cutoff, k_max=max_bond, q=q)
                else:
                    U, s, VH = rand_linalg.rsvd(x, max_bond, q=q)
            else:
                U, s, VH = rand_linalg.rsvd(x, cutoff, q=q)
            # guard: a clean randomized result has a finite, non-empty spectrum and,
            # when a cutoff is active, resolved it *inside* the kept rank (the sketch
            # was wide enough).  Hitting the bond cap with an active cutoff, or any
            # nan/empty, means it may be under-resolved -> verify with exact SVD.
            n = int(getattr(s, "shape", [0])[0])
            capped = max_bond > 0 and n >= max_bond and cutoff > 0.0
            if n == 0 or not bool(np.all(np.isfinite(s))) or capped:
                return exact(x, cutoff, cutoff_mode, max_bond, absorb, renorm)
        except Exception:
            return exact(x, cutoff, cutoff_mode, max_bond, absorb, renorm)
        U, s, VH, _ = decomp._trim_and_renorm_svd_result_numba(
            U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm)
        return U, s, VH

    _REGISTERED["done"] = True


def compress_opts_for(decomp_mode: str, q: int):
    """Return the ``compress_opts`` dict selecting the per-bond decomposition."""
    if decomp_mode == "exact":
        return {}  # quimb's default (full SVD / eigh for dm)
    if decomp_mode == "rsvd":
        _register_edm_rsvd()
        set_rsvd_q(q)
        return {"method": "edm_rsvd"}
    raise ValueError(f"unknown compress_decomp {decomp_mode!r}; choose 'exact' or 'rsvd'")


def canonize_opts_for(canon_mode: str):
    """Return the ``canonize_opts`` dict selecting the canonicalisation QR."""
    if canon_mode not in _CANON_METHOD:
        raise ValueError(
            f"unknown compress_canon {canon_mode!r}; choose {sorted(_CANON_METHOD)}")
    method = _CANON_METHOD[canon_mode]
    return {} if method is None else {"method": method}
