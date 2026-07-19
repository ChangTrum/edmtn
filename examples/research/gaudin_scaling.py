"""Gaudin recompute (stage G) -- math diagnostic + physical observable from ONE streaming fold.

Clean re-derivation of the separable-bath scaling law for the standard isotropic Gaudin
model: fold sub-baths strongest-first (sorted named profiles) and, per fold ``L -> L+1``,
record with ``x = g_{L+1}^2 / gbar_L^2``:

* **math diagnostic** -- the per-bond subspace increment (``eta_rms``/``eta_max``, chordal
  distance, ``n_new``), same machinery as ``coupling_distributions.py``;
* **physical observable** -- the central-spin polarization ``P_L(t) = <S_z(t)>`` (+z initial
  state, coupling channel 3, Eq. F2/F3 sweep) on the solver's PUBLIC axis ``eps..T``
  (mirrors ``EDMSolver._solve_separable``: drop the t=0 point, append ``Tr[S_z rho(T)]``),
  and its per-fold increment ``dP_L = RMS_t |P_{L+1}(t) - P_L(t)|``;
* **derived diagnostic** -- the *mean-polarization spectrum* ``S_L(w) = eps^2 |rfft(P_L -
  mean)|^2``.  This is the power spectrum of the deterministic relaxation trace, i.e. a
  functional of the SAME trajectory -- it is NOT a spin-noise PSD and NOT an independent
  observable channel (a true noise spectrum needs the two-time connected correlation,
  deferred pending a derivation).

Execution model
---------------
The sweep is **streaming**: one fold pass per configuration, at most two canonical host
snapshots resident (the raw host copy is released right after extraction), each snapshot
left-canonicalized exactly once.  Resume is **configuration-level only**: a completed
configuration is skipped, an interrupted one restarts from ``L = 1`` (mid-configuration
resume is NOT implemented).  Per ``L``, the science data (trajectory, spectrum, bond
profile, per-bond diagnostics) is atomically written to a shard under ``<name>.shards/``
so an interrupted run keeps its sampled data.  Every actual (re)run -- ``--force``
included -- first ARCHIVES all artifacts of any previous run of the same name (JSON, NPZ,
progress, shards, run marker) under one shared timestamped ``.archived-`` suffix:
interrupted shards are preserved, and a stale complete result can never be mistaken for
the new run's output.  A ``<name>.running.json`` marker exists while a run is in flight;
shards are consolidated into the final NPZ and removed only after the NPZ and JSON are
written.  Skipping a completed result first verifies the requested protocol parameters,
the code git commit and the NPZ's existence against the stored metadata -- any mismatch
is an error (exit 4), never a silent reuse.

Fits and gates
--------------
Power law ``y ~ c x^alpha`` (full range + smallest-x tail).  The float64 roundoff floor
mask (``--eta-floor``, default 1e-6) applies to the MATH eta fits only -- the physical /
spectrum increments have no self-comparison-established floor, so they are fitted unmasked
and their minimum value is reported.

``--anchor`` runs evaluate a PRE-REGISTERED acceptance contract and set the exit code:

* **fit gate** (from the archived K=49 envelope alpha_tail in [0.98, 1.01] over 10
  configs, widened by +-0.01): tail alpha in [0.97, 1.02], R2_tail >= 0.95, >= 8 tail
  points, on math eta_rms AND physical dP;
* **quality gate**: no ``cap_hit`` at any ``L`` (a hit cap means rank-limited truncation,
  not a natural bond); no non-finite value in trajectories, spectra, diagnostics, trace
  deviations or discarded weights; and, under ``compress_decomp='exact'``, a NUMERIC
  discarded weight on every fold -- a ``None`` there means the P1-15 metric chain failed
  (``rsvd`` legitimately reports ``None``, but G1 does not use rsvd).  The max discarded
  weight and max trace deviation are recorded; trace deviation is REPORTED only (no
  pre-registered threshold exists, so none is invented).

Exit codes: 0 = OK / gates passed; 3 = anchor gates FAILED (also when skipping a
previously failed anchor result); 4 = reuse/consistency mismatch.  The sbatch runs under
``set -e``, so a failed anchor stops the queue ("out of gate -> stop and report").

Usage
-----
    python examples/research/gaudin_scaling.py --smoke                    # fast validation
    python examples/research/gaudin_scaling.py --check                    # cross-validation
    python examples/research/gaudin_scaling.py --selftest                 # exit-code paths
    python examples/research/gaudin_scaling.py --K 49 --T 3 --eps 0.1 \\
        --coupling linear --anchor --device gpu --name g1_linear          # G1 anchor arm
    python examples/research/gaudin_scaling.py --pool g1_random_s0,... --name g1_random_pooled
    python examples/research/gaudin_scaling.py --compare g1_linear,g1_linear_fine \\
        --name g1_eps_compare                                             # report-only
    python examples/research/gaudin_scaling.py --replot g1_linear         # figures from disk
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from edmtn.driver.auto_config import SolverConfig
from edmtn.evolution.quimb_edm import QuimbEDM
from edmtn.evolution.separable_bath import SeparableBathEvolution
from edmtn.expansion.first_order import FirstOrderExpander
from edmtn.expansion.second_order import SecondOrderExpander
from edmtn.kernels.separable_mpo import SeparableKernelEngine
from edmtn.models import GaudinModel
from edmtn.observables.extractor import ObservableExtractor

_DIR_OUT = Path(__file__).resolve().parent / "data" / "gaudin_recompute"

CHANNEL = 3       # coupling channel 3 = S_z (1-based contract)
_ITEMSIZE = 16    # complex128

EXIT_OK = 0
EXIT_GATE_FAIL = 3       # pre-registered anchor gates failed (fresh run or skip-reload)
EXIT_REUSE_MISMATCH = 4  # completed result exists but protocol/commit/NPZ inconsistent

# Pre-registered acceptance gates (--anchor): archived envelope alpha_tail in [0.98, 1.01]
# (10 configs, K=49 / T=3 / eps=0.1 / cutoff=1e-8 / max_bond=500) widened by +-0.01 slack.
_GATE_ALPHA_BAND = (0.97, 1.02)
_GATE_R2_MIN = 0.95
_GATE_N_MIN = 8

# Protocol fields that must match for a completed result to be reused (skip); pool
# members must additionally agree on EVERYTHING except ``seed`` (see run_pool).
_PROTOCOL_KEYS = ("g", "K", "coupling", "beta", "seed", "rho", "L0", "T", "eps", "order",
                  "cutoff", "cutoff_mode", "max_bond", "method", "decomp", "decomp_q",
                  "canon", "device", "tail_frac", "eta_floor")


def model_for(kind, g, K, *, beta=0.1, seed=0, rho=0.8):
    """Build a GaudinModel with the named coupling profile (first-class model option)."""
    params = {}
    if kind == "exp":
        params["beta"] = beta
    elif kind == "random":
        params["seed"] = seed
    elif kind == "ou":
        params["rho"], params["seed"] = rho, seed
    return GaudinModel(g=g, K=K, coupling=kind, coupling_params=params)


# --------------------------------------------------------------------------
# host transfer + per-bond subspace diagnostics
# (same maths as examples/research/coupling_distributions.py; the transition
#  analysis takes ALREADY left-canonical snapshots so each snapshot is
#  canonicalized exactly once per sweep instead of twice per transition)
# --------------------------------------------------------------------------

def _to_numpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _edm_to_numpy(edm):
    """Bring a (possibly CuPy-backed) EDMMPS to host NumPy."""
    from edmtn.evolution.mps_utils import EDMMPS  # noqa: PLC0415
    return EDMMPS(tensors=[_to_numpy(t) for t in edm.tensors], d=edm.d,
                  d_phys=edm.d_phys, rho0_vec=_to_numpy(edm.rho0_vec))


def left_canonicalize(mps):
    """Left-canonical copy (sites 0..n-2 isometric over ``(phi, chi_left)``)."""
    cm = mps.copy()
    Ts = cm.tensors
    for p in range(len(Ts) - 1):
        phi, l, r = Ts[p].shape
        Q, R = np.linalg.qr(Ts[p].reshape(phi * l, r))
        Ts[p] = Q.reshape(phi, l, Q.shape[1])
        Ts[p + 1] = np.einsum("kr,prs->pks", R, Ts[p + 1], optimize=True)
    return cm


def _cross_left_overlaps(A, B):
    """``E_tau = Q_A(tau)^H Q_B(tau)`` at every internal bond (left-canonical A, B)."""
    la0 = A.tensors[0].shape[1]
    E = np.eye(la0, dtype=np.complex128)
    Es = []
    for p in range(A.num_sites):
        T1 = np.einsum("pla,lm->pam", np.conj(A.tensors[p]), E, optimize=True)
        E = np.einsum("pam,pmc->ac", T1, B.tensors[p], optimize=True)
        if p < A.num_sites - 1:
            Es.append(E)
    return Es


def _right_bond_density(B):
    """Right-environment bond density matrices of left-canonical ``B`` (``{tau: rho}``)."""
    R = np.eye(B.tensors[-1].shape[2], dtype=np.complex128)
    out = {}
    for p in range(B.num_sites - 1, -1, -1):
        Bp = B.tensors[p]
        T = np.einsum("plr,rs->pls", Bp, R, optimize=True)
        R = np.einsum("pls,pms->lm", T, np.conj(Bp), optimize=True)
        if p >= 1:
            out[p] = R
    return out


def analyse_transition_canon(A, B, xi):
    """Per-bond subspace diagnostics for the fold ``L -> L+1``.

    ``A``/``B`` must be **already left-canonical** snapshots (L and L+1); the maths is
    identical to ``coupling_distributions.analyse_transition``, which canonicalizes
    internally -- verified equal by ``--check``.
    """
    if A.num_sites != B.num_sites:
        raise ValueError("L and L+1 MPS have different site counts")
    Es, rhos = _cross_left_overlaps(A, B), _right_bond_density(B)
    deltas = {"xi": xi, "sqrt(xi)": np.sqrt(xi)}
    rec = {"dD": [], "resid_ratio": [], "chordal_norm": [],
           **{f"n_new[{k}]": [] for k in deltas}}
    for tau in range(1, A.num_sites):
        E, R = Es[tau - 1], rhos[tau]
        DL, DL1 = E.shape
        cos = np.clip(np.linalg.svd(E, compute_uv=False), 0.0, 1.0)
        total = float(np.trace(R).real)
        captured = float(np.einsum("ab,bc,ac->", E, R, np.conj(E), optimize=True).real)
        resid = np.sqrt(max(0.0, 1.0 - captured / total)) if total > 0 else 0.0
        sin2 = np.clip(1.0 - cos**2, 0.0, None)
        rec["dD"].append(DL1 - DL)
        rec["resid_ratio"].append(resid)
        rec["chordal_norm"].append(float(np.sqrt(sin2.sum())) / np.sqrt(DL))
        for name, delta in deltas.items():
            rec[f"n_new[{name}]"].append(DL1 - int(np.count_nonzero(cos >= 1.0 - delta)))
    return {k: np.asarray(v) for k, v in rec.items()}


def aggregate(rec):
    resid = rec["resid_ratio"]
    return {
        "eta_max": float(resid.max()),
        "eta_rms": float(np.sqrt(np.mean(resid**2))),
        "chord": float(rec["chordal_norm"].max()),
        "n_new_xi": int(rec["n_new[xi]"].max()),
        "n_new_rtxi": int(rec["n_new[sqrt(xi)]"].max()),
        "max_dD": int(rec["dD"].max()),
    }


# --------------------------------------------------------------------------
# observable extraction (public-axis mirror of EDMSolver._solve_separable)
# --------------------------------------------------------------------------

def public_axis_polarization(snap, eps, order, Sop, n_steps):
    """``P(t) = <S_z(t)>`` on the solver's public axis ``eps, 2eps, ..., T``.

    The Eq.-F2/F3 sweep yields values at ``t = 0, eps, ..., (N-1) eps`` (measured before
    each Trotter step); mirror the driver exactly: drop the t=0 point and append the
    final-time ``Tr[S_z rho(T)]`` read from the same snapshot.
    """
    _, raw = ObservableExtractor.coupling_polarization_history(
        snap, eps, channel=CHANNEL, order=order)
    p_T = float(ObservableExtractor.expectation(snap, Sop).real)
    P = np.concatenate((raw[1:], np.asarray([p_T], dtype=np.float64)))
    times = eps * np.arange(1, n_steps + 1, dtype=np.float64)
    return times, P


def spectrum(P, eps):
    """Mean-polarization spectrum (derived diagnostic; fixed, recorded convention).

    Demeaned trace, rectangular window (no other detrend), ``S(w_j) = eps^2 |rfft(P -
    mean)|^2`` at ``w_j = 2 pi rfftfreq(N, eps)``; the DC bin is kept (== 0 after
    demeaning).  NOT a spin-noise PSD -- see the module docstring.
    """
    P = np.asarray(P, dtype=np.float64)
    F = np.fft.rfft(P - P.mean())
    S = (np.abs(F) ** 2) * (eps ** 2)
    omega = 2.0 * np.pi * np.fft.rfftfreq(P.size, d=eps)
    return omega, S


# --------------------------------------------------------------------------
# power-law fits (full + smallest-x tail), floor mask optional
# --------------------------------------------------------------------------

def powerlaw_fit(x, y, floor=None):
    """Fit ``y ~ c * x^alpha`` on positive finite points; optional roundoff-floor mask.

    Returns alpha/c/R2 plus the bookkeeping the audit requires: ``n_total`` (positive
    finite points), ``n_floored`` (dropped as ``y <= floor``), ``n_used``.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    m0 = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    n_total = int(m0.sum())
    if floor is not None:
        m = m0 & (y > floor)
    else:
        m = m0
    n_used = int(m.sum())
    out = {"alpha": np.nan, "c": np.nan, "R2": np.nan,
           "n_total": n_total, "n_used": n_used, "n_floored": n_total - n_used}
    if n_used < 3:
        return out
    lx, ly = np.log(x[m]), np.log(y[m])
    alpha, lc = np.polyfit(lx, ly, 1)
    resid = ly - (alpha * lx + lc)
    ss_res, ss_tot = float(np.sum(resid**2)), float(np.sum((ly - ly.mean()) ** 2))
    out.update(alpha=float(alpha), c=float(np.exp(lc)),
               R2=(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan))
    return out


def fit_block(x, y, *, tail_frac, floor=None):
    """Full-range + smallest-x-tail fits, with the tail x-range and min value recorded."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    block = {"full": powerlaw_fit(x, y, floor=floor)}
    pos = (x > 0) & np.isfinite(x)
    ypos = y[(y > 0) & np.isfinite(y)]
    block["y_min"] = float(ypos.min()) if ypos.size else np.nan
    if pos.sum() < 6:
        block["tail"] = powerlaw_fit([], [], floor=floor)
        block["tail"].update(x_max_used=np.nan, tail_frac=tail_frac)
        return block
    thr = float(np.quantile(x[pos], tail_frac))
    m = pos & (x <= thr)
    block["tail"] = powerlaw_fit(x[m], y[m], floor=floor)
    block["tail"].update(x_max_used=thr, tail_frac=tail_frac)
    return block


def acceptance_report(fits, quality, decomp):
    """Pre-registered acceptance: fit gate AND quality gate (--anchor runs).

    ``quality`` comes from :func:`run_sweep`: cap hits, non-finite occurrences, max
    discarded weight / trace deviation (the latter reported, not gated).  Under
    ``decomp='exact'`` every fold must have measured a numeric discarded weight
    (``n_disc_unmeasured == 0``); under ``rsvd`` ``None`` is legitimate.
    """
    lo, hi = _GATE_ALPHA_BAND
    fit_gate = {"band": list(_GATE_ALPHA_BAND), "R2_min": _GATE_R2_MIN, "n_min": _GATE_N_MIN}
    ok_fit = True
    for key in ("math_eta_rms", "phys_dP"):
        t = fits[key]["tail"]
        g = {"alpha": t["alpha"],
             "in_band": bool(np.isfinite(t["alpha"]) and lo <= t["alpha"] <= hi),
             "R2_ok": bool(np.isfinite(t["R2"]) and t["R2"] >= _GATE_R2_MIN),
             "n_ok": bool(t["n_used"] >= _GATE_N_MIN)}
        g["pass"] = g["in_band"] and g["R2_ok"] and g["n_ok"]
        ok_fit = ok_fit and g["pass"]
        fit_gate[key] = g
    fit_gate["pass"] = ok_fit
    disc_metric_ok = (decomp != "exact") or (quality["n_disc_unmeasured"] == 0)
    quality_gate = {
        "cap_hit_Ls": quality["cap_hit_Ls"],
        "nonfinite": quality["nonfinite"],
        "disc_weight_max": quality["disc_weight_max"],
        "n_disc_unmeasured": quality["n_disc_unmeasured"],
        "disc_metric_ok": disc_metric_ok,
        "trace_dev_max": quality["trace_dev_max"],   # reported only -- no gate threshold
        "pass": (not quality["cap_hit_Ls"] and not quality["nonfinite"]
                 and disc_metric_ok),
    }
    return {"fit_gate": fit_gate, "quality_gate": quality_gate,
            "pass": fit_gate["pass"] and quality_gate["pass"]}


def _exit_code(accept, anchor: bool) -> int:
    if not anchor:
        return EXIT_OK
    return EXIT_OK if (accept is not None and accept["pass"]) else EXIT_GATE_FAIL


# --------------------------------------------------------------------------
# provenance + resource sampling + atomic IO
# --------------------------------------------------------------------------

def _rss_peak_bytes():
    """Process-lifetime peak RSS (ru_maxrss), NOT the instantaneous value at this L."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(ru if sys.platform == "darwin" else ru * 1024)  # linux reports KiB


def _gpu_snapshot():
    import cupy as cp  # noqa: PLC0415
    pool = cp.get_default_memory_pool()
    free, total = cp.cuda.runtime.memGetInfo()
    return {"pool_used": int(pool.used_bytes()), "pool_total": int(pool.total_bytes()),
            "dev_used": int(total - free)}


_GPU_SAMPLE_INTERVAL = 0.01  # s -- GPU peaks are SAMPLED at this cadence, not exact maxima


class GpuPeakSampler:
    """Background thread sampling CUDA device/pool usage -> a SAMPLED per-fold peak.

    ``cudaMemGetInfo`` + the CuPy pool counters are read every ``interval`` seconds while
    active; ``stop()`` returns the max seen (plus one final synchronous sample).  This is a
    sampled peak, NOT an exact upper bound -- transients shorter than ``interval`` can be
    missed -- but unlike a single post-fold snapshot it covers the transient fold/compress
    workspace.  Capacity projections must keep a safety margin on top of it.
    """

    def __init__(self, interval=_GPU_SAMPLE_INTERVAL):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.peak = {"dev_used": 0, "pool_used": 0, "pool_total": 0}

    def _sample(self):
        s = _gpu_snapshot()
        for k in self.peak:
            self.peak[k] = max(self.peak[k], s[k])

    def _loop(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def start(self):
        self.peak = {"dev_used": 0, "pool_used": 0, "pool_total": 0}
        self._stop.clear()
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._sample()
        return {f"peak_{k}": v for k, v in self.peak.items()}


def _git(*cmd):
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(cmd, cwd=root, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001 -- provenance is best-effort, never fatal
        return None


def _provenance(device):
    root = Path(__file__).resolve().parents[2]
    import quimb  # noqa: PLC0415
    prov = {
        "script": str(Path(__file__).relative_to(root)),
        "argv": sys.argv,
        "git_commit": _git("git", "rev-parse", "HEAD"),
        "git_dirty": bool(_git("git", "status", "--porcelain")),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "quimb": quimb.__version__,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if device == "gpu":
        import cupy as cp  # noqa: PLC0415
        prov["cupy"] = cp.__version__
        prov["gpu_name"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    return prov


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _atomic_npz(path: Path, arrays: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _archive_group(paths) -> str | None:
    """Move every existing artifact of a previous run aside under ONE shared stamp.

    ``<p>.archived-<stamp>`` for each existing path (files and the shards dir alike), so
    the old JSON/NPZ/progress/shards/marker remain recognisable as one run's remains.
    Nothing is deleted -- interrupted shards survive, and the reuse check (which reads
    only the unsuffixed names) can never mistake a stale result for the new run's output.
    Returns the stamp, or ``None`` if nothing existed.
    """
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for p in existing:
        target = p.with_name(f"{p.name}.archived-{stamp}")
        n = 0
        while target.exists():
            n += 1
            target = p.with_name(f"{p.name}.archived-{stamp}.{n}")
        p.rename(target)
    return stamp


# --------------------------------------------------------------------------
# the streaming sweep
# --------------------------------------------------------------------------

def stream_sweep(model, p, on_progress=None):
    """Fold sub-baths one at a time; yield one record per ``L = 1..K``.

    Memory discipline: the raw host snapshot is released right after extraction; at most
    the previous and current left-canonical snapshots are resident.
    """
    ke = SeparableKernelEngine.from_model(model, T=p["T"], eps=p["eps"])
    order = p["order"]
    expander = FirstOrderExpander() if order == 1 else SecondOrderExpander()
    ev = SeparableBathEvolution(
        expander=expander, compress_method=p["method"], compress_decomp=p["decomp"],
        compress_decomp_q=p["decomp_q"], compress_canon=p["canon"])
    d, d_phys, K = model.system_dim, ke.d_phys, ke.K
    D_a = int(ke.corr.bond_dim)
    n_steps = SolverConfig(eps=p["eps"], T=p["T"]).n_steps  # validated grid, no silent round
    n_sites = order * n_steps
    rho0 = model.initial_system_state().reshape(-1).astype(np.complex128)
    gpu = p["device"] == "gpu"
    if gpu:
        import cupy as cp  # noqa: PLC0415
        convert = cp.asarray
        sampler = GpuPeakSampler()
    else:
        convert = lambda a: a  # noqa: E731
        sampler = None
    mps = QuimbEDM.from_edmmps(
        ev._build_system_mps(model, p["eps"], n_steps, order, d, d_phys, rho0, convert))
    Sop = model.coupling_operators_at(n_steps * p["eps"])[CHANNEL - 1]
    gk = model.couplings
    dims = {"D_a": D_a, "d_phys": d_phys, "n_steps": n_steps, "n_sites": n_sites,
            "dtype": "complex128", "itemsize": _ITEMSIZE}
    prev = None  # (canonical snapshot, P, S) of the previous L
    for k in range(K):
        t0 = time.perf_counter()
        if sampler is not None:
            sampler.start()
        try:
            mpo_sites = [convert(s)
                         for s in ke.for_sub_bath(k).get_kernel_mpo(n_sites).site_tensors]
            mps = mps.fold(mpo_sites, cutoff=p["cutoff"], cutoff_mode=p["cutoff_mode"],
                           method=p["method"], max_bond=p["max_bond"], decomp=p["decomp"],
                           decomp_q=p["decomp_q"], canon=p["canon"])
            w_disc = mps.max_discarded_weight  # per-fold local (P1-15); None = unmeasurable
            snap = _edm_to_numpy(mps.to_edmmps())
            L = k + 1
            bonds = [int(b) for b in snap.bond_dims]
            # projected size of the NEXT fold's raw (uncompressed) MPS from the actual
            # per-site (D_left, D_right) -- storage only, compression workspace excluded
            next_raw = sum(d_phys * (t.shape[1] * D_a) * (t.shape[2] * D_a) * _ITEMSIZE
                           for t in snap.tensors)
            times, P = public_axis_polarization(snap, p["eps"], order, Sop, n_steps)
            trace_dev = float(ObservableExtractor.trace_deviation(snap))
            omega, S = spectrum(P, p["eps"])
            canon = left_canonicalize(snap)
            del snap  # raw copy released; only canonical snapshots stay resident
            out = {
                "L": L, "times": times, "P": P, "omega": omega, "S": S,
                "bonds": bonds, "Dmax": (max(bonds) if bonds else 1),
                "cap_hit": bool(p["max_bond"] is not None and bonds
                                and max(bonds) >= p["max_bond"]),
                "trace_dev": trace_dev,
                "disc_weight": (float(w_disc) if w_disc is not None else None),
                "next_fold_raw_bytes": int(next_raw),
                "dims": dims, "rec": None, "row": None,
            }
            if prev is not None:
                rec = analyse_transition_canon(prev[0], canon, p["cutoff"])
                row = aggregate(rec)
                row["L"] = k  # transition (L=k) -> (L+1); labeling as coupling_distributions
                row["x"] = float(gk[k] ** 2 / model.effective_coupling(k) ** 2)
                row["dP_rms"] = float(np.sqrt(np.mean((P - prev[1]) ** 2)))
                row["dS_rms"] = float(np.sqrt(np.mean((S - prev[2]) ** 2)))
                out["rec"], out["row"] = rec, row
            prev = (canon, P, S)
        except BaseException:
            # a capacity probe must not die silently: keep the sampled-so-far peak and
            # the failed L in the progress record before propagating (e.g. OOM)
            fail = {"L": k + 1, "failed": True, "Dmax": None, "cap_hit": None,
                    "trace_dev": None, "disc_weight": None, "next_fold_raw_bytes": None,
                    "wall_s": time.perf_counter() - t0,
                    "rss_peak_so_far_bytes": _rss_peak_bytes()}
            if sampler is not None:
                fail["gpu_peak"] = sampler.stop()
            if on_progress is not None:
                on_progress(fail)
            raise
        if sampler is not None:
            out["gpu_peak"] = sampler.stop()
            out["gpu_post_fold"] = {f"post_fold_{k2}": v
                                    for k2, v in _gpu_snapshot().items()}
        out["wall_s"] = time.perf_counter() - t0
        out["rss_peak_so_far_bytes"] = _rss_peak_bytes()
        if on_progress is not None:
            on_progress(out)
        yield out


def _fits_from_rows(rows, p):
    x = np.array([r["x"] for r in rows])
    fits = {
        "math_eta_rms": fit_block(x, [r["eta_rms"] for r in rows],
                                  tail_frac=p["tail_frac"], floor=p["eta_floor"]),
        "math_eta_max": fit_block(x, [r["eta_max"] for r in rows],
                                  tail_frac=p["tail_frac"], floor=p["eta_floor"]),
        "chord": fit_block(x, [r["chord"] for r in rows], tail_frac=p["tail_frac"]),
        "phys_dP": fit_block(x, [r["dP_rms"] for r in rows], tail_frac=p["tail_frac"]),
        "spec_dS": fit_block(x, [r["dS_rms"] for r in rows], tail_frac=p["tail_frac"]),
    }
    fits["spec_dS"]["derived_diagnostic"] = True  # functional of P(t), not independent
    return fits


def _nonfinite_check(out):
    """Collect non-finite occurrences in this L's science data (quality gate input)."""
    bad = []
    for label, arr in (("P", out["P"]), ("S", out["S"])):
        if not np.all(np.isfinite(arr)):
            bad.append(f"L={out['L']}: non-finite in {label}")
    if not np.isfinite(out["trace_dev"]):
        bad.append(f"L={out['L']}: non-finite trace_dev")
    if out["disc_weight"] is not None and not np.isfinite(out["disc_weight"]):
        bad.append(f"L={out['L']}: non-finite disc_weight")
    if out["rec"] is not None:
        for key in ("resid_ratio", "chordal_norm"):
            if not np.all(np.isfinite(out["rec"][key])):
                bad.append(f"L={out['L']}: non-finite in {key}")
    return bad


def _reuse_check(json_path: Path, args):
    """Validate a completed result before skipping it.

    Returns ``(status, accepted)`` with status ``'ok'`` / ``'incomplete'`` /
    ``'mismatch'``.  ``'ok'`` requires: every protocol field equal to the request, the
    stored git commit equal to the current one, and the referenced NPZ present.
    """
    try:
        data = json.load(open(json_path))
    except Exception:  # noqa: BLE001 -- unreadable = not complete
        return "incomplete", None
    if not data.get("complete"):
        return "incomplete", None
    meta = data.get("meta", {})
    mismatches = [k for k in _PROTOCOL_KEYS if meta.get(k) != getattr(args, k)]
    stored_commit = meta.get("provenance", {}).get("git_commit")
    if stored_commit != _git("git", "rev-parse", "HEAD"):
        mismatches.append("git_commit")
    npz_name = data.get("npz")
    if not npz_name or not (json_path.parent / npz_name).exists():
        mismatches.append("npz_missing")
    if data.get("anchor_requested", False) != args.anchor:
        mismatches.append("anchor")
    if mismatches:
        print(f"[reuse-mismatch] {json_path.name}: {mismatches} -- refusing silent reuse "
              f"(use --force or a new --name)")
        return "mismatch", None
    return "ok", data.get("accepted")


def run_sweep(args) -> int:
    p = _params(args)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / f"{args.name}.json"
    npz_path = outdir / f"{args.name}.npz"
    progress_path = outdir / f"{args.name}.progress.jsonl"
    shards_dir = outdir / f"{args.name}.shards"

    if json_path.exists() and not args.force:
        status, accepted = _reuse_check(json_path, args)
        if status == "mismatch":
            return EXIT_REUSE_MISMATCH
        if status == "ok":
            rc = _exit_code({"pass": bool(accepted)}, args.anchor)
            print(f"[skip] {json_path} already complete "
                  f"(accepted={accepted}, exit {rc}; --force to redo)")
            return rc

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    marker_path = outdir / f"{args.name}.running.json"
    # one archiving event moves EVERY remain of a previous run (complete or interrupted)
    # aside: interrupted shards are preserved, and a stale JSON/NPZ can never satisfy the
    # reuse check as if it were this run's output
    _archive_group([json_path, npz_path, progress_path, shards_dir, marker_path])
    shards_dir.mkdir(parents=True)
    _atomic_json(marker_path, {"run_id": run_id, "pid": os.getpid(),
                               "hostname": socket.gethostname(),
                               "started": datetime.now(timezone.utc)
                               .isoformat(timespec="seconds")})

    model = model_for(args.coupling, args.g, args.K,
                      beta=args.beta, seed=args.seed, rho=args.rho)
    prov = _provenance(p["device"])
    meta = {**{k: v for k, v in vars(args).items()
               if k not in ("pool", "check", "compare", "replot", "selftest")},
            "run_id": run_id,
            "couplings": model.couplings.tolist(), "channel": CHANNEL,
            "spectrum_convention": "demeaned, rectangular window, S=eps^2|rfft|^2, "
                                   "omega=2*pi*rfftfreq(N,eps), DC bin kept",
            "provenance": prov}
    if p["device"] == "gpu":
        meta["gpu_peak_sampling"] = {
            "interval_s": _GPU_SAMPLE_INTERVAL, "sampled_peak": True,
            "caveat": "transients shorter than the sampling interval may be missed; "
                      "capacity projections must keep a safety margin"}
    print(f"[{args.name}] run_id={run_id} K={args.K} coupling={args.coupling} T={args.T} "
          f"eps={args.eps} order={args.order} cutoff={args.cutoff:g} "
          f"max_bond={args.max_bond} {args.method}/{args.decomp}(q{args.decomp_q})/"
          f"{args.canon} device={args.device}")

    def on_progress(out):
        line = {"run_id": run_id, "name": args.name, "L": out["L"],
                "failed": out.get("failed", False), "Dmax": out["Dmax"],
                "cap_hit": out["cap_hit"], "trace_dev": out["trace_dev"],
                "disc_weight": out["disc_weight"],
                "next_fold_raw_bytes": out["next_fold_raw_bytes"],
                "wall_s": round(out["wall_s"], 3),
                "rss_peak_so_far_bytes": out["rss_peak_so_far_bytes"],
                **out.get("gpu_peak", {}), **out.get("gpu_post_fold", {}),
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with open(progress_path, "a") as f:
            f.write(json.dumps(line) + "\n")

    t0 = time.perf_counter()
    rows, per_L, arrays = [], [], {}
    quality = {"cap_hit_Ls": [], "nonfinite": [], "disc_weight_max": None,
               "n_disc_unmeasured": 0, "trace_dev_max": 0.0}
    dims = None
    for out in stream_sweep(model, p, on_progress=on_progress):
        L = out["L"]
        dims = out["dims"]
        entry = {k: out[k] for k in
                 ("L", "Dmax", "cap_hit", "trace_dev", "disc_weight",
                  "next_fold_raw_bytes", "wall_s", "rss_peak_so_far_bytes")}
        for gk_ in ("gpu_peak", "gpu_post_fold"):
            if gk_ in out:
                entry[gk_] = out[gk_]
        per_L.append(entry)
        if out["cap_hit"]:
            quality["cap_hit_Ls"].append(L)
        quality["nonfinite"].extend(_nonfinite_check(out))
        if out["disc_weight"] is None:
            quality["n_disc_unmeasured"] += 1
        elif quality["disc_weight_max"] is None or out["disc_weight"] > quality["disc_weight_max"]:
            quality["disc_weight_max"] = out["disc_weight"]
        quality["trace_dev_max"] = max(quality["trace_dev_max"], out["trace_dev"])
        shard = {"P": out["P"], "S": out["S"],
                 "bonds": np.asarray(out["bonds"], dtype=np.int64)}
        arrays[f"P_L{L:03d}"], arrays[f"S_L{L:03d}"] = out["P"], out["S"]
        arrays[f"bonds_L{L:03d}"] = shard["bonds"]
        if L == 1:
            arrays["times"], arrays["omega"] = out["times"], out["omega"]
            shard["times"], shard["omega"] = out["times"], out["omega"]
        if out["row"] is not None:
            rows.append(out["row"])
            rec, tl = out["rec"], out["row"]["L"]
            for key, aname in (("resid_ratio", "resid"), ("chordal_norm", "chord"),
                               ("dD", "dD"), ("n_new[xi]", "nnew_xi"),
                               ("n_new[sqrt(xi)]", "nnew_rtxi")):
                arrays[f"{aname}_T{tl:03d}"] = rec[key]
                shard[f"{aname}_T{tl:03d}"] = rec[key]
        _atomic_npz(shards_dir / f"L{L:03d}.npz", shard)  # science data survives interrupts
        print(f"  L={L:>3}  Dmax={out['Dmax']:>4}{'*' if out['cap_hit'] else ' '} "
              f"|Tr-1|={out['trace_dev']:.2e}  "
              f"w_disc={out['disc_weight'] if out['disc_weight'] is not None else 'n/a'}  "
              f"{out['wall_s']:.1f}s", flush=True)

    fit_rows = [r for r in rows if r["L"] >= args.L0]
    fits = _fits_from_rows(fit_rows, p)
    accept = acceptance_report(fits, quality, args.decomp) if args.anchor else None
    wall = time.perf_counter() - t0
    prov["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    _atomic_npz(npz_path, arrays)
    _atomic_json(json_path, {
        "complete": True, "anchor_requested": args.anchor,
        "accepted": (accept["pass"] if accept is not None else None),
        "kind": "sweep", "meta": meta, "dims": dims, "per_L": per_L, "rows": rows,
        "L0": args.L0, "fits": fits, "quality": quality, "acceptance": accept,
        "wall_s": wall, "npz": npz_path.name})
    shutil.rmtree(shards_dir)  # this run's shards, consolidated into the final NPZ
    marker_path.unlink(missing_ok=True)
    _report_fits(args.name, fits, accept)
    print(f"[{args.name}] done in {wall:.1f}s -> {json_path}")
    return _exit_code(accept, args.anchor)


def _report_fits(name, fits, accept):
    for key in ("math_eta_rms", "math_eta_max", "chord", "phys_dP", "spec_dS"):
        f, t = fits[key]["full"], fits[key]["tail"]
        print(f"  [{name}] {key:>12}: alpha={f['alpha']:.3f} (R2={f['R2']:.3f}, "
              f"n={f['n_used']}/{f['n_total']}, floored={f['n_floored']}) | "
              f"tail alpha={t['alpha']:.3f} (R2={t['R2']:.3f}, n={t['n_used']}, "
              f"floored={t['n_floored']}) | y_min={fits[key]['y_min']:.3g}")
    if accept is not None:
        fg, qg = accept["fit_gate"], accept["quality_gate"]
        print(f"  [{name}] ACCEPTANCE (pre-registered): "
              f"{'PASS' if accept['pass'] else 'FAIL -- stop and report'}  "
              f"fit_gate={fg['pass']} "
              + " ".join(f"{k}:a={fg[k]['alpha']:.3f}" for k in ("math_eta_rms", "phys_dP"))
              + f"  quality_gate={qg['pass']} (cap_hit_Ls={qg['cap_hit_Ls']}, "
                f"nonfinite={len(qg['nonfinite'])}, trace_dev_max={qg['trace_dev_max']:.2e})")


# --------------------------------------------------------------------------
# pooled fits (per-seed first, then pooled -- audit requirement)
# --------------------------------------------------------------------------

def run_pool(args) -> int:
    """Per-member + pooled fits over seed realisations.

    Members must be complete sweeps of a **seed-bearing profile** (``random``/``ou``)
    agreeing on EVERY protocol field except ``seed`` (the coupling array differs only
    through the seed) and produced by the same git commit; an anchor member must be
    ``accepted`` -- a FAILed result is refused.  Any violation exits 4.
    """
    outdir = Path(args.outdir)
    names = [s for s in args.pool.split(",") if s.strip()]
    members, all_rows = {}, []
    p, ref_proto, ref_name = None, None, None
    for n in names:
        data = json.load(open(outdir / f"{n}.json"))
        if not data.get("complete"):
            raise ValueError(f"member {n} is not a complete sweep")
        meta = data["meta"]
        if meta.get("coupling") not in ("random", "ou"):
            print(f"[pool-mismatch] {n}: coupling {meta.get('coupling')!r} is not a "
                  f"seed-bearing profile (random/ou) -- refusing to pool")
            return EXIT_REUSE_MISMATCH
        if data.get("anchor_requested") and not data.get("accepted"):
            print(f"[pool-mismatch] {n}: anchor member was NOT accepted -- a FAILed "
                  f"result cannot enter a pooled fit")
            return EXIT_REUSE_MISMATCH
        proto = {k: meta.get(k) for k in _PROTOCOL_KEYS if k != "seed"}
        proto["git_commit"] = meta.get("provenance", {}).get("git_commit")
        if ref_proto is None:
            ref_proto, ref_name = proto, n
        elif proto != ref_proto:
            diff = [k for k in proto if proto[k] != ref_proto[k]]
            print(f"[pool-mismatch] {n} differs from {ref_name} on {diff} -- refusing to pool")
            return EXIT_REUSE_MISMATCH
        rows = [r for r in data["rows"] if r["L"] >= data["L0"]]
        all_rows.extend(rows)
        members[n] = {"math_tail_alpha": data["fits"]["math_eta_rms"]["tail"]["alpha"],
                      "phys_tail_alpha": data["fits"]["phys_dP"]["tail"]["alpha"],
                      "n_rows": len(rows)}
        if p is None:
            p = {"tail_frac": meta["tail_frac"], "eta_floor": meta["eta_floor"]}
    fits = _fits_from_rows(all_rows, p)
    spread = {}
    for key, mkey in (("math_eta_rms", "math_tail_alpha"), ("phys_dP", "phys_tail_alpha")):
        vals = sorted(m[mkey] for m in members.values())
        spread[key] = {"min": vals[0], "median": float(np.median(vals)), "max": vals[-1]}
    out = {"complete": True, "kind": "pool", "members": members, "spread": spread,
           "pooled_fits": fits, "n_rows_pooled": len(all_rows),
           "protocol": ref_proto,  # shared member protocol incl. git_commit, minus seed
           "provenance": _provenance("cpu")}
    path = outdir / f"{args.name}.json"
    _atomic_json(path, out)
    print(f"[{args.name}] pooled {len(names)} members, {len(all_rows)} rows -> {path}")
    for key in ("math_eta_rms", "phys_dP"):
        t = fits[key]["tail"]
        s = spread[key]
        print(f"  {key}: per-seed tail alpha min/med/max = "
              f"{s['min']:.3f}/{s['median']:.3f}/{s['max']:.3f}; "
              f"pooled tail alpha={t['alpha']:.3f} (R2={t['R2']:.3f}, n={t['n_used']})")
    return EXIT_OK


# --------------------------------------------------------------------------
# coarse/fine eps comparison (report-only: deviations, NOT a convergence verdict)
# --------------------------------------------------------------------------

def run_compare(args) -> int:
    """Align a coarse and a fine sweep on common physical times; report deviations.

    Protocol fields other than ``eps`` must match; the eps ratio must be an integer, and
    the coarse grid is a subset of the fine one (public axis ``eps..T``).  Output is a
    comparison JSON with per-``L`` max/RMS ``P_L`` deviations, row-level ``dP``/``eta``
    deltas and the tail-alpha shifts.  REPORT-ONLY: no pre-registered convergence
    criterion exists, so none is applied.
    """
    outdir = Path(args.outdir)
    coarse_name, fine_name = [s.strip() for s in args.compare.split(",")]
    dc = json.load(open(outdir / f"{coarse_name}.json"))
    df = json.load(open(outdir / f"{fine_name}.json"))
    for d, n in ((dc, coarse_name), (df, fine_name)):
        if not d.get("complete"):
            raise ValueError(f"{n} is not a complete sweep")
    mc, mf = dc["meta"], df["meta"]
    diff = [k for k in _PROTOCOL_KEYS if k != "eps" and mc.get(k) != mf.get(k)]
    if diff:
        print(f"[compare-mismatch] {coarse_name} vs {fine_name} differ on {diff}")
        return EXIT_REUSE_MISMATCH
    ratio = mc["eps"] / mf["eps"]
    if abs(ratio - round(ratio)) > 1e-9 or round(ratio) < 2:
        raise ValueError(f"eps ratio must be an integer >= 2, got {ratio!r}")
    ratio = int(round(ratio))
    zc = np.load(outdir / dc["npz"])
    zf = np.load(outdir / df["npz"])
    K = mc["K"]
    idx = np.arange(1, len(zc["times"]) + 1) * ratio - 1  # coarse t_m == fine t_{m*ratio}
    if not np.allclose(zc["times"], zf["times"][idx]):
        raise ValueError("aligned time axes do not match -- grids are inconsistent")
    per_L, overall_max = [], 0.0
    sq_sum, n_pts = 0.0, 0
    for L in range(1, K + 1):
        dev = zc[f"P_L{L:03d}"] - zf[f"P_L{L:03d}"][idx]
        mx, rms = float(np.max(np.abs(dev))), float(np.sqrt(np.mean(dev**2)))
        per_L.append({"L": L, "P_max_dev": mx, "P_rms_dev": rms})
        overall_max = max(overall_max, mx)
        sq_sum += float(np.sum(dev**2))
        n_pts += dev.size
    rows_c = {r["L"]: r for r in dc["rows"]}
    rows_f = {r["L"]: r for r in df["rows"]}
    row_delta = [{"L": L,
                  "dP_rms_coarse": rows_c[L]["dP_rms"], "dP_rms_fine": rows_f[L]["dP_rms"],
                  "eta_rms_coarse": rows_c[L]["eta_rms"], "eta_rms_fine": rows_f[L]["eta_rms"]}
                 for L in sorted(set(rows_c) & set(rows_f))]
    alphas = {key: {"coarse": dc["fits"][key]["tail"]["alpha"],
                    "fine": df["fits"][key]["tail"]["alpha"],
                    "delta": df["fits"][key]["tail"]["alpha"] - dc["fits"][key]["tail"]["alpha"]}
              for key in ("math_eta_rms", "phys_dP", "spec_dS")}
    out = {"complete": True, "kind": "eps_compare", "coarse": coarse_name,
           "fine": fine_name, "eps_coarse": mc["eps"], "eps_fine": mf["eps"],
           "ratio": ratio, "verdict": "report-only (no pre-registered criterion)",
           "overall": {"P_max_dev": overall_max,
                       "P_rms_dev": float(np.sqrt(sq_sum / n_pts))},
           "per_L": per_L, "row_delta": row_delta, "tail_alphas": alphas,
           "provenance": _provenance("cpu")}
    path = outdir / f"{args.name}.json"
    _atomic_json(path, out)
    print(f"[{args.name}] eps compare {coarse_name} (eps={mc['eps']}) vs {fine_name} "
          f"(eps={mf['eps']}): P max dev={overall_max:.3e}, "
          f"rms dev={out['overall']['P_rms_dev']:.3e} (report-only)")
    for key, a in alphas.items():
        print(f"  {key}: tail alpha {a['coarse']:.3f} -> {a['fine']:.3f} "
              f"(delta {a['delta']:+.3f})")
    return EXIT_OK


# --------------------------------------------------------------------------
# figure regeneration from retrieved JSON + NPZ (local, matplotlib Agg)
# --------------------------------------------------------------------------

def run_replot(args) -> int:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; cannot replot")
        return 1
    outdir = Path(args.outdir)
    picdir = outdir / "pictures"
    picdir.mkdir(parents=True, exist_ok=True)
    name = args.replot
    data = json.load(open(outdir / f"{name}.json"))
    kind = data.get("kind")

    def _save(fig, stem):
        fig.tight_layout()
        png = picdir / f"{stem}.png"
        fig.savefig(png, dpi=130)
        plt.close(fig)
        print(f"saved {png}")

    if kind == "sweep":
        rows = [r for r in data["rows"] if r["L"] >= data["L0"]]
        x = np.array([r["x"] for r in rows])
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for key, marker in (("eta_rms", "o"), ("dP_rms", "s"), ("dS_rms", "^")):
            y = np.array([r[key] for r in rows])
            ax.loglog(x, y, marker, ms=4, alpha=0.7, label=key)
        fitkeys = {"eta_rms": "math_eta_rms", "dP_rms": "phys_dP", "dS_rms": "spec_dS"}
        for key, fkey in fitkeys.items():
            t = data["fits"][fkey]["tail"]
            if np.isfinite(t.get("alpha", np.nan)) and np.isfinite(t.get("x_max_used", np.nan)):
                xs = np.array([x[x > 0].min(), t["x_max_used"]])
                ax.loglog(xs, t["c"] * xs ** t["alpha"], "-", lw=1.0,
                          label=f"{fkey} tail a={t['alpha']:.3f}")
        xr = np.array([x[x > 0].min(), x.max()])
        ax.loglog(xr, xr / xr.max() * 1e-2, "k--", lw=1.0, label="slope 1")
        ax.set_xlabel(r"$x = g_{L+1}^2/\bar g_L^2$")
        ax.set_ylabel("increment")
        ax.set_title(f"{name}: scaling (full range + tail fits)")
        ax.legend(fontsize=7)
        _save(fig, f"{name}_scaling")

        per = data["per_L"]
        L = [e["L"] for e in per]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        a = axes[0][0]
        a.plot(L, [e["Dmax"] for e in per], "o-", ms=3)
        cap = data["meta"].get("max_bond")
        if cap:
            a.axhline(cap, color="r", ls="--", lw=1, label=f"max_bond={cap}")
        hitL = [e["L"] for e in per if e["cap_hit"]]
        if hitL:
            a.plot(hitL, [cap] * len(hitL), "rx", ms=8, label="cap_hit")
        a.set_xlabel("L"); a.set_ylabel("Dmax"); a.legend(fontsize=7)
        a = axes[0][1]
        a.semilogy(L, [e["trace_dev"] for e in per], "o-", ms=3, label="|Tr-1|")
        wd = [(e["L"], e["disc_weight"]) for e in per if e["disc_weight"]]
        if wd:
            a.semilogy(*zip(*wd), "s-", ms=3, label="disc weight")
        a.set_xlabel("L"); a.legend(fontsize=7)
        a = axes[1][0]
        a.plot(L, [e["wall_s"] for e in per], "o-", ms=3)
        a.set_xlabel("L"); a.set_ylabel("wall [s]")
        a = axes[1][1]
        gp = [(e["L"], e["gpu_peak"]["peak_dev_used"] / 2**30)
              for e in per if "gpu_peak" in e]
        if gp:
            a.plot(*zip(*gp), "o-", ms=3, label="GPU sampled peak [GiB]")
        a.plot(L, [e["rss_peak_so_far_bytes"] / 2**30 for e in per], "s-", ms=3,
               label="host RSS peak-so-far [GiB]")
        a.set_xlabel("L"); a.legend(fontsize=7)
        fig.suptitle(f"{name}: per-L records")
        _save(fig, f"{name}_perL")

    elif kind == "pool":
        fig, ax = plt.subplots(figsize=(7, 4.5))
        names = list(data["members"])
        for i, key in enumerate(("math_tail_alpha", "phys_tail_alpha")):
            vals = [data["members"][n][key] for n in names]
            ax.plot(range(len(names)), vals, "os"[i], ms=6, label=key)
        for key, mk in (("math_eta_rms", "--"), ("phys_dP", ":")):
            ax.axhline(data["pooled_fits"][key]["tail"]["alpha"], ls=mk, lw=1,
                       label=f"pooled {key}")
        ax.set_xticks(range(len(names)), names, rotation=20, fontsize=7)
        ax.set_ylabel("tail alpha"); ax.legend(fontsize=7)
        ax.set_title(f"{name}: per-seed vs pooled tail alpha")
        _save(fig, f"{name}_pool")

    elif kind == "eps_compare":
        per = data["per_L"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.semilogy([e["L"] for e in per], [e["P_max_dev"] for e in per], "o-", ms=3,
                    label="max dev")
        ax.semilogy([e["L"] for e in per], [e["P_rms_dev"] for e in per], "s-", ms=3,
                    label="rms dev")
        ax.set_xlabel("L"); ax.set_ylabel(r"$|P^{coarse}_L - P^{fine}_L|$")
        ax.set_title(f"{name}: eps {data['eps_coarse']} vs {data['eps_fine']} (report-only)")
        ax.legend(fontsize=8)
        _save(fig, f"{name}_epscompare")

    else:
        print(f"unknown kind {kind!r} in {name}.json")
        return 1
    return EXIT_OK


# --------------------------------------------------------------------------
# --check: cross-validate the streaming path against the established machinery
# --------------------------------------------------------------------------

def run_check(args) -> int:
    """Small-K CPU cross-validation (audit requirement).

    A. streaming per-bond diagnostics == coupling_distributions.analyse_transition on
       all-snapshot folds (same parameters, same canon), per key, atol 1e-10;
    B. P_L(t) == EDMSolver(sub_baths=L).solve(channel=3).polarization on the public axis
       for L in {1, K//2, K}, atol 1e-8 (identical code path mirrored);
    C. order-2 axis: times == eps * (1..N), length N.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import coupling_distributions as cd  # noqa: PLC0415 -- sibling module, check only
    from edmtn.driver import EDMSolver  # noqa: PLC0415

    K, T, eps, order = 8, 1.0, 0.25, 2
    cutoff, max_bond, method, decomp, q, canon = 1e-8, 200, "direct", "exact", 2, "quimb"
    p = {"T": T, "eps": eps, "order": order, "cutoff": cutoff, "cutoff_mode": "rel",
         "max_bond": max_bond, "method": method, "decomp": decomp, "decomp_q": q,
         "canon": canon, "device": "cpu", "tail_frac": 0.4, "eta_floor": 1e-6}
    model = model_for("linear", 1.0, K)
    n_steps = SolverConfig(eps=eps, T=T).n_steps
    failures = []

    stream = {out["L"]: out for out in stream_sweep(model, p)}

    # A -- against the all-snapshot reference implementation (canon hardcoded 'quimb' there)
    snaps, _ = cd.fold_snapshots(model, T=T, eps=eps, order=order, cutoff=cutoff,
                                 cutoff_mode="rel", max_bond=max_bond, method=method,
                                 decomp=decomp, decomp_q=q, device="cpu")
    for L in range(1, K):
        ref = cd.analyse_transition(snaps[L], snaps[L + 1], cutoff)
        got = stream[L + 1]["rec"]
        for key in ref:
            if not np.allclose(ref[key], got[key], atol=1e-10, rtol=1e-8):
                failures.append(f"A: transition {L}->{L + 1} key {key} max diff "
                                f"{np.max(np.abs(ref[key] - got[key])):.3e}")
    print(f"check A (streaming vs all-snapshot diagnostics, {K - 1} transitions): "
          + ("PASS" if not any(f.startswith('A') for f in failures) else "FAIL"))

    # B -- against the driver's public polarization contract
    for L in (1, K // 2, K):
        res = EDMSolver.from_model(
            model, T=T, eps=eps, expansion_order=order, cutoff=cutoff, max_bond=max_bond,
            sub_baths=L, backend="cpu", compress_method=method, compress_decomp=decomp,
            compress_canon=canon).solve(channel=CHANNEL)
        got = stream[L]
        if not np.array_equal(res.times, got["times"]):
            failures.append(f"B: L={L} time axes differ")
        dmax = float(np.max(np.abs(res.polarization - got["P"])))
        if dmax > 1e-8:
            failures.append(f"B: L={L} polarization max diff {dmax:.3e}")
        else:
            print(f"check B (P_L vs EDMSolver sub_baths={L}): PASS (max diff {dmax:.2e})")

    # C -- axis semantics
    t1 = stream[1]["times"]
    if not (t1.size == n_steps and np.allclose(t1, eps * np.arange(1, n_steps + 1))):
        failures.append("C: public time axis is not eps*(1..N)")
    else:
        print(f"check C (public axis eps..T, N={n_steps}): PASS")

    if failures:
        print("CROSS-VALIDATION FAILED:")
        for f in failures:
            print("  " + f)
        return 1
    print("cross-validation: ALL PASS")
    return 0


# --------------------------------------------------------------------------
# --selftest: exit-code contract (PASS / FAIL / skip-reload / reuse-mismatch)
# --------------------------------------------------------------------------

def run_selftest(args) -> int:
    """Assert the exit-code + data-safety contracts (synthetic gates + real tiny runs)."""
    import tempfile  # noqa: PLC0415
    failures = []

    def expect(label, got, want):
        ok = got == want
        print(f"selftest {label}: got {got} (want {want}) -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(label)

    # 1 -- pure gate logic on synthetic fits/quality
    def synth_fits(alpha):
        t = {"alpha": alpha, "R2": 0.999, "n_used": 10, "n_total": 10, "n_floored": 0}
        return {k: {"tail": dict(t)} for k in ("math_eta_rms", "phys_dP")}
    clean_q = {"cap_hit_Ls": [], "nonfinite": [], "disc_weight_max": 1e-16,
               "n_disc_unmeasured": 0, "trace_dev_max": 1e-11}
    expect("synthetic PASS",
           _exit_code(acceptance_report(synth_fits(1.00), clean_q, "exact"), True), EXIT_OK)
    expect("synthetic band-FAIL",
           _exit_code(acceptance_report(synth_fits(0.50), clean_q, "exact"), True),
           EXIT_GATE_FAIL)
    expect("synthetic quality-FAIL (cap)",
           _exit_code(acceptance_report(synth_fits(1.00), {**clean_q, "cap_hit_Ls": [7]},
                                        "exact"), True), EXIT_GATE_FAIL)
    expect("synthetic exact-metric-missing FAILs",
           _exit_code(acceptance_report(synth_fits(1.00),
                                        {**clean_q, "n_disc_unmeasured": 2}, "exact"),
                      True), EXIT_GATE_FAIL)
    expect("synthetic rsvd metric None allowed",
           _exit_code(acceptance_report(synth_fits(1.00),
                                        {**clean_q, "n_disc_unmeasured": 2}, "rsvd"),
                      True), EXIT_OK)
    expect("non-anchor ignores gates",
           _exit_code(acceptance_report(synth_fits(0.50), clean_q, "exact"), False),
           EXIT_OK)

    with tempfile.TemporaryDirectory(prefix="gaudin_selftest_") as tmp:
        tmpp = Path(tmp)
        base = ["--K", "10", "--T", "1.5", "--eps", "0.25", "--max-bond", "200",
                "--outdir", tmp]
        ap = build_parser()

        # 2 -- exit-code paths on real tiny runs (anchor FAILS the n>=8 gate by scale)
        a1 = ap.parse_args(base + ["--anchor", "--name", "st_anchor"])
        expect("tiny anchor run FAILs", run_sweep(a1), EXIT_GATE_FAIL)
        expect("skip of FAILed anchor re-fails", run_sweep(a1), EXIT_GATE_FAIL)
        a2 = ap.parse_args(base + ["--name", "st_plain"])
        expect("non-anchor run OK", run_sweep(a2), EXIT_OK)
        expect("skip of OK run OK", run_sweep(a2), EXIT_OK)
        a3 = ap.parse_args(["--K", "10", "--T", "1.5", "--eps", "0.125", "--max-bond",
                            "200", "--outdir", tmp, "--name", "st_plain"])
        expect("protocol-mismatch reuse rejected", run_sweep(a3), EXIT_REUSE_MISMATCH)

        # 3 -- --force + interruption: the OLD result must never resurface as the new one,
        #      and the interrupted run's shards must survive
        a4 = ap.parse_args(base + ["--name", "st_force"])
        expect("st_force initial run OK", run_sweep(a4), EXIT_OK)
        run_id0 = json.load(open(tmpp / "st_force.json"))["meta"]["run_id"]
        orig_stream = globals()["stream_sweep"]

        def _interrupted(model, pp, on_progress=None):
            for i, out in enumerate(orig_stream(model, pp, on_progress=on_progress)):
                if i >= 2:
                    raise RuntimeError("simulated interruption")
                yield out

        globals()["stream_sweep"] = _interrupted
        try:
            try:
                run_sweep(ap.parse_args(base + ["--name", "st_force", "--force"]))
                failures.append("interrupted run did not raise")
            except RuntimeError:
                pass
        finally:
            globals()["stream_sweep"] = orig_stream
        expect("no current JSON after interruption (reuse impossible)",
               (tmpp / "st_force.json").exists(), False)
        expect("old complete JSON archived, not deleted",
               len(list(tmpp.glob("st_force.json.archived-*"))) >= 1, True)
        expect("interrupted shards kept on disk",
               len(list((tmpp / "st_force.shards").glob("L*.npz"))) >= 1, True)
        expect("post-interruption rerun recomputes OK",
               run_sweep(ap.parse_args(base + ["--name", "st_force"])), EXIT_OK)
        expect("rerun is a NEW run (fresh run_id)",
               json.load(open(tmpp / "st_force.json"))["meta"]["run_id"] != run_id0, True)
        expect("interrupted shards archived by rerun",
               any((d / "L001.npz").exists()
                   for d in tmpp.glob("st_force.shards.archived-*")), True)

        # 4 -- pool consistency rules
        for s in (0, 1):
            run_sweep(ap.parse_args(base + ["--coupling", "random", "--seed", str(s),
                                            "--name", f"st_rand{s}"]))
        expect("pool of matching random seeds OK",
               run_pool(ap.parse_args(["--outdir", tmp, "--pool", "st_rand0,st_rand1",
                                       "--name", "st_pool_ok"])), EXIT_OK)
        expect("pool rejects mixed/non-seed-bearing coupling",
               run_pool(ap.parse_args(["--outdir", tmp, "--pool", "st_rand0,st_plain",
                                       "--name", "st_pool_bad1"])), EXIT_REUSE_MISMATCH)
        for rho, nm in (("0.8", "st_ou8"), ("0.5", "st_ou5")):
            run_sweep(ap.parse_args(base + ["--coupling", "ou", "--rho", rho,
                                            "--seed", "0", "--name", nm]))
        expect("pool rejects differing rho",
               run_pool(ap.parse_args(["--outdir", tmp, "--pool", "st_ou8,st_ou5",
                                       "--name", "st_pool_bad2"])), EXIT_REUSE_MISMATCH)
        for s in (0, 1):
            run_sweep(ap.parse_args(base + ["--coupling", "random", "--seed", str(s),
                                            "--anchor", "--name", f"st_arand{s}"]))
        expect("pool rejects FAILed anchor member",
               run_pool(ap.parse_args(["--outdir", tmp, "--pool", "st_arand0,st_arand1",
                                       "--name", "st_pool_bad3"])), EXIT_REUSE_MISMATCH)

    print("selftest: " + ("ALL PASS" if not failures else f"FAILED {failures}"))
    return 0 if not failures else 1


# --------------------------------------------------------------------------

def _params(args) -> dict:
    return {"T": args.T, "eps": args.eps, "order": args.order, "cutoff": args.cutoff,
            "cutoff_mode": args.cutoff_mode, "max_bond": args.max_bond,
            "method": args.method, "decomp": args.decomp, "decomp_q": args.decomp_q,
            "canon": args.canon, "device": args.device, "tail_frac": args.tail_frac,
            "eta_floor": args.eta_floor}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=49)
    ap.add_argument("--coupling", default="linear",
                    choices=("linear", "uniform", "exp", "random", "ou"))
    ap.add_argument("--beta", type=float, default=0.1, help="exp profile decay rate")
    ap.add_argument("--seed", type=int, default=0, help="random/ou profile seed")
    ap.add_argument("--rho", type=float, default=0.8, help="ou profile correlation")
    ap.add_argument("--L0", type=int, default=2, help="first transition row used in fits")
    ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--order", type=int, default=2, choices=(1, 2))
    ap.add_argument("--cutoff", type=float, default=1e-8)
    ap.add_argument("--cutoff-mode", default="rel")
    ap.add_argument("--max-bond", type=int, default=500)
    ap.add_argument("--method", default="direct", choices=("direct", "zipup", "dm"))
    ap.add_argument("--decomp", default="exact", choices=("exact", "rsvd"))
    ap.add_argument("--decomp-q", type=int, default=2)
    ap.add_argument("--canon", default="quimb", choices=("quimb", "householder", "cholqr"))
    ap.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    ap.add_argument("--tail-frac", type=float, default=0.4)
    ap.add_argument("--eta-floor", type=float, default=1e-6,
                    help="roundoff-floor mask for the MATH eta fits only")
    ap.add_argument("--anchor", action="store_true",
                    help="evaluate the pre-registered acceptance gates; exit 3 on FAIL")
    ap.add_argument("--name", default="gaudin_scaling")
    ap.add_argument("--outdir", default=str(_DIR_OUT))
    ap.add_argument("--force", action="store_true", help="redo a completed configuration")
    ap.add_argument("--smoke", action="store_true", help="tiny fast config for validation")
    ap.add_argument("--check", action="store_true",
                    help="cross-validate against coupling_distributions + EDMSolver (CPU)")
    ap.add_argument("--selftest", action="store_true",
                    help="assert the acceptance exit-code paths (CPU, temp dir)")
    ap.add_argument("--pool", metavar="NAMES",
                    help="comma list of completed sweep names: per-member + pooled fits")
    ap.add_argument("--compare", metavar="COARSE,FINE",
                    help="align two sweeps differing only in eps; report deviations")
    ap.add_argument("--replot", metavar="NAME",
                    help="regenerate figures from a retrieved JSON+NPZ (local)")
    return ap


def main():
    args = build_parser().parse_args()
    if args.check:
        sys.exit(run_check(args))
    if args.selftest:
        sys.exit(run_selftest(args))
    if args.pool:
        sys.exit(run_pool(args))
    if args.compare:
        sys.exit(run_compare(args))
    if args.replot:
        sys.exit(run_replot(args))
    if args.smoke:
        args.K, args.T, args.eps, args.max_bond = 10, 1.5, 0.25, 200
        if args.name == "gaudin_scaling":
            args.name = f"smoke_{args.device}"
    sys.exit(run_sweep(args))


if __name__ == "__main__":
    main()
