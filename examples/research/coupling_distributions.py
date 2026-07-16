"""Is the EDM scaling law ``eta ~ c * x^alpha`` intrinsic to the Gaudin model, or
an artefact of the paper's linearly-decaying couplings?

Prior finding (linear couplings ``g_k ~ (K+1-k)/K``): folding sub-baths in
descending strength, the per-fold "new-direction" strength obeys a power law
``eta ~ c * (g_{L+1}^2 / gbar_L^2)^alpha`` with ``alpha ~ 0.85``, and later
sub-baths matter progressively less -- the bond dimension stops growing (~L=15),
and eventually pure projection (no SVD) suffices (~L=24).  **Open question:** is
that driven by the *shape* of ``g_k`` or by the model itself?

We test four coupling distributions, every one normalised ``sum_k g_k^2 = g^2``
(same total coupling) and folded **strongest-first** (descending ``g_k``) so the
critical-``L`` and ``x`` are comparable across them:

  linear   g_k = g * sqrt(6K/(2K^2+3K+1)) * (K+1-k)/K   (paper)
  uniform  g_k = g / sqrt(K)                            (flat -- x = 1/L exactly)
  exp      g_k ~ exp(-beta * k)                         (fast geometric decay)
  random   g_k ~ Uniform(0, g_max)                      (disordered; several seeds)

Per fold ``L -> L+1`` we record, with ``x = g_{L+1}^2 / gbar_L^2`` the control:

  eta_max, eta_rms  -- max / rms residual ratio over bonds (new-direction strength)
  n_new(xi), n_new(sqrt(xi))  -- count of genuinely new directions per threshold
  d_chord / sqrt(D)  -- chordal distance between consecutive left subspaces

Then we fit ``eta ~ c * x^alpha`` (and the chordal distance) per distribution and
report ``c, alpha, R^2`` plus the critical ``L*`` (pure-projection-feasible), to
see whether the exponent / prefactor / critical layer move with the distribution.

Pure CPU / NumPy.

Usage
-----
    python examples/coupling_distributions.py                       # K=28 default
    python examples/coupling_distributions.py --K 30 --seeds 6 --beta 0.2
    python examples/coupling_distributions.py --smoke                # tiny, fast
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from edmtn.evolution.quimb_edm import QuimbEDM  # noqa: E402
from edmtn.evolution.separable_bath import SeparableBathEvolution  # noqa: E402
from edmtn.expansion.first_order import FirstOrderExpander  # noqa: E402
from edmtn.expansion.second_order import SecondOrderExpander  # noqa: E402
from edmtn.kernels.separable_mpo import SeparableKernelEngine  # noqa: E402
from edmtn.models import GaudinModel  # noqa: E402

_DIR_DATA = Path(__file__).resolve().parent / "data" / "coupling_dist"
_DIR_PICS = Path(__file__).resolve().parent / "pictures" / "coupling_dist"

DISTRIBUTIONS = ("linear", "uniform", "exp", "random", "ou")


def _floats(s):
    """Parse a comma-separated list of floats (argparse type)."""
    return [float(v) for v in str(s).split(",") if v.strip()]


def model_for(kind, g, K, *, beta=0.15, seed=0, rho=0.8):
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
# fold + per-bond subspace diagnostics (self-contained; current QuimbEDM API)
#
# Ports the snapshot-fold + transfer-matrix subspace comparison that used to live
# in examples/edm_incremental.py (stale after the quimb re-platform).  The fold is
# driven through QuimbEDM.fold; the diagnostics are pure NumPy on EDMMPS tensors
# (phi_up, chi_left, chi_right), with a local QR left-canonicaliser.
# --------------------------------------------------------------------------

def _to_numpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _edm_to_numpy(edm):
    """Bring an (possibly CuPy-backed) EDMMPS to host NumPy for the diagnostic."""
    from edmtn.evolution.mps_utils import EDMMPS  # noqa: PLC0415
    return EDMMPS(tensors=[_to_numpy(t) for t in edm.tensors], d=edm.d,
                  d_phys=edm.d_phys, rho0_vec=_to_numpy(edm.rho0_vec))


def fold_snapshots(model, *, T, eps, order, cutoff, cutoff_mode, max_bond,
                   method, decomp, decomp_q, device="cpu"):
    """Stream the separable fold; return ``({L: compressed EDMMPS for L=1..K}, D_a)``.

    The heavy fold/compression runs on ``device`` (``'gpu'`` -> CuPy on an A800);
    each snapshot is then moved to host NumPy because the per-bond subspace
    diagnostic is many tiny QR/SVDs that are faster on CPU than on the GPU.
    """
    ke = SeparableKernelEngine.from_model(model, T=T, eps=eps)
    expander = FirstOrderExpander() if order == 1 else SecondOrderExpander()
    ev = SeparableBathEvolution(expander=expander, compress_method=method,
                                compress_decomp=decomp, compress_decomp_q=decomp_q)
    d, d_phys, K = model.system_dim, ke.d_phys, ke.K
    n_steps = int(round(T / eps))
    n_sites = order * n_steps
    rho0 = model.initial_system_state().reshape(-1).astype(np.complex128)
    if device == "gpu":
        import cupy as cp  # noqa: PLC0415
        convert = lambda a: cp.asarray(a)  # noqa: E731
    else:
        convert = lambda a: a  # noqa: E731
    mps = QuimbEDM.from_edmmps(
        ev._build_system_mps(model, eps, n_steps, order, d, d_phys, rho0, convert))
    snaps = {}
    for k in range(K):
        mpo_sites = [convert(s) for s in ke.for_sub_bath(k).get_kernel_mpo(n_sites).site_tensors]
        mps = mps.fold(mpo_sites, cutoff=cutoff, cutoff_mode=cutoff_mode, method=method,
                       max_bond=max_bond, decomp=decomp, decomp_q=decomp_q, canon="quimb")
        snaps[k + 1] = _edm_to_numpy(mps.to_edmmps())
    return snaps, int(ke.corr.bond_dim)


def left_canonicalize(mps):
    """Left-canonical copy (sites 0..n-2 isometric over ``(phi, chi_left)``)."""
    cm = mps.copy()
    Ts = cm.tensors
    for p in range(len(Ts) - 1):
        phi, l, r = Ts[p].shape
        Q, R = np.linalg.qr(Ts[p].reshape(phi * l, r))      # rows (phi,l) -> isometry
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


def analyse_transition(mps_L, mps_L1, xi):
    """Per-bond subspace diagnostics for the fold ``L -> L+1`` (arrays over bonds)."""
    if mps_L.num_sites != mps_L1.num_sites:
        raise ValueError("L and L+1 MPS have different site counts")
    A, B = left_canonicalize(mps_L), left_canonicalize(mps_L1)
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


# --------------------------------------------------------------------------
# per-transition aggregation + power-law fit (mirrors critical_L_and_scaling)
# --------------------------------------------------------------------------

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


def powerlaw_fit(x, y):
    """Fit ``y ~ c * x^alpha`` on positive points; return ``(alpha, c, R2, n)``."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return (np.nan, np.nan, np.nan, int(m.sum()))
    lx, ly = np.log(x[m]), np.log(y[m])
    alpha, lc = np.polyfit(lx, ly, 1)
    resid = ly - (alpha * lx + lc)
    ss_res, ss_tot = float(np.sum(resid**2)), float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return (float(alpha), float(np.exp(lc)), float(r2), int(m.sum()))


def tail_fit(x, y, frac=0.5):
    """Power-law fit on the small-``x`` tail (lowest ``frac`` of x) -- tests whether
    the exponent approaches the theoretical asymptotic alpha=1 as x -> 0."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    pos = x > 0
    if pos.sum() < 6:
        return (np.nan, np.nan, np.nan, int(pos.sum()))
    thr = np.quantile(x[pos], frac)
    m = pos & (x <= thr)
    return powerlaw_fit(x[m], y[m])


def critical_Ls(rows, D_a):
    """Smallest L whose fold L->L+1 meets each pure-projection-feasibility criterion."""
    def first(pred):
        for r in rows:
            if pred(r):
                return r["L"]
        return None
    return {
        "eta_max<1e-3": first(lambda r: r["eta_max"] < 1e-3),
        "eta_max<1e-4": first(lambda r: r["eta_max"] < 1e-4),
        "n_new(sqrt xi)=0": first(lambda r: r["n_new_rtxi"] == 0),
        "n_new(xi)<=D_a": first(lambda r: r["n_new_xi"] <= D_a),
        "max dD=0": first(lambda r: r["max_dD"] == 0),
    }


# --------------------------------------------------------------------------
# run one fold sweep for a given coupling vector
# --------------------------------------------------------------------------

def sweep(model, *, T, eps, order, cutoff, cutoff_mode, max_bond, method, decomp,
          decomp_q, device, L0):
    snaps, D_a = fold_snapshots(model, T=T, eps=eps, order=order, cutoff=cutoff,
                                cutoff_mode=cutoff_mode, max_bond=max_bond,
                                method=method, decomp=decomp, decomp_q=decomp_q, device=device)
    gk, K = model.couplings, model.K
    rows = []
    for L in range(max(1, L0), K):
        rec = analyse_transition(snaps[L], snaps[L + 1], cutoff)
        agg = aggregate(rec)
        gbarL = model.effective_coupling(L)
        agg.update(L=int(L), x=float(gk[L] ** 2 / gbarL**2))
        rows.append(agg)
    return rows, int(D_a)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=28)
    ap.add_argument("--L0", type=int, default=2)
    ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--cutoff", type=float, default=1e-6)
    ap.add_argument("--cutoff-mode", default="rel")
    ap.add_argument("--max-bond", type=int, default=1024)
    ap.add_argument("--order", type=int, default=2, choices=(1, 2))
    # Track-1 compression recipe (matches the HPC benchmark): direct + cold rSVD.
    # NOTE: rSVD is approximate -> adds a little noise to the subspace diagnostic;
    # pass --decomp exact for the cleanest scaling-law fit.
    ap.add_argument("--method", default="direct", choices=("direct", "zipup", "dm"))
    # exact decomposition by default: removes rSVD noise from the small-x tail so the
    # asymptotic alpha->1 test is clean (the restrained precision bump).
    ap.add_argument("--decomp", default="exact", choices=("rsvd", "exact"))
    ap.add_argument("--decomp-q", type=int, default=2)
    ap.add_argument("--device", default="cpu", choices=("cpu", "gpu"),
                    help="run the fold on CPU (NumPy) or GPU (CuPy / A800)")
    ap.add_argument("--exp-betas", type=_floats, default=[0.05, 0.1, 0.2, 0.4],
                    help="comma-separated exp-decay rates (one curve each)")
    ap.add_argument("--ou-rhos", type=_floats, default=[0.5, 0.9],
                    help="comma-separated OU correlations (one curve each)")
    ap.add_argument("--tail-frac", type=float, default=0.4,
                    help="fraction of smallest-x points used for the asymptotic-alpha fit")
    ap.add_argument("--seeds", type=int, default=4, help="random/OU realisations (pooled)")
    ap.add_argument("--name", default="coupling_dist")
    ap.add_argument("--smoke", action="store_true", help="tiny fast config for validation")
    ap.add_argument("--replot", metavar="JSON",
                    help="skip compute: reload a saved results JSON and (re)draw the figures")
    args = ap.parse_args()

    if args.replot:                       # local figure regeneration from cluster JSON
        with open(args.replot) as f:
            saved = json.load(f)
        results = {k: {"rows": v["rows"], "fits": v["fits"], "D_a": v["D_a"],
                       "couplings": v["couplings"], "mode": v.get("mode", k)}
                   for k, v in saved["distributions"].items()}
        _plot(results, saved["xi"], saved["meta"].get("name", "coupling_dist"))
        return

    if args.smoke:
        args.K, args.T, args.eps, args.seeds, args.max_bond = 10, 1.0, 0.25, 2, 200
        args.exp_betas, args.ou_rhos = [0.1, 0.4], [0.9]

    xi = args.cutoff
    common = dict(T=args.T, eps=args.eps, order=args.order, cutoff=xi,
                  cutoff_mode=args.cutoff_mode, max_bond=args.max_bond, method=args.method,
                  decomp=args.decomp, decomp_q=args.decomp_q, device=args.device, L0=args.L0)
    print(f"Coupling-distribution scaling study (Gaudin): K={args.K}, T={args.T} g^-1, "
          f"eps={args.eps} g^-1, order={args.order}, xi={xi:g}, "
          f"exp_betas={args.exp_betas}, ou_rhos={args.ou_rhos}, seeds={args.seeds}, "
          f"compress={args.method}/{args.decomp}(q{args.decomp_q}), device={args.device}")

    # Build the sample groups: a few parameter values per mode (each its own fit);
    # random/ou pool several seeds into one fit.  ou folds in generation order
    # (correlation preserved -> x non-monotonic, no critical-L).
    g, K, S = args.g, args.K, args.seeds
    groups = [("linear", "linear", [model_for("linear", g, K)]),
              ("uniform", "uniform", [model_for("uniform", g, K)])]
    for b in args.exp_betas:
        groups.append((f"exp b={b:g}", "exp", [model_for("exp", g, K, beta=b)]))
    groups.append(("random", "random", [model_for("random", g, K, seed=s) for s in range(S)]))
    for rho in args.ou_rhos:
        groups.append((f"ou rho={rho:g}", "ou", [model_for("ou", g, K, rho=rho, seed=s)
                                                 for s in range(S)]))

    results = {}   # label -> dict(rows, fits, crit, members, couplings, D_a, mode)
    for label, mode, models in groups:
        t0 = time.perf_counter()
        all_rows, members, D_a = [], [], None
        for m in models:
            rows, D_a = sweep(m, **common)
            all_rows.extend(rows)
            members.append(critical_Ls(rows, D_a))
        x = np.array([r["x"] for r in all_rows])
        fits = {
            "eta_max": powerlaw_fit(x, [r["eta_max"] for r in all_rows]),
            "eta_rms": powerlaw_fit(x, [r["eta_rms"] for r in all_rows]),
            "eta_rms_tail": tail_fit(x, [r["eta_rms"] for r in all_rows], args.tail_frac),
            "chord":   powerlaw_fit(x, [r["chord"] for r in all_rows]),
        }
        results[label] = dict(rows=all_rows, fits=fits, members=members, mode=mode,
                              couplings=models[0].couplings.tolist(), D_a=D_a)
        a, c_, r2, n = fits["eta_rms"]
        at = fits["eta_rms_tail"][0]
        print(f"  [{label:>10}] eta_rms ~ {c_:.3g}*x^{a:.3f} (R2={r2:.3f},n={n}) "
              f"| tail a={at:.3f}  ({time.perf_counter()-t0:.1f}s)")

    _report(results, xi)
    _save(results, args, xi)
    _plot(results, xi, args.name)


# --------------------------------------------------------------------------
# reporting / saving / plotting
# --------------------------------------------------------------------------

def _report(results, xi):
    print("\n=== Scaling law  eta ~ c * x^alpha  (x = g_{L+1}^2 / gbar_L^2;  theory: alpha=1) ===")
    print(f"{'group':>11} | {'a(eta_rms)':>10} {'c':>8} {'R2':>6} | "
          f"{'a_tail(x->0)':>12} {'R2':>6} {'n':>4} | {'a(chord)':>9} | {'a(eta_max)':>10}")
    print("-" * 92)
    for label, r in results.items():
        ar, cr, r2r, _ = r["fits"]["eta_rms"]
        at, _, r2t, nt = r["fits"]["eta_rms_tail"]
        ac, _, _, _ = r["fits"]["chord"]
        am, _, _, _ = r["fits"]["eta_max"]
        print(f"{label:>11} | {ar:>10.3f} {cr:>8.3g} {r2r:>6.3f} | "
              f"{at:>12.3f} {r2t:>6.3f} {nt:>4} | {ac:>9.3f} | {am:>10.3f}")
    print("  (a_tail = slope on the smallest-x tail; theory predicts it -> 1 for every "
          "non-degenerate mode)")

    print("\n=== Critical L* (pure-projection feasible; ou unsorted -> not meaningful) ===")
    for label, r in results.items():
        members = r["members"]
        keys = members[0].keys()
        if len(members) == 1:
            print(f"  [{label}] " + ", ".join(
                f"{k}={members[0][k] if members[0][k] is not None else '—'}" for k in keys))
        else:
            print(f"  [{label}] across {len(members)} seeds:")
            for key in keys:
                shown = [mm[key] for mm in members if mm[key] is not None]
                print(f"      {key:<20} L* = {(f'{min(shown)}..{max(shown)}' if shown else 'never')}")


def _save(results, args, xi):
    _DIR_DATA.mkdir(parents=True, exist_ok=True)
    out = {"meta": vars(args), "xi": xi,
           "distributions": {k: {"fits": r["fits"], "members": r["members"],
                                 "mode": r["mode"], "D_a": r["D_a"],
                                 "couplings": r["couplings"], "rows": r["rows"]}
                             for k, r in results.items()}}
    path = _DIR_DATA / f"{args.name}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {path}")


def _plot(results, xi, name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return
    labels = list(results)
    cmap = plt.get_cmap("turbo")
    colors = {lab: cmap(i / max(1, len(labels) - 1)) for i, lab in enumerate(labels)}

    # --- Figure 1: scaling collapse (eta_rms vs x) + coupling profiles --------
    fig, (ax_c, ax_s) = plt.subplots(1, 2, figsize=(14, 5.5))
    for lab in labels:
        r = results[lab]
        ax_c.plot(range(1, len(r["couplings"]) + 1), r["couplings"], "-", lw=1.2,
                  color=colors[lab], label=lab)
    ax_c.set_xlabel("sub-bath k (fold order)"); ax_c.set_ylabel(r"$g_k$")
    ax_c.set_title(r"coupling profiles ($\sum g_k^2=g^2$)"); ax_c.legend(fontsize=7, ncol=2)

    for lab in labels:
        r = results[lab]
        x = np.array([row["x"] for row in r["rows"]])
        y = np.array([row["eta_rms"] for row in r["rows"]])
        a, c, r2, _ = r["fits"]["eta_rms"]
        at = r["fits"]["eta_rms_tail"][0]
        ax_s.loglog(x, y, "o", ms=3, color=colors[lab], alpha=0.55,
                    label=fr"{lab}: $\alpha$={a:.2f} (tail {at:.2f})")
        xs = np.array([x[x > 0].min(), x.max()])
        ax_s.loglog(xs, c * xs**a, "-", color=colors[lab], lw=1.0)
    # theory slope-1 guide
    xall = np.concatenate([[row["x"] for row in r["rows"]] for r in results.values()])
    xall = xall[xall > 0]
    xr = np.array([xall.min(), xall.max()])
    ax_s.loglog(xr, xr / xr.max() * 1e-2, "k--", lw=1.0, label=r"slope 1 (theory)")
    ax_s.set_xlabel(r"$x = g_{L+1}^2 / \bar g_L^2$"); ax_s.set_ylabel(r"$\eta_{\rm rms}$")
    ax_s.set_title(r"scaling law $\eta \sim c\,x^\alpha$  (theory $\alpha=1$)")
    ax_s.legend(fontsize=6, ncol=2)
    _savefig(fig, plt, f"{name}_scaling")

    # --- Figure 2: per-distribution eta & n_new vs L (one representative per mode) --
    reps, seen = [], set()
    for lab in labels:
        mode = results[lab].get("mode", lab)
        if mode not in seen:
            seen.add(mode); reps.append(lab)
    fig, axes = plt.subplots(2, len(reps), figsize=(4 * len(reps), 8), squeeze=False)
    for j, kind in enumerate(reps):
        r = results[kind]
        L = np.array([row["L"] for row in r["rows"]])
        a0, a1 = axes[0][j], axes[1][j]
        a0.semilogy(L, [row["eta_max"] for row in r["rows"]], "o-", ms=3, label=r"$\eta_{\max}$")
        a0.semilogy(L, [row["eta_rms"] for row in r["rows"]], "s-", ms=3, label=r"$\eta_{\rm rms}$")
        a0.semilogy(L, [row["chord"] for row in r["rows"]], "^-", ms=3, label=r"$d_{\rm ch}/\sqrt{D}$")
        a0.axhline(xi, color="k", ls="--", lw=1)
        a0.set_title(f"{kind}"); a0.set_xlabel("L"); a0.set_ylabel("diagnostic")
        if j == 0:
            a0.legend(fontsize=8)
        a1.plot(L, [row["n_new_xi"] for row in r["rows"]], "o-", ms=3, label=r"$n_{\rm new}(\xi)$")
        a1.plot(L, [row["n_new_rtxi"] for row in r["rows"]], "s-", ms=3, label=r"$n_{\rm new}(\sqrt{\xi})$")
        a1.axhline(r["D_a"], color="k", ls="--", lw=1, label=f"$D_a$={r['D_a']}")
        a1.set_xlabel("L"); a1.set_ylabel("new directions")
        if j == 0:
            a1.legend(fontsize=8)
    _savefig(fig, plt, f"{name}_perL")


def _savefig(fig, plt, stem):
    try:
        fig.tight_layout()
    except Exception:  # noqa: BLE001
        pass
    _DIR_PICS.mkdir(parents=True, exist_ok=True)
    png = _DIR_PICS / f"{stem}.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"saved {png}")


if __name__ == "__main__":
    main()
