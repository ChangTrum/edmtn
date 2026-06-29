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

DISTRIBUTIONS = ("linear", "uniform", "exp", "random")


def model_for(kind, g, K, *, beta=0.15, seed=0):
    """Build a GaudinModel with the named coupling profile (first-class model option)."""
    params = {}
    if kind == "exp":
        params["beta"] = beta
    elif kind == "random":
        params["seed"] = seed
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
    ap.add_argument("--decomp", default="rsvd", choices=("rsvd", "exact"))
    ap.add_argument("--decomp-q", type=int, default=2)
    ap.add_argument("--device", default="cpu", choices=("cpu", "gpu"),
                    help="run the fold on CPU (NumPy) or GPU (CuPy / A800)")
    ap.add_argument("--beta", type=float, default=0.15, help="exp-decay rate")
    ap.add_argument("--seeds", type=int, default=4, help="random-distribution realisations")
    ap.add_argument("--name", default="coupling_dist")
    ap.add_argument("--smoke", action="store_true", help="tiny fast config for validation")
    ap.add_argument("--replot", metavar="JSON",
                    help="skip compute: reload a saved results JSON and (re)draw the figures")
    args = ap.parse_args()

    if args.replot:                       # local figure regeneration from cluster JSON
        with open(args.replot) as f:
            saved = json.load(f)
        results = {k: {"rows": v["rows"], "fits": v["fits"], "D_a": v["D_a"],
                       "couplings": v["couplings"]}
                   for k, v in saved["distributions"].items()}
        _plot(results, saved["xi"], saved["meta"].get("name", "coupling_dist"))
        return

    if args.smoke:
        args.K, args.T, args.eps, args.seeds, args.max_bond = 10, 1.0, 0.25, 2, 200

    xi = args.cutoff
    common = dict(T=args.T, eps=args.eps, order=args.order, cutoff=xi,
                  cutoff_mode=args.cutoff_mode, max_bond=args.max_bond, method=args.method,
                  decomp=args.decomp, decomp_q=args.decomp_q, device=args.device, L0=args.L0)
    print(f"Coupling-distribution scaling study (Gaudin): K={args.K}, T={args.T} g^-1, "
          f"eps={args.eps} g^-1, order={args.order}, xi={xi:g}, beta={args.beta}, "
          f"seeds={args.seeds}, compress={args.method}/{args.decomp}(q{args.decomp_q}), "
          f"device={args.device}")

    results = {}   # kind -> dict(rows, fits, crit, couplings, D_a, [per_seed])
    for kind in DISTRIBUTIONS:
        t0 = time.perf_counter()
        if kind == "random":
            seed_rows, per_seed = [], []
            D_a = None
            for s in range(args.seeds):
                model = model_for("random", args.g, args.K, seed=s)
                rows, D_a = sweep(model, **common)
                seed_rows.extend(rows)
                per_seed.append({"seed": s, "crit": critical_Ls(rows, D_a)})
            rows_for_fit = seed_rows
            couplings = model_for("random", args.g, args.K, seed=0).couplings
            crit = {"per_seed": per_seed}
        else:
            model = model_for(kind, args.g, args.K, beta=args.beta)
            rows_for_fit, D_a = sweep(model, **common)
            couplings = model.couplings
            crit = critical_Ls(rows_for_fit, D_a)

        x = np.array([r["x"] for r in rows_for_fit])
        fits = {
            "eta_max": powerlaw_fit(x, [r["eta_max"] for r in rows_for_fit]),
            "eta_rms": powerlaw_fit(x, [r["eta_rms"] for r in rows_for_fit]),
            "chord":   powerlaw_fit(x, [r["chord"] for r in rows_for_fit]),
        }
        results[kind] = dict(rows=rows_for_fit, fits=fits, crit=crit,
                             couplings=couplings.tolist(), D_a=D_a)
        wall = time.perf_counter() - t0
        a, c_, r2, n = fits["eta_rms"]
        print(f"  [{kind:>7}] eta_rms ~ {c_:.3g} * x^{a:.3f}  (R2={r2:.3f}, n={n}) "
              f"  ({wall:.1f}s)")

    _report(results, xi)
    _save(results, args, xi)
    _plot(results, xi, args.name)


# --------------------------------------------------------------------------
# reporting / saving / plotting
# --------------------------------------------------------------------------

def _report(results, xi):
    print("\n=== Scaling law  eta ~ c * x^alpha  (x = g_{L+1}^2 / gbar_L^2) ===")
    print(f"{'dist':>8} | {'alpha(eta_max)':>14} {'c':>9} {'R2':>6} | "
          f"{'alpha(eta_rms)':>14} {'c':>9} {'R2':>6} | {'alpha(chord)':>12} {'R2':>6}")
    print("-" * 100)
    for kind, r in results.items():
        am, cm, r2m, _ = r["fits"]["eta_max"]
        ar, cr, r2r, _ = r["fits"]["eta_rms"]
        ac, _, r2c, _ = r["fits"]["chord"]
        print(f"{kind:>8} | {am:>14.3f} {cm:>9.3g} {r2m:>6.3f} | "
              f"{ar:>14.3f} {cr:>9.3g} {r2r:>6.3f} | {ac:>12.3f} {r2c:>6.3f}")

    print("\n=== Critical L* (pure-projection feasible) ===")
    for kind, r in results.items():
        crit = r["crit"]
        if "per_seed" in crit:
            keys = crit["per_seed"][0]["crit"].keys()
            print(f"  [{kind}] across {len(crit['per_seed'])} seeds:")
            for key in keys:
                vals = [ps["crit"][key] for ps in crit["per_seed"]]
                shown = [v for v in vals if v is not None]
                summ = f"{min(shown)}..{max(shown)}" if shown else "never"
                print(f"      {key:<20} L* = {summ}")
        else:
            print(f"  [{kind}] " + ", ".join(
                f"{k}={v if v is not None else '—'}" for k, v in crit.items()))


def _save(results, args, xi):
    _DIR_DATA.mkdir(parents=True, exist_ok=True)
    out = {"meta": vars(args), "xi": xi,
           "distributions": {k: {"fits": r["fits"], "crit": r["crit"],
                                 "D_a": r["D_a"], "couplings": r["couplings"],
                                 "rows": r["rows"]}
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
    colors = {"linear": "C0", "uniform": "C1", "exp": "C2", "random": "C3"}

    # --- Figure 1: scaling collapse (eta_rms vs x) + coupling profiles --------
    fig, (ax_c, ax_s) = plt.subplots(1, 2, figsize=(13, 5))
    for kind, r in results.items():
        ax_c.plot(range(1, len(r["couplings"]) + 1), r["couplings"], "o-", ms=3,
                  color=colors[kind], label=kind)
    ax_c.set_xlabel("sub-bath k (descending)"); ax_c.set_ylabel(r"$g_k$")
    ax_c.set_title(r"coupling profiles ($\sum g_k^2=g^2$)"); ax_c.legend(fontsize=9)

    for kind, r in results.items():
        x = np.array([row["x"] for row in r["rows"]])
        y = np.array([row["eta_rms"] for row in r["rows"]])
        a, c, r2, _ = r["fits"]["eta_rms"]
        ax_s.loglog(x, y, "o", ms=4, color=colors[kind], alpha=0.6,
                    label=fr"{kind}: $\alpha$={a:.2f}, $R^2$={r2:.2f}")
        xs = np.array([x[x > 0].min(), x.max()])
        ax_s.loglog(xs, c * xs**a, "-", color=colors[kind], lw=1.2)
    ax_s.set_xlabel(r"$x = g_{L+1}^2 / \bar g_L^2$"); ax_s.set_ylabel(r"$\eta_{\rm rms}$")
    ax_s.set_title(r"scaling law $\eta \sim c\,x^\alpha$"); ax_s.legend(fontsize=8)
    _savefig(fig, plt, f"{name}_scaling")

    # --- Figure 2: per-distribution eta & n_new vs L --------------------------
    fig, axes = plt.subplots(2, len(results), figsize=(4 * len(results), 8), squeeze=False)
    for j, (kind, r) in enumerate(results.items()):
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
