"""How far is the EDM representation from the information-theoretic floor?

Anchors the diagnostic on the paper's framework. For each (L, tau) we expose the
**full Schmidt spectrum** of bond tau after folding L sub-baths (build the fold with a
tiny cutoff so almost nothing is discarded), then compare:

  * D_tau^EDM(L)  -- the carried bond dimension (what the representation stores);
  * D_tau^eff(L)  -- the minimal bond to hit accuracy xi, i.e. the smallest D_trunc with
        eps_loss(D_trunc) = sqrt(sum_{i>D_trunc} sigma_i^2 / sum_i sigma_i^2) < xi;
  * redundancy  R(L, tau) = 1 - D_tau^eff / D_tau^EDM.

Aggregate vs Theorem 1: N_actual = max_tau D_tau^eff(K)  vs  N_T = d^2 T / eps (the
Theorem-1 upper bound) vs the paper's hard cap D_c = 400. The gaps quantify how loose
the bound is and how much the paper over-provisioned.

For a few typical bonds it plots eps_loss(D_trunc) vs D_trunc, marking D_eff (the
free/lossy boundary) and D_c = 400 -- i.e. "how much the paper wasted, how much is
recoverable". Pure Track 1 (compressed quimb fold); CPU by default, --device gpu for CuPy.
The fold mirrors examples/studies/coupling_distributions.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from edmtn.evolution.quimb_edm import QuimbEDM
from edmtn.evolution.separable_bath import SeparableBathEvolution
from edmtn.expansion.first_order import FirstOrderExpander
from edmtn.expansion.second_order import SecondOrderExpander
from edmtn.kernels.separable_mpo import SeparableKernelEngine
from edmtn.models import GaudinModel

_DIR_DATA = Path(__file__).resolve().parent / "data" / "redundancy"
_DIR_PICS = Path(__file__).resolve().parent / "pictures" / "redundancy"


# --------------------------------------------------------------------------
# fold + bond Schmidt spectra (self-contained; mirrors coupling_distributions.py)
# --------------------------------------------------------------------------

def _to_numpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def fold_snapshots(model, *, T, eps, order, cutoff, max_bond, method, decomp, device):
    """Stream the separable fold (generous representation); return {L: EDMMPS (host)}."""
    ke = SeparableKernelEngine.from_model(model, T=T, eps=eps)
    expander = FirstOrderExpander() if order == 1 else SecondOrderExpander()
    ev = SeparableBathEvolution(expander=expander, compress_method=method,
                                compress_decomp=decomp)
    from edmtn.evolution.mps_utils import EDMMPS  # noqa: PLC0415
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
        mpo = [convert(s) for s in ke.for_sub_bath(k).get_kernel_mpo(n_sites).site_tensors]
        mps = mps.fold(mpo, cutoff=cutoff, cutoff_mode="rel", method=method,
                       max_bond=max_bond, decomp=decomp, canon="quimb")
        e = mps.to_edmmps()
        snaps[k + 1] = EDMMPS(tensors=[_to_numpy(t) for t in e.tensors], d=e.d,
                              d_phys=e.d_phys, rho0_vec=_to_numpy(e.rho0_vec))
    return snaps, n_sites


def bond_spectra(mps):
    """Schmidt spectra {tau: sigma^2 (desc)} at every internal bond, via the
    right-environment density matrix of a left-canonical copy (its eigenvalues are
    the squared Schmidt values)."""
    cm = mps.copy()
    Ts = cm.tensors
    for p in range(len(Ts) - 1):                       # left-canonicalise
        phi, l, r = Ts[p].shape
        Q, Rm = np.linalg.qr(Ts[p].reshape(phi * l, r))
        Ts[p] = Q.reshape(phi, l, Q.shape[1])
        Ts[p + 1] = np.einsum("kr,prs->pks", Rm, Ts[p + 1], optimize=True)
    spectra = {}
    R = np.eye(Ts[-1].shape[2], dtype=np.complex128)
    for p in range(len(Ts) - 1, -1, -1):
        Bp = Ts[p]
        Tn = np.einsum("plr,rs->pls", Bp, R, optimize=True)
        R = np.einsum("pls,pms->lm", Tn, np.conj(Bp), optimize=True)
        if p >= 1:
            w = np.clip(np.linalg.eigvalsh(R).real, 0.0, None)
            spectra[p] = np.sort(w)[::-1]              # sigma^2 descending
    return spectra


# --------------------------------------------------------------------------
# information-theoretic metrics on a spectrum
# --------------------------------------------------------------------------

def eps_loss_curve(s2):
    """eps_loss(D_trunc) for D_trunc = 0..len, the relative discarded weight."""
    tot = float(s2.sum())
    if tot <= 0:
        return np.zeros(len(s2) + 1)
    tail = tot - np.concatenate([[0.0], np.cumsum(s2)])  # tail[D] = weight beyond D kept
    return np.sqrt(np.clip(tail, 0.0, None) / tot)


def d_eff(s2, xi):
    """Smallest D_trunc with eps_loss(D_trunc) < xi (the minimal bond at accuracy xi)."""
    tot = float(s2.sum())
    if tot <= 0:
        return 0
    tail = tot - np.cumsum(s2)                 # tail[k] = eps_loss^2*tot after keeping k+1
    below = np.nonzero(tail < (xi * xi * tot))[0]
    return int(below[0]) + 1 if below.size else len(s2)


# --------------------------------------------------------------------------
# study
# --------------------------------------------------------------------------

def run(model, *, T, eps, order, xi, build_cutoff, build_max_bond, method, decomp, device):
    snaps, n_sites = fold_snapshots(model, T=T, eps=eps, order=order, cutoff=build_cutoff,
                                    max_bond=build_max_bond, method=method, decomp=decomp,
                                    device=device)
    K = model.K
    per_L = {}            # L -> {tau: (D_EDM, D_eff, R)}
    for L, mps in snaps.items():
        spec = bond_spectra(mps)
        per_L[L] = {tau: (len(s2), d_eff(s2, xi),
                          1.0 - d_eff(s2, xi) / max(1, len(s2)))
                    for tau, s2 in spec.items()}
    spectra_K = bond_spectra(snaps[K])         # full spectra at L=K (for the curves)
    return per_L, spectra_K, n_sites


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=49)
    ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--order", type=int, default=2, choices=(1, 2))
    ap.add_argument("--xi", type=float, default=1e-6, help="target accuracy for D_eff")
    ap.add_argument("--build-cutoff", type=float, default=1e-12,
                    help="tiny cutoff so the fold exposes the near-full spectrum")
    ap.add_argument("--build-max-bond", type=int, default=512)
    ap.add_argument("--method", default="direct", choices=("direct", "zipup", "dm"))
    ap.add_argument("--decomp", default="exact", choices=("exact", "rsvd"))
    ap.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    ap.add_argument("--coupling", default="linear", help="coupling profile (paper: linear)")
    ap.add_argument("--Dc", type=int, default=400, help="paper's bond cap to mark")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--name", default="redundancy")
    ap.add_argument("--replot", metavar="JSON")
    args = ap.parse_args()

    if args.replot:
        with open(args.replot) as f:
            saved = json.load(f)
        _plot(saved, args.name if args.name != "redundancy" else saved["meta"]["name"])
        return
    if args.smoke:
        args.K, args.T, args.eps, args.build_max_bond = 12, 1.0, 0.25, 128

    model = GaudinModel(g=args.g, K=args.K, coupling=args.coupling)
    N_T = model.system_dim ** 2 * args.T / args.eps    # Theorem-1 upper bound (d^2 T/eps)
    print(f"Redundancy / lower-bound study (Gaudin {args.coupling}): K={args.K}, T={args.T}, "
          f"eps={args.eps}, xi={args.xi:g}, build(cutoff={args.build_cutoff:g},"
          f"max_bond={args.build_max_bond}), N_T=d^2 T/eps={N_T:g}, D_c={args.Dc}")

    t0 = time.perf_counter()
    per_L, spectra_K, n_sites = run(model, T=args.T, eps=args.eps, order=args.order, xi=args.xi,
                                    build_cutoff=args.build_cutoff,
                                    build_max_bond=args.build_max_bond, method=args.method,
                                    decomp=args.decomp, device=args.device)
    wall = time.perf_counter() - t0

    atK = per_L[args.K]
    taus = sorted(atK)
    D_EDM = {t: atK[t][0] for t in taus}
    D_eff = {t: atK[t][1] for t in taus}
    Rk = {t: atK[t][2] for t in taus}
    N_actual = max(D_eff.values())
    tau_star = max(D_eff, key=D_eff.get)
    # Theorem-2 per-bond bound D_tau <= d^2 * tau (paper Eq.14, per PHYSICAL step).
    # site tau maps to physical step tau/order; the both-sides (Schmidt) bound also caps
    # by the future block: d^2 * min(tau, n_sites-tau)/order.
    d2 = model.system_dim ** 2
    bound = {t: d2 * min(t, n_sites - t) / args.order for t in taus}   # tent (both sides)
    bound_mono = {t: d2 * t / args.order for t in taus}               # paper's literal d^2*tau
    over_eff = max(D_eff[t] / bound[t] for t in taus if bound[t] > 0)
    over_edm = max(D_EDM[t] / bound[t] for t in taus if bound[t] > 0)
    over_mono = max(D_eff[t] / bound_mono[t] for t in taus if bound_mono[t] > 0)
    print(f"  folded K={args.K} (order {args.order}) in {wall:.1f}s; bond profile at L=K:")
    print(f"    max carried  D_EDM = {max(D_EDM.values())} (at tau={max(D_EDM, key=D_EDM.get)})")
    print(f"    N_actual = max_tau D_eff(K) = {N_actual} (at tau={tau_star})")
    print(f"    Theorem-2 per-bond bound d^2*min(tau,N-tau)[/order]: "
          f"max D_eff/bound = {over_eff:.2f}, max D_EDM/bound = {over_edm:.2f}")
    print(f"    vs paper's literal d^2*tau:   max D_eff/(d^2*tau) = {over_mono:.2f}  "
          f"({'WITHIN bound' if over_mono <= 1.0 else 'EXCEEDS bound -> Thm-2 gap?'})")
    print(f"    (context) N_T = d^2 T/eps = {N_T:g} (this run, N={args.T/args.eps:g} steps); "
          f"paper Gaudin uses eps=0.03,T=15 -> N=500, N_T=2000, cap D_c={args.Dc}")
    print(f"    mean redundancy R over bonds at L=K = {np.mean(list(Rk.values())):.3f}")

    out = {"meta": vars(args), "N_T": N_T, "N_actual": int(N_actual), "Dc": args.Dc,
           "n_sites": n_sites, "tau_star": int(tau_star), "d2": d2,
           "over_eff": float(over_eff), "over_edm": float(over_edm), "over_mono": float(over_mono),
           "atK": {int(t): {"D_EDM": int(D_EDM[t]), "D_eff": int(D_eff[t]), "R": float(Rk[t])}
                   for t in taus},
           "R_LT": {int(L): {int(t): float(per_L[L][t][2]) for t in per_L[L]} for L in per_L},
           # full spectra at a few representative bonds (for the eps_loss curves)
           "spectra_K": {int(t): spectra_K[t].tolist()
                         for t in _pick_taus(taus, D_EDM)}}
    _DIR_DATA.mkdir(parents=True, exist_ok=True)
    path = _DIR_DATA / f"{args.name}.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"saved {path}")
    _plot(out, args.name)


def _pick_taus(taus, D_EDM):
    """A few representative bonds: quartiles of the chain + the max-bond bond."""
    if not taus:
        return []
    qs = [taus[int(f * (len(taus) - 1))] for f in (0.1, 0.35, 0.6, 0.85)]
    qs.append(max(D_EDM, key=D_EDM.get))
    return sorted(set(qs))


def _plot(out, name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return
    xi = out["meta"]["xi"]; Dc = out["Dc"]; N_T = out["N_T"]; N_actual = out["N_actual"]
    atK = {int(t): v for t, v in out["atK"].items()}
    taus = sorted(atK)

    # --- Figure 1: eps_loss(D_trunc) for representative bonds; mark D_eff & D_c -----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    spec = {int(t): np.asarray(s) for t, s in out["spectra_K"].items()}
    cmap = plt.get_cmap("viridis")
    sk = sorted(spec)
    for i, t in enumerate(sk):
        curve = eps_loss_curve(spec[t])
        D = np.arange(len(curve))
        col = cmap(i / max(1, len(sk) - 1))
        de = atK[t]["D_eff"]
        ax.semilogy(D, np.clip(curve, 1e-18, None), "-", color=col,
                    label=fr"$\tau$={t}: $D_{{\rm eff}}$={de}, $D_{{\rm EDM}}$={atK[t]['D_EDM']}")
        ax.plot(de, max(xi, 1e-18), "o", color=col, ms=6)        # D_eff marker
    ax.axhline(xi, color="k", ls=":", lw=1, label=fr"$\xi$={xi:g}")
    ax.axvline(Dc, color="r", ls="--", lw=1.3, label=fr"paper $D_c$={Dc}")
    ax.set_xlabel(r"$D_{\rm trunc}$ (kept singular values)")
    ax.set_ylabel(r"$\epsilon_{\rm loss}(D_{\rm trunc})$")
    ax.set_title("free vs lossy zone: $D_{\\rm eff}$ (dots) is all you need; "
                 f"$D_c$={Dc} is what the paper carries")
    ax.legend(fontsize=8)
    _savefig(fig, plt, f"{name}_epsloss")

    # --- Figure 2: bond profile vs the Theorem-2 per-bond bound d^2*tau ------------
    d2 = out.get("d2", 4); order = out["meta"]["order"]; nsites = out["n_sites"]
    tent = [d2 * min(t, nsites - t) / order for t in taus]      # both-sides (Schmidt) bound
    mono = [d2 * t / order for t in taus]                       # paper's literal d^2*tau
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(taus, [atK[t]["D_EDM"] for t in taus], "s-", ms=3, label=r"$D_\tau^{\rm EDM}$ (carried)")
    ax.plot(taus, [atK[t]["D_eff"] for t in taus], "o-", ms=3,
            label=fr"$D_\tau^{{\rm eff}}$ (need @ $\xi$={xi:g})")
    ax.plot(taus, tent, "k-", lw=1.4, label=r"Thm 2: $d^2\min(\tau,N{-}\tau)$")
    ax.plot(taus, mono, "k:", lw=1.0, label=r"$d^2\tau$ (paper's literal)")
    ax.axhline(Dc, color="r", ls="--", lw=1.0, label=fr"$D_c$={Dc} (paper cap, N=500 run)")
    ax.set_xlabel(r"bond $\tau$ (site index)"); ax.set_ylabel("bond dimension")
    ax.set_title(f"bond profile vs Theorem-2 bound (order {order}); "
                 f"$D_{{\\rm eff}}$/bound$_{{\\max}}$={out.get('over_eff', 0):.1f}")
    ax.legend(fontsize=8)
    _savefig(fig, plt, f"{name}_profile")

    # --- Figure 3: redundancy heatmap R(L, tau) -----------------------------------
    R_LT = {int(L): {int(t): v for t, v in d.items()} for L, d in out["R_LT"].items()}
    Ls = sorted(R_LT); allt = sorted({t for L in R_LT for t in R_LT[L]})
    M = np.full((len(Ls), len(allt)), np.nan)
    for i, L in enumerate(Ls):
        for j, t in enumerate(allt):
            if t in R_LT[L]:
                M[i, j] = R_LT[L][t]
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1,
                   extent=[allt[0], allt[-1], Ls[0], Ls[-1]])
    fig.colorbar(im, ax=ax, label=r"redundancy $R=1-D_{\rm eff}/D_{\rm EDM}$")
    ax.set_xlabel(r"bond $\tau$"); ax.set_ylabel("sub-baths folded $L$")
    ax.set_title(r"redundancy $R(L,\tau)$")
    _savefig(fig, plt, f"{name}_redundancy")


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
