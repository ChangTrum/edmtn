# Achieving global second order for a time-dependent, dissipative EDM

## Why this note exists

The spin-boson and Gaudin pipelines inherit their time discretisation from the EDM paper.
The Dicke pipeline does not: it is the first model in `edmtn` whose interaction-picture
coupling is **time dependent on both sides** (system *and* bath), and the first that carries
**local Lindblad dissipation**. Neither feature is covered by the paper's analysis, so the
order of the resulting scheme has to be derived and measured here rather than cited.

The conclusion is that a genuinely **globally second-order** scheme is available, that it
needs two ingredients beyond the existing second-order algebraic split, and that it leaves
the EDM algebra (the `A`, `D` and Kraus tensors) **completely unchanged** — only the time
argument at which they are evaluated, and the placement of the dissipative half-steps, differ.

## 1. What one physical step has to approximate

Work in the interaction picture of `H_0 = omega_c a^dag a + sum_k (omega_k/2) sigma_{k,z}`.
The state obeys

```
d rho / dt = ( Hm(t) + LD ) rho,
Hm(t) rho  = -i [ H_I(t), rho ],
H_I(t)     = sum_k S(t) (x) B_k(t) = S(t) (x) B(t),      B(t) = sum_k B_k(t),
S(t)       = a e^{-i omega_c t} + a^dag e^{i omega_c t},
B_k(t)     = g_k [ cos(omega_k t) sigma_{k,x} - sin(omega_k t) sigma_{k,y} ],
LD         = kappa D[a] + sum_k ( w_k D[sigma_k^+] + gamma_k D[sigma_k^-] + (gamma_k^phi/2) D[sigma_{k,z}] ),
D[L] rho   = L rho L^dag - (1/2) { L^dag L, rho }.
```

The single system operator `S(t)` couples to every sub-bath, so the bath is separable: each
`B_k` acts on its own spin and the sub-baths are folded one at a time. All bath dissipators
above are **local to one spin**; a collective jump operator would destroy separability and
invalidate the whole transfer-tensor construction.

`LD` is time independent in the interaction picture, because each of these jump operators
picks up only a phase under `H_0` (`D[a e^{-i omega_c t}] = D[a]`,
`D[e^{i omega_k t} sigma_k^+] = D[sigma_k^+]`, `D[sigma_{k,z}]` invariant). This is a property
of *these* jump operators, not a general fact.

The exact propagator over one physical step `[t_{n-1}, t_n]`, `t_n = n eps`, is the
time-ordered exponential of `Hm(t) + LD`. A discretisation is *globally* `p`-th order when its
per-step error is `O(eps^{p+1})`.

## 2. Two independent sources of error

Writing `Q_n` for the map the pipeline actually applies over step `n`, the error splits into
two contributions that must **both** be `O(eps^3)`:

1. **Freezing.** `Hm(t)` is sampled once per step and held fixed; the sampling point decides
   the quadrature order of the Magnus/Dyson series.
2. **Splitting.** The frozen Hamiltonian part and the dissipative part are applied as separate
   factors; their interleaving decides the commutator error.

The existing second-order algebraic split contributes no error at this order at all:

```
A_1 = I + c_1 eps Hm,   A_2 = I + c_2 eps Hm,   c_1 = (1-i)/2,  c_2 = (1+i)/2
A_2 A_1 = I + eps Hm + (eps^2/2) Hm^2      exactly (c_1 + c_2 = 1, c_1 c_2 = 1/2)
        = exp(eps Hm) + O(eps^3).
```

So the polynomial is already good to `O(eps^3)`. Whatever limits the scheme to first order
comes from (1) and (2), never from the Taylor order.

## 3. Ingredient one — sample at the midpoint

Let `t_n^* ` be the point at which `Hm` is frozen. Expanding the Dyson series of the exact
propagator against `exp(eps Hm(t_n^*))` gives

| sampling point | per-step error | global order |
|---|---|---|
| right endpoint `t_n^* = n eps` | `-(eps^2/2) dHm/dt + O(eps^3)` | 1 |
| left endpoint `t_n^* = (n-1) eps` | `+(eps^2/2) dHm/dt + O(eps^3)` | 1 |
| **midpoint `t_n^* = (n - 1/2) eps`** | `O(eps^3)` | **2** |

The midpoint is the only choice among the three for which the first-order Magnus term
`Omega_1 = integral Hm(s) ds` is reproduced to `O(eps^3)` (midpoint quadrature), while the
second Magnus term `Omega_2 = -(1/2) integral integral [Hm(s_1), Hm(s_2)]` is `O(eps^3)`
on its own because the commutator vanishes linearly as `s_1 -> s_2`.

**Both sides must use the same `t_n^*`.** The system superoperators `S^phi` are built from
`S(t_n^*)` and the bath transfer tensors from `B_k(t_n^*)`; for `order = 2` the two algebraic
sub-steps of the same physical step share that single midpoint. A mismatch between the two
sides is not a loss of order — it is the wrong Hamiltonian.

## 4. Ingredient two — Strang placement of the dissipators

Both dissipative maps are exact exponentials of time-independent generators, hence exact
one-parameter semigroups:

```
M(dt) = exp(dt kappa D[a])          (cavity, acting on the system Liouville space)
D_k(dt) = exp(dt L_k)               (bath spin k, acting on its 4-dim Liouville space)
M(h) M(h) = M(2h),   D_k(h) D_k(h) = D_k(2h)     exactly.
```

Write `h = eps/2` for the half-step, and collect **all** the half-step channels into one
full-space map

```
E_h = M(h) (x) D_1(h) (x) ... (x) D_K(h) = exp(h LD),
```

so that `E_h` is the half-step counterpart of the complete `LD` of Section 1 — the cavity
channel `M(h)` alone is not, and comparing `M(h)`-only products against `exp(eps(Hm + LD))`
would be an inconsistent statement. `E_h` factorises across the system and the `K` spins
because those generators act on disjoint tensor factors and commute.

Let `F_1`, `F_2` be the **full-space** (system-and-bath) interaction factors of the two
algebraic sub-steps,

```
F_j = I + c_j eps Hm(t_n^*),      c_1 = (1-i)/2,  c_2 = (1+i)/2,
```

so that `F_2 F_1 = exp(eps Hm(t_n^*)) + O(eps^3)`. The two candidate placements are

```
naive   Q_n = E_h F_2 E_h F_1        (one half-channel inside each algebraic sub-step)
Strang  Q_n = E_h F_2 F_1 E_h        (the two halves at the two ends of the physical step)
```

Both contain two exact half-channel factors whose durations sum to exactly `eps`, so neither
double-counts nor loses dissipation; but only the second has the right operator ordering:

```
naive  - exp(eps (Hm + LD)) = ((1-i)/4) eps^2 [LD, Hm] + O(eps^3)      -> global order 1
Strang - exp(eps (Hm + LD)) = O(eps^3)                                 -> global order 2
```

The naive placement is therefore first order whenever `[LD, Hm] != 0`, i.e. always in the
dissipative case. It degenerates correctly only when all rates vanish, where `E_h = I`
and the two placements coincide.

**Discretisation map, order 2** (rightmost factor acts first):

```
Q_n = E_h  F_{2,n}  F_{1,n}  E_h ,        h = eps/2,   t_n^* = (n - 1/2) eps
```

**Discretisation map, order 1** (rightmost factor acts first):

```
Q_n = E_h  F_{1,n}  E_h ,                 h = eps/2,   t_n^* = (n - 1/2) eps
      F_{1,n} = I + eps Hm(t_n^*)
```

Order 1 stays globally first order — the polynomial `I + eps Hm` is only `O(eps^2)` accurate —
but the symmetric placement keeps one uniform rule and the exact per-step dissipative
increment. Note `h = eps/2` at **both** orders: the two halves belong to the physical step,
not to the algebraic sub-steps, so `h` does not scale with `order`.

Across consecutive steps the trailing and leading halves merge exactly into one full-step
channel,

```
Q_N ... Q_1 = E_h F_2 F_1 E(eps) F_2 F_1 E(eps) ... E(eps) F_2 F_1 E_h,      E(eps) = exp(eps LD)
```

so the dissipative time increment consumed per physical step is exactly `eps` — none is
double-counted and none is lost. This is **not** the statement that the product equals
`E(N eps)`: the `E(eps)` factors remain separated by interaction factors and cannot be
combined further. Only in the non-interacting limit (`g_k = 0`, all `F_j = I`) do they
collapse to `E(N eps)`. With interaction the composite map is a Strang-interleaved
approximation, second-order accurate by the estimate above.

## 5. How this lands on the MPS sites

Two orderings coexist in the code and must not be confused:

* **storage order** — `EDMMPS.tensors[0]` and `KernelMPO.site_tensors[0]` are the **newest**
  site; index `p` runs newest to oldest;
* **action order** — the state is `M_0 M_1 ... M_{n-1} rho0_vec`, so the **rightmost
  (oldest)** factor acts first.

With `n_sites = order * n_steps` and the sub-step index `g = n_sites - p` (oldest `g = 1`),
the physical step is `n = (g - 1) // order + 1` and the algebraic sub-step is
`sub = (g - 1) % order` (`sub = 0` is the earlier one). Therefore, for `order = 2`:

| sub-step | `g` parity | family | system site tensor | bath site tensor |
|---|---|---|---|---|
| earlier (`sub = 0`) | odd | `S_1` | `S^phi_1(t_n^*) M(h)` | `A_k(t_n^*) D_k(h)` |
| later (`sub = 1`) | even | `S_2` | `M(h) S^phi_2(t_n^*)` | `D_k(h) A_k(t_n^*)` |

and for `order = 1` the single site carries `M(h) S^phi(t_n^*) M(h)` (system) and
`D_k(h) A_k(t_n^*) D_k(h)` (bath). The site-level factors are exactly the tensor factors of
`E_h`: the system side carries `M(h)`, the site of sub-bath `k` carries `D_k(h)`, and their
product across the system and all `K` spins is `E_h`. Multiplying the sites of one physical
step in *action* order reproduces `Q_n` of Section 4 exactly.

**The complex coefficients belong to the system side only.** `S^phi_1` and `S^phi_2` are the
first-order system superoperators with the non-identity entries (`phi != 0`) scaled by `c_1`
and `c_2` respectively; the identity entry `phi = 0` is never scaled. The **bath transfer
tensor is the same object in both sub-steps**, `A_k(t_n^*)`, with no `c_j` factor anywhere —
the `c_j` enter the full-space factors `F_j` of Section 4 solely through the system family.
Multiplying `c_j` into the bath transfer as well would square them and is a silent error.

The bath transfer array is stored oldest-first, `transfer[g - 1]` for `g = 1 .. n_sites`, so
the kernel's newest-first site list is its reverse:

```
site_tensors[p] = operatorised( transfer[n_sites - 1 - p] ),      p = 0 .. n_sites - 1
```

with the newest site's left lateral index fixed to `0` and the oldest site's right lateral
index contracted with the bath Bloch vector `r_k`.

That the per-site families reconstruct the full-space step at all rests on the superoperator
factorisation

```
S^+ (x) B^-  +  S^- (x) B^+  =  -i [ S (x) B , . ],
S^+ = (1/2){S, .},  S^- = -i[S, .],  B^+ = (1/2){B, .},  B^- = -i[B, .],
```

which fixes the `phi = 2a-1 <-> (S^+, B^-)`, `phi = 2a <-> (S^-, B^+)` pairing used
throughout `edmtn`.

## 6. Numerical stability of the bath channel

The bath generator for spin `k`, in the Pauli basis `(r_0, r_x, r_y, r_z)`, is

```
Gamma_1 = w_k + gamma_k,      Gamma_2 = Gamma_1 / 2 + gamma_k^phi
L_k = [[0,0,0,0], [0,-Gamma_2,0,0], [0,0,-Gamma_2,0], [w_k - gamma_k, 0, 0, -Gamma_1]]
D_k(dt) = exp(dt L_k) = [[1,0,0,0], [0,e^{-Gamma_2 dt},0,0], [0,0,e^{-Gamma_2 dt},0],
                         [d_k, 0, 0, e^{-Gamma_1 dt}]]
```

The affine entry must be evaluated as

```
d_k = (w_k - gamma_k) * phi1(Gamma_1, dt),      phi1(G, dt) = -expm1(-G dt) / G,  phi1(0, dt) = dt
```

rather than `(w_k - gamma_k) (1 - e^{-Gamma_1 dt}) / Gamma_1`. The naive form is `0/0` at
`Gamma_1 = 0` — which is the **default** configuration, all rates zero — and suffers
cancellation for small `Gamma_1`. Pure dephasing (`Gamma_1 = 0`, `gamma^phi > 0`) is a
distinct regime and must be covered separately: there `d_k = 0` but `Gamma_2 > 0`.

## 7. Verification record

**Symbolic (Mathematica, generic symbolic matrices, no numerical substitution).**

| statement | result |
|---|---|
| `A_2 A_1 = I + eps Hm + (eps^2/2) Hm^2` exactly | identity |
| exact propagator `-` `exp(eps Hm(midpoint))` | leading `eps^3` |
| exact propagator `-` `exp(eps Hm(right endpoint))` | leading `eps^2`, coefficient `-dHm/dt / 2` |
| exact propagator `-` `exp(eps Hm(left endpoint))` | leading `eps^2` |
| `E_h F_2 F_1 E_h - exp(eps(Hm + LD))` | leading `eps^3` |
| `E_h F_2 E_h F_1 - exp(eps(Hm + LD))` | leading `eps^2`, coefficient `((1-i)/4) [LD, Hm]` |
| `M(h) M(h) = M(2h)`, `D_k(h) D_k(h) = D_k(2h)` | identity |
| `S^+ (x) B^- + S^- (x) B^+ = -i[S (x) B, .]` | identity |
| closed-form `D_k(dt)` vs `MatrixExp[dt L_k]` | identity |
| truncated-Fock Kraus sum vs the closed-form cavity channel | identity |
| cavity channel trace preserving on the truncated space | identity |

**Numerical exploratory check.** Dense full-space integration, no tensor network involved.

* system: `n_fock = 3`, `K = 2`, deliberately asymmetric — `omega_c = 1.0`,
  `omega_k = (0.8, 1.3)`, `g_k = (0.35, 0.22)`, `kappa = 0.13`, `w_k = (0.05, 0.02)`,
  `gamma_k = (0.11, 0.07)`, `gamma_k^phi = (0.04, 0.09)`;
* initial state: truncated coherent cavity (`alpha = 0.3 + 0.2i`, renormalised) tensor two
  off-axis Bloch spins `(0.2, -0.3, 0.5)` and `(0.0, 0.4, -0.4)`;
* final time `T = 0.8`;
* reference: `scipy.integrate.solve_ivp`, DOP853, `rtol = 1e-13`, `atol = 1e-15`, applied to
  the continuous-time master equation — **not** to any discretised map;
* error norm: maximum elementwise absolute deviation of the vectorised **full system-and-bath**
  density matrix at `T` (the exploratory check runs on the full space; the runnable regression
  test fits on the pipeline's final **reduced** density matrix instead);
* fit: least squares of `log(error)` against `log(eps)` over `N = 16, 32, 64, 128, 256`
  (5 points), all well above the reference error floor (`~1e-13`).

| scheme | measured order | error at `N = 256` |
|---|---|---|
| order 2, right endpoint, no dissipation | 1.00 | 1.70e-04 |
| order 2, left endpoint, no dissipation | 1.00 | 1.70e-04 |
| **order 2, midpoint, no dissipation** | **2.00** | 2.71e-07 |
| order 1, midpoint, no dissipation | 1.00 | 1.77e-04 |
| order 2, midpoint, naive dissipator placement | 1.00 | 1.77e-05 |
| **order 2, midpoint, Strang placement** | **2.00** | 2.54e-07 |
| order 2, right endpoint, Strang placement | 1.00 | 1.52e-04 |
| order 1, midpoint, full-step dissipator | 1.00 | 1.46e-04 |

Both ingredients are necessary: dropping either one returns the scheme to global first order.

## 8. What is and is not claimed

* **Claimed, under the stated conditions.** With midpoint sampling and Strang placement,
  `expansion_order = 2` is globally second order in `eps`, in the closed and the dissipative
  case alike. The claim is supported by a symbolic per-step error expansion and by a measured
  convergence order against an independent continuous-time reference — not by analogy with
  the paper. The conditions are: a **fixed finite Fock truncation** `n_fock` held constant as
  `eps` is refined; a smooth, bounded time-dependent generator; an interaction-picture
  dissipator that is **time independent** (true for these jump operators, not in general);
  `compress = False`; the density-matrix norm stated in Section 7; and an `eps` range well
  above the reference solver's own error floor.
* **Claimed.** `expansion_order = 1` is globally first order. The dissipative time increment
  consumed per physical step is exactly `eps` at both orders.
* **Not claimed.** Nothing beyond second order. Third order would require the Magnus
  commutator terms, which are not implemented.
* **Not claimed.** Any statement about the *compressed* pipeline's accuracy. The order above
  is the discretisation order at `compress = False`. Truncation is a **separate error source**,
  not an additive correction: compression error and time-discretisation error interact in
  general, and the measured order can degrade once truncation is active.
* **Not claimed.** Any statement about Fock truncation adequacy. The scheme's order says
  nothing about whether the retained Fock space is large enough for the state being evolved;
  that is a separate physical convergence question, checked from the photon-number
  distribution `p_n = rho_nn` rather than from the time-step order.
* **Not claimed.** That this document substitutes for tests. It records the derivation and a
  one-off measurement; the runnable regression suite keeps the algebraic anchor and the
  convergence-order anchor as two separate tests against two different references.

## 9. Consequence for the EDM derivation

The derivation of the transfer tensors is untouched. `A_{k;phi}`, the `D_k` tensor, the
picking tensor and the cavity Kraus channel are the same objects as before; only

1. their time argument becomes `t_n^* = (n - 1/2) eps` instead of `n eps`, with both algebraic
   sub-steps of a physical step sharing it, and
2. the dissipative factors are split into two half-steps placed at the ends of the physical
   step rather than one per sub-step,

which means an existing hand derivation carries over by substituting the midpoint and
re-reading Section 4 for the placement.
