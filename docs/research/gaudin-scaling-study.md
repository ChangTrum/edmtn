# The Gaudin scaling study: the separable-bath law from anchor to deep tail

This document records the 2026-07 end-to-end derivation of the separable-bath scaling
law for the standard isotropic Gaudin (central-spin) model.  It is written to be
self-contained for a reader with an undergraduate quantum-mechanics background: every
model parameter, every measured quantity, the full numerical protocol, the
pre-registered acceptance rules, and the honest caveats are spelled out.  Tooling:
`examples/research/gaudin_scaling.py`; data:
`examples/research/data/gaudin_scaling/`; figures:
`examples/research/pictures/gaudin_scaling/`.

**Headline result.**  Two independent readouts of the fold-by-fold increment — a
mathematical subspace diagnostic and a physically measurable polarization trace — both
follow the power law (increment) $\propto x^\alpha$ in the coupling-weight variable
$x$, with small-$x$ tail exponent $\alpha \to 1$.  At the anchor scale ($K=49$) every
pre-registered acceptance gate passed, reproducing the archived finding
($\alpha_\text{phys} = 1.000$ on all seven arms).  At the scale-up ($K=100$) the
physical observable carries $\alpha = 1.0007$ with $R^2 = 1.0000$ down to
$x \approx 3\times10^{-6}$ — **below** the float64 resolution floor of the mathematical
diagnostic — extending the clean slope-1 window to more than five decades.

---

## 1. Purpose and scientific context

### 1.1 The separability question

An open quantum system is a small system $S$ (here: one spin-$\tfrac12$) coupled to a
large environment ("bath") $B$.  A bath is called **separable** when it decomposes into
mutually independent sub-baths: the bath Hamiltonian has no coupling *between*
sub-baths, the system couples to each sub-bath additively, and the initial bath state is
a product over sub-baths.  Separability is a strong structural property — it underlies
most tractable decoherence models — yet it is essentially unmeasurable directly: one
cannot check operator commutation relations or initial-state factorisation of an
environment experimentally.

The research programme this study belongs to pursues an indirect, falsifiable
criterion: **measure the scaling exponent of the system's response to adding one more
sub-bath**.  For a separable bath each sub-bath enters independently, and the empirical
observation across every separable model examined so far (Gaudin, discretised
spin-boson, Dicke) is that the per-sub-bath increment of any probe scales as

$$ \text{(increment when adding sub-bath } L{+}1\text{)} \;\approx\; c\, x^{\alpha},
   \qquad x \;=\; \frac{g_{L+1}^2}{\bar g_L^2}, \qquad \alpha \to 1
   \text{ in the small-}x\text{ tail,} $$

where $g_{L+1}$ is the coupling of the newly added sub-bath and
$\bar g_L^2 = \sum_{k\le L} g_k^2$ is the accumulated coupling weight.  The exponent
$\alpha = 1$ is the **trivial** value expected when contributions simply add;
correlations between sub-baths are expected to *renormalise* the exponent
(an **anomalous** $\alpha \ne 1$), in analogy with mean-field versus anomalous critical
exponents.

**Epistemic status (important).**  The $\alpha_\text{tail} \to 1$ law is an *empirical*
regularity, found in the data first.  The conjectured mechanism — additivity of bath
cumulants across independent sub-baths implying a first-order ($\propto g^2$) response
— has **not** been derived; an analytic proof from the influence functional is in
progress separately.  Nothing in this document should be read as assuming the law; the
computations below *test* it.

### 1.2 Why this study, and what is new

The earlier Gaudin results were produced during an exploratory phase.  This stage
derives them from scratch on the audited codebase under a stricter discipline:

* a **pre-registered protocol**: acceptance tolerances fixed *before* running, taken
  from the archived envelope, with out-of-tolerance results contractually meaning
  "stop and report" (enforced by process exit codes), not "retune";
* **two independent increment readouts** (mathematical and physical) plus one derived
  consistency spectrum, computed from a single streaming fold pass;
* full machine-checkable **provenance** (git commit, cluster job id, environment, GPU
  identity, per-fold resource profile in every result file);
* a capacity-measured **scale-up** (stage G2) that pushes the probe one decade deeper
  in $x$ than the anchor scale could reach.

## 2. The model

### 2.1 Hamiltonian and initial state

The standard isotropic Gaudin ("central spin") model: one central spin-$\tfrac12$
$\mathbf S$ coupled isotropically to $K$ independent bath spins-$\tfrac12$
$\mathbf J_k$ (paper Eq. 22 of arXiv:2509.00424):

$$ H \;=\; \sum_{k=1}^{K} g_k\, \mathbf S\cdot\mathbf J_k
      \;=\; \sum_{k=1}^{K} g_k \left( S_x J_{k;x} + S_y J_{k;y} + S_z J_{k;z} \right), $$

with **no self-Hamiltonian** on either the central spin or the bath spins.  Two
consequences matter throughout:

* $H_S = 0$: the interaction picture is trivial and the system coupling operators
  $S_\alpha$ are time-independent;
* the bath is static, so its correlation functions never decay — the **memory time is
  infinite**.  This is what makes Gaudin the *hard* case for time-axis compression (the
  temporal entanglement never saturates with evolution time), and hence a good
  stress-test.

The initial state is a product state

$$ \rho(0) \;=\; \underbrace{\left(\tfrac12 + S_z\right)}_{\lvert\uparrow\rangle
   \langle\uparrow\rvert} \;\otimes\; \bigotimes_{k=1}^{K} \tfrac{\mathbb 1}{2}, $$

i.e. the central spin polarised along $+z$ and every bath spin at infinite temperature
(maximally mixed, unpolarised).  Each bath spin is one **sub-bath**: the bath is
separable by construction (independent sub-baths, product initial state).  It is also
**non-Gaussian**: the cumulant expansion of a spin bath does not truncate (all orders
are non-zero), so the pipeline uses the exact per-sub-bath correlation-tensor route
(paper Eq. F1), not a Gaussian/cumulant approximation.  Physically this model describes,
e.g., an electron spin relaxing through hyperfine contact with a bath of nuclear spins;
the measurable consequence used here is the decay of the central-spin polarization
$\langle S_z(t)\rangle$.

### 2.2 Coupling profiles

All couplings are normalised so that $\sum_k g_k^2 = g^2$ with $g = 1$, which fixes the
time unit ($t$ is measured in $1/g$).  Three profile families are used in this stage
(the fuller profile study, including `uniform` and the correlated `ou` family, is
documented in [coupling-scaling-law](coupling-scaling-law.md)):

| profile | $g_k$ (before normalisation) | parameters | ordering |
|---|---|---|---|
| `linear` (paper's) | $g\sqrt{\dfrac{6K}{2K^2+3K+1}}\,\dfrac{K+1-k}{K}$ | — | descending |
| `exp` | $\propto e^{-\beta k}$ | $\beta = 0.1$ | descending |
| `random` | $\sim \mathrm{Uniform}(0,1)$, sorted | seed $\in \{0,1,2,3\}$ | sorted descending |

All three are sorted strongest-first, and the fold below proceeds in stored order, so
"the first $L$ sub-baths" always means the $L$ strongest.  Under this ordering $x$
decreases monotonically with $L$: late folds probe the small-$x$ tail.

## 3. Method: the EDM fold in brief

### 3.1 What is computed

The evolution is carried by the **EDM (evolution density matrix)** tensor-network
method (Chong Chen and Ren-Bao Liu, *Polynomial complexity of open quantum system
problems*, arXiv:2509.00424, — the same codebase whose
documentation lives in this repository).  For a reader who has not met tensor networks:
the joint influence of the bath on the system over a discretised time grid
$t_m = m\,\varepsilon$, $m = 1..N$, is stored as a **matrix-product state (MPS) along
the time axis** — a chain of small tensors, one (order 1) or two (order 2) per Trotter
step, connected by "bond" indices.  The **bond dimension** $D$ at each link measures how
much temporal correlation the state carries across that time cut; compression consists
of discarding negligible singular values at each bond.  Every quantity below is a
function of this temporal MPS.

Discretisation: symmetric second-order Trotter splitting (`expansion_order = 2`), so
the chain has $n_\text{sites} = 2N$ sites, each with physical dimension
$d_\text{phys} = 7$ (the identity channel plus two directions for each of the three
coupling operators $S_{x,y,z}$).

### 3.2 The separable outer loop

For a separable bath the EDM factorises per sub-bath (paper Eq. 21): the full-bath EDM
is built **fold by fold**,

$$ \text{MPS}_{L+1} \;=\; \mathrm{compress}\big( \text{MPO}_{L+1} \times \text{MPS}_L \big), $$

where $\text{MPO}_{L+1}$ is the exact single-sub-bath kernel (bond dimension $D_a = 4$)
of bath spin $L+1$.  Each fold multiplies every bond dimension by $D_a$ and the
compression truncates it back.  After fold $L$ the MPS *is* the exact (up to truncation)
EDM of the system coupled to the $L$ strongest sub-baths — so one streaming pass over
$L = 1..K$ yields the entire family of intermediate physical states, which is exactly
what the increment analysis needs.

### 3.3 Compression protocol

One fixed compression protocol is used for every run in this stage:

| knob | value | meaning |
|---|---|---|
| `compress_method` | `direct` | quimb's sequential bond-by-bond 1D compression |
| `compress_decomp` | `exact` | full SVD at every bond (no randomised sketching) |
| `compress_canon` | `quimb` | quimb's own canonicalisation |
| `cutoff` | $10^{-8}$, mode `rel` | discard singular values below $10^{-8}\,\sigma_\max$ per bond |
| `max_bond` | 500 | **safety cap only** — never reached in any run of this stage |
| precision | float64/complex128 | no mixed precision |

The truncation quality is *measured*, not assumed: every fold records the maximum
per-bond **discarded weight** $\max_b \sum_{i\,\text{disc}} \sigma_i^2$ (P1-15 metric),
the deviation of the reconstructed $\mathrm{Tr}\,\rho$ from 1, and the full per-site
bond profile.  A fold whose bond dimension touches `max_bond` is flagged (`cap_hit`) —
a hit cap would mean rank-limited truncation rather than a natural bond, and (at the
anchor stage) fails the run.

### 3.4 Implementation and cross-validation

The sweep is **streaming**: one fold pass per configuration, with at most two host
snapshots resident, each snapshot left-canonicalised exactly once.  The implementation
composes the packaged layers (`GaudinModel`, `SeparableKernelEngine`, `QuimbEDM.fold`,
`ObservableExtractor`) rather than the top-level driver, for two reasons: the driver
recomputes from scratch per $L$ (an $O(K^2)$-folds cost vs. $O(K)$ streaming), and the
increment diagnostic consumes the intermediate MPS tensors themselves, which are not a
driver product.  Correctness is anchored by `--check` cross-validation, all green:

* the streaming per-bond diagnostics equal the established all-snapshot reference
  implementation (`coupling_distributions.analyse_transition`) to $10^{-10}$;
* the extracted $P_L(t)$ equals the packaged solver
  `EDMSolver(sub_baths=L).solve(channel=3)` on the public time axis with
  **max difference 0.0** (bit-identical code path);
* the public axis is verified to be $t = \varepsilon, 2\varepsilon, \dots, T$.

## 4. The three readouts

All three are computed for every $L$, and their per-fold increments are fitted against
the same scale variable $x = g_{L+1}^2/\bar g_L^2$.

### 4.1 Mathematical diagnostic: the subspace increment $\eta$

At each internal bond $\tau$ of the temporal MPS, the state after $L$ folds spans a
left singular subspace $Q_A(\tau)$; after fold $L{+}1$ it spans $Q_B(\tau)$.  The
overlap matrix $E_\tau = Q_A(\tau)^\dagger Q_B(\tau)$ has singular values
$\cos\theta_i$ — the cosines of the **principal angles** between the two subspaces (the
natural generalisation of "the angle between two lines" to subspaces).  The residual
ratio at bond $\tau$,

$$ \eta(\tau) \;=\; \sqrt{\,1 - \frac{\mathrm{Tr}\!\left(E_\tau\, \rho_\tau\,
   E_\tau^\dagger\right)}{\mathrm{Tr}\,\rho_\tau}\,}, $$

weights the new state's bond density matrix $\rho_\tau$ (so directions the state
actually occupies count more) and measures the fraction of the new state that the old
subspace **cannot represent** — "how much genuinely new temporal correlation did
sub-bath $L{+}1$ bring?".  Aggregates over bonds give $\eta_\text{rms}$ (the primary
fit target), $\eta_\max$, the normalised chordal distance, and the counts of new
directions $n_\text{new}$.  Details and the profile-dependence study:
[coupling-scaling-law](coupling-scaling-law.md).

### 4.2 Physical observable: the polarization trace $P_L(t)$

$P_L(t) = \langle S_z(t)\rangle$ — the central-spin polarization, with the bath
truncated to the $L$ strongest sub-baths — read from the same MPS snapshot via the
Eq. F2/F3 sweep, on the public axis $t = \varepsilon..T$ (the last point is
$\mathrm{Tr}[S_z\rho(T)]$ from the same state; the convention mirrors the packaged
solver exactly).  Its per-fold increment is

$$ \delta P_L \;=\; \mathrm{RMS}_t\, \lvert P_{L+1}(t) - P_L(t)\rvert . $$

This is the experimentally meaningful channel: $P(t)$ is a standard relaxation trace,
and $\delta P_L$ asks "how much does the measured curve change when one more nuclear
spin is included?" — the single-spin-sensing resolve-sub-baths-one-by-one methodology.
**Scope note:** in the *isotropic* Gaudin model the transverse-coherence readout
($\langle S_+\rangle$ from a $+x$ initial state) coincides with the longitudinal one by
symmetry, so this stage carries **one** independent physical channel, not two.

### 4.3 Derived consistency check: the mean-polarization spectrum

$S_L(\omega) = \varepsilon^2 \lvert \mathrm{rfft}(P_L - \bar P_L)\rvert^2$ at
$\omega_j = 2\pi\,\mathrm{rfftfreq}(N, \varepsilon)$ — demeaned, rectangular window, DC
bin kept.  This is the power spectrum of the *deterministic mean trace*: a nonlinear
functional of the same $P_L(t)$, **not** an independent observable and **not** a
spin-noise PSD (a genuine noise spectrum requires the two-time connected correlation
$\langle \delta S_z(t_1)\,\delta S_z(t_2)\rangle$, whose EDM extraction needs a
time-ordering derivation that has not been done; it is deliberately out of scope).  It
is recorded as a consistency diagnostic: if the increments scale, their frequency-domain
image should scale too.

### 4.4 Fits, tail, and the roundoff floor

Every fit is ordinary least squares of $\log y$ on $\log x$ over the rows
$L \ge L_0 = 2$; reported per fit: $\alpha$, $c$, $R^2$, the number of points total /
used / floored.  The **tail fit** restricts to the smallest-$x$ 40% quantile — the
$\alpha_\text{tail}\to1$ claim concerns this asymptotic regime, not the full window
(the full-window exponent is profile-dependent; see
[coupling-scaling-law](coupling-scaling-law.md)).

The subspace diagnostic $\eta$ has a hard resolution floor: QR/SVD subspace comparison
in float64 bottoms out at $\eta \sim 10^{-7}$ (machine-roundoff, established
empirically by self-comparison in the archived study).  Therefore a **floor mask at
$\eta \le 10^{-6}$** removes floored points from the *mathematical* fits only (masked
counts are reported).  The physical and spectral increments have **no**
self-comparison-established floor and are fitted unmasked, with their minimum values
reported — whether they *also* floor is precisely one of the questions (see §8).

## 5. Pre-registration and quality gates (stage G1)

G1 runs are **anchor** runs: acceptance is evaluated against tolerances fixed before
execution, and the batch queue *stops* on the first out-of-gate result (enforced by
process exit code 3 under `set -e`).

**Fit gate** (basis: the archived K=49 envelope $\alpha_\text{tail}\in[0.98,1.01]$ over
10 configurations at identical settings, widened by $\pm0.01$ measurement slack):

* $\alpha_\text{tail} \in [0.97, 1.02]$ for **both** $\eta_\text{rms}$ and
  $\delta P$;
* $R^2_\text{tail} \ge 0.95$; at least 8 tail points.

**Quality gate:**

* no `cap_hit` at any $L$ (bond dimensions must be natural, not rank-capped);
* no non-finite value in any trajectory, spectrum, diagnostic array, trace deviation
  or discarded weight;
* under the exact-SVD path, a *measured* discarded weight on every fold (an
  unmeasured fold would mean the metric chain silently failed);
* the maximum trace deviation is recorded and *reported only* — no threshold was
  pre-registered, so none is applied.

Additional process guarantees (all exercised by a 23-assertion selftest): completed
results are only reused after the full parameter set, code commit and data files are
verified to match; re-runs archive all previous artifacts (never silently overwrite or
delete, including interrupted partial data); random-seed pooling refuses members that
differ in anything but the seed or that failed their gates.

## 6. Stage G1 — the anchor (K = 49)

Protocol: $K=49$, $T=3$, $\varepsilon=0.1$ ($N=30$, 60 sites; the fine arm
$\varepsilon=0.05$, 120 sites), all other knobs as in §3.3.  Seven arms, strictly
sequential on one A800 (cluster job 48316, 21m48s total, commit `9fae4ea`).

| arm | $\alpha_\text{tail}^{\eta}$ | $\alpha_\text{tail}^{\delta P}$ | $R^2$ | tail pts (floored) | $D_\max$ | wall |
|---|---|---|---|---|---|---|
| linear | 0.994 | **1.000** | 1.000 | 19 (0) | 103 | 137 s |
| exp $\beta=0.1$ | 0.998 | **1.000** | 1.000 | 16 (3) | 118 | 181 s |
| random s0 | 0.993 | **1.000** | 1.000 | 17 (2) | 99 | 185 s |
| random s1 | 0.993 | **1.000** | 1.000 | 19 (0) | 103 | 171 s |
| random s2 | 0.990 | **1.000** | 1.000 | 19 (0) | 96 | 189 s |
| random s3 | 0.992 | **1.000** | 1.000 | 18 (1) | 99 | 164 s |
| linear fine ($\varepsilon=0.05$) | 0.992 | **1.000** | 1.000 | 18 (1) | 78 | 246 s |

**Every arm passed every gate** (`accepted = true`): zero cap hits (natural
$D_\max \le 118 \ll 500$), zero non-finite values, discarded weight measured on every
fold (maximum $2.4\times10^{-15}$ — truncation at these settings is far below every
other error scale), $\lvert\mathrm{Tr}\,\rho - 1\rvert \le 8.6\times10^{-9}$.

* **Pooled random seeds** (per-seed fits first, then pooled rows): per-seed
  $\alpha^{\eta}$ 0.990/0.992/0.993 (min/median/max), $\alpha^{\delta P}$ = 1.000 for
  every seed; pooled $\alpha^{\eta} = 0.992$, $\alpha^{\delta P} = 1.000$.
* **Time-step comparison** (report-only, no convergence verdict pre-registered):
  aligning $\varepsilon = 0.1$ vs $0.05$ on common physical times, the trajectories
  differ by at most $4.7\times10^{-4}$ (RMS $2.4\times10^{-4}$) and every tail
  exponent shifts by $\le 0.0015$.
* The spectrum exponent tracks the physical one on every arm
  ($\alpha^{\delta S}_\text{tail} \approx 0.98$–1.00).

The archived K=49 finding — physical $\alpha_\text{tail} = 1.000$ across observables
and coupling profiles, mathematical $\eta$ slightly below 1 — is **reproduced in full**
under the pre-registered protocol.  Figures: `g1_linear_scaling.png` (three increment
channels parallel to slope 1 over $\sim3.5$ decades), `g1_linear_perL.png`,
`g1_random_pooled_pool.png`, `g1_eps_compare_epscompare.png`.

## 7. The capacity probe (K = 24, uncapped)

Gaudin's infinite memory time makes the temporal bond grow with evolution time $T$; an
earlier benchmark (`docs/benchmarks/gpu-scaling-benchmark.md`) measured, uncapped at
$\xi=10^{-8}$, $\varepsilon=0.2$: $D_\max = 191$ (T=3, 0.68 GB) $\to$ 643 (T=6,
13.3 GB) $\to$ OOM $>80$ GB (T=9).  A fixed large-scale configuration therefore cannot
be assumed feasible; stage G2 was gated on a probe (job 48319, K=24, uncapped
`max_bond`, $T=6$):

| grid | sites | natural $D_\max$ | next-raw-fold estimate | host RSS peak | wall/fold (plateau) |
|---|---|---|---|---|---|
| $\varepsilon=0.2$ (benchmark) | 60 | 643 | — | — | — |
| $\varepsilon=0.1$ | 120 | 311 | 11.3 GiB | 4.2 GiB | $\sim$40–60 s |
| $\varepsilon=0.05$ | 240 | 222 | 12.2 GiB | 4.3 GiB | $\sim$40 s |

The probe's central finding is counter-intuitive and capacity-friendly: **at fixed $T$,
refining the Trotter step *lowers* the natural per-bond dimension** (643 → 311 → 222
across $\varepsilon = 0.2 \to 0.1 \to 0.05$) — the temporal correlations spread over
more sites, each bond carrying less.  The G2 target grid (240 sites) is thus
comfortably inside a single 80 GB card.  (The recorded "GPU peak" of $\sim$79 GiB is
the CuPy memory-pool *retained* high-water mark — the pool keeps freed blocks for
reuse and releases them under pressure; the live working set, bounded by the raw-fold
estimate plus decomposition workspace, is far smaller.  GPU peaks are 10 ms *sampled*
values with that caveat stored in every result file.)

The probe also returned K=24 tail exponents ($\delta P$: 1.009 on both grids; $\eta$:
0.965/0.932) — noted but not interpreted: 9 tail points at a non-anchor scale (see
§8.3).

## 8. Stage G2 — the deep-tail test (K = 100)

### 8.1 Rationale and protocol

The scientific point of the scale-up: at $K=49$ the smallest reachable
$x \approx 2.5\times10^{-5}$, and the mathematical $\eta$ already brushes its
$10^{-7}$ roundoff floor there.  At $K=100$ (linear profile) $x$ reaches
$3\times10^{-6}$ — but $\eta$ at those $x$ is *predicted* (by the law itself) to sit at
$\sim10^{-7}$, i.e. **in** the floor.  The question only the physical observable can
answer: **does the $\alpha=1$ law continue below the mathematical diagnostic's
resolution?**

Protocol: $K=100$, $T=6$, $\varepsilon=0.05$ (240 sites), `max_bond = 500` as a pure
safety cap (probe-measured natural $D_\max = 222$ at K=24; K=100's normalised couplings
are individually weaker still), all other knobs unchanged.  Arms: linear, linear at
$\varepsilon=0.1$ (for the report-only step comparison), random s0/s1 + pooled.
Cluster job 48320, 2h23m, single A800.

G2 arms are **not** anchor runs.  The pre-registered gates of §5 are bound to the
archived K=49/T=3 envelope; no archived envelope exists at this scale, and the
protocol forbids inventing thresholds after seeing data.  All fits and quality
records are reported instead, and the quality *signals* (cap hits, non-finite values,
unmeasured weights) were all clean: natural $D_\max = 158$–161 ($\ll 500$; K=100's
plateau is even lower than the probe's 222), discarded weight $\le 3.5\times10^{-15}$,
$\lvert\mathrm{Tr}\,\rho-1\rvert \le 3\times10^{-8}$.

### 8.2 Results

| arm | $\alpha_\text{tail}^{\eta}$ (floored) | $\alpha_\text{tail}^{\delta P}$ | $\alpha_\text{tail}^{\delta S}$ | $D_\max$ | wall |
|---|---|---|---|---|---|
| linear $\varepsilon=0.05$ | 0.978 (3) | **1.003** | 0.996 | 161 | 40 m |
| linear $\varepsilon=0.1$ | 0.994 (2) | **1.004** | 0.996 | 221 | 32 m |
| random s0 | 0.980 (5) | **1.003** | 0.997 | 158 | 36 m |
| random s1 | 0.979 (4) | **1.003** | 0.997 | 158 | 36 m |

(All tail fits: $R^2 = 1.000$, 34–39 points.  Pooled random: $\eta$ 0.980,
$\delta P$ 1.003, seeds statistically indistinguishable.  Step comparison at $K=100$:
trajectory deviation $\le 5.4\times10^{-4}$; $\delta P$ and $\delta S$ tail exponents
shift by $\le 0.0002$ between the two grids.)

**The deep tail.**  On the linear arm the fit window spans
$x \in [3.0\times10^{-6},\, 0.49]$.  The physical increment $\delta P$ reaches
$3.6\times10^{-7}$ — well below where $\eta$ floors ($\eta_\min = 1.1\times10^{-7}$
against its $10^{-6}$ mask; 3 points masked) — and restricting the fit to the deepest
decade alone,

$$ x \le 10^{-4} \; (5 \text{ points}): \qquad \alpha_{\delta P} = 1.0007, \quad
   R^2 = 1.0000. $$

The scaling figure (`g2_linear_scaling.png`) shows $\delta P$ hugging slope 1 over
**more than five decades of $x$** with no visible curvature, while the deepest $\eta$
points visibly lift off their fit line as they approach the roundoff wall.  The
physical observable carries the separable-bath law past the resolution limit of the
mathematical diagnostic — which is simultaneously the pragmatic message for
experiment-facing use: relaxation-trace increments are the more robust probe.

### 8.3 The $\eta$ depression at fine grids — an interpretation, flagged as such

The mathematical exponent at $K=100$ reads 0.978–0.980 on the $\varepsilon=0.05$ arms
but 0.994 at $\varepsilon=0.1$, a $-0.016$ shift, while the physical exponent moves by
$-0.0001$ between the same two grids.  The reading most consistent with the data is
**floor contamination**: unmasked $\eta$ points between $10^{-6}$ and
$\sim10^{-5}$ already sit partially on the roundoff shoulder, biasing the tail slope
down — the same mechanism previously established for fast-decaying profiles, and
consistent with the K=24 probe's still-lower values (0.93–0.97 from only 9 tail
points).  This is an *interpretation*, not a gated conclusion: no pre-registered
criterion covers it, and it is recorded here with that status.  What is *not* in doubt
is the asymmetry: under every grid change the physical exponent is rock-stable at
$1.000$–$1.004$ while the mathematical one degrades exactly where its floor is
approached.

## 9. Honest caveats and scope

1. **The law remains empirical.**  $\alpha_\text{tail}\to1$ for separable baths is an
   observed regularity with a conjectured (underived) mechanism; this stage adds a
   pre-registered reproduction and a five-decade extension for one model class, not a
   proof.
2. **One independent physical channel.**  Isotropy makes coherence and polarization
   readouts coincide; the spectrum is a derived functional of the same trace.  A
   genuinely independent second channel (e.g. a two-time noise spectrum or an echo
   protocol) requires derivations that are deliberately out of scope here.
3. **$\eta$ near its floor** (§8.3) — interpretation, not conclusion.
4. **Numerical scope.**  Single-GPU float64, exact-SVD compression only; `rsvd`,
   other canonisation/compression variants, and the `uniform`/`ou` profiles were not
   re-run in this stage (profile dependence is covered by the earlier committed study).
   `max_bond` never bound anywhere, so no rank-cap effects are present in any quoted
   number.
5. **Spectrum head feature.**  The $\delta S$ channel shows a dip near
   $x \approx 4\times10^{-2}$ on the K=100 linear arm (visible in
   `g2_linear_scaling.png`), i.e. in the strong-coupling head far outside the tail
   window; it does not affect any fitted exponent and is left unexplained here.
6. **Trace deviation** grows mildly with accumulated folds
   ($\le 3\times10^{-8}$ at K=100, 240 sites); it is recorded per fold and reported,
   with no pre-registered threshold.

## 10. Data, provenance, and reproduction

**Data** (all retrieved from the cluster and preserved locally; the repository's
`.gitignore` deliberately keeps regenerable example outputs — `examples/**/data/`,
`examples/**/pictures/` — out of version control, so the files below live beside the
repo and on the cluster, and every quoted number traces to them):

* `examples/research/data/gaudin_scaling/g1_*.json|npz` — seven G1 arms + GPU smoke
  + pooled + step comparison;
* `g2_probe_T6_e10|e05.*` — the capacity probe; `g2_*.json|npz` — four G2 arms +
  pooled + step comparison;
* every sweep JSON carries: the full argument set, the coupling array, the git commit
  (`9fae4ea`) and dirty flag, hostname, SLURM job id, GPU name, per-$L$ bond profiles,
  discarded weights, trace deviations, wall times, sampled GPU peaks and host RSS, and
  the fit/acceptance blocks.

**Cluster jobs** (CUHK physics cluster, node c1, 1× NVIDIA A800-SXM4-80GB per job,
`edmtn-gpu` env: CuPy 14.1.1 / quimb 1.14.0 / numpy 2.4.6): 48316 (G1), 48319 (probe),
48320 (G2).

**Reproduction.**  Each arm is one CLI invocation of
`examples/research/gaudin_scaling.py` (see the module docstring); e.g. the G2 linear
arm:

```bash
PYTHONPATH=src python examples/research/gaudin_scaling.py \
  --K 100 --g 1.0 --T 6 --eps 0.05 --order 2 --cutoff 1e-8 --max-bond 500 \
  --method direct --decomp exact --canon quimb --device gpu \
  --coupling linear --name g2_linear
```

`--check` reruns the cross-validation, `--selftest` the process-contract tests, and
`--replot <name>` regenerates every figure from the committed JSON+NPZ without any
recomputation.
