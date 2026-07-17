# nanoindent — design notes

Why the library is built the way it is: the modelling decisions, the numerical
choices behind them, and the places where a decision is a *judgement call* rather
than a consequence of the physics.

> **Status of this document.** It was **reconstructed from the implementation**
> (2026-07-17) to replace references to an earlier design document that was never
> committed to this repository and could not be recovered. It therefore records
> what the code does and why that is defensible — it is not the specification the
> code was originally written against, and it claims no authority the code doesn't
> already have. Where the original was cited as *sanctioning* a choice
> (e.g. the scikit-fem fallback), the honest reading is that the choice is
> justified on its merits below. See issue #22.

The problem: a rigid axisymmetric tip indenting a linear-elastic film of finite
thickness `h`, bonded to (or resting on) a rigid substrate, under frictionless
unilateral contact. The observable is the force–indentation curve `F(δ)` and, in
particular, the **bottom-effect stiffening** relative to the Hertz half-space.

---

## 1. Scope and framing

Deliberately **linear elastic, small strain, frictionless, quasi-static**. That is
enough to quantify the bottom effect — which is a *geometric* consequence of a
finite-thickness bonded layer, not a material nonlinearity — and it keeps every
result checkable against a closed form.

The consequence to keep in view: a 1 nm indent into a 7 nm film is `δ/h ≈ 14%`,
at the edge of small-strain validity. Expect a few-percent finite-strain error
that this library does not model. That is a known, bounded limitation, not an
oversight (see §7).

---

## 2. Units

Length **nm**, modulus/stress **MPa**, force **pN**. The force unit is not a
choice, it follows:

```
1 MPa · nm² = 1e6 Pa · (1e-9 m)² = 1e-12 N = 1 pN
```

so integrating a stress in MPa over an area in nm² yields pN directly, with no
conversion factor anywhere in the solver. `MPA_NM2_TO_PN = 1.0` exists only to
document that. Sanity anchor: Hertz at `E* ≈ 143 MPa`, `R = 17 nm`, `δ = 1 nm`
gives `F ≈ 786 pN` — a sensible AFM force scale.

---

## 3. Axisymmetry

The model is 2D axisymmetric in `(r, z)`. This is what makes fine contact meshes
affordable, and it constrains the indenter: **the tip must be a surface of
revolution**.

Sphere, paraboloid, cone, and blunted cone are exact. **Pyramids (Berkovich,
Vickers, cube-corner) are not axisymmetric and cannot be meshed here.** The
accepted workaround is an *equivalent cone* whose half-angle reproduces the
projected-area-vs-depth law (`contact.equivalent_cone_half_angle`). This is an
approximation, and it is deliberately explicit rather than hidden behind a
`tip_shape="berkovich"` that would silently mean something else.

The hoop strain `ε_tt = u_r / r` is carried explicitly in the constitutive
matrix — that term is precisely what separates the axisymmetric operator from
plane strain, and dropping it would silently solve a different problem.

### 3.1 Exact sphere, not the parabola

`RigidSphere` uses the **exact** spherical gap, not `z = r²/2R`. The parabola is
available separately as `Paraboloid` because it is the tip the Hertz and
Garcia–Garcia derivations actually assume — so comparing FEM against those models
is a like-for-like comparison only with `tip_shape="paraboloid"`.

Keeping both is the point: the difference between them is a real modelling error
term, and it should be visible rather than assumed negligible.

---

## 4. Discretisation

**Vector P2 (quadratic) displacement elements.** For a thin film in bending under
a concentrated contact, P1 is poor: it needs many more elements for the same
accuracy and handles the contact patch badly. P2 resolves both with few elements,
which matters because the mesh must be fine near contact and the film is thin.

### 4.1 Meshing

The domain is a graded rectangle in `(r, z)`, built directly as a tensor-product
triangle mesh with size clustered at the contact point `(r=0, z=h)`
(`mesh_size_contact`) and coarsening geometrically (growth 1.3) toward the lateral
edge (`mesh_size_edge`). Vertical grading guarantees several element layers across
even a 7 nm film.

gmsh is **not** required: the geometry is a rectangle, so a mesher would add a
hard dependency for no benefit. A final-sliver guard removes a degenerate last
element where the grading would otherwise leave one.

**On the "0.1 nm edge size" figure.** A 0.1 nm element size at the far lateral
edge of a 100 nm domain is rejected here as unreasonable: it implies an enormous
element count in the region that contributes least to the contact force. The
default is a few-nm edge size. This is a deliberate departure, and the arbiter is
not authority but **the convergence study** (§6) — if `F(δ_max)` is insensitive to
edge refinement, the coarse edge is correct and the fine one is waste.

### 4.2 Domain size `L`

`L` is interpreted as the domain **radius**, not diameter. A stated "lateral size
of 100 nm" is genuinely ambiguous, so the interpretation is a documented default
and `L` is a free parameter — the honest handling of an ambiguity is to name it
and expose it, not to pick silently.

The lateral edge is traction-free. `L` must be large enough that the edge does not
perturb the contact solution; `expected_contact_radius = √(Rδ)` is the scale to
compare against, and the lipid bridge's `to_simulation_config` uses
`L = max(20, 10·√(Rδ))` on that basis.

---

## 5. Near-incompressibility and the mixed formulation

Soft biological films are often modelled as nearly or fully incompressible, and
this is the sharpest numerical trap in the library.

**The problem.** The single-field ("primal") P2 displacement formulation **locks**
as `ν → 0.5`: the discrete space cannot represent the isochoric constraint, and the
result is spurious over-stiffening. That is a silent failure — it produces a
plausible force curve that is simply wrong.

**The response, in three parts:**

1. **`ν ≥ 0.45` emits a locking warning** in `SimulationConfig._validate`. Loud
   beats subtle for a failure whose signature is "looks fine, reads high."
2. **`formulation="mixed"`** selects a **Taylor–Hood** (P2 displacement / P1
   pressure) two-field formulation that is locking-free and supports `ν = 0.5`
   exactly. It assembles the saddle-point block system

   ```
   [ A + K_contact   Bᵀ ] [u]   [f_contact]
   [ B              −C  ] [p] = [    0    ]
   ```

   with `A(u,v)=∫2μ ε:ε r`, `B(u,q)=∫(∇·u) q r`, `C(p,q)=∫(1/λ) p q r` — and `C → 0`
   exactly at `ν = ½`, which is why the mixed form reaches full incompressibility
   while the primal form cannot.
3. **The primal form refuses `ν = 0.5` outright** (`ValueError`), rather than
   returning a locked answer.

**Cross-check.** The mixed formulation reproduces the primal result at `ν = 0.3`
to < 0.5%, so it is not a different physics — it is the same physics in a space
that can represent the constraint. And because it reaches `ν = ½`, it enables the
Chadwick verification (§6), which the primal form structurally could not do.

Taylor–Hood is chosen because it is inf-sup stable without stabilisation
parameters — consistent with the rule that there are no magic numbers in solver
code.

---

## 6. Contact, and how it is trusted

**Penalty, not Lagrange multipliers.** The penalty keeps the system size fixed and
the assembly simple. Its cost is a free parameter, and free parameters need
discipline:

- **Refinement-consistent stiffness.** `k_pen = penalty_alpha · E / h_local`, with
  `h_local` the *local facet size*. Scaling by the local size means the penalty
  does not silently change meaning as the mesh refines — `penalty_alpha` is
  dimensionless and mesh-independent by construction.
- **Active-set Newton.** The only nonlinearity is the contact active set. The
  elastic stiffness `K` is assembled once; each Newton iteration re-assembles only
  the contact residual and its active-set tangent from the current penetration.
- **The penalty study** (§6.1) is what justifies `penalty_alpha`: `F` must plateau
  and interpenetration must → 0. A penalty accepted without that sweep is an
  unvalidated fudge factor.

**Two independent force measures.** The reaction force is computed twice — by
integrating contact pressure over the top surface, and by integrating `σ_zz` over
the bonded bottom — and their relative mismatch is reported as
`equilibrium_residual`, warning above `equilibrium_tol` (~1% on a converged mesh).
Two routes to the same number, from opposite ends of the domain, is a check that a
single measure cannot provide: it catches assembly and boundary-condition errors
that would otherwise look like plausible physics.

### 6.1 Convergence studies

`nanoindent.convergence` provides three, and they are the library's internal
arbiter for every numerical parameter above:

1. **mesh study** — refine `mesh_size_contact`, watch `F(δ_max)` (converges < 0.5%);
2. **penalty study** — sweep `penalty_alpha`, watch `F` plateau and interpenetration → 0;
3. **load-step study** — sweep `n_steps`; the elastic problem is path-independent,
   so this is a null test that should show *no* dependence. A null test that fails
   indicates a bug in the stepping, which is exactly why it is worth running.

### 6.2 Validation against closed forms

The primary validation is **analytic**, not comparative:

- **Hertz** (half-space limit) — exact;
- **Dimitriadis 2002** — bonded/free finite-thickness corrections;
- **Garcia & Garcia 2018** — bottom-effect correction, paraboloidal tip;
- **Chadwick 2002** — thin incompressible layer, reachable only via the mixed
  formulation.

This is a deliberate choice of standard. Another FEM code (COMSOL or otherwise) is
*not* ground truth — it carries its own discretisation error and its own unstated
mesh decisions, so agreeing with it demonstrates that two codes made similar
choices, not that either is right. Closed forms are ground truth in their domain of
validity. Where an analytic model and this library disagree, the disagreement
should be traceable to a stated assumption (tip shape, `δ/h` range, compressibility)
rather than waved at.

Note the two 2002 papers are distinct: Dimitriadis et al. is the empirical
polynomial for finite-thickness bonded/free films; Chadwick is the thin
incompressible layer law.

### 6.3 One cross-check against another code, and what it was worth

§6.2 argues another FEM code is not ground truth. That argument was tested rather
than assumed, once, against [MoFEM](https://mofem.eng.gla.ac.uk) (MIT, Glasgow).
Its ADV-1 atom test `hertz_mfront_2d_axisymm_flat` is this problem — an exact-sphere
SDF indenting a flat axisymmetric linear-elastic body — reached by a different
route: augmented-Lagrangian contact with an H(div) traction field, against our
penalty and active-set Newton. `examples/mofem_cross_check.py` runs our solver at
its configuration (`R=100, δ=0.01, E=1e3, ν=0.4`, domain radius 50 × depth 100).

| | `F/F_Hertz` |
|---|---|
| MoFEM (augmented Lagrangian, H(div)) | `0.99703` (−0.30%) |
| this library (penalty, P2) | `0.99591 … 1.00281` (±0.34%) |

**The verdict is that §6.2 was right.** MoFEM's −0.30% lands *inside* our own mesh
scatter, so the second code separated nothing: it could not distinguish our error
from its own, and agreeing with it added no information that Hertz had not already
supplied. Both codes were measured against the analytic ruler; the ruler did the
work. This is recorded as a null result, not a validation.

Three things did come out of it, none of which required MoFEM to be right:

- **The penalty is not setting the answer.** `penalty_alpha` 400 → 1000 moves `F`
  by 0.02%. §6.1's penalty study says to earn the free parameter; this is that
  study's conclusion stated as a number. It also retires augmented-Lagrangian
  contact as a thing worth implementing here — the formulation MoFEM uses to avoid
  the free parameter buys nothing our sweep does not already show is absent.
- **The mesh sweep does not converge, it oscillates** (1.0028 → 1.0002 → 0.9970 →
  0.9959 → 1.0000 → 1.0015). Refinement does not settle it because the active set
  changes *discretely* — facets enter and leave contact — so `F` carries a noise
  floor rather than a converging error. §6.1's "converges < 0.5%" is satisfied but
  reads more favourably than the data deserves: it is ±0.34% of scatter, not
  0.34% of remaining discretisation error. **Do not quote this solver tighter than
  ~±0.35% on `F`.** A monotonicity assertion guards this in the example.
- **The exact-sphere term of §3.1 is real and now has a size.** At MoFEM's shallow
  `δ/R = 1e-4` we sit on Hertz (±0.3%); at the library's own `δ/R = 0.059`
  (`R=17, δ=1`) we sit at +1.1%, above the noise floor. An exact sphere is stiffer
  than the parabola Hertz assumes, and the deviation appears only with depth —
  which is the signature §3.1 predicts, and evidence that the +1.1% is physics
  rather than error.

**Reproducing the MoFEM side** (not needed for the example, which records its
result): `likask/mofem-spack-jupyterhub:0.15.0`, amd64 only. Two traps. Its meshes
are pre-partitioned — `hertz_2d_4p.h5m` needs `mpirun -np 4`, and on one rank it
returns zero force *silently* rather than erroring. And its axisymmetric path is
hard-gated behind MFront (`MOFEM_NOT_IMPLEMENTED` otherwise), whose shipped
`libBehaviour.so` is compiled with AVX2/FMA that **Rosetta 2 cannot execute**;
either disable Rosetta in Docker Desktop so qemu handles it, or rebuild the
behaviour (`mfront --obuild --interface=generic LinearElasticity.mfront` with
`-march=x86-64 -mno-avx -mno-avx2 -mno-fma`).

---

## 7. Judgement calls, stated

Decisions where the physics does not force an answer. Each is a parameter, not a
constant, and each is worth revisiting per problem:

| Choice | Default | Why, and what would change it |
|---|---|---|
| Poisson's ratio | `ν = 0.3` | Not determined by the problem statement. Material-dependent and consequential: `E* = E/(1−ν²)`, so ν = 0.3 → `E* ≈ 143 MPa` vs ν = 0.5 → `E* ≈ 173 MPa` at `E = 130 MPa` — ~21% on force. **Sweep it rather than trust it.** |
| `L` as radius | radius | "Lateral size" is ambiguous (§4.2). Check against `√(Rδ)`. |
| Edge mesh size | few nm | The convergence study is the arbiter, not a quoted figure (§4.1). |
| Small strain | assumed | `δ/h ≈ 14%` at `δ=1 nm, h=7 nm` costs a few percent. Finite strain is out of scope for v1. |
| Substrate | `bonded` | Matches an SLB on mica/glass. A bonded incompressible film is markedly stiffer than an unbonded one, which can shear material out along the substrate — this is a physics-changing switch, not a detail. |
| Penalty `α` | 100–600 | Problem-dependent; justified per-case by the penalty study (§6.1). |

---

## 8. Force volume and viscoelasticity, from one solve

Both extensions rest on a single measured fact: **the indentation force is exactly
proportional to the modulus.** At fixed geometry, `F(δ; E) = E · φ(δ)`, where `φ`
is a purely geometric kernel (tip radius, film thickness, ν, mesh — but not E).
This is not an approximation to check later; it is exact linear elasticity, and
the solver reproduces it to `0.0000%` across a 400× range of E (the penalty
`k_pen = α·E/h_local` scales with E too, so the whole discrete system scales and
`φ = F/E` is E-independent). The von Mises **field** is likewise `E`-linear at
fixed δ (displacements are E-independent under displacement-controlled contact;
stress `= C(E):ε ∝ E`).

Everything below is bookkeeping on top of that one fact.

### 8.1 Force volume (`forcemap.py`)

A force map is an `n×n` grid of force curves at spatial pitch `X` and ramp
frequency `Y`. Because `F = E·φ`, one FEM solve gives `φ`, and every pixel — any
modulus field, heterogeneous or continuous — is an exact analytic scaling of it.
A 64×64 map costs **one solve, not 4096**. `simulate_force_volume` emits raw
`(z, deflection)` curves (the calibration inverted: `d = F/k`, `z = z₀+δ+d`), so a
`ForceVolume` is a drop-in for the ordinary inverse analysis and the map
*round-trips*: generate at a known modulus field, recover it, compare.

Two recovery paths, and what each is for:
- **`model="kernel"`** fits against `φ` itself. Since the curves *are* `E·φ`, this
  recovers E to the contact-point scan's resolution (~a few %). It validates the
  machinery — timing, calibration inversion, scaling — independently of any
  analytic model.
- **`model="hertz"` / `"dimitriadis"` / …** fit against the closed forms, and so
  surface *their* error. On a thin bonded film Hertz reads ~2–4× high (the bottom
  effect of §3/§6.2, now as a whole map); the finite-thickness corrections much
  less. The force-map view makes the artefact spatial and unmissable.

**Frequency `Y` sets the time base only** for an elastic sample: ramp velocity
`2·z_amp·Y`, per-pixel time `1/Y`, scan time, timestamps. It does *not* change the
forces — there is no rate term in the elastic constitutive law. Rate dependence is
§8.2, and keeping the two separate is deliberate: conflating acquisition timing
with material response is a common way to misread fast force maps.

### 8.2 Viscoelasticity by correspondence (`viscoelastic.py`)

Rate/frequency dependence of the modulus itself is viscoelasticity. The bridge is
again `F = E·φ`: the elastic–viscoelastic **correspondence principle** (Lee & Radok
1960) replaces the constant modulus by a hereditary convolution with the material's
relaxation function `E(t)`:

    F(t) = ∫₀ᵗ E(t − t') d/dt'[ φ(δ(t')) ] dt'.

This is exact for a linear viscoelastic solid during monotonic loading (growing
contact; Ting 1966 handles receding contact on unload, out of scope), and it
inherits the FEM kernel's geometry — finite thickness, bottom effect, exact sphere
— which the textbook Hertz-based viscoelastic formulae do not carry. `E(t)` is a
Prony series (SLS, generalized Maxwell); the convolution is evaluated by the exact
first-order recurrence for exponential kernels (one internal variable per branch,
O(N), no growing sum).

Driving `δ(t)` sinusoidally and projecting the steady-state force onto sine/cosine
recovers the storage and loss moduli `E'(ω)`, `E''(ω)`. **Validation is against the
closed-form complex modulus** `E*(ω) = E_inf + Σ Eₖ (iωτₖ)/(1+iωτₖ)`, keeping to
§6.2's standard — recovery matches to `0.10%` even through the FEM kernel's
geometric nonlinearity (small-amplitude oscillation about an operating depth,
normalised by the local kernel slope). The time-domain view is a ramp-and-hold
stress-relaxation curve decaying from the glassy `E₀` toward the equilibrium
`E_inf`.

### 8.3 Recovery between consecutive indentations (`recovery.py`)

An AFM force map indents, retracts, steps a few nm laterally, and indents again.
On an elastic sample that is fine — the surface springs back the instant the tip
lifts, so consecutive pokes are independent. On a **viscoelastic** sample it is
not: the surface recovers over `tau`, so a poke that comes too soon (fast scan) or
too close (small lateral step) lands on material still dented by the previous one.
This is a real source of correlation/hysteresis in fast or dense force volume, and
`recovery.py` locates its limits.

Two ingredients, each honest about where it comes from:

- **Spatial footprint `s(r)`** — the normalised surface depression `u_z(r)/u_z(0)`,
  read from one real FEM solve. It carries the bottom effect: on a bonded thin film
  the dent is localised (here `s` is down to ~10% by ~1.5 contact radii and shows a
  faint pile-up ring just outside contact), far tighter than a half-space. So the
  *spatial* limit — the pitch beyond which the previous dent simply does not reach —
  is ~1.5 a, set by the FEM, not assumed.
- **Temporal recovery** — an **imprint-then-recover** model: contact is fast (the
  material deforms with its glassy modulus, imprinting `δ_max·s`), then the residual
  relaxes freely as `exp(−t/τ)`. This is the right asymmetry for AFM (fast load, slow
  recovery); a single symmetric `dA/dt` law would wrongly make the material fail to
  deform during a quick contact.

Because the solver is axisymmetric, it cannot place two laterally-separated indents
in one solve. The raster is built by **linear superposition** of the FEM footprint
at each site — valid in the same small-strain linear regime as everything else, and
labelled a reduced-order model, not a 3-D solve. The time-domain raster
(`raster_surface`) and the closed-form steady-state overlap (`residual_fraction`,
summing `s(k·pitch)·exp(−(k/f)/τ)` over prior pokes) agree, and `recovery_limit_map`
sweeps pitch × frequency to draw the boundary. The upshot is a two-limit rule:
a poke sees fresh material if the step exceeds ~1.5 a **or** the interval exceeds a
few `τ`; it does not if both are small.

### 8.4 The inverse problem — reading parameters off a measurement

The forward model runs backwards to interpret real data. Because a force curve
factorises as `F(t) = ∫ E(t−t′) d/dt′[φ(δ(t′))] dt′` — geometry `φ` times material
`E(t)` — calibrating `φ` (one FEM solve, the tip/contact) turns the curve into a
pure convolution of the relaxation function, and the SLS parameters fall out by
least squares. `fit_relaxation` fits a ramp-and-hold decay (`F = φ·(E_inf +
(E0−E_inf)e^{−t/τ})`); `fit_dynamic_spectrum` fits a measured `E′(ω)`, `E″(ω)`
(loss peak locates `τ`, the storage plateaus give `E_inf`, `E0`). A synthetic
round-trip recovers known parameters to a percent through realistic noise. Both
protocols live on the loading/hold branch, where the theory is exact — preferred
over hysteresis-loop fitting, which sits on the approximate retract (§8.5). The
practical requirements are a good `φ` calibration, a detected contact point, and
an experiment whose timescale brackets `τ`.

### 8.5 What is *not* modelled

The force-volume timing is kinematic, not a coupled cantilever dynamics model
(no tip inertia or hydrodynamic drag — irrelevant at the elastic quasi-static
limit, but they matter for true high-speed AFM and are not here). Viscoelastic
unloading uses the loading (Lee–Radok) form, not Ting's receding-contact
correction. The recovery raster (§8.3) is a linear superposition of the
axisymmetric footprint with a single-`τ` recovery law — a reduced-order stand-in
for a genuine 3-D two-indent solve, good for scoping the limits, not for the exact
stress field between two adjacent dents. None of the three extensions adds a new
solver: all are transforms of the one elastic kernel, which is why they are cheap
and why they validate against closed forms (or, for recovery, an internal
ODE-vs-analytic cross-check) rather than against another code.

---

## 9. Backends

**Target: FEniCSx/dolfinx + gmsh. Shipped: scikit-fem.**

dolfinx installs cleanly only via conda-forge. Making it a hard requirement would
mean this library could not be `pip install`-ed, could not run in a plain CI
matrix, and could not be a dependency of anything pip-based. scikit-fem is pure
pip, runs everywhere, and is sufficient for axisymmetric P2 elasticity with
penalty contact — the whole problem here.

The cost is bounded by design: **`solver.py` is the only backend-specific module**
(weak-form assembly). `config`, `contact`, `geometry`, `material`, `analytics`, and
`postproc` are backend-agnostic, so a dolfinx backend can be added without touching
the API. That containment is what makes the fallback a defensible engineering
choice rather than a compromise — and it is why `geometry.py` builds its mesh
directly instead of requiring gmsh.

---

## 10. Related documents

- `README.md` — usage, results, validation tables.
- `docs/methodology_msnl_bio_dc.md` — Methods text (Biophysical Journal style) for
  the MSNL-Bio-DC cantilever case study.
- Issue #22 — the missing citation for the reference COMSOL setup, and the history
  of this document.
