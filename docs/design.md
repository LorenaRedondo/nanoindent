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

## 8. Backends

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

## 9. Related documents

- `README.md` — usage, results, validation tables.
- `docs/methodology_msnl_bio_dc.md` — Methods text (Biophysical Journal style) for
  the MSNL-Bio-DC cantilever case study.
- Issue #22 — the missing citation for the reference COMSOL setup, and the history
  of this document.
