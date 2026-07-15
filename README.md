# nanoindent

FEM simulation of **AFM nanoindentation of thin elastic films** by a rigid
spherical tip. An open-source Python replacement for a COMSOL setup: a rigid
sphere (R = 17 nm) indenting a linear-elastic film (E = 130 MPa, thickness
h = 7 / 14 nm, lateral size 100 nm) bonded to a rigid substrate, frictionless
contact, indentation up to 1 nm.

The library computes force–indentation curves **F(δ)**, quantifies the
**bottom-effect stiffening** of a finite-thickness bonded film relative to the
Hertz half-space, and ships analytic benchmarks (Hertz, Dimitriadis 2002,
Garcia & Garcia 2018) for validation.

---

## Highlights

- 2D **axisymmetric** small-strain linear elasticity, vector **P2** elements.
- **Exact spherical gap** function (not the parabolic approximation).
- **Penalty contact** with an active-set Newton solver, displacement-controlled
  load stepping.
- Two independent **reaction-force measures** (contact-pressure integral and
  bonded-bottom σ_zz integral) cross-checked for equilibrium.
- **Analytic benchmarks** and built-in **convergence studies** (mesh, penalty,
  load step).
- Clean, parametrized, fully-tested modules — no magic numbers in solver code.

## Bottom-effect result

For E = 130 MPa, ν = 0.3, R = 17 nm, a bonded film is markedly stiffer than the
Hertz half-space, and thinner films are stiffer still. At δ = 1 nm the FEM
gives F/F_Hertz ≈ **1.88** for h = 7 nm and ≈ **1.34** for h = 14 nm, in close
agreement with the bonded Dimitriadis 2002 correction:

![comparison](docs/comparison.png)

---

## Installation

### Primary backend (design target): FEniCSx / dolfinx

The design targets **dolfinx** with the **gmsh** mesher. dolfinx is best
installed via conda-forge (do **not** attempt a pip-only install):

```bash
conda install -c conda-forge fenics-dolfinx mpich pyvista gmsh python-gmsh
```

### Default backend here: scikit-fem (pure pip)

This repository's shipped backend uses **scikit-fem**, the design's sanctioned
pure-pip fallback, so it installs and runs anywhere without conda:

```bash
pip install -e .
# or with test deps:
pip install -e ".[test]"
```

Only `solver.py` is backend-specific (weak-form assembly); `config`, `contact`,
`geometry`, `analytics`, and `postproc` are backend-agnostic, so a dolfinx
backend can be dropped in without touching the rest of the API.

---

## Quick start

```python
from nanoindent import SimulationConfig, IndentationSolver

cfg = SimulationConfig(
    R=17.0,            # nm, tip radius
    E=130.0,           # MPa, sample Young's modulus
    nu=0.3,            # Poisson's ratio (the paper does not specify one!)
    h=7.0,             # nm, film thickness
    L=100.0,           # nm, lateral size (radius)
    delta_max=1.0,     # nm
    n_steps=20,
    mesh_size_contact=0.15,  # nm, near the tip
    mesh_size_edge=6.0,      # nm, at the far edge (see "Meshing" below)
    penalty_alpha=400.0,
)
result = IndentationSolver(cfg).run()          # -> ForceCurve
result.save("h7.csv")
result.plot(compare=["hertz", "dimitriadis_bonded", "garcia_bec"])
```

Reproduce both paper thicknesses with one command:

```bash
python examples/paper_reproduction.py           # writes results/{h7,h14}.csv + PNGs
python examples/hertz_validation.py             # thick-film Hertz check
```

---

## Units

A single consistent system is used everywhere:

| Quantity          | Unit |
|-------------------|------|
| length            | nm   |
| modulus / stress  | MPa  |
| force             | pN   |

The force unit follows from the other two, since
`1 MPa·nm² = 1e6 Pa · (1e-9 m)² = 1e-12 N = 1 pN`. Any force obtained by
integrating a stress over an area therefore comes out directly in **pN**, a
natural AFM scale (the Hertz estimate at δ = 1 nm is ≈ 786 pN).

---

## Physics & numerics

- **Boundary conditions**: bottom `z = 0` bonded (`u_r = u_z = 0`); axis
  `r = 0` symmetry (`u_r = 0`); lateral edge `r = L` traction-free; top `z = h`
  frictionless unilateral contact against the rigid sphere.
- **Gap function**: `g(r) = (z_c − √(R²−r²)) − (h + u_z)`, sphere centre
  `z_c = h + R − δ`, penetration `[−g]₊`. Contact is impossible for `r ≥ R`.
- **Penalty**: local stiffness `k_pen = α · E / h_local` with `h_local` the
  local facet size, so the penalty stays consistent under refinement. The
  contact residual `∫ k_pen [−g]₊ v_z · 2πr dr` is added to the elastic
  residual; the only nonlinearity is the contact active set, handled by a
  Newton iteration per load step.
- **Reaction force**: measured two ways — integrating the contact pressure over
  the top surface, and integrating σ_zz over the bonded bottom — and the
  relative mismatch is reported as a solution-quality metric (warns above a
  configurable tolerance, ~1% on a converged mesh).

### Meshing

The mesh is a graded axisymmetric rectangle, clustered at the contact point
`(r = 0, z = h)` (element size `mesh_size_contact`) and coarsening toward the
lateral edge (`mesh_size_edge`). The paper's quoted **0.1 nm edge size** on a
100 nm domain is unreasonably fine (a huge element count for negligible
benefit far from contact); the default here is a saner few-nm edge size, with
the convergence study as the arbiter. Vertical grading guarantees several
element layers across even a 7 nm film, and quadratic (P2) displacement
elements resolve bending and contact accurately with few elements.

### Near-incompressibility

For `ν ≥ 0.45` the P2 displacement formulation starts to lock; the config emits
a warning. A bonded near-incompressible film is physically much stiffer than
`ν = 0.3`, so a mixed (Taylor–Hood) displacement–pressure formulation is the
natural v2 extension. `ν = 0.5` exactly is not supported by the displacement
formulation.

---

## Convergence & validation

`nanoindent.convergence` provides three studies mirroring the paper:

```python
from nanoindent import SimulationConfig, convergence
cfg = SimulationConfig(h=14.0)
convergence.mesh_study(cfg)        # refine mesh_size_contact; F converges < 0.5%
convergence.penalty_study(cfg)     # alpha sweep; F plateaus, interpenetration -> 0
convergence.load_step_study(cfg)   # n_steps sweep; elastic => path-independent
```

Physics-validation tests (pytest `-m slow`) check:

1. **Hertz limit** — a 200 nm film recovers `F = (4/3) E* √R δ^{3/2}` within 2%.
2. **Bottom effect** — h = 7 / 14 nm track the bonded Dimitriadis 2002 model.
3. **Sanity** — zero force at δ = 0, monotone F(δ), `a(δ) ≈ √(Rδ)`, and a
   thinner bonded film is stiffer than a thicker one.

```bash
pytest -m "not slow"     # fast unit suite
pytest -m slow           # physics validation (~15 s)
pytest                   # everything
```

---

## Package layout

```
src/nanoindent/
├── config.py       # SimulationConfig dataclass (validated; units nm/MPa/pN)
├── geometry.py     # graded axisymmetric mesh + boundary tags
├── material.py     # LinearElastic: Lame params, axisym constitutive matrix
├── contact.py      # RigidSphere: exact gap, penetration, penalty stiffness
├── solver.py       # IndentationSolver: assembly, Newton, reaction extraction
├── postproc.py     # ForceCurve: CSV IO, diagnostics, plotting
├── analytics.py    # Hertz; Dimitriadis 2002; Garcia & Garcia 2018 BEC
└── convergence.py  # mesh / penalty / load-step studies
examples/
├── paper_reproduction.py
└── hertz_validation.py
tests/               # unit + physics-validation (slow) suites
```

## Modelling choices (documented, parametrized)

- **Poisson's ratio** is not given in the paper → default `ν = 0.3`, exposed
  prominently; run a sensitivity sweep by varying `cfg.nu`.
- **"Lateral size 100 nm"** is interpreted as the domain **radius** by default
  (`L`), and is a parameter.
- **Small strain** is assumed (as in the paper); a 1 nm indent on a 7 nm film
  is near the edge of small-strain validity, so expect a few-percent
  finite-strain error — out of scope for v1.

## References

- Hertz, *J. Reine Angew. Math.* **92**, 156 (1882).
- Dimitriadis, Horkay, Maresca, Kachar, Chadwick, *Biophys. J.* **82**, 2798
  (2002) — finite-thickness corrections for bonded / free soft films.
- Garcia & Garcia, *Biophys. J.* **114**, 2923 (2018) — bottom-effect
  correction for paraboloidal tips.

## License

MIT.
