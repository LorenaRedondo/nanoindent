# nanoindent

[![CI](https://github.com/alpibrusl/afm-finite-elements/actions/workflows/ci.yml/badge.svg)](https://github.com/alpibrusl/afm-finite-elements/actions/workflows/ci.yml)

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

![bottom effect](docs/bottom_effect.png)

*Biophysical Journal-style figure from `examples/publication_figure.py`:
(A) F(δ) for h = 7 / 14 nm — FEM (markers) vs. Hertz (dotted) and the bonded
Dimitriadis 2002 correction (dashed); (B) the stiffening ratio F/F_Hertz.*

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
python examples/synthetic_analysis.py           # realistic dataset -> modulus recovery
python examples/publication_figure.py           # journal-ready multi-panel figure (PDF/PNG/SVG)
python examples/model_comparison.py              # Hertz/Garcia/Dimitriadis/Chadwick vs FEM
python examples/chadwick_verification.py         # incompressible (nu=0.5) mixed FEM -> Chadwick
python examples/field_heatmap.py                 # von Mises / displacement field heatmaps
```

### Publication-quality figures

`nanoindent.plotting` produces journal-ready figures: **vector output**
(PDF / SVG / EPS) plus high-dpi PNG, a **colour-blind-safe** Okabe-Ito palette,
embedded (editable) fonts, and minimal chartjunk. **Journal presets** set the
right font, panel-label convention, and physical column widths — the default is
**Biophysical Journal** (Cell Press: Arial, 7 pt, uppercase bold panel letters,
85 / 114 / 174 mm columns; Arial falls back to metric-identical Liberation Sans
on systems without it).

```python
from nanoindent import plotting
plotting.set_publication_style(journal="biophysical_journal")   # or "generic"
w = plotting.journal_widths("biophysical_journal")["double"]    # 174 mm in inches
fig = plotting.plot_force_curves({7.0: c7, 14.0: c14},
                                 compare=["hertz", "dimitriadis_bonded"])
plotting.panel_label(fig.axes[0], 0)             # "A" (Cell Press) / "(a)" generic
plotting.save_figure(fig, "figures/fig2", formats=("pdf", "png", "svg"))
```

Add a journal by dropping an entry in `plotting.JOURNAL_PRESETS`.

Available plotters: `plot_force_curves`, `plot_stiffening_ratio`,
`plot_modulus_recovery`, `plot_model_comparison`, `plot_convergence`,
`plot_field` (field heatmaps), plus `save_figure` (multi-format) and the
`SINGLE_COLUMN` / `DOUBLE_COLUMN` width constants.

### Field heatmaps

`plot_field` renders any recovered field over the axisymmetric (r, z)
cross-section — von Mises stress, `sigma_zz`, mechanical `pressure`,
displacement magnitude, `uz`, `ur` — mirrored across the axis with the rigid
tip outline overlaid. Fields are L2-projected to the mesh vertices via
`solver.nodal_field(name)`.

```python
solver = IndentationSolver(cfg); solver.run()
plotting.plot_field(solver, field="von_mises")            # heatmap
plotting.plot_field(solver, field="uz", cmap="RdBu_r", deformed=True)
vm = solver.nodal_field("von_mises")                      # raw nodal values
```

![field heatmap](docs/field_heatmap.png)

*`examples/field_heatmap.py`: von Mises stress concentrates under the tip
(~21 MPa at h=7 nm, δ=1 nm) while u_z reaches −δ at the apex and 0 at the
bonded substrate.*

---

## Indenter geometry

The indenter shape is a configuration parameter. Because the model is **2D
axisymmetric**, the tip must be a surface of revolution — four exact shapes are
built in:

| `tip_shape` | parameters | analytic benchmark |
|---|---|---|
| `"sphere"` (default) | `R` | Hertz |
| `"paraboloid"` | `R` | Hertz (parabolic tip) |
| `"cone"` | `tip_half_angle_deg` **or** `tip_pyramid` | Sneddon `F=(2/π)E*tanθ·δ²` |
| `"blunted_cone"` | `R`, `tip_half_angle_deg`/`tip_pyramid` | sphere→cone crossover |

```python
SimulationConfig(tip_shape="cone", tip_half_angle_deg=70.3)
SimulationConfig(tip_shape="blunted_cone", R=17.0, tip_half_angle_deg=35.0)
```

**Pyramids** (Berkovich, Vickers, cube-corner) are *not* axisymmetric and
cannot be meshed directly in this model. The standard workaround — an
**equivalent cone** with a half-angle reproducing the same projected-area-vs-
depth law — is built in:

```python
SimulationConfig(tip_shape="cone", tip_pyramid="berkovich")   # 70.3° equiv. cone
```

Each shape is validated against its analytic law in the thick-film limit
(sphere/paraboloid → Hertz, cone → Sneddon, all within a few percent). A sharp
cone reproduces its δ² force law; the blunted cone reduces to the sphere while
contact stays on the spherical cap and stiffens toward the cone beyond it.

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

### Near-incompressibility and the mixed formulation

For `ν ≥ 0.45` the single-field ("primal") P2 displacement formulation locks
and over-stiffens. Set `formulation="mixed"` for a **Taylor–Hood** (P2
displacement / P1 pressure) two-field formulation that is locking-free and
supports fully incompressible **`ν = 0.5`** exactly:

```python
SimulationConfig(nu=0.5, formulation="mixed", tip_shape="paraboloid", ...)
```

The mixed solve assembles the block system

```
[ A + K_contact   Bᵀ ] [u]   [f_contact]
[ B              −C   ] [p] = [   0     ]
```

with `A(u,v)=∫2μ ε:ε r`, `B(u,q)=∫(∇·u) q r`, `C(p,q)=∫(1/λ) p q r` (→ 0 at
`ν=½`), Newton-iterated over the contact active set. It reproduces the primal
result at `ν=0.3` (agreement < 0.5%).

**Verifying Chadwick.** Because it handles `ν = ½`, the mixed formulation lets
us check the Chadwick 2002 thin-incompressible-layer law directly. Sweeping the
thin-film parameter `χ = √(Rδ)/h`, the incompressible FEM force converges to
`F=(2π/3)E R²δ³/h³` as `χ` grows — recovering both the scaling and the
coefficient (`examples/chadwick_verification.py`):

![chadwick verification](docs/chadwick_verification.png)

| χ_max | 1.4 | 2.9 | 4.8 | 8.3 |
|---|---|---|---|---|
| F_FEM / F_Chadwick | 2.90 | 1.46 | 1.16 | 0.95 |

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

### Comparing contact models

`nanoindent.analytics` bundles the standard AFM contact models and lets you
compare them (and the FEM) on one grid. The default set is **Hertz**,
**Garcia & Garcia 2018**, **Dimitriadis 2002 (bonded)**, and **Chadwick 2002**:

```python
from nanoindent import SimulationConfig, analytics, plotting
cfg = SimulationConfig(h=7.0)
table = analytics.compare_models(cfg)                 # dict: delta + each model
plotting.plot_model_comparison(cfg, fem_curve=result) # overlay all + FEM
```

| key | model | regime |
|---|---|---|
| `hertz` | Hertz half-space | thick film / baseline |
| `garcia_bec` | Garcia & Garcia 2018 BEC (paraboloid) | bonded, finite h |
| `dimitriadis_bonded` | Dimitriadis 2002 bonded polynomial (general ν) | bonded, crossover |
| `chadwick_2002` | Chadwick 2002 thin **incompressible** layer, `F=(2π/3)E R²δ³/h³` | bonded, deep thin-film (ν→½) |

![model comparison](docs/model_comparison.png)

`analytics.BEC_MODELS` is the default list; pass your own `models=[...]` to
either function. Note the two 2002 papers are **distinct**: Dimitriadis *et al.*
(*Biophys. J.* **82**, 2798) is the empirical polynomial that interpolates
Hertz→thin for general ν; Chadwick (*SIAM J. Appl. Math.* **62**, 1520) is the
asymptotic thin-**incompressible**-layer law it is built on — it stiffens as
`1/h³` and only applies in the deep thin-film limit (it does *not* reduce to
Hertz for thick films). `examples/model_comparison.py` overlays all four vs. the
FEM across thicknesses.

### Ingesting real AFM experiments

`nanoindent.experiment` reads **raw instrument force curves**, calibrates them
to force-indentation, and fits the modulus. An AFM records cantilever
**deflection** vs. **piezo-Z**, so you supply the spring constant `k` (N/m) and,
if the deflection is in volts, the sensitivity InvOLS (nm/V); the module handles
baseline subtraction, contact-point detection, and the cantilever-compliance
correction `δ = (Z−Z₀) − (d−d₀)`.

Supported formats (dispatched by extension via `load_afm`):

| format | extension | reader |
|---|---|---|
| JPK | `.jpk-force`, `.jpk` | `read_jpk` (ZIP: segment headers + channel data) |
| Bruker / Veeco Nanoscope | `.spm`, `.pfc` | `read_bruker` (ASCII header + int16 data) |
| Asylum Research | `.ibw` | `read_ibw` (Igor wave; needs `pip install igor2`) |
| generic | `.csv`, `.txt`, `.tsv` | `read_csv` (Z, deflection columns) |

```python
from nanoindent import load_afm, analyze_experiment
curve = load_afm("cell_force.jpk-force")          # or .spm / .ibw / .csv
fit, (delta, force) = analyze_experiment(
    curve, R=17.0, nu=0.5, h=7.0, model="dimitriadis_bonded")
print(fit.E_fit)                                   # recovered modulus (MPa)
```

The whole path is validated by a synthetic round-trip: `synthetic_raw_curve`
generates the `(Z, deflection)` an AFM would record at a known `E`, and
`to_force_indentation` → `fit_modulus` recovers it (`examples/analyze_afm_experiment.py`
— demo curve at E=130 MPa recovers ~125 MPa with Dimitriadis while naive Hertz
reads ~250 MPa). Instrument readers target the documented layouts; the CSV path
and calibration pipeline are backend-free and fully tested — verify the binary
readers against your own exports.

### Synthetic data & inverse analysis (end-to-end double-check)

`nanoindent.dataset` closes the loop: it generates a **realistic AFM force
curve** (measurement noise, baseline offset, unknown contact point) from a
ground-truth source — including the FEM itself — and then **fits the modulus
back out** with a chosen contact model, contact-point determination included.

```python
from nanoindent import SimulationConfig, synthetic_force_curve, fit_modulus
cfg = SimulationConfig(h=7.0, E=130.0)
ds = synthetic_force_curve(cfg, source="dimitriadis", noise_pN=10.0)
naive  = fit_modulus(ds.delta, ds.force, "hertz",       cfg.R, cfg.nu, cfg.h, fit_contact_point=True)
bec    = fit_modulus(ds.delta, ds.force, "dimitriadis", cfg.R, cfg.nu, cfg.h, fit_contact_point=True)
print(naive.E_fit, bec.E_fit)   # ~267 MPa vs ~130 MPa
```

Running `examples/synthetic_analysis.py` (ground truth **E = 130 MPa**, data
generated from the **FEM**, ~10 pN noise) demonstrates the bottom-effect
artefact and its correction:

| h (nm) | Hertz fit E | Dimitriadis fit E |
|-------:|------------:|------------------:|
|   7    | 267 MPa (**+106 %**) | 134 MPa (+3 %) |
|  14    | 182 MPa (**+40 %**)  | 130 MPa (−0 %) |
| 200    | 131 MPa (+0 %)       | 128 MPa (−1 %) |

The naive Hertz fit dramatically over-estimates the modulus on thin bonded
films (and correctly agrees with Hertz for the 200 nm half-space, an
independent check on the solver), while the Dimitriadis bonded correction
recovers the true modulus at every thickness — confirming the forward FEM and
the analytic corrections are mutually consistent.

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
├── contact.py      # tip zoo (sphere/paraboloid/cone/blunted): gap, penalty
├── solver.py       # IndentationSolver: assembly, Newton, reaction extraction
├── postproc.py     # ForceCurve: CSV IO, diagnostics, plotting
├── analytics.py    # Hertz; Dimitriadis 2002; Garcia 2018; Chadwick 2002; compare_models
├── convergence.py  # mesh / penalty / load-step studies
├── dataset.py      # synthetic AFM data + modulus-recovery inverse analysis
├── experiment.py   # raw AFM ingestion (JPK/Bruker/Asylum/CSV) + calibration
└── plotting.py     # publication-quality figures (vector output, CB-safe)
examples/
├── paper_reproduction.py       ├── model_comparison.py
├── hertz_validation.py         ├── chadwick_verification.py
├── synthetic_analysis.py       ├── field_heatmap.py
├── analyze_afm_experiment.py   ├── publication_figure.py
├── msnl_bio_dc.py              └── (case study: R=20 nm, h=6.2 nm)
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
