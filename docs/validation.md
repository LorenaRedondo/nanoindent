# Validation matrix

What the solver is checked against, over what range, and which test enforces it.
Tolerances below are the **asserted** thresholds in the test suite, not measured
accuracy claims. Agreement is bounded by those thresholds, and only within the
stated configuration. See [`design.md`](design.md) section 6 for the reasoning.

Tests marked *slow* run with `pytest -m slow`.

## Model assumptions

Linear isotropic elasticity, small-strain kinematics, quasi-static loading,
frictionless unilateral contact, axisymmetric geometry, rigid indenter and
substrate. Every result exposes these through
`ForceCurve.applicability()` together with dimensionless ratios and warnings
(see `nanoindent.quality`).

## Analytical benchmarks

| Benchmark | Reference | Configuration | Asserted tolerance | Test |
|---|---|---|---|---|
| Hertz, half-space limit (sphere) | Hertz | thick film | per-step ratio within 5%, mean within 2% | `tests/test_hertz.py::test_hertz_thick_film_limit` |
| Hertz, paraboloid tip | Hertz | thick film | mean ratio within 3% | `tests/test_hertz.py::test_paraboloid_recovers_hertz_thick_film` |
| Sneddon, cone | Sneddon 1965 | thick film | mean relative error below 5% | `tests/test_hertz.py::test_cone_recovers_sneddon_thick_film` |
| Bottom effect, bonded | Dimitriadis et al. 2002 | finite thickness | mean relative error below 10% | `tests/test_hertz.py::test_bottom_effect_matches_dimitriadis` |
| Thin incompressible layer | Chadwick 2002 | mixed formulation | ratio to law decreases monotonically with the thickness parameter, and the last ratio is within 15% of 1 | `tests/test_hertz.py::test_chadwick_limit_mixed_incompressible` |
| Contact radius | Hertz | thick film | see test | `tests/test_hertz.py::test_contact_radius_thick_film` |
| Garcia and Garcia 2018 (analytic form) | Garcia and Garcia 2018 | golden values of the formula | exact to stored values | `tests/test_golden.py::test_analytic_golden` |

## Qualitative and structural checks

| Property | Test |
|---|---|
| Thinner film is stiffer | `tests/test_hertz.py::test_thinner_film_is_stiffer` |
| Incompressible film stiffer than compressible | `tests/test_hertz.py::test_incompressible_film_stiffer_than_compressible` |
| Unbonded softer than bonded | `tests/test_solver.py::test_unbonded_softer_than_bonded` |
| Mixed matches primal when compressible | `tests/test_solver.py::test_mixed_matches_primal_compressible` |
| Force is monotone, zero at zero depth | `tests/test_solver.py::test_force_monotone_increasing` |
| Contact vs substrate reaction equilibrium (below 1.5%, refined mesh) | `tests/test_hertz.py::test_equilibrium_under_one_percent_refined` |
| Interpenetration below 1% of depth | `tests/test_hertz.py::test_penalty_interpenetration_below_one_percent` |
| Exact sphere geometry, not the parabola | `tests/test_contact.py::test_gap_matches_exact_sphere` |
| Load-step path independence | `tests/test_convergence.py::test_load_step_study_is_path_independent` |
| Penalty plateau | `tests/test_convergence.py::test_penalty_study_plateaus` |
| Mesh convergence study | `tests/test_convergence.py::test_mesh_study_converges` |
| FEM regression (golden values) | `tests/test_golden.py::test_fem_golden_sphere_h14`, `::test_fem_golden_paraboloid_h7` |
| Applicability and quality reports | `tests/test_quality.py` |

## What this does not establish

- Agreement with experiment. Synthetic recovery is not experimental validation.
- Accuracy outside the listed configurations, in particular at large
  depth-to-thickness ratios where small-strain kinematics no longer holds.
- A universal error percentage. Variation across meshes is *mesh sensitivity*;
  contact problems can show small non-monotonic changes as the active set
  changes.
