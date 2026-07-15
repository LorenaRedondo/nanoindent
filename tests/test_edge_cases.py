"""Edge-case and boundary-condition tests across all modules.

These target the corners that ordinary "happy path" tests miss: parameter
boundaries, degenerate inputs, geometric singularities (r = R, r = 0, delta = 0),
empty/malformed data, and API misuse. They are deliberately fast (no heavy FEM
solves) so they can run in the CI fast suite.
"""

import math
import numpy as np
import pytest

from nanoindent import (
    SimulationConfig, analytics, fit_modulus, synthetic_force_curve,
    ForceCurveData, load_afm, read_csv,
)
from nanoindent.contact import (
    RigidSphere, Paraboloid, Cone, BluntedCone, make_tip,
    equivalent_cone_half_angle,
)
from nanoindent.geometry import _graded_coords, build_mesh
import nanoindent.experiment as exp


# ===========================================================================
# config: parameter boundaries
# ===========================================================================
@pytest.mark.parametrize("kw", [
    {"R": 0}, {"R": -1}, {"h": 0}, {"E": -5}, {"L": 0},
    {"delta_max": 0}, {"n_steps": 0}, {"penalty_alpha": 0},
    {"mesh_size_contact": 0}, {"element_order": 3},
])
def test_config_rejects_nonpositive_or_bad(kw):
    with pytest.raises(ValueError):
        SimulationConfig(**kw)


def test_nu_boundaries():
    SimulationConfig(nu=0.49)                       # allowed (warns)
    with pytest.raises(ValueError):
        SimulationConfig(nu=0.5)                     # primal rejects incompressible
    SimulationConfig(nu=0.5, formulation="mixed")    # mixed allows incompressible
    with pytest.raises(ValueError):
        SimulationConfig(nu=0.6, formulation="mixed")   # nu > 0.5 always invalid
    with pytest.raises(ValueError):
        SimulationConfig(nu=-1.0)                        # nu <= -1 invalid


def test_delta_max_vs_R_only_for_sphere():
    with pytest.raises(ValueError):
        SimulationConfig(tip_shape="sphere", delta_max=20.0, R=17.0)
    # paraboloid and cone have no r<R restriction
    SimulationConfig(tip_shape="paraboloid", delta_max=20.0, R=17.0)
    SimulationConfig(tip_shape="cone", tip_half_angle_deg=70.0,
                     delta_max=20.0, R=17.0)


def test_mesh_edge_must_exceed_contact():
    with pytest.raises(ValueError):
        SimulationConfig(mesh_size_contact=1.0, mesh_size_edge=0.5)


def test_invalid_substrate():
    with pytest.raises(ValueError):
        SimulationConfig(substrate="glued")
    SimulationConfig(substrate="bonded")
    SimulationConfig(substrate="unbonded")


def test_invalid_formulation_and_tip():
    with pytest.raises(ValueError):
        SimulationConfig(formulation="explicit")
    with pytest.raises(ValueError):
        SimulationConfig(tip_shape="wedge")
    with pytest.raises(ValueError):
        SimulationConfig(tip_shape="cone")          # angle/pyramid missing


# ===========================================================================
# contact geometry: singular radii
# ===========================================================================
def test_sphere_at_and_beyond_R():
    s = RigidSphere(R=17.0, h=7.0)
    # exactly at R: no contact (surface_z is +inf)
    assert not np.isfinite(s.surface_z(np.array([17.0]), 0.5)[0])
    # beyond R: infinite gap, zero penetration
    assert s.penetration(np.array([20.0]), np.array([0.0]), 0.5)[0] == 0.0


def test_zero_indentation_no_penetration():
    for tip in (RigidSphere(17.0, 7.0), Paraboloid(17.0, 7.0),
                Cone(70.0, 7.0), BluntedCone(17.0, 50.0, 7.0)):
        r = np.linspace(0.0, 10.0, 20)
        pen = tip.penetration(r, np.zeros_like(r), delta=0.0)
        assert np.all(pen <= 1e-12)


def test_tip_surface_handles_2d_quadrature_arrays():
    # solver passes (n_facets, n_qp) arrays; surface_z must be shape-preserving
    r = np.linspace(0.1, 5.0, 12).reshape(3, 4)
    for tip in (RigidSphere(17.0, 7.0), Paraboloid(17.0, 7.0),
                Cone(70.0, 7.0), BluntedCone(17.0, 50.0, 7.0)):
        assert tip.surface_z(r, 0.5).shape == (3, 4)


def test_cone_angle_limits():
    with pytest.raises(ValueError):
        Cone(0.0, 7.0)
    with pytest.raises(ValueError):
        Cone(90.0, 7.0)


def test_make_tip_unknown_shape():
    class C:
        tip_shape = "torus"; R = 17.0; h = 7.0
    with pytest.raises(ValueError):
        make_tip(C())


# ===========================================================================
# analytics: limits and clipping
# ===========================================================================
def test_all_models_zero_at_zero_indentation():
    cfg = SimulationConfig(h=7.0)
    for key, m in analytics.model_registry().items():
        assert m.func(np.array([0.0]), cfg)[0] == pytest.approx(0.0, abs=1e-12)


def test_negative_indentation_clipped():
    # models clip delta<0 to 0 (no imaginary forces from delta**1.5 etc.)
    for f in (
        lambda: analytics.hertz_force(-1.0, 17, 130, 0.3),
        lambda: analytics.dimitriadis_force(-1.0, 17, 130, 0.3, 7.0),
        lambda: analytics.chadwick_force(-1.0, 17, 130, 7.0),
        lambda: analytics.sneddon_cone_force(-1.0, 70.0, 130, 0.3),
    ):
        val = f()
        assert np.isfinite(val) and float(val) == 0.0


def test_bec_to_hertz_thick_limit_various_nu():
    d = np.linspace(0.05, 1.0, 6)
    for nu in (0.0, 0.3, 0.45):
        fh = analytics.hertz_force(d, 17, 130, nu)
        fd = analytics.dimitriadis_force(d, 17, 130, nu, 1e5)
        assert np.allclose(fd, fh, rtol=3e-3)


def test_compare_models_scalar_delta():
    cfg = SimulationConfig(h=7.0)
    out = analytics.compare_models(cfg, delta=0.5)
    assert np.ndim(out["hertz"]) == 0 or out["hertz"].size == 1


# ===========================================================================
# geometry
# ===========================================================================
def test_graded_coords_focus_must_be_endpoint():
    with pytest.raises(ValueError):
        _graded_coords(0.0, 10.0, focus=5.0, size_min=0.1, size_max=1.0)


def test_graded_coords_endpoints_exact():
    c = _graded_coords(0.0, 10.0, focus=0.0, size_min=0.1, size_max=1.0)
    assert c[0] == 0.0 and c[-1] == 10.0
    assert np.all(np.diff(c) > 0)                   # strictly increasing


def test_very_thin_film_mesh_has_layers():
    dom = build_mesh(SimulationConfig(h=2.0, L=50.0, mesh_size_contact=0.1,
                                      mesh_size_edge=5.0))
    z = np.unique(np.round(dom.mesh.p[1], 8))
    assert len(z) >= 4


# ===========================================================================
# dataset / fit_modulus
# ===========================================================================
def test_fit_too_few_points():
    with pytest.raises(ValueError):
        fit_modulus(np.array([0.1, 0.2]), np.array([1.0, 2.0]), "hertz",
                    17, 0.3, 7.0)


def test_fit_zero_force_gives_zero_modulus():
    d = np.linspace(0.1, 1.0, 10)
    fit = fit_modulus(d, np.zeros_like(d), "hertz", 17, 0.3, 7.0)
    assert fit.E_fit == pytest.approx(0.0, abs=1e-9)


def test_fit_recovers_known_modulus_exact():
    cfg = SimulationConfig(h=7.0, E=130.0, nu=0.3, delta_max=1.0)
    ds = synthetic_force_curve(cfg, source="hertz", noise_pN=0.0,
                               include_precontact=False)
    fit = fit_modulus(ds.delta, ds.force, "hertz", cfg.R, cfg.nu, cfg.h)
    assert fit.E_fit == pytest.approx(130.0, rel=1e-6)


def test_synthetic_force_unknown_model():
    with pytest.raises(ValueError):
        synthetic_force_curve(SimulationConfig(), source="bogus")


# ===========================================================================
# experiment ingestion
# ===========================================================================
def test_force_curve_length_mismatch():
    with pytest.raises(ValueError):
        ForceCurveData(z=np.arange(5.0), deflection=np.arange(4.0))


def test_missing_spring_constant():
    fc = ForceCurveData(z=np.arange(5.0), deflection=np.arange(5.0))
    with pytest.raises(ValueError):
        fc.to_force_indentation()                    # no k anywhere


def test_volts_without_inv_ols():
    fc = ForceCurveData(z=np.arange(5.0), deflection=np.arange(5.0),
                        deflection_unit="V", spring_constant=0.1)
    with pytest.raises(ValueError):
        fc.deflection_nm()


def test_load_afm_unknown_extension():
    with pytest.raises(ValueError):
        load_afm("data.bogusext")


def test_csv_drops_header_and_nan_rows(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("z,defl\n0,0\n1,1\n\n2,2\n")
    c = read_csv(str(p), z_col=0, deflection_col=1, spring_constant=0.1)
    # header + blank line dropped -> 3 numeric rows
    assert len(c.z) == 3 and np.allclose(c.z, [0, 1, 2])


def test_contact_point_detection_flat_curve():
    # a curve that never rises (all baseline) should not crash; falls back
    z = np.linspace(0, 10, 50)
    d = np.full_like(z, 3.0)
    fc = ForceCurveData(z=z, deflection=d, spring_constant=0.1)
    delta, force = fc.to_force_indentation()
    assert np.all(np.abs(force) < 1e-6)              # ~zero force everywhere


# ===========================================================================
# solver API misuse (no heavy solves)
# ===========================================================================
def test_nodal_field_before_run_raises():
    from nanoindent import IndentationSolver
    solver = IndentationSolver(SimulationConfig(h=7.0))
    with pytest.raises(Exception):
        solver.nodal_field("von_mises")


def test_equivalent_cone_unknown():
    with pytest.raises(ValueError):
        equivalent_cone_half_angle("spherical")
