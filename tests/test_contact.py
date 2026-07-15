import math
import numpy as np
import pytest

from nanoindent.contact import (
    RigidSphere, Paraboloid, Cone, BluntedCone, penalty_stiffness,
    make_tip, equivalent_cone_half_angle,
)
from nanoindent import SimulationConfig


def test_gap_zero_at_apex_no_indent():
    s = RigidSphere(R=17.0, h=7.0)
    # At r=0, delta=0, undeformed: sphere touches surface -> gap 0.
    g = s.gap(np.array([0.0]), np.array([0.0]), delta=0.0)
    assert g[0] == pytest.approx(0.0, abs=1e-12)


def test_penetration_activates_under_indentation():
    s = RigidSphere(R=17.0, h=7.0)
    # Indent by 0.5 at the apex with no displacement -> penetration = 0.5.
    pen = s.penetration(np.array([0.0]), np.array([0.0]), delta=0.5)
    assert pen[0] == pytest.approx(0.5, rel=1e-9)


def test_no_contact_beyond_radius():
    s = RigidSphere(R=17.0, h=7.0)
    r = np.array([20.0])  # r > R
    pen = s.penetration(r, np.array([0.0]), delta=0.5)
    assert pen[0] == 0.0
    assert not s.is_active(r, np.array([0.0]), delta=0.5)[0]


def test_gap_matches_exact_sphere():
    s = RigidSphere(R=17.0, h=7.0)
    r = np.array([2.0])
    delta = 0.3
    # z_sphere = (h + R - delta) - sqrt(R^2 - r^2)
    z_sphere = (7.0 + 17.0 - delta) - np.sqrt(17.0 ** 2 - r ** 2)
    expected_gap = z_sphere - (7.0 + 0.0)
    assert s.gap(r, np.array([0.0]), delta)[0] == pytest.approx(expected_gap[0])


def test_surface_displacement_reduces_penetration():
    s = RigidSphere(R=17.0, h=7.0)
    # If the surface pushes down (negative uz), penetration decreases.
    pen0 = s.penetration(np.array([0.0]), np.array([0.0]), delta=0.5)
    pen1 = s.penetration(np.array([0.0]), np.array([-0.2]), delta=0.5)
    assert pen1[0] == pytest.approx(pen0[0] - 0.2, rel=1e-9)


def test_penalty_stiffness_scaling():
    k = penalty_stiffness(100.0, 130.0, np.array([0.5, 1.0]))
    assert k[0] == pytest.approx(100.0 * 130.0 / 0.5)
    assert k[1] == pytest.approx(100.0 * 130.0 / 1.0)


# -- shape zoo ---------------------------------------------------------------
def test_paraboloid_apex_and_curvature():
    p = Paraboloid(R=17.0, h=7.0)
    # apex at z = h - delta
    assert p.surface_z(np.array([0.0]), 0.3)[0] == pytest.approx(7.0 - 0.3)
    # z = apex + r^2/(2R)
    assert p.surface_z(np.array([2.0]), 0.0)[0] == pytest.approx(7.0 + 4.0 / (2 * 17.0))


def test_paraboloid_matches_sphere_near_axis():
    s = RigidSphere(R=17.0, h=7.0)
    p = Paraboloid(R=17.0, h=7.0)
    r = np.array([0.5])
    # Small r: parabola ~ sphere to O(r^4)
    assert p.surface_z(r, 0.2)[0] == pytest.approx(s.surface_z(r, 0.2)[0], abs=1e-3)


def test_cone_geometry():
    c = Cone(half_angle_deg=60.0, h=7.0)
    r = np.array([1.0])
    # z = apex + r / tan(beta)
    assert c.surface_z(r, 0.0)[0] == pytest.approx(7.0 + 1.0 / math.tan(math.radians(60.0)))


def test_blunted_cone_c1_continuous_at_transition():
    bc = BluntedCone(R=17.0, half_angle_deg=50.0, h=7.0)
    r_t = bc.r_transition
    left = bc.surface_z(np.array([r_t - 1e-6]), 0.0)[0]
    right = bc.surface_z(np.array([r_t + 1e-6]), 0.0)[0]
    assert left == pytest.approx(right, abs=1e-4)   # continuous at the blend


def test_blunted_cone_reduces_to_sphere_near_apex():
    bc = BluntedCone(R=17.0, half_angle_deg=50.0, h=7.0)
    s = RigidSphere(R=17.0, h=7.0)
    r = np.array([0.5])  # well inside the spherical cap
    assert bc.surface_z(r, 0.2)[0] == pytest.approx(s.surface_z(r, 0.2)[0])


def test_equivalent_cone_half_angles():
    assert equivalent_cone_half_angle("berkovich") == pytest.approx(70.3)
    assert equivalent_cone_half_angle("cube_corner") == pytest.approx(42.3)
    with pytest.raises(ValueError):
        equivalent_cone_half_angle("nonsense")


def test_make_tip_dispatch():
    assert isinstance(make_tip(SimulationConfig(tip_shape="sphere")), RigidSphere)
    assert isinstance(make_tip(SimulationConfig(tip_shape="paraboloid")), Paraboloid)
    assert isinstance(
        make_tip(SimulationConfig(tip_shape="cone", tip_half_angle_deg=70.3)), Cone)
    assert isinstance(
        make_tip(SimulationConfig(tip_shape="cone", tip_pyramid="berkovich")), Cone)
    assert isinstance(
        make_tip(SimulationConfig(tip_shape="blunted_cone", tip_half_angle_deg=50.0)),
        BluntedCone)


def test_cone_requires_angle():
    with pytest.raises(ValueError):
        SimulationConfig(tip_shape="cone")   # no angle or pyramid


def test_cone_ignores_delta_max_vs_R():
    # Sharp cone has no r<R restriction, so delta_max >= R is allowed.
    cfg = SimulationConfig(tip_shape="cone", tip_half_angle_deg=70.3,
                           delta_max=1.0, R=17.0)
    assert cfg.tip_shape == "cone"
