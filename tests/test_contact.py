import numpy as np
import pytest

from nanoindent.contact import RigidSphere, penalty_stiffness


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
