import numpy as np
import pytest

from nanoindent import analytics


def test_hertz_scaling():
    R, E, nu = 17.0, 130.0, 0.3
    d = np.array([0.25, 1.0])
    F = analytics.hertz_force(d, R, E, nu)
    # F ~ delta^1.5 -> ratio for 4x delta is 8x
    assert F[1] / F[0] == pytest.approx(8.0, rel=1e-9)
    # closed-form value
    Estar = E / (1 - nu ** 2)
    assert F[1] == pytest.approx((4 / 3) * Estar * np.sqrt(R) * 1.0 ** 1.5)


def test_hertz_zero_at_zero():
    assert analytics.hertz_force(0.0, 17.0, 130.0, 0.3) == 0.0


def test_bec_models_reduce_to_hertz_for_thick_film():
    R, E, nu = 17.0, 130.0, 0.3
    d = np.linspace(0.05, 1.0, 8)
    h = 5000.0  # extremely thick
    fh = analytics.hertz_force(d, R, E, nu)
    fd = analytics.dimitriadis_force(d, R, E, nu, h, bonded=True)
    fg = analytics.garcia_bec_force(d, R, E, nu, h)
    assert np.allclose(fd, fh, rtol=2e-3)
    assert np.allclose(fg, fh, rtol=2e-3)


def test_bec_models_stiffen_for_thin_film():
    R, E, nu = 17.0, 130.0, 0.3
    d = np.linspace(0.1, 1.0, 5)
    fh = analytics.hertz_force(d, R, E, nu)
    f7 = analytics.dimitriadis_force(d, R, E, nu, 7.0, bonded=True)
    f14 = analytics.dimitriadis_force(d, R, E, nu, 14.0, bonded=True)
    # thinner film stiffer than thicker stiffer than half-space
    assert np.all(f7 > f14)
    assert np.all(f14 > fh)


def test_incompressible_tabulated_coefficients():
    # At chi -> 0 the incompressible bonded formula reduces to (16/9) E sqrt(R) d^1.5
    R, E, h = 17.0, 130.0, 1e6
    d = 0.5
    F = analytics.dimitriadis_bonded_incompressible(d, R, E, h)
    expected = (16 / 9) * E * np.sqrt(R) * d ** 1.5
    assert F == pytest.approx(expected, rel=1e-4)


def test_sneddon_cone_scaling():
    # F ~ delta^2 for a cone; doubling delta quadruples F.
    F = analytics.sneddon_cone_force(np.array([0.5, 1.0]), 70.3, 130.0, 0.3)
    assert F[1] / F[0] == pytest.approx(4.0, rel=1e-9)
    Estar = 130.0 / (1 - 0.3 ** 2)
    expected = (2 / np.pi) * Estar * np.tan(np.deg2rad(70.3)) * 1.0 ** 2
    assert F[1] == pytest.approx(expected)


def test_paraboloid_equals_hertz():
    d = np.array([0.3, 0.7])
    assert np.allclose(
        analytics.paraboloid_force(d, 17.0, 130.0, 0.3),
        analytics.hertz_force(d, 17.0, 130.0, 0.3))


def test_contact_radius_hertz():
    a = analytics.contact_radius_hertz(np.array([1.0, 4.0]), 17.0)
    assert a[0] == pytest.approx(np.sqrt(17.0))
    assert a[1] == pytest.approx(np.sqrt(68.0))


def test_registry_keys():
    reg = analytics.model_registry()
    for k in ("hertz", "dimitriadis_bonded", "garcia_bec"):
        assert k in reg
        cfg = type("C", (), dict(R=17.0, E=130.0, nu=0.3, h=7.0))()
        val = reg[k].func(np.array([0.5]), cfg)
        assert np.all(val > 0)
