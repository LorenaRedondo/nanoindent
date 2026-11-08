import numpy as np
import pytest

from nanoindent import analytics, SimulationConfig


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


# -- Dimitriadis 2002 coefficient anchors ------------------------------------
# The general-nu alpha0/beta0 machinery must reproduce the paper's own
# tabulations at nu = 1/2. These caught three real errors: the bonded chi^4
# sign, free-film coefficients wrong by 4x at leading order, and
# DIMITRIADIS_BONDED_INCOMPRESSIBLE holding Garcia's series instead.

def _poly_coeffs(nu, bonded):
    """Recover [c0..c4] of P(chi) by fitting the evaluated polynomial."""
    x = np.linspace(0.01, 1.0, 9)
    return np.polyfit(x, analytics._dimitriadis_poly(x, nu, bonded), 4)[::-1]


@pytest.mark.parametrize("bonded,published", [
    (True, analytics.DIMITRIADIS_BONDED_INCOMPRESSIBLE),
    (False, analytics.DIMITRIADIS_NOT_BONDED_INCOMPRESSIBLE),
])
def test_general_nu_series_reproduces_published_tabulation(bonded, published):
    got = _poly_coeffs(0.5, bonded)
    # chi^4 is the least certain term (see _ORDER4_SIGN); it is worth <2.5% of
    # the correction at chi = 1, so it gets a looser tolerance.
    np.testing.assert_allclose(got[:4], published[:4], rtol=1e-3)
    assert got[4] == pytest.approx(published[4], abs=5e-4)


def test_bonded_chi4_coefficient_is_positive():
    """A bonded film stiffens monotonically; a negative chi^4 term was a bug."""
    assert _poly_coeffs(0.5, True)[4] > 0


def test_dimitriadis_is_not_garcia():
    """The two papers give different series; conflating them was the old bug."""
    dim = analytics.DIMITRIADIS_BONDED_INCOMPRESSIBLE
    gar = analytics.GARCIA_BONDED_COEFFS
    assert dim[:2] == gar[:2]           # they agree to first order
    assert dim[2:] != gar[2:]           # and diverge from chi^2 onward
    assert sum(gar) / sum(dim) > 1.2    # ~30% apart at chi = 1


def test_free_film_is_softer_than_bonded():
    R, E, h = 17.0, 130.0, 7.0
    d = np.linspace(0.1, 1.0, 8)
    free = analytics.dimitriadis_force(d, R, E, 0.5, h, bonded=False)
    bond = analytics.dimitriadis_force(d, R, E, 0.5, h, bonded=True)
    hertz = analytics.hertz_force(d, R, E, 0.5)
    assert np.all(hertz < free)
    assert np.all(free < bond)


def test_free_film_warns_away_from_incompressible():
    with pytest.warns(UserWarning, match="only for nu = 1/2"):
        analytics.dimitriadis_force(0.5, 17.0, 130.0, 0.3, 7.0, bonded=False)




def test_sneddon_cone_scaling():
    # F ~ delta^2 for a cone; doubling delta quadruples F.
    F = analytics.sneddon_cone_force(np.array([0.5, 1.0]), 70.3, 130.0, 0.3)
    assert F[1] / F[0] == pytest.approx(4.0, rel=1e-9)
    Estar = 130.0 / (1 - 0.3 ** 2)
    expected = (2 / np.pi) * Estar * np.tan(np.deg2rad(70.3)) * 1.0 ** 2
    assert F[1] == pytest.approx(expected)


def test_connell_area_compressibility():
    # F = pi * kappa_A * R * z0^2 / (2d - z0)^2 ; units: mN/m, nm -> pN
    F = analytics.connell_force(1.0, 20.0, 250.0, 5.0)
    assert float(F) == pytest.approx(np.pi * 250.0 * 20.0 * 1.0 / (5.0 - 1.0) ** 2)
    # diverges toward the substrate (z0 -> 2d) and is 0 at z0 = 0
    assert analytics.connell_force(0.0, 20.0, 250.0, 5.0) == 0.0
    assert analytics.connell_force(4.9, 20.0, 250.0, 5.0) > \
        analytics.connell_force(2.0, 20.0, 250.0, 5.0)


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
    for k in ("hertz", "dimitriadis_bonded", "garcia_bec", "chadwick_2002"):
        assert k in reg
        cfg = type("C", (), dict(R=17.0, E=130.0, nu=0.3, h=7.0))()
        val = reg[k].func(np.array([0.5]), cfg)
        assert np.all(val > 0)


def test_chadwick_scaling_and_coefficient():
    R, E, h = 17.0, 130.0, 5.0
    d = np.array([0.5, 1.0])
    F = analytics.chadwick_force(d, R, E, h)
    # F ~ delta^3: doubling delta gives 8x
    assert F[1] / F[0] == pytest.approx(8.0, rel=1e-9)
    # exact coefficient F = (2 pi/3) E R^2 delta^3 / h^3
    assert F[1] == pytest.approx((2 * np.pi / 3) * E * R ** 2 * 1.0 ** 3 / h ** 3)


def test_chadwick_inverse_cube_thickness():
    # Bonded incompressible thin layer stiffens as 1/h^3.
    F_h = analytics.chadwick_force(1.0, 17.0, 130.0, 5.0)
    F_2h = analytics.chadwick_force(1.0, 17.0, 130.0, 10.0)
    assert F_h / F_2h == pytest.approx(8.0, rel=1e-9)


def test_chadwick_dominates_in_thin_limit():
    # As h shrinks, Chadwick overtakes Hertz (thin-film stiffening).
    d = 1.0
    thin = analytics.chadwick_force(d, 17.0, 130.0, 2.0)
    hertz = analytics.hertz_force(d, 17.0, 130.0, 0.3)
    assert thin > hertz


def test_compare_models_returns_all_keys():
    cfg = SimulationConfig(h=7.0, R=17.0, E=130.0, nu=0.3, delta_max=1.0)
    out = analytics.compare_models(cfg, delta=np.array([0.5, 1.0]))
    assert "delta" in out
    for k in analytics.BEC_MODELS:
        assert k in out and out[k].shape == (2,)


def test_compare_models_unknown_raises():
    cfg = SimulationConfig(h=7.0)
    with pytest.raises(ValueError):
        analytics.compare_models(cfg, models=["hertz", "nonsense"])
