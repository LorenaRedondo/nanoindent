"""Property-based tests (hypothesis).

Instead of fixed inputs, these assert invariants that must hold for *any* valid
parameters: force is non-negative and finite, Hertz is monotone and zero at
zero, a bonded film is never softer than the half-space, and the modulus fit
inverts the forward model. Fuzzing the whole valid parameter box catches
crashes and NaNs that hand-picked cases miss.
"""

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st

from nanoindent import SimulationConfig, analytics, fit_modulus, synthetic_force_curve


# bounded, finite strategies over the valid parameter box
R_ = st.floats(1.0, 100.0, allow_nan=False, allow_infinity=False)
E_ = st.floats(0.5, 5000.0, allow_nan=False, allow_infinity=False)
NU_ = st.floats(-0.9, 0.49, allow_nan=False, allow_infinity=False)
H_ = st.floats(1.0, 5000.0, allow_nan=False, allow_infinity=False)
D_ = st.floats(0.0, 5.0, allow_nan=False, allow_infinity=False)

SET = settings(max_examples=60, deadline=None)


@SET
@given(R=R_, h=st.floats(1.0, 500.0), L=st.floats(1.0, 500.0), E=E_, nu=NU_,
       dmax=st.floats(0.01, 5.0))
def test_valid_config_constructs_with_finite_derived(R, h, L, E, nu, dmax):
    # keep delta_max < R so the default sphere gap is valid
    dmax = min(dmax, 0.9 * R)
    cfg = SimulationConfig(R=R, h=h, L=max(L, 0.2), E=E, nu=nu, delta_max=dmax,
                           mesh_size_contact=0.05, mesh_size_edge=5.0)
    assert np.isfinite(cfg.E_star) and cfg.E_star > 0
    assert np.isfinite(cfg.lame_mu) and cfg.lame_mu > 0


@SET
@given(d=D_, R=R_, E=E_, nu=NU_)
def test_hertz_nonneg_and_zero_at_zero(d, R, E, nu):
    f = float(analytics.hertz_force(d, R, E, nu))
    assert np.isfinite(f) and f >= 0.0
    assert analytics.hertz_force(0.0, R, E, nu) == pytest.approx(0.0, abs=1e-12)


@SET
@given(R=R_, E=E_, nu=NU_, d=st.floats(0.01, 4.0))
def test_hertz_monotone_increasing(R, E, nu, d):
    lo = analytics.hertz_force(d, R, E, nu)
    hi = analytics.hertz_force(d * 1.5, R, E, nu)
    assert hi >= lo


@SET
@given(R=R_, E=E_, nu=NU_, d=st.floats(0.01, 2.0), h=H_)
def test_bonded_film_not_softer_than_halfspace(R, E, nu, d, h):
    # restrict to the small-contact regime (chi <= 1) where the bonded
    # Dimitriadis correction is a monotone stiffening (poly >= 1)
    h = max(h, np.sqrt(R * d))
    fh = float(analytics.hertz_force(d, R, E, nu))
    fd = float(analytics.dimitriadis_force(d, R, E, nu, h))
    fc = float(analytics.chadwick_force(d, R, E, h))
    assert np.isfinite(fd) and np.isfinite(fc)
    assert fd >= fh - 1e-9
    assert fc >= -1e-9


@SET
@given(E=st.floats(1.0, 1000.0), R=st.floats(2.0, 60.0),
       h=st.floats(50.0, 2000.0), nu=st.floats(0.0, 0.45))
def test_fit_modulus_inverts_forward_model(E, R, h, nu):
    # noiseless data from the bonded model must invert back to the same E
    cfg = SimulationConfig(R=R, E=E, nu=nu, h=h, delta_max=1.0)
    ds = synthetic_force_curve(cfg, source="dimitriadis", noise_pN=0.0,
                               include_precontact=False)
    fit = fit_modulus(ds.delta, ds.force, "dimitriadis", R, nu, h)
    assert fit.E_fit == pytest.approx(E, rel=1e-4)
