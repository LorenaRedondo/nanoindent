"""Tests for recovery between consecutive indentations."""

import warnings
import numpy as np
import pytest

from nanoindent import SimulationConfig
from nanoindent import recovery as rec


def _cfg():
    return SimulationConfig(R=20.0, E=5.0, nu=0.5, h=6.0, L=120.0, delta_max=1.2,
                            formulation="mixed", mesh_size_contact=0.4,
                            mesh_size_edge=10.0)


@pytest.fixture(scope="module")
def footprint():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rec.surface_footprint(_cfg())


# -- footprint -------------------------------------------------------------
def test_footprint_normalised_and_decays(footprint):
    assert footprint.s[0] == pytest.approx(1.0)
    a = footprint.contact_radius
    # decays: near-axis > far-field
    assert footprint.shape(0.5 * a) > footprint.shape(2.0 * a)
    # localised by the bonded bottom: essentially gone a few radii out
    assert abs(footprint.shape(3.0 * a)) < 0.1
    # zero beyond the solved radial extent
    assert footprint.shape(1e6) == 0.0


def test_footprint_symmetric(footprint):
    a = footprint.contact_radius
    assert footprint.shape(a) == pytest.approx(footprint.shape(-a))


# -- residual fraction limits ----------------------------------------------
def test_residual_far_pitch_recovers(footprint):
    # A lateral step well beyond the footprint => next poke sees fresh material.
    r = rec.residual_fraction(footprint, tau=1.0, pitch_nm=6.0 * footprint.contact_radius,
                              ramp_freq_hz=100.0)
    assert abs(r) < 0.05


def test_residual_slow_scan_recovers(footprint):
    # Very slow scan (Dt >> tau) => fully recovered even at small pitch.
    r = rec.residual_fraction(footprint, tau=0.01, pitch_nm=footprint.contact_radius,
                              ramp_freq_hz=1.0)
    assert abs(r) < 0.05


def test_residual_fast_dense_does_not_recover(footprint):
    # Fast + dense (Dt << tau, pitch ~ a) => substantial residual overlap.
    r = rec.residual_fraction(footprint, tau=1.0, pitch_nm=footprint.contact_radius,
                              ramp_freq_hz=50.0)
    assert r > 0.15


def test_residual_higher_for_faster_scan(footprint):
    # Faster scanning leaves less time to recover, so more residual overlap.
    # (Not strictly step-monotone: the footprint's small pile-up ring can dip
    # the tail, so compare well-separated frequencies.)
    a = footprint.contact_radius
    slow = rec.residual_fraction(footprint, tau=0.5, pitch_nm=a, ramp_freq_hz=1.0)
    fast = rec.residual_fraction(footprint, tau=0.5, pitch_nm=a, ramp_freq_hz=100.0)
    assert fast > slow


# -- ODE raster agrees with the analytic residual --------------------------
def test_raster_matches_analytic(footprint):
    a = footprint.contact_radius
    tau, pitch, freq = 0.5, 1.2 * a, 5.0
    raster = rec.raster_surface(footprint, tau=tau, pitch_nm=pitch,
                                ramp_freq_hz=freq, n_insertions=8,
                                steps_per_cycle=120)
    ode = raster.residual_at_starts()[-1]          # steady-state site
    analytic = rec.residual_fraction(footprint, tau, pitch, freq)
    assert ode == pytest.approx(analytic, abs=0.05)


def test_raster_first_site_is_fresh(footprint):
    raster = rec.raster_surface(footprint, tau=0.5, pitch_nm=5.0,
                                ramp_freq_hz=25.0, n_insertions=6)
    # nothing has happened before the first poke
    assert raster.residual_at_starts()[0] == pytest.approx(0.0, abs=1e-6)


def test_viscous_fraction_scales_residual(footprint):
    # beta = 0 is purely elastic: no residual regardless of scan.
    a = footprint.contact_radius
    r_elastic = rec.residual_fraction(footprint, tau=1.0, pitch_nm=a,
                                      ramp_freq_hz=50.0, viscous_fraction=0.0)
    assert r_elastic == pytest.approx(0.0, abs=1e-12)
    # residual is linear in beta
    full = rec.residual_fraction(footprint, tau=1.0, pitch_nm=a, ramp_freq_hz=50.0,
                                 viscous_fraction=1.0)
    half = rec.residual_fraction(footprint, tau=1.0, pitch_nm=a, ramp_freq_hz=50.0,
                                 viscous_fraction=0.5)
    assert half == pytest.approx(0.5 * full, rel=1e-9)


def test_elastic_raster_recovers_between_pokes(footprint):
    # A more elastic material (small beta, fast tau) leaves ~no residual.
    r = rec.raster_surface(footprint, tau=0.008, pitch_nm=5.0, ramp_freq_hz=25.0,
                           n_insertions=6, viscous_fraction=0.2)
    assert abs(r.residual_at_starts()[-1]) < 0.03


def test_limit_map_shape_and_trend(footprint):
    a = footprint.contact_radius
    pitches = np.array([a, 2 * a, 4 * a])
    freqs = np.array([1.0, 100.0])
    m = rec.recovery_limit_map(footprint, tau=0.5, pitches=pitches, freqs=freqs)
    assert m.shape == (3, 2)
    # larger pitch -> less residual (down each column)
    assert m[0, 1] >= m[2, 1] - 1e-9
    # higher frequency -> more residual (across each row) at small pitch
    assert m[0, 1] >= m[0, 0] - 1e-9
