"""Tests for synthetic force-volume generation."""

import warnings
import numpy as np
import pytest

from nanoindent import SimulationConfig
from nanoindent import forcemap as fm


def _cfg():
    return SimulationConfig(R=20.0, nu=0.5, h=6.0, L=100.0, delta_max=1.5,
                            formulation="mixed", n_steps=8,
                            mesh_size_contact=0.4, mesh_size_edge=12.0)


@pytest.fixture(scope="module")
def kernel():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fm.geometric_kernel(_cfg())


# -- kernel: force is exactly linear in E ----------------------------------
def test_kernel_force_proportional_to_E(kernel):
    d = np.linspace(0.1, 1.4, 20)
    f1 = kernel.force(d, 3.0)
    f2 = kernel.force(d, 30.0)
    # F(E=30) == 10 * F(E=3) to round-off
    assert np.allclose(f2, 10.0 * f1, rtol=1e-10)


def test_kernel_zero_at_zero_and_monotone(kernel):
    assert kernel.phi_of(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-9)
    d = np.linspace(0.0, kernel.delta_max, 30)
    phi = kernel.phi_of(d)
    assert np.all(np.diff(phi) >= -1e-9)          # non-decreasing


def test_kernel_clips_beyond_depth(kernel):
    # Evaluating past the solved depth clamps rather than extrapolating.
    beyond = kernel.phi_of(np.array([kernel.delta_max * 2]))[0]
    assert beyond == pytest.approx(kernel.phi_of(np.array([kernel.delta_max]))[0])


# -- spec timing -----------------------------------------------------------
def test_spec_timing():
    spec = fm.ForceVolumeSpec(16, 16, pitch_nm=10.0, ramp_freq_hz=4.0,
                              z_amplitude_nm=2.0)
    assert spec.n_pixels == 256
    assert spec.ramp_period_s == pytest.approx(0.25)
    # triangle: v = 2 * amplitude * freq
    assert spec.approach_velocity_nm_s == pytest.approx(16.0)
    assert spec.total_scan_time_s == pytest.approx(256 * 0.25)
    w, h = spec.field_of_view_nm
    assert w == pytest.approx(150.0) and h == pytest.approx(150.0)


def test_spec_dwell_adds_to_period():
    spec = fm.ForceVolumeSpec(4, 4, pitch_nm=8.0, ramp_freq_hz=2.0, dwell_s=0.1)
    assert spec.ramp_period_s == pytest.approx(0.5 + 0.1)


# -- modulus fields --------------------------------------------------------
def test_field_helpers_shapes_and_values():
    assert fm.uniform_field(5, 7, 12.0).shape == (5, 7)
    assert np.all(fm.uniform_field(5, 7, 12.0) == 12.0)

    tp = fm.two_phase_field(20, 20, E_soft=2.0, E_stiff=20.0,
                            stiff_fraction=0.4, seed=0)
    vals = set(np.unique(tp))
    assert vals == {2.0, 20.0}
    # roughly the requested stiff fraction
    assert 0.25 < np.mean(tp == 20.0) < 0.55

    gr = fm.gradient_field(6, 8, E_left=1.0, E_right=9.0, axis=1)
    assert gr[0, 0] == pytest.approx(1.0)
    assert gr[0, -1] == pytest.approx(9.0)


def test_field_callable_and_bad_shape():
    E = fm._resolve_field(lambda i, j: float(i + j), 3, 3)
    assert E[2, 2] == 4.0
    with pytest.raises(ValueError):
        fm._resolve_field(np.zeros((2, 2)), 3, 3)


# -- generation ------------------------------------------------------------
def test_simulate_one_solve_and_metadata(kernel):
    spec = fm.ForceVolumeSpec(6, 6, pitch_nm=8.0, ramp_freq_hz=2.0,
                              z_amplitude_nm=1.2, points_per_ramp=48)
    E = fm.uniform_field(6, 6, 10.0)
    vol = fm.simulate_force_volume(spec, E, _cfg(), spring_constant=0.1,
                                   kernel=kernel)
    assert len(vol) == 36
    assert vol.E_true.shape == (6, 6)
    assert vol.timestamps.shape == (36,)
    # timestamps advance by the per-pixel period
    dt = np.diff(vol.timestamps)
    assert np.allclose(dt, spec.per_pixel_time_s)
    # each curve is a raw ForceCurveData carrying its pixel + true E
    c0 = vol.curves[0]
    assert c0.metadata["pixel"] == 0
    assert c0.metadata["true_E"] == pytest.approx(10.0)


def test_z_amplitude_exceeding_kernel_raises(kernel):
    spec = fm.ForceVolumeSpec(3, 3, pitch_nm=8.0, ramp_freq_hz=2.0,
                              z_amplitude_nm=kernel.delta_max + 1.0)
    with pytest.raises(ValueError):
        fm.simulate_force_volume(spec, fm.uniform_field(3, 3, 5.0), _cfg(),
                                 kernel=kernel)


# -- round trip: recover the modulus that was put in -----------------------
def test_roundtrip_kernel_recovers_modulus(kernel):
    spec = fm.ForceVolumeSpec(8, 8, pitch_nm=8.0, ramp_freq_hz=2.0,
                              z_amplitude_nm=1.2, points_per_ramp=80)
    E = fm.two_phase_field(8, 8, E_soft=2.0, E_stiff=20.0, seed=2)
    vol = fm.simulate_force_volume(spec, E, _cfg(), spring_constant=0.08,
                                   kernel=kernel, noise_nm=0.0)
    Erec = vol.recover_modulus_map(model="kernel", fit_max_delta=1.2)
    assert np.all(np.isfinite(Erec))
    rel = np.abs(Erec - E) / E
    # kernel fit is limited only by the contact-point scan; comfortably < 10%
    assert np.median(rel) < 0.10
    # phases stay separated
    assert np.nanmedian(Erec[E < 10]) < np.nanmedian(Erec[E >= 10]) / 3


def test_hertz_overestimates_on_thin_bonded_film(kernel):
    # The bottom-effect artefact: half-space Hertz reads high on a thin film.
    spec = fm.ForceVolumeSpec(6, 6, pitch_nm=8.0, ramp_freq_hz=2.0,
                              z_amplitude_nm=1.2, points_per_ramp=64)
    E = fm.uniform_field(6, 6, 10.0)
    vol = fm.simulate_force_volume(spec, E, _cfg(), spring_constant=0.08,
                                   kernel=kernel)
    E_hertz = vol.recover_modulus_map(model="hertz", fit_max_delta=1.2)
    assert np.nanmedian(E_hertz) > 1.5 * 10.0
