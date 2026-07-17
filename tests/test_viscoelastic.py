"""Tests for the linear viscoelastic correspondence module."""

import numpy as np
import pytest

from nanoindent import viscoelastic as ve


# -- relaxation models -----------------------------------------------------
def test_sls_limits_and_complex_modulus():
    sls = ve.standard_linear_solid(E0=30.0, E_inf=10.0, tau=1e-3)
    assert sls.E0 == pytest.approx(30.0)
    assert sls.E_inf == pytest.approx(10.0)
    # E(0) = glassy, E(inf) = equilibrium
    assert sls.E_relax(np.array([0.0]))[0] == pytest.approx(30.0)
    assert sls.E_relax(np.array([10.0]))[0] == pytest.approx(10.0, abs=1e-6)
    # complex modulus: low w -> E_inf (real), high w -> E0 (real), loss -> 0 at both
    lo = sls.E_complex(np.array([1e-3 / 1e3]))[0]
    hi = sls.E_complex(np.array([1e12]))[0]
    assert lo.real == pytest.approx(10.0, rel=1e-3)
    assert hi.real == pytest.approx(30.0, rel=1e-3)
    assert abs(hi.imag) < 1e-2


def test_loss_peaks_at_corner():
    sls = ve.standard_linear_solid(E0=30.0, E_inf=10.0, tau=1e-3)
    w = np.logspace(0, 6, 4000)
    loss = sls.loss(w)
    w_peak = w[np.argmax(loss)]
    # loss maximum at omega * tau = 1
    assert w_peak * 1e-3 == pytest.approx(1.0, rel=0.02)
    # peak height = (E0 - Einf)/2
    assert loss.max() == pytest.approx(10.0, rel=0.02)


def test_sls_requires_E0_above_Einf():
    with pytest.raises(ValueError):
        ve.standard_linear_solid(E0=10.0, E_inf=20.0, tau=1e-3)


def test_generalized_maxwell_reduces_to_prony_sum():
    gm = ve.generalized_maxwell(E_inf=5.0, weights=[15.0, 8.0], taus=[1e-2, 1e-4])
    assert gm.E0 == pytest.approx(28.0)
    assert gm.E_relax(np.array([0.0]))[0] == pytest.approx(28.0)


# -- hereditary integral ---------------------------------------------------
def test_hereditary_elastic_limit():
    # A single branch with tau -> infinity never relaxes: F(t) = E0 * phi.
    relax = ve.RelaxationModel(E_inf=0.0, weights=np.array([20.0]),
                               taus=np.array([1e9]))
    phi = lambda d: np.asarray(d, float) ** 1.5
    t = np.linspace(0, 1, 50)
    d = t.copy()
    F = ve.hereditary_force(t, d, relax, phi)
    assert np.allclose(F, 20.0 * phi(d), rtol=1e-3)


def test_stress_relaxation_decays_to_equilibrium():
    sls = ve.standard_linear_solid(E0=30.0, E_inf=10.0, tau=1e-3)
    phi = lambda d: np.asarray(d, float)
    # Near-instantaneous ramp then a long hold: peak ~ glassy, end ~ equilibrium.
    t, d, F = ve.ramp_hold_curve(sls, phi, delta_max=1.0,
                                 approach_velocity_nm_s=1e5, hold_s=20e-3)
    assert F.max() == pytest.approx(30.0, rel=0.05)          # E0 * phi(1)
    assert F[-1] == pytest.approx(10.0, rel=0.02)            # E_inf * phi(1)


# -- frequency sweep: recovered E'(w), E''(w) match closed form ------------
def test_frequency_sweep_matches_analytic_linear_kernel():
    sls = ve.standard_linear_solid(E0=30.0, E_inf=10.0, tau=1e-3)
    phi = lambda d: np.asarray(d, float)
    omegas = 2 * np.pi * np.logspace(0, 4.5, 12)
    spec = ve.frequency_sweep(sls, phi, lambda d0: 1.0, omegas,
                              delta0=1.0, delta_amp=0.02)
    assert spec.max_rel_error() < 0.01


def test_frequency_sweep_nonlinear_kernel():
    # Hertz-like kernel phi ~ delta^1.5: the small-amplitude linearisation with
    # a local slope should still recover the complex modulus well.
    sls = ve.standard_linear_solid(E0=25.0, E_inf=8.0, tau=2e-3)
    phi = lambda d: np.asarray(d, float) ** 1.5
    slope = lambda d0: ve.kernel_slope(phi, d0)
    omegas = 2 * np.pi * np.logspace(0, 4, 10)
    spec = ve.frequency_sweep(sls, phi, slope, omegas, delta0=1.0,
                              delta_amp=0.01)
    assert spec.max_rel_error() < 0.02


def test_kernel_slope_matches_analytic():
    phi = lambda d: np.asarray(d, float) ** 1.5
    # d/dd delta^1.5 = 1.5 delta^0.5; at delta=1 -> 1.5
    assert ve.kernel_slope(phi, 1.0) == pytest.approx(1.5, rel=1e-3)


def test_dynamic_spectrum_freq_hz():
    spec = ve.DynamicSpectrum(omega=np.array([2 * np.pi]),
                              storage=np.array([1.0]), loss=np.array([0.1]),
                              storage_exact=np.array([1.0]),
                              loss_exact=np.array([0.1]))
    assert spec.freq_hz[0] == pytest.approx(1.0)
