import numpy as np
import pytest

from nanoindent import SimulationConfig, synthetic_force_curve, fit_modulus
from nanoindent import analytics


def test_noiseless_hertz_roundtrip():
    # Data generated from Hertz at known E must be recovered exactly (no noise).
    cfg = SimulationConfig(h=200.0, E=130.0, nu=0.3, delta_max=1.0)
    ds = synthetic_force_curve(cfg, source="hertz", noise_pN=0.0,
                               include_precontact=False, seed=0)
    fit = fit_modulus(ds.delta, ds.force, "hertz", cfg.R, cfg.nu, cfg.h)
    assert fit.E_fit == pytest.approx(130.0, rel=1e-6)
    assert fit.rmse_pN < 1e-6


def test_noiseless_dimitriadis_roundtrip():
    cfg = SimulationConfig(h=7.0, E=130.0, nu=0.3, delta_max=1.0)
    ds = synthetic_force_curve(cfg, source="dimitriadis", noise_pN=0.0,
                               include_precontact=False, seed=0)
    fit = fit_modulus(ds.delta, ds.force, "dimitriadis", cfg.R, cfg.nu, cfg.h)
    assert fit.E_fit == pytest.approx(130.0, rel=1e-6)


def test_hertz_fit_overestimates_on_thin_bonded_film():
    # Data from the bonded model, fit with naive Hertz -> inflated modulus.
    cfg = SimulationConfig(h=7.0, E=130.0, nu=0.3, delta_max=1.0)
    ds = synthetic_force_curve(cfg, source="dimitriadis", noise_pN=0.0,
                               include_precontact=False, seed=0)
    hertz_fit = fit_modulus(ds.delta, ds.force, "hertz", cfg.R, cfg.nu, cfg.h)
    dim_fit = fit_modulus(ds.delta, ds.force, "dimitriadis", cfg.R, cfg.nu, cfg.h)
    assert hertz_fit.E_fit > 1.3 * cfg.E          # clear over-estimate
    assert dim_fit.E_fit == pytest.approx(130.0, rel=1e-6)


def test_baseline_and_contact_point_recovered_with_scan():
    cfg = SimulationConfig(h=14.0, E=130.0, nu=0.3, delta_max=1.0)
    ds = synthetic_force_curve(cfg, source="dimitriadis", noise_pN=0.0,
                               baseline_pN=40.0, contact_point=0.08,
                               include_precontact=True, seed=3)
    fit = fit_modulus(ds.delta, ds.force, "dimitriadis", cfg.R, cfg.nu, cfg.h,
                      fit_contact_point=True)
    assert fit.E_fit == pytest.approx(130.0, rel=0.05)
    assert fit.contact_point == pytest.approx(0.08, abs=0.05)


def test_noise_reflected_in_rmse():
    cfg = SimulationConfig(h=14.0, E=130.0, nu=0.3, delta_max=1.0)
    ds = synthetic_force_curve(cfg, source="dimitriadis", noise_pN=15.0,
                               include_precontact=False, seed=7)
    fit = fit_modulus(ds.delta, ds.force, "dimitriadis", cfg.R, cfg.nu, cfg.h)
    # RMS residual should be on the order of the injected noise.
    assert 5.0 < fit.rmse_pN < 30.0
    # Modulus still close despite noise (many points average it down).
    assert fit.E_fit == pytest.approx(130.0, rel=0.1)


def test_fit_uncertainty_positive():
    cfg = SimulationConfig(h=14.0, E=130.0, nu=0.3, delta_max=1.0)
    ds = synthetic_force_curve(cfg, source="dimitriadis", noise_pN=10.0, seed=2)
    fit = fit_modulus(ds.delta, ds.force, "dimitriadis", cfg.R, cfg.nu, cfg.h)
    assert fit.E_std > 0.0


def test_save_dataset(tmp_path):
    cfg = SimulationConfig(h=7.0, E=130.0, nu=0.3, delta_max=1.0)
    ds = synthetic_force_curve(cfg, source="dimitriadis", noise_pN=5.0, seed=0)
    p = tmp_path / "ds.csv"
    ds.save(str(p))
    assert p.exists() and p.stat().st_size > 0


def test_unknown_model_raises():
    cfg = SimulationConfig(h=7.0)
    with pytest.raises(ValueError):
        synthetic_force_curve(cfg, source="nope")
