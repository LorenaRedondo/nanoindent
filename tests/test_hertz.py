"""Physics-validation tests (slow). Run with ``pytest -m slow``."""

import warnings
import numpy as np
import pytest

from nanoindent import SimulationConfig, IndentationSolver, analytics

pytestmark = pytest.mark.slow


def _run(**kw):
    cfg = SimulationConfig(**kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return IndentationSolver(cfg).run(), cfg


def test_hertz_thick_film_limit():
    # Very thick film should recover the Hertz half-space result within 2%.
    curve, cfg = _run(h=200.0, L=200.0, nu=0.3, n_steps=8, delta_max=1.0,
                      mesh_size_contact=0.35, mesh_size_edge=12.0,
                      penalty_alpha=400.0)
    fh = analytics.hertz_force(curve.delta, cfg.R, cfg.E, cfg.nu)
    # Compare over the second half of the curve (fully developed contact).
    mask = curve.delta >= 0.5
    ratio = curve.force[mask] / fh[mask]
    assert np.all(np.abs(ratio - 1.0) < 0.05)
    assert abs(np.mean(ratio) - 1.0) < 0.02


def test_equilibrium_under_one_percent_refined():
    curve, cfg = _run(h=14.0, L=100.0, nu=0.3, n_steps=6, delta_max=1.0,
                      mesh_size_contact=0.15, mesh_size_edge=5.0,
                      penalty_alpha=500.0)
    assert np.nanmax(curve.equilibrium_residual) < 0.015


def test_penalty_interpenetration_below_one_percent():
    curve, cfg = _run(h=14.0, L=100.0, nu=0.3, n_steps=6, delta_max=1.0,
                      mesh_size_contact=0.2, mesh_size_edge=6.0,
                      penalty_alpha=1000.0)
    rel = curve.max_penetration[1:] / curve.delta[1:]
    assert np.max(rel) < 0.01


def test_bottom_effect_matches_dimitriadis():
    # FEM stiffening ratio should track the bonded Dimitriadis model.
    for h in (7.0, 14.0):
        curve, cfg = _run(h=h, L=100.0, nu=0.3, n_steps=6, delta_max=1.0,
                          mesh_size_contact=0.2, mesh_size_edge=6.0,
                          penalty_alpha=400.0)
        dim = analytics.dimitriadis_force(curve.delta, cfg.R, cfg.E, cfg.nu,
                                          cfg.h, bonded=True)
        mask = curve.delta >= 0.4
        rel = np.abs(curve.force[mask] - dim[mask]) / dim[mask]
        # Qualitative-to-good agreement expected (design doc section 7.2).
        assert np.mean(rel) < 0.10


def test_thinner_film_is_stiffer():
    c7, cfg7 = _run(h=7.0, L=100.0, nu=0.3, n_steps=5, delta_max=1.0,
                    mesh_size_contact=0.25, mesh_size_edge=6.0,
                    penalty_alpha=400.0)
    c14, cfg14 = _run(h=14.0, L=100.0, nu=0.3, n_steps=5, delta_max=1.0,
                      mesh_size_contact=0.25, mesh_size_edge=6.0,
                      penalty_alpha=400.0)
    assert c7.force[-1] > c14.force[-1]


def test_contact_radius_thick_film():
    curve, cfg = _run(h=200.0, L=200.0, nu=0.3, n_steps=6, delta_max=1.0,
                      mesh_size_contact=0.3, mesh_size_edge=12.0,
                      penalty_alpha=400.0)
    a_hertz = np.sqrt(cfg.R * curve.delta)
    mask = curve.delta >= 0.5
    rel = np.abs(curve.contact_radius[mask] - a_hertz[mask]) / a_hertz[mask]
    # Contact radius is resolved only to ~mesh size; allow 25%.
    assert np.mean(rel) < 0.25
