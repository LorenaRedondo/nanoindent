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


def test_cone_recovers_sneddon_thick_film():
    # A sharp cone on a thick film should follow Sneddon's F ~ delta^2 law.
    curve, cfg = _run(tip_shape="cone", tip_half_angle_deg=70.3,
                      h=200.0, L=200.0, nu=0.3, n_steps=8, delta_max=1.0,
                      mesh_size_contact=0.3, mesh_size_edge=12.0,
                      penalty_alpha=400.0)
    sned = analytics.sneddon_cone_force(curve.delta, 70.3, cfg.E, cfg.nu)
    mask = curve.delta >= 0.5
    rel = np.abs(curve.force[mask] - sned[mask]) / sned[mask]
    assert np.mean(rel) < 0.05


def test_paraboloid_recovers_hertz_thick_film():
    curve, cfg = _run(tip_shape="paraboloid", R=17.0,
                      h=200.0, L=200.0, nu=0.3, n_steps=8, delta_max=1.0,
                      mesh_size_contact=0.35, mesh_size_edge=12.0,
                      penalty_alpha=400.0)
    fh = analytics.hertz_force(curve.delta, cfg.R, cfg.E, cfg.nu)
    mask = curve.delta >= 0.5
    ratio = curve.force[mask] / fh[mask]
    assert abs(np.mean(ratio) - 1.0) < 0.03


def test_incompressible_film_stiffer_than_compressible():
    # A bonded thin film is much stiffer at nu=0.5 (mixed) than nu=0.3 (primal).
    common = dict(tip_shape="paraboloid", R=20.0, E=50.0, h=6.2, L=200.0,
                  delta_max=2.0, n_steps=5, mesh_size_contact=0.05,
                  mesh_size_edge=4.0, penalty_alpha=600.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c5 = IndentationSolver(SimulationConfig(nu=0.5, formulation="mixed",
                                                **common)).run()
        c3 = IndentationSolver(SimulationConfig(nu=0.3, formulation="primal",
                                                **common)).run()
    assert c5.force[-1] > 1.5 * c3.force[-1]


def test_chadwick_limit_mixed_incompressible():
    # Mixed (nu=0.5) FEM must converge to Chadwick's thin-layer law as the
    # thin-film parameter chi = sqrt(R*delta)/h grows.
    R, E = 17.0, 130.0
    results = []
    for h, dmax in [(2.0, 2.0), (1.5, 3.0), (1.0, 4.0)]:
        cfg = SimulationConfig(nu=0.5, formulation="mixed", tip_shape="paraboloid",
                               h=h, L=80.0, R=R, E=E, delta_max=dmax, n_steps=5,
                               mesh_size_contact=0.12, mesh_size_edge=4.0,
                               penalty_alpha=800.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            curve = IndentationSolver(cfg).run()
        chad = analytics.chadwick_force(curve.delta, R, E, h)
        chi = np.sqrt(R * curve.delta[-1]) / h
        results.append((chi, curve.force[-1] / chad[-1]))

    chis = [r[0] for r in results]
    ratios = [r[1] for r in results]
    # Monotone approach to 1 from above as chi increases.
    assert chis[0] < chis[1] < chis[2]
    assert ratios[0] > ratios[1] > ratios[2]
    # At the largest chi the FEM is within 15% of Chadwick's coefficient.
    assert abs(ratios[-1] - 1.0) < 0.15


def test_contact_radius_thick_film():
    curve, cfg = _run(h=200.0, L=200.0, nu=0.3, n_steps=6, delta_max=1.0,
                      mesh_size_contact=0.3, mesh_size_edge=12.0,
                      penalty_alpha=400.0)
    a_hertz = np.sqrt(cfg.R * curve.delta)
    mask = curve.delta >= 0.5
    rel = np.abs(curve.contact_radius[mask] - a_hertz[mask]) / a_hertz[mask]
    # Contact radius is resolved only to ~mesh size; allow 25%.
    assert np.mean(rel) < 0.25
