import warnings
import numpy as np
import pytest

from nanoindent import SimulationConfig, IndentationSolver, analytics


@pytest.fixture(scope="module")
def small_curve():
    # A deliberately small/coarse run so the fast suite stays quick.
    cfg = SimulationConfig(h=14.0, L=60.0, nu=0.3, n_steps=4, delta_max=1.0,
                           mesh_size_contact=0.6, mesh_size_edge=12.0,
                           penalty_alpha=300.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return IndentationSolver(cfg).run()


def test_zero_force_at_zero_indentation(small_curve):
    assert small_curve.delta[0] == 0.0
    assert small_curve.force[0] == 0.0


def test_force_monotone_increasing(small_curve):
    assert np.all(np.diff(small_curve.force) > 0)


def test_equilibrium_reasonable(small_curve):
    # Coarse mesh: allow a few percent, but the two force measures must agree.
    assert np.nanmax(small_curve.equilibrium_residual) < 0.05


def test_penetration_small(small_curve):
    rel = small_curve.max_penetration[1:] / small_curve.delta[1:]
    assert np.max(rel) < 0.05


def test_stiffer_than_hertz_thin_film(small_curve):
    fh = analytics.hertz_force(small_curve.delta, small_curve.cfg.R,
                               small_curve.cfg.E, small_curve.cfg.nu)
    # Bonded thin film must be stiffer than the half-space at delta_max.
    assert small_curve.force[-1] > fh[-1]


def test_mixed_formulation_runs_incompressible():
    # The mixed formulation must accept nu = 0.5 and produce a sane curve.
    cfg = SimulationConfig(nu=0.5, formulation="mixed", h=14.0, L=50.0,
                           n_steps=3, delta_max=1.0, mesh_size_contact=0.6,
                           mesh_size_edge=10.0, penalty_alpha=400.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        curve = IndentationSolver(cfg).run()
    assert curve.force[0] == 0.0
    assert np.all(np.diff(curve.force) > 0)          # monotone
    assert np.nanmax(curve.equilibrium_residual) < 0.05


def test_mixed_matches_primal_compressible():
    # At nu = 0.3 (well away from incompressibility) mixed ~ primal.
    common = dict(nu=0.3, h=60.0, L=60.0, n_steps=3, delta_max=1.0,
                  mesh_size_contact=0.6, mesh_size_edge=12.0, penalty_alpha=400.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cp = IndentationSolver(SimulationConfig(formulation="primal", **common)).run()
        cm = IndentationSolver(SimulationConfig(formulation="mixed", **common)).run()
    rel = np.abs(cm.force[-1] - cp.force[-1]) / cp.force[-1]
    assert rel < 0.02


def test_csv_roundtrip(small_curve, tmp_path):
    from nanoindent import ForceCurve
    p = tmp_path / "curve.csv"
    small_curve.save(str(p))
    loaded = ForceCurve.load(str(p))
    assert np.allclose(loaded.delta, small_curve.delta)
    assert np.allclose(loaded.force, small_curve.force)
