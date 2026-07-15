"""Convergence-study tests. The multi-run studies are marked slow."""

import warnings
import numpy as np
import pytest

from nanoindent import SimulationConfig
from nanoindent import convergence


def _base_cfg():
    return SimulationConfig(h=14.0, L=80.0, nu=0.3, n_steps=4, delta_max=1.0,
                            mesh_size_contact=0.5, mesh_size_edge=10.0,
                            penalty_alpha=300.0)


def test_relative_changes_and_convergence_helpers():
    recs = [
        convergence.ConvergenceRecord("m", 0.5, 100.0, 10, 0.0, 0.0),
        convergence.ConvergenceRecord("m", 0.25, 101.0, 20, 0.0, 0.0),
        convergence.ConvergenceRecord("m", 0.125, 101.2, 40, 0.0, 0.0),
    ]
    ch = convergence.relative_changes(recs)
    assert ch.shape == (2,)
    assert ch[-1] < 0.005
    assert convergence.is_converged(recs, tol=0.005)


@pytest.mark.slow
def test_load_step_study_is_path_independent():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recs = convergence.load_step_study(_base_cfg(), n_steps_list=(4, 12))
    f = np.array([r.force_max for r in recs])
    # Elastic problem: final force must be essentially step-independent.
    assert abs(f[1] - f[0]) / f[0] < 0.02


@pytest.mark.slow
def test_penalty_study_plateaus():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recs = convergence.penalty_study(_base_cfg(), alphas=(100.0, 1000.0))
    f = np.array([r.force_max for r in recs])
    # Force should change little once the penalty is stiff enough.
    assert abs(f[1] - f[0]) / f[0] < 0.05
    # Interpenetration must drop with higher penalty.
    assert recs[1].max_penetration <= recs[0].max_penetration + 1e-9


@pytest.mark.slow
def test_mesh_study_converges():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recs = convergence.mesh_study(_base_cfg(), sizes=(0.5, 0.25, 0.125))
    ch = convergence.relative_changes(recs)
    assert ch[-1] < 0.03  # tightening as mesh refines
