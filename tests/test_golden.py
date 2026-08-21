"""Golden-value regression tests.

Freeze known-good numbers so an accidental change in the physics (a wrong
coefficient, a sign flip, a units slip) is caught even when nothing raises.
Analytic anchors are exact (tight tolerance); the small FEM runs are
deterministic and pinned to +/-0.5%, loose enough to absorb LU/BLAS
platform differences but tight enough to catch real drift.

If a golden value changes intentionally, update it in the same commit as the
physics change and say why in the message.
"""

import warnings
import numpy as np
import pytest

from nanoindent import SimulationConfig, IndentationSolver, analytics


# -- analytic anchors (exact closed forms) -----------------------------------
@pytest.mark.parametrize("fn, args, expected", [
    (analytics.hertz_force, (1.0, 17.0, 130.0, 0.3), 785.353452498602),
    # Updated 2026-08-21: the previous value 1412.8746095837907 was produced
    # with a sign error on the chi^4 term of the Dimitriadis series, which
    # made a bonded film soften at 4th order. +3.96% at these parameters.
    (analytics.dimitriadis_force, (1.0, 17.0, 130.0, 0.3, 7.0), 1468.850251298797),
    (analytics.chadwick_force, (1.0, 17.0, 130.0, 7.0), 229.40648395601272),
    (analytics.sneddon_cone_force, (1.0, 70.3, 130.0, 0.3), 254.00143853054578),
])
def test_analytic_golden(fn, args, expected):
    assert float(fn(*args)) == pytest.approx(expected, rel=1e-9)


# -- FEM golden (deterministic small runs) -----------------------------------
def _fem(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return IndentationSolver(SimulationConfig(**kw)).run()


def test_fem_golden_sphere_h14():
    c = _fem(h=14.0, L=60.0, nu=0.3, n_steps=4, delta_max=1.0,
             mesh_size_contact=0.6, mesh_size_edge=12.0, penalty_alpha=400.0)
    assert float(c.force[-1]) == pytest.approx(1056.86, rel=5e-3)


def test_fem_golden_paraboloid_h7():
    c = _fem(tip_shape="paraboloid", h=7.0, L=60.0, nu=0.3, n_steps=4,
             delta_max=1.0, mesh_size_contact=0.5, mesh_size_edge=10.0,
             penalty_alpha=400.0)
    assert float(c.force[-1]) == pytest.approx(1495.78, rel=5e-3)
