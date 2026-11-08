"""Applicability metadata and numerical-quality report."""
import json

import numpy as np

from nanoindent.config import SimulationConfig
from nanoindent.postproc import ForceCurve
from nanoindent.quality import applicability, quality_report, with_sensitivity, ASSUMPTIONS
from nanoindent.solver import IndentationSolver
from nanoindent.convergence import mesh_study


def _small(**kw):
    base = dict(h=7.0, R=17.0, E=130.0, nu=0.3, delta_max=0.5, n_steps=4,
                mesh_size_contact=0.5, L=60.0)
    base.update(kw)
    return SimulationConfig(**base)


def test_applicability_ratios_and_assumptions():
    a = applicability(_small(delta_max=0.35))
    assert abs(a.depth_over_thickness - 0.05) < 1e-12
    assert a.within_documented_limits
    assert "small-strain kinematics" in a.assumptions


def test_depth_over_thickness_warning():
    # 1 nm into a 7 nm film is ~14 % -> flagged
    a = applicability(_small(delta_max=1.0))
    assert a.depth_over_thickness > 0.10
    assert not a.within_documented_limits
    assert any("delta_max/h" in w for w in a.warnings)


def test_locking_warning_for_primal_near_incompressible():
    a = applicability(_small(nu=0.45))
    assert any("locking" in w for w in a.warnings)


def test_run_carries_applicability_and_quality():
    curve = IndentationSolver(_small()).run()
    assert curve.newton_log and len(curve.newton_log) == 4
    app = curve.applicability()
    assert app.contact_radius_over_thickness is not None
    q = curve.quality_report()
    assert q.newton_converged is True
    assert q.failed_steps == []
    assert q.max_equilibrium_residual is not None
    assert "nanoindent" in q.software and "scikit-fem" in q.software
    json.dumps(q.to_dict())
    json.dumps(app.to_dict())


def test_quality_without_log_is_graceful():
    c = ForceCurve(delta=np.array([0.0, 1.0]), force=np.array([0.0, 1.0]))
    q = quality_report(c)
    assert q.newton_converged is None


def test_with_sensitivity_is_labelled_not_error():
    cfg = _small()
    recs = mesh_study(cfg, sizes=[0.6, 0.45])
    q = with_sensitivity(quality_report(IndentationSolver(cfg).run()), "mesh", recs)
    s = q.sensitivity["mesh"]
    assert s["n_runs"] == 2
    assert "not an error bound" in s["interpretation"]
