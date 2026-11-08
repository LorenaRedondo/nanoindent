"""Applicability metadata and numerical-quality reporting.

A converged Newton solve is not automatically a physically valid result.  This
module attaches two machine-readable records to a run:

* :func:`applicability` -- the model's physical assumptions, the dimensionless
  ratios that govern whether they hold, and warnings when a configuration
  approaches a documented limit.
* :func:`quality_report` -- numerical diagnostics (equilibrium residual,
  interpenetration, Newton convergence).  Mesh and penalty sensitivity come
  from :mod:`nanoindent.convergence` and can be merged in with
  :func:`with_sensitivity`.

Both are plain dataclasses with ``to_dict`` for JSON export.  They report
*what was measured*; they do not claim an error bound.  Variation across
meshes is *mesh sensitivity*, not a discretisation-error estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .config import SimulationConfig
    from .postproc import ForceCurve

#: Physical assumptions of the solver (docs/design.md).
ASSUMPTIONS = (
    "linear isotropic elasticity",
    "small-strain kinematics",
    "quasi-static loading",
    "frictionless unilateral contact",
    "axisymmetric geometry",
    "rigid indenter and rigid substrate",
)

#: Documented thresholds.  They flag *approach to a limit*, not invalidity.
DEPTH_TO_THICKNESS_WARN = 0.10
CONTACT_TO_THICKNESS_WARN = 1.0
NU_LOCKING = 0.45
PENETRATION_WARN = 0.05


@dataclass
class Applicability:
    """Assumptions, governing ratios and warnings for one configuration."""

    assumptions: tuple = ASSUMPTIONS
    depth_over_thickness: float = 0.0
    """Maximum indentation depth / film thickness (delta_max / h)."""
    contact_radius_over_thickness: Optional[float] = None
    """Maximum FEM contact radius / film thickness (None if not available)."""
    lateral_domain_over_contact: Optional[float] = None
    """Lateral domain radius / maximum contact radius."""
    warnings: list = field(default_factory=list)

    @property
    def within_documented_limits(self) -> bool:
        return not self.warnings

    def to_dict(self) -> dict:
        d = asdict(self)
        d["assumptions"] = list(self.assumptions)
        d["within_documented_limits"] = self.within_documented_limits
        return d


def applicability(cfg: "SimulationConfig",
                  curve: Optional["ForceCurve"] = None) -> Applicability:
    """Assess how far ``cfg`` is from the model's documented limits."""
    a = Applicability()
    a.depth_over_thickness = float(cfg.delta_max / cfg.h)
    if a.depth_over_thickness > DEPTH_TO_THICKNESS_WARN:
        a.warnings.append(
            f"delta_max/h = {a.depth_over_thickness:.2f} exceeds "
            f"{DEPTH_TO_THICKNESS_WARN:.2f}: small-strain kinematics may "
            "under-predict stiffening or misrepresent the response.")
    if curve is not None and curve.contact_radius.size:
        amax = float(np.nanmax(curve.contact_radius))
        a.contact_radius_over_thickness = amax / cfg.h
        if amax > 0:
            a.lateral_domain_over_contact = float(cfg.L / amax)
            if a.lateral_domain_over_contact < 5.0:
                a.warnings.append(
                    f"lateral domain is only {a.lateral_domain_over_contact:.1f}x "
                    "the contact radius: far-field boundary may affect the result.")
        if a.contact_radius_over_thickness > CONTACT_TO_THICKNESS_WARN:
            a.warnings.append(
                f"contact radius/h = {a.contact_radius_over_thickness:.2f} "
                f"exceeds {CONTACT_TO_THICKNESS_WARN:.1f}: strongly "
                "substrate-dominated regime; check against the validation matrix.")
    if cfg.nu >= NU_LOCKING and cfg.formulation == "primal":
        a.warnings.append(
            f"nu = {cfg.nu} with the primal formulation risks volumetric "
            "locking; use the mixed formulation.")
    return a


@dataclass
class QualityReport:
    """Numerical diagnostics of one run."""

    software: dict = field(default_factory=dict)
    max_equilibrium_residual: Optional[float] = None
    """max over steps of |F_contact - F_bottom| / F_contact."""
    max_penetration_over_delta: Optional[float] = None
    newton_converged: Optional[bool] = None
    newton_total_iterations: Optional[int] = None
    newton_max_iterations: Optional[int] = None
    failed_steps: list = field(default_factory=list)
    """Load steps (delta, nm) where Newton did not converge."""
    sensitivity: dict = field(default_factory=dict)
    """Optional mesh / penalty sensitivity summaries (see with_sensitivity)."""
    warnings: list = field(default_factory=list)

    @property
    def numerically_ok(self) -> bool:
        return not self.warnings

    def to_dict(self) -> dict:
        d = asdict(self)
        d["numerically_ok"] = self.numerically_ok
        return d


def software_versions() -> dict:
    """Versions that identify how a result was produced."""
    import sys
    from importlib import metadata
    out = {"python": sys.version.split()[0], "numpy": np.__version__}
    for pkg in ("nanoindent", "scikit-fem", "scipy"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "unknown"
    return out


def quality_report(curve: "ForceCurve") -> QualityReport:
    """Build the numerical-quality report for ``curve``."""
    q = QualityReport(software=software_versions())
    cfg = curve.cfg
    res = curve.equilibrium_residual
    if res.size > 1:
        q.max_equilibrium_residual = float(np.nanmax(res[1:]))
        if cfg is not None and q.max_equilibrium_residual > cfg.equilibrium_tol:
            q.warnings.append(
                f"equilibrium residual {q.max_equilibrium_residual:.1%} exceeds "
                f"{cfg.equilibrium_tol:.0%}")
    if curve.max_penetration.size > 1:
        rel = curve.max_penetration[1:] / np.maximum(curve.delta[1:], 1e-30)
        q.max_penetration_over_delta = float(np.max(rel))
        if q.max_penetration_over_delta > PENETRATION_WARN:
            q.warnings.append(
                f"interpenetration reaches {q.max_penetration_over_delta:.1%} of delta")
    if curve.newton_log:
        its = [s["iterations"] for s in curve.newton_log]
        q.newton_total_iterations = int(sum(its))
        q.newton_max_iterations = int(max(its))
        q.failed_steps = [s["delta"] for s in curve.newton_log if not s["converged"]]
        q.newton_converged = not q.failed_steps
        if q.failed_steps:
            q.warnings.append(
                f"Newton did not converge at {len(q.failed_steps)} load step(s)")
    return q


def with_sensitivity(report: QualityReport, name: str, records) -> QualityReport:
    """Attach a sensitivity study (mesh / penalty / load-step) to ``report``.

    ``records`` is a list of :class:`nanoindent.convergence.ConvergenceRecord`.
    The summary is the observed spread of the final force across the study,
    reported as *sensitivity*, not as a discretisation-error estimate.
    """
    f = np.array([r.force_max for r in records], dtype=float)
    spread = float((f.max() - f.min()) / abs(f[-1])) if f.size and f[-1] else float("nan")
    report.sensitivity[name] = {
        "n_runs": int(f.size),
        "final_force_spread_relative": spread,
        "interpretation": "observed variation across the study; "
                          "not an error bound",
    }
    return report
