"""IndentationSolver: axisymmetric penalty-contact FEM driver.

Backend: scikit-fem (pure-pip fallback sanctioned by the design doc when
dolfinx is unavailable).  The weak-form assembly is the only backend-specific
part of the library; everything else (config, contact geometry, analytics,
post-processing) is backend-agnostic.

Formulation
-----------
Axisymmetric small-strain linear elasticity with vector P2 displacement.  All
integrals carry the factor ``r`` (the ``2*pi`` from the azimuthal integration
is applied once, at force extraction, so scalar forces come out in pN).

Contact is enforced by a penalty on the top surface: the residual gains

    R_contact(v) = integral over top of  k_pen * [-g(u)]_+ * v_z * r dr

with ``k_pen = alpha * E / h_local``.  The only nonlinearity is the contact
active set, so each load step is solved by an active-set Newton iteration:
the elastic stiffness ``K`` is assembled once, and at every iteration the
contact residual and its (active-set) tangent are assembled on the top facets
from the current penetration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import warnings

import numpy as np
from skfem import (
    Basis, FacetBasis, ElementVector, ElementTriP1, ElementTriP2,
    BilinearForm, LinearForm, Functional, asm, condense, solve,
)

from .config import SimulationConfig
from .material import LinearElastic
from .geometry import build_mesh, Domain
from .contact import make_tip, penalty_stiffness
from .postproc import ForceCurve
from . import analytics


class IndentationSolver:
    """Displacement-controlled nanoindentation solver.

    Parameters
    ----------
    cfg:
        Fully specified :class:`SimulationConfig`.
    verbose:
        Print per-step convergence information.
    """

    def __init__(self, cfg: SimulationConfig, verbose: bool = False) -> None:
        self.cfg = cfg
        self.verbose = verbose
        self.material = LinearElastic(cfg.E, cfg.nu)
        self.tip = make_tip(cfg)

        self.domain: Optional[Domain] = None
        self.basis: Optional[Basis] = None
        self._K = None
        self._dirichlet = None

    # -- setup -------------------------------------------------------------
    def setup(self) -> None:
        """Build the mesh, function space, elastic stiffness, and BC dofs."""
        self.domain = build_mesh(self.cfg)
        mesh = self.domain.mesh
        elem = ElementVector(
            ElementTriP2() if self.cfg.element_order == 2 else ElementTriP1()
        )
        self.basis = Basis(mesh, elem)

        C = self.material.constitutive_matrix()
        self._K = self._assemble_stiffness(self.basis, C)
        self._dirichlet = self._dirichlet_dofs(self.basis)

    def _dirichlet_dofs(self, basis: Basis) -> np.ndarray:
        # Bottom bonded: u_r = u_z = 0.  Axis: u_r = 0 (u^1 is the r-component).
        bottom = basis.get_dofs("bottom").all()
        axis_r = basis.get_dofs("axis").all("u^1")
        return np.unique(np.concatenate([bottom, axis_r]))

    @staticmethod
    def _assemble_stiffness(basis: Basis, C: np.ndarray):
        """Assemble the axisymmetric elastic stiffness matrix (factor r)."""

        @BilinearForm
        def stiffness(u, v, w):
            r = w.x[0]
            # trial strains
            e_rr = u.grad[0][0]
            e_zz = u.grad[1][1]
            e_tt = u[0] / r
            g_rz = u.grad[0][1] + u.grad[1][0]
            # test strains
            f_rr = v.grad[0][0]
            f_zz = v.grad[1][1]
            f_tt = v[0] / r
            g_rz_v = v.grad[0][1] + v.grad[1][0]
            # stress from trial strain
            s_rr = C[0, 0] * e_rr + C[0, 1] * e_zz + C[0, 2] * e_tt
            s_zz = C[1, 0] * e_rr + C[1, 1] * e_zz + C[1, 2] * e_tt
            s_tt = C[2, 0] * e_rr + C[2, 1] * e_zz + C[2, 2] * e_tt
            s_rz = C[3, 3] * g_rz
            return (s_rr * f_rr + s_zz * f_zz + s_tt * f_tt
                    + s_rz * g_rz_v) * r

        return stiffness.assemble(basis)

    # -- contact assembly --------------------------------------------------
    def _contact_forms(self, fb: FacetBasis, u: np.ndarray, delta: float):
        """Return (residual_vector, tangent_matrix) for the penalty contact at
        displacement ``u`` and indentation ``delta``."""
        cfg = self.cfg
        tip = self.tip

        # Interpolate current vertical displacement at facet quadrature points
        # to evaluate the penetration and the active set there.
        uh = fb.interpolate(u)

        @LinearForm
        def residual(v, w):
            r = w.x[0]
            uz = w["uz"]
            k_pen = penalty_stiffness(cfg.penalty_alpha, cfg.E, w.h)
            # penetration = (h + uz) - z_tip(r); z_tip is +inf where the tip
            # cannot contact, so pen -> -inf -> pen_pos = 0 there.
            z_tip = tip.surface_z(r, delta)
            pen = (tip.h + uz) - z_tip
            pen_pos = np.maximum(pen, 0.0)
            # contact pressure pushes the surface down: force on the test
            # function's z-component is -k_pen*pen (traction), but the reaction
            # on the indenter is +; we assemble the internal force so that the
            # residual K u + R_c = 0 balances.  Pressure acts on +z dof.
            return k_pen * pen_pos * v[1] * r

        @BilinearForm
        def tangent(u_, v_, w):
            r = w.x[0]
            uz = w["uz"]
            k_pen = penalty_stiffness(cfg.penalty_alpha, cfg.E, w.h)
            z_tip = tip.surface_z(r, delta)
            pen = (tip.h + uz) - z_tip
            active = (pen > 0.0).astype(float)
            return k_pen * active * u_[1] * v_[1] * r

        Rc = residual.assemble(fb, uz=uh[1])
        Kc = tangent.assemble(fb, uz=uh[1])
        return Rc, Kc

    # -- single load step --------------------------------------------------
    def solve_step(self, u0: np.ndarray, delta: float, fb: FacetBasis):
        """Active-set Newton solve for one indentation depth.

        Residual: ``F(u) = K u + R_contact(u) = 0`` with homogeneous Dirichlet
        constraints.  Returns the converged displacement vector.
        """
        cfg = self.cfg
        K = self._K
        D = self._dirichlet
        u = u0.copy()

        # Reference residual scale for the relative tolerance.
        r0 = None
        for it in range(cfg.newton_max_it):
            Rc, Kc = self._contact_forms(fb, u, delta)
            residual = K @ u + Rc
            J = K + Kc

            # Newton update with homogeneous Dirichlet on constrained dofs.
            du = solve(*condense(J, -residual, D=D))
            u = u + du

            rnorm = np.linalg.norm(residual[self._free_mask()])
            if r0 is None:
                r0 = max(rnorm, 1e-30)
            dnorm = np.linalg.norm(du)
            if self.verbose:
                print(f"    Newton it={it:2d} |R|={rnorm:.3e} |du|={dnorm:.3e}")
            if rnorm <= cfg.newton_rtol * r0 or dnorm < 1e-12:
                break
        else:
            warnings.warn(
                f"Newton did not converge at delta={delta:.4f} nm "
                f"(|R|/|R0|={rnorm / r0:.2e})", stacklevel=2,
            )
        return u

    def _free_mask(self) -> np.ndarray:
        mask = np.ones(self.basis.N, dtype=bool)
        mask[self._dirichlet] = False
        return mask

    # -- force extraction --------------------------------------------------
    def _contact_force(self, fb: FacetBasis, u: np.ndarray, delta: float) -> float:
        """Total reaction force (pN) = 2*pi * integral k_pen [-g]_+ r dr."""
        cfg = self.cfg
        tip = self.tip
        uh = fb.interpolate(u)

        @Functional
        def integ(w):
            r = w.x[0]
            uz = w["uz"]
            k_pen = penalty_stiffness(cfg.penalty_alpha, cfg.E, w.h)
            z_tip = tip.surface_z(r, delta)
            pen = (tip.h + uz) - z_tip
            pen_pos = np.maximum(np.where(np.isfinite(pen), pen, 0.0), 0.0)
            return k_pen * pen_pos * r

        val = integ.assemble(fb, uz=uh[1])
        return 2.0 * np.pi * float(val)

    def _bottom_force(self, u: np.ndarray) -> float:
        """Vertical reaction over the bonded bottom (pN), equilibrium check.

        Total vertical reaction = 2*pi * integral sig_zz(z=0) r dr.
        """
        cfg = self.cfg
        C = self.material.constitutive_matrix()
        fb = FacetBasis(self.domain.mesh, self.basis.elem, facets="bottom")
        uh = fb.interpolate(u)

        @Functional
        def integ(w):
            r = w.x[0]
            e_rr = w["u"].grad[0][0]
            e_zz = w["u"].grad[1][1]
            e_tt = w["u"][0] / r
            s_zz = C[1, 0] * e_rr + C[1, 1] * e_zz + C[1, 2] * e_tt
            return s_zz * r

        # The substrate reaction on the sample is the traction t_z = -sig_zz
        # integrated over the bottom (outward normal -z), giving a positive
        # force that balances the downward contact pressure.
        val = integ.assemble(fb, u=uh)
        return -2.0 * np.pi * float(val)

    def _contact_radius(self, fb: FacetBasis, u: np.ndarray, delta: float) -> float:
        """Largest radius still in contact (nm), from facet quadrature points."""
        uh = fb.interpolate(u)
        r = fb.global_coordinates()[0]
        active = self.tip.is_active(r, uh[1], delta)
        if not np.any(active):
            return 0.0
        return float(np.max(r[active]))

    def _max_penetration(self, fb: FacetBasis, u: np.ndarray, delta: float) -> float:
        uh = fb.interpolate(u)
        r = fb.global_coordinates()[0]
        pen = self.tip.penetration(r, uh[1], delta)
        pen = pen[np.isfinite(pen)]
        return float(np.max(pen)) if pen.size else 0.0

    # -- driver ------------------------------------------------------------
    def run(self) -> ForceCurve:
        """Run the full load-stepping simulation and return a ForceCurve."""
        if self.basis is None:
            self.setup()
        cfg = self.cfg
        fb = FacetBasis(self.domain.mesh, self.basis.elem, facets="top")

        deltas = [0.0] + cfg.delta_schedule()
        n = len(deltas)
        force = np.zeros(n)
        force_bottom = np.zeros(n)
        radius = np.zeros(n)
        maxpen = np.zeros(n)

        u = np.zeros(self.basis.N)
        for i, delta in enumerate(deltas):
            if delta == 0.0:
                continue
            if self.verbose:
                print(f"  step {i}/{n - 1}: delta = {delta:.4f} nm")
            u = self.solve_step(u, delta, fb)
            force[i] = self._contact_force(fb, u, delta)
            force_bottom[i] = self._bottom_force(u)
            radius[i] = self._contact_radius(fb, u, delta)
            maxpen[i] = self._max_penetration(fb, u, delta)

        self._last_u = u
        curve = ForceCurve(
            delta=np.array(deltas),
            force=force,
            force_bottom=force_bottom,
            contact_radius=radius,
            max_penetration=maxpen,
            cfg=cfg,
            n_dofs=self.basis.N,
        )
        self._check_quality(curve)
        return curve

    def _check_quality(self, curve: ForceCurve) -> None:
        """Emit warnings if equilibrium or penalty diagnostics look poor."""
        cfg = self.cfg
        res = curve.equilibrium_residual
        if res.size:
            worst = float(np.nanmax(res[1:])) if res.size > 1 else 0.0
            if worst > cfg.equilibrium_tol:
                warnings.warn(
                    f"equilibrium check: contact vs bottom force differ by "
                    f"{worst * 100:.1f}% (> {cfg.equilibrium_tol * 100:.0f}%). "
                    "Consider refining the mesh or raising penalty_alpha.",
                    stacklevel=2,
                )
        if curve.max_penetration.size:
            rel = curve.max_penetration / np.maximum(curve.delta, 1e-30)
            worst_pen = float(np.max(rel[1:])) if rel.size > 1 else 0.0
            if worst_pen > 0.05:
                warnings.warn(
                    f"penalty interpenetration reaches {worst_pen * 100:.1f}% "
                    "of delta (> 5%). Increase penalty_alpha.",
                    stacklevel=2,
                )
