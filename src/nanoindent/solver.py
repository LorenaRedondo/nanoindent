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
        self.mixed = str(cfg.formulation).strip().lower() == "mixed"
        # Shear modulus is well defined at nu = 0.5; the primal constitutive
        # matrix (which needs lambda) is only built when not mixed.
        self._mu = cfg.E / (2.0 * (1.0 + cfg.nu))
        self.material = None if self.mixed else LinearElastic(cfg.E, cfg.nu)
        self.tip = make_tip(cfg)

        self.history: list = []      # per-step {delta, u, p} when run(store_history=True)
        self.domain: Optional[Domain] = None
        self.basis: Optional[Basis] = None
        self.basis_p: Optional[Basis] = None
        self._K = None
        self._A = self._B = self._C = None
        self._dirichlet = None

    # -- setup -------------------------------------------------------------
    def setup(self) -> None:
        """Build the mesh, function space(s), operators, and BC dofs."""
        self.domain = build_mesh(self.cfg)
        mesh = self.domain.mesh
        elem = ElementVector(
            ElementTriP2() if self.cfg.element_order == 2 else ElementTriP1()
        )
        if self.mixed:
            # Taylor-Hood: P2 displacement, P1 pressure, shared quadrature.
            self.basis = Basis(mesh, elem, intorder=4)
            self.basis_p = Basis(mesh, ElementTriP1(), intorder=4)
            self._assemble_mixed_blocks()
        else:
            self.basis = Basis(mesh, elem)
            C = self.material.constitutive_matrix()
            self._K = self._assemble_stiffness(self.basis, C)
        self._dirichlet = self._dirichlet_dofs(self.basis)

    def _assemble_mixed_blocks(self) -> None:
        """Assemble the constant (u,p) Taylor-Hood blocks A, B, C.

        System per Newton step::

            [ A + Kc   B^T ] [du]   = -[ A u + B^T p + Rc ]
            [ B       -C   ] [dp]      [ B u - C p        ]

        with ``A(u,v)=int 2 mu eps(u):eps(v) r``, ``B(u,q)=int (div u) q r`` and
        ``C(p,q)=int (1/lambda) p q r`` (C = 0 for nu = 0.5).
        """
        mu = self._mu
        nu = self.cfg.nu
        inv_lambda = 0.0 if nu >= 0.5 - 1e-12 else 1.0 / self.material_lambda()

        @BilinearForm
        def A_form(u, v, w):
            r = w.x[0]
            e_rr = u.grad[0][0]; e_zz = u.grad[1][1]
            e_tt = u[0] / r;     g_rz = u.grad[0][1] + u.grad[1][0]
            f_rr = v.grad[0][0]; f_zz = v.grad[1][1]
            f_tt = v[0] / r;     g_rz_v = v.grad[0][1] + v.grad[1][0]
            # 2 mu eps:eps ; eps:eps uses 1/2 on the engineering shear product
            return 2 * mu * (e_rr * f_rr + e_zz * f_zz + e_tt * f_tt
                             + 0.5 * g_rz * g_rz_v) * r

        @BilinearForm
        def B_form(u, q, w):        # trial u (vector), test q (scalar)
            r = w.x[0]
            div_u = u.grad[0][0] + u.grad[1][1] + u[0] / r
            return div_u * q * r

        @BilinearForm
        def C_form(p, q, w):
            r = w.x[0]
            return inv_lambda * p * q * r

        self._A = A_form.assemble(self.basis)
        self._B = B_form.assemble(self.basis, self.basis_p)
        self._C = C_form.assemble(self.basis_p)

    def material_lambda(self) -> float:
        """First Lame parameter (only used by the mixed C block for nu<0.5)."""
        nu, E = self.cfg.nu, self.cfg.E
        return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def _dirichlet_dofs(self, basis: Basis) -> np.ndarray:
        # Axis: u_r = 0 (u^1 is the r-component). Bottom: bonded fixes both
        # components (u_r = u_z = 0); unbonded fixes only the normal component
        # (u_z = 0) so the film slides frictionlessly on the rigid substrate.
        axis_r = basis.get_dofs("axis").all("u^1")
        if str(self.cfg.substrate).strip().lower() == "unbonded":
            bottom = basis.get_dofs("bottom").all("u^2")
        else:
            bottom = basis.get_dofs("bottom").all()
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

        Primal residual ``K u + R_contact(u) = 0``; mixed residual is the
        Taylor-Hood block system (see :meth:`_assemble_mixed_blocks`).  Returns
        the converged displacement vector (pressure is stored in ``self._p``).
        """
        if self.mixed:
            return self._solve_step_mixed(u0, delta, fb)
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

    def _solve_step_mixed(self, u0: np.ndarray, delta: float, fb: FacetBasis):
        """Active-set Newton on the coupled (u, p) Taylor-Hood system."""
        from scipy.sparse import bmat

        cfg = self.cfg
        Nu, Np = self.basis.N, self.basis_p.N
        A, B, C = self._A, self._B, self._C
        D = self._dirichlet            # displacement dofs only
        u = u0.copy()
        p = self._p.copy() if getattr(self, "_p", None) is not None \
            else np.zeros(Np)

        r0 = None
        for it in range(cfg.newton_max_it):
            Rc, Kc = self._contact_forms(fb, u, delta)
            # Residual blocks
            res_u = A @ u + B.T @ p + Rc
            res_p = B @ u - C @ p
            residual = np.concatenate([res_u, res_p])
            J = bmat([[A + Kc, B.T], [B, -C]], format="csr")

            dx = solve(*condense(J, -residual, D=D))
            u = u + dx[:Nu]
            p = p + dx[Nu:]

            free = np.ones(Nu + Np, dtype=bool)
            free[D] = False
            rnorm = np.linalg.norm(residual[free])
            if r0 is None:
                r0 = max(rnorm, 1e-30)
            dnorm = np.linalg.norm(dx)
            if self.verbose:
                print(f"    [mixed] it={it:2d} |R|={rnorm:.3e} |dx|={dnorm:.3e}")
            if rnorm <= cfg.newton_rtol * r0 or dnorm < 1e-12:
                break
        else:
            warnings.warn(
                f"Newton (mixed) did not converge at delta={delta:.4f} nm "
                f"(|R|/|R0|={rnorm / r0:.2e})", stacklevel=2,
            )
        self._p = p
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

        Total vertical reaction = 2*pi * integral sig_zz(z=0) r dr.  In the
        mixed formulation sig_zz = 2 mu eps_zz + p (the pressure carries the
        volumetric stress); in the primal formulation it is the full C row.
        """
        # In mixed mode the u and p facet bases must share quadrature.
        io = 4 if self.mixed else None
        fb = FacetBasis(self.domain.mesh, self.basis.elem, facets="bottom",
                        intorder=io)
        uh = fb.interpolate(u)

        if self.mixed:
            mu = self._mu
            fb_p = FacetBasis(self.domain.mesh, self.basis_p.elem,
                              facets="bottom", intorder=io)
            ph = fb_p.interpolate(self._p)

            @Functional
            def integ(w):
                r = w.x[0]
                e_zz = w["u"].grad[1][1]
                s_zz = 2.0 * mu * e_zz + w["p"]
                return s_zz * r

            val = integ.assemble(fb, u=uh, p=ph)
        else:
            C = self.material.constitutive_matrix()

            @Functional
            def integ(w):
                r = w.x[0]
                e_rr = w["u"].grad[0][0]
                e_zz = w["u"].grad[1][1]
                e_tt = w["u"][0] / r
                s_zz = C[1, 0] * e_rr + C[1, 1] * e_zz + C[1, 2] * e_tt
                return s_zz * r

            val = integ.assemble(fb, u=uh)

        # The substrate reaction on the sample is the traction t_z = -sig_zz
        # integrated over the bottom (outward normal -z), giving a positive
        # force that balances the downward contact pressure.
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

    # -- field recovery ----------------------------------------------------
    def nodal_field(self, name: str = "von_mises", u: Optional[np.ndarray] = None,
                    p: Optional[np.ndarray] = None):
        """Return a scalar field sampled at the mesh vertices (for heatmaps).

        The field is recovered from the converged solution by an L2 projection
        onto continuous P1 (so it aligns with ``self.domain.mesh.p``).

        Supported ``name`` values: ``"von_mises"``, ``"sigma_zz"``,
        ``"pressure"`` (mechanical, ``-tr(sigma)/3``), ``"displacement"``
        (magnitude), ``"uz"``, ``"ur"``. Stresses are in MPa, displacements nm.

        ``u`` / ``p`` override the stored solution (used when animating a stored
        load-step history); ``p`` is only consulted in the mixed formulation.
        """
        from skfem import ElementTriP1, solve as sksolve

        if u is None:
            u = getattr(self, "_last_u", None)
        if u is None:
            raise RuntimeError("run() the solver before requesting fields")

        mesh = self.domain.mesh
        io = 4
        p1 = Basis(mesh, ElementTriP1(), intorder=io)
        ub = Basis(mesh, self.basis.elem, intorder=io)
        uh = ub.interpolate(u)
        mu = self._mu
        mixed = self.mixed
        C = None if mixed else self.material.constitutive_matrix()
        ph = None
        if mixed:
            pb = Basis(mesh, self.basis_p.elem, intorder=io)
            ph = pb.interpolate(self._p if p is None else p)

        @BilinearForm
        def mass(a, b, w):
            return a * b * w.x[0]

        @LinearForm
        def load(v, w):
            r = w.x[0]
            U = w["u"]
            e_rr = U.grad[0][0]; e_zz = U.grad[1][1]
            e_tt = U[0] / r;     g_rz = U.grad[0][1] + U.grad[1][0]
            if mixed:
                pp = w["p"]
                s_rr = 2 * mu * e_rr + pp; s_zz = 2 * mu * e_zz + pp
                s_tt = 2 * mu * e_tt + pp; s_rz = mu * g_rz
            else:
                s_rr = C[0, 0] * e_rr + C[0, 1] * e_zz + C[0, 2] * e_tt
                s_zz = C[1, 0] * e_rr + C[1, 1] * e_zz + C[1, 2] * e_tt
                s_tt = C[2, 0] * e_rr + C[2, 1] * e_zz + C[2, 2] * e_tt
                s_rz = C[3, 3] * g_rz
            if name == "von_mises":
                f = np.sqrt(0.5 * ((s_rr - s_zz) ** 2 + (s_zz - s_tt) ** 2
                                   + (s_tt - s_rr) ** 2) + 3 * s_rz ** 2)
            elif name == "sigma_zz":
                f = s_zz
            elif name == "pressure":
                f = -(s_rr + s_zz + s_tt) / 3.0
            elif name == "uz":
                f = U[1]
            elif name == "ur":
                f = U[0]
            elif name == "displacement":
                f = np.sqrt(U[0] ** 2 + U[1] ** 2)
            else:
                raise ValueError(f"unknown field {name!r}")
            return f * v * r

        M = mass.assemble(p1)
        kwargs = {"u": uh}
        if mixed:
            kwargs["p"] = ph
        b = load.assemble(p1, **kwargs)
        return sksolve(M, b)

    def nodal_displacement(self, u: Optional[np.ndarray] = None):
        """Return (ur, uz) at the mesh vertices (nm), for deformed-shape plots."""
        return self.nodal_field("ur", u), self.nodal_field("uz", u)

    # -- driver ------------------------------------------------------------
    def run(self, store_history: bool = False) -> ForceCurve:
        """Run the full load-stepping simulation and return a ForceCurve.

        With ``store_history=True`` the per-step solution ``{delta, u, p}`` is
        kept in ``self.history`` so the field evolution can be animated.
        """
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

        self.history = []
        u = np.zeros(self.basis.N)
        if self.mixed:
            self._p = np.zeros(self.basis_p.N)
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
            if store_history:
                self.history.append({
                    "delta": delta,
                    "u": u.copy(),
                    "p": (self._p.copy() if self.mixed else None),
                })

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
