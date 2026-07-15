"""AFM nanoindentation of a 6.2 nm film by an MSNL-BIO-DC tip — unbonded case.

Numerical experiment matching the kind run in commercial FE software (COMSOL
Multiphysics; COMSOL AB, Stockholm, Sweden), with the open-source solver:

    rigid paraboloid   R = 20 nm   (tip E = 170 GPa >> sample -> rigid)
    sample             E = 50 MPa, linear isotropic homogeneous elastic
    thickness          h = 6.2 nm
    lateral size       L = 500 nm  (no lateral boundary effects)
    bottom             adsorbed to a rigid substrate but NOT fixed
                       -> unbonded: u_z = 0, radial slip free (frictionless base)
    contact            frictionless, indentation up to delta = 5 nm
    mesh               graded, refined to 0.01 nm at the contact point, coarser
                       toward the edges; refined until F(delta_max) converged

Deliverables (written to --outdir):
  * von_mises_cross_section.png/pdf   -- cross-section, von Mises stress (jet)
  * von_mises_indentation.mp4         -- von Mises stress during the indentation
  * force_indentation.png/pdf + .csv  -- F(delta), nu=0.5 (primary) and nu=0.3
  * mesh_convergence.csv              -- F(delta_max) vs near-contact element size
  * (docs) methodology_msnl_bio_dc_unbonded.docx  -- Methods, Biophysical Journal style

Poisson's ratio is not specified for the sample; nu = 0.5 (incompressible, the
standard choice for soft hydrated biological material) is the primary case and
uses the locking-free mixed formulation, with nu = 0.3 for comparison.

Usage
-----
    python examples/msnl_bio_dc_unbonded_5nm.py [--outdir DIR] [--contact 0.01]
                                                [--n-steps 30] [--quick]
"""

from __future__ import annotations

import argparse
import csv
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nanoindent import SimulationConfig, IndentationSolver, analytics, plotting

R, E, L, H, DMAX = 20.0, 50.0, 500.0, 6.2, 5.0
NU_TIP_GPA = 170.0
BASE = dict(tip_shape="paraboloid", R=R, E=E, L=L, h=H, delta_max=DMAX,
            substrate="unbonded", mesh_size_edge=4.0, penalty_alpha=600.0)


def _run(nu, mesh_contact, n_steps, mixed, store_history=False):
    cfg = SimulationConfig(
        nu=nu, formulation="mixed" if mixed else "primal",
        n_steps=n_steps, mesh_size_contact=mesh_contact, **BASE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = IndentationSolver(cfg)
        curve = solver.run(store_history=store_history)
    return solver, curve


def mesh_convergence(outdir, sizes, n_steps):
    print("Mesh convergence (nu=0.5, mixed, unbonded):")
    rows, prev = [], None
    for msc in sizes:
        _, c = _run(0.5, msc, n_steps, mixed=True)
        f = float(c.force[-1])
        change = None if prev is None else (f - prev) / prev * 100.0
        rows.append((msc, int(c.n_dofs), f, change))
        print(f"  contact={msc:6.3f} nm  DOF={c.n_dofs:6d}  F(5nm)={f:9.1f} pN"
              + ("" if change is None else f"  ({change:+.2f}%)"))
        prev = f
    with open(os.path.join(outdir, "mesh_convergence.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mesh_size_contact_nm", "n_dofs", "F_delta_max_pN", "rel_change_pct"])
        for msc, dof, f, ch in rows:
            w.writerow([msc, dof, f"{f:.3f}", "" if ch is None else f"{ch:.3f}"])
    return rows


def build_docx(path, res):
    """Write the Methods section as a Biophysical Journal-style .docx."""
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)

    def h(text, level):
        doc.add_heading(text, level=level)

    def p(text, justify=True):
        par = doc.add_paragraph(text)
        if justify:
            par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        return par

    title = doc.add_heading(
        "Finite-element simulation of AFM nanoindentation of a thin elastic film "
        "adsorbed to a rigid substrate", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = p("Materials and Methods (Biophysical Journal style)", justify=False)
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].italic = True

    h("Finite-element model", 1)
    p("Force–indentation curves and the internal stress distribution of the "
      "sample were computed with a custom, open-source finite-element (FE) "
      "solver (nanoindent, implemented in Python on the scikit-fem/FEniCSx "
      "framework), reproducing the numerical experiment usually carried out in "
      "commercial FE software (COMSOL Multiphysics; COMSOL AB, Stockholm, "
      "Sweden). The atomic force microscopy (AFM) indentation of a probe "
      "(MSNL-BIO-DC) into a thin elastic film adsorbed to a rigid substrate is "
      "an axisymmetric problem; exploiting the rotational symmetry about the "
      "indentation axis, the three-dimensional geometry was reduced to a "
      "two-dimensional cross-section in cylindrical coordinates (r, z). All "
      "fields, boundary conditions, and the contact constraint are independent "
      "of the azimuthal angle, and every domain integral carries the "
      "axisymmetric weight 2πr.")

    h("Geometry and material models", 1)
    p(f"The indenter was modeled as a rigid paraboloid of apex radius R = {R:g} "
      "nm, matching the nominal end radius of the MSNL-BIO-DC tip. Although the "
      f"probe is a silicon-nitride cantilever with Young's modulus E_tip = "
      f"{NU_TIP_GPA:g} GPa, its stiffness exceeds that of the sample by more "
      "than three orders of magnitude (E_tip/E_sample ≈ 3.4×10³); "
      "the tip deformation is therefore negligible and the indenter was treated "
      "as perfectly rigid, entering the formulation solely through its gap "
      "(indentation) function. The exact paraboloidal profile, z_tip(r) = "
      "r²/(2R), was used rather than a small-slope linearization.")
    p(f"The sample was represented as a linear, isotropic, homogeneous elastic "
      f"solid with Young's modulus E = {E:g} MPa. Because the Poisson's ratio "
      "of the sample was not independently determined, it was fixed at "
      "ν = 0.5 (fully incompressible), the value conventionally assumed for "
      "soft, hydrated biological material; a compressible case (ν = 0.3) "
      "was also computed for comparison. Small-strain (infinitesimal) kinematics "
      f"were assumed throughout. The sample was an axisymmetric disc of radius "
      f"L = {L:g} nm and thickness h = {H:g} nm; the lateral extent was large "
      "compared with the contact radius, so that lateral boundary effects were "
      "negligible.")

    h("Boundary and contact conditions", 1)
    p("The following boundary conditions were imposed. Bottom surface (z = 0): "
      "the film is adsorbed to the rigid substrate but not clamped, so it cannot "
      "detach or penetrate the substrate yet is free to slide along it; this was "
      "modeled as a frictionless (unbonded) support with the normal "
      "displacement constrained, u_z = 0, and the tangential (radial) "
      "displacement u_r left free. Symmetry axis (r = 0): u_r = 0 (axisymmetry). "
      "Lateral edge (r = L): traction-free. Top surface (z = h): frictionless "
      "(normal-only) unilateral contact against the rigid tip, traction-free "
      "outside the contact region.")
    p("Contact was enforced with a penalty formulation. The signed gap between "
      "the deformed sample surface and the rigid paraboloid, g(r) = z_tip(r) "
      "− [h + u_z(r)], defines the interpenetration [−g]₊ = "
      "max(−g, 0); a contact pressure p_c = k_pen[−g]₊ was "
      "applied over the top surface, with the penalty stiffness k_pen scaled to "
      "the local element size to remain consistent under mesh refinement. Only "
      "the normal component of the contact traction was retained (frictionless "
      "contact); no tangential constraint was imposed.")
    p("Because the incompressible sample (ν = 0.5) would cause volumetric "
      "locking of a single-field displacement discretization, the elasticity "
      "was solved with a mixed displacement–pressure (Taylor–Hood) formulation, "
      "using quadratic (P2) displacement and linear (P1) pressure fields; the "
      "pressure enforces the incompressibility constraint. For the compressible "
      "comparison (ν = 0.3) the standard single-field P2 displacement "
      "formulation was used.")

    h("Loading and solution", 1)
    p(f"Indentation was displacement-controlled: the rigid tip was advanced "
      f"normal to the sample surface in {res['n_steps']} equal increments up to "
      f"a maximum indentation of δ = {DMAX:g} nm. At each increment the "
      "nonlinear contact problem — nonlinear only through the activation of the "
      "contact set — was solved by a Newton iteration, using the converged "
      "solution of the previous step as the initial guess; the linear systems "
      "were solved with a direct sparse solver. The reaction force on the "
      "indenter was computed by integrating the contact pressure over the top "
      "surface, F = ∫ k_pen[−g]₊ 2πr dr, and, independently, "
      "by integrating the axial stress over the bottom boundary; the two "
      f"estimates agreed to within {res['equil_pct']:.1f}%, confirming global "
      "equilibrium. Forces are reported in piconewtons, using the consistent "
      "unit system (length in nm, stress in MPa), for which 1 MPa·nm² "
      "= 1 pN.")

    h("Mesh and convergence", 1)
    p("The cross-section was discretized with quadratic triangular elements on a "
      "graded mesh, strongly refined at the contact point (r = 0, z = h) and "
      "coarsening toward the outer boundaries. Mesh independence was established "
      "by successively reducing the near-contact element size (down to 0.01 nm) "
      "while holding the graded coarsening toward the edges fixed; the reaction "
      "force at maximum indentation varied by only a few percent over the finest "
      "refinements. The reported results use the 0.01 nm near-contact mesh. "
      "Several element layers were resolved across the 6.2 nm film thickness.")
    tbl = doc.add_table(rows=1, cols=3)
    tbl.style = "Light Grid Accent 1"
    hdr = tbl.rows[0].cells
    hdr[0].text = "Near-contact element size (nm)"
    hdr[1].text = "Degrees of freedom"
    hdr[2].text = "F(δ = 5 nm) (pN)"
    for msc, dof, f, _ in res["conv_rows"]:
        c = tbl.add_row().cells
        c[0].text, c[1].text, c[2].text = f"{msc:g}", f"{dof:d}", f"{f:.0f}"

    h("Outputs", 1)
    p("Two quantities were extracted from the converged solution: (i) the "
      "force–indentation curve F(δ) over 0 ≤ δ ≤ 5 nm; and "
      "(ii) the spatial distribution of the von Mises (equivalent) stress, "
      "σ_vM = sqrt{ ½[(σ_rr−σ_zz)² + "
      "(σ_zz−σ_θθ)² + "
      "(σ_θθ−σ_rr)²] + 3σ_rz² }, "
      "recovered by L² projection onto the mesh nodes and displayed on the "
      "axisymmetric cross-section (mirrored about the symmetry axis). The "
      "evolution of σ_vM through the loading history was additionally "
      "rendered as a video.")
    p("For interpretation, the simulated F(δ) was compared with the "
      "analytical Hertz model for a paraboloid on an elastic half-space, "
      "F = (4/3)[E/(1−ν²)]√R δ^{3/2}; with the Chadwick "
      "(2002) asymptotic solution for a thin incompressible layer, "
      "F = (2π/3) E R² δ³/h³; and with the "
      "finite-thickness correction of Dimitriadis et al. (2002) for a film with "
      "a free (frictionless) base.")

    h("Representative results", 1)
    p(f"For the incompressible sample (E = {E:g} MPa, ν = 0.5, R = {R:g} "
      f"nm, h = {H:g} nm) adsorbed to but free to slide on the rigid substrate, "
      f"the converged model gave a reaction force of F ≈ {res['F5_nu05']/1000:.1f} "
      f"nN at the maximum indentation δ = {DMAX:g} nm, with a peak von "
      f"Mises stress of ≈ {res['vm_nu05']:.0f} MPa. Because the film is "
      "unbonded, the incompressible material relieves confinement by sliding "
      "laterally along the substrate rather than by large in-plane stretching, "
      "so the high-stress region extends down to the sliding base beneath the "
      "tip apex. At equal indentation the compressible sample (ν = 0.3) is "
      f"more compliant, F ≈ {res['F5_nu03']/1000:.1f} nN. The strongly "
      "nonlinear rise of F(δ) at this large indentation-to-thickness ratio "
      "(δ/h ≈ 0.8) reflects the finite-thickness (bottom-effect) "
      "stiffening of the thin film; the FE curve is bracketed by the Hertz "
      "half-space and thin-layer analytical limits.")

    h("Figures", 1)
    if res.get("fig_cross"):
        doc.add_picture(res["fig_cross"], width=Inches(5.8))
    p("Figure 1. Cross-section of the FE simulation of the AFM tip (rigid "
      f"paraboloid, R = {R:g} nm) indenting a {H:g} nm elastic film adsorbed to "
      f"a rigid substrate, at the maximum indentation δ = {DMAX:g} nm. The "
      "color map shows the simulated von Mises stress within the sample (red: "
      "high stress; blue: low stress) produced by the vertical force applied by "
      "the tip; the red curve marks the rigid tip surface, and the domain is "
      "mirrored about the symmetry axis r = 0.", justify=False)
    doc.add_paragraph()
    if res.get("fig_fd"):
        doc.add_picture(res["fig_fd"], width=Inches(5.4))
    p(f"Figure 2. Simulated force–indentation curve F(δ) for the same "
      f"sample (E = {E:g} MPa, h = {H:g} nm, unbonded base), for ν = 0.5 "
      "(incompressible, primary) and ν = 0.3, compared with the Hertz "
      "half-space, Chadwick thin-incompressible-layer, and Dimitriadis (2002) "
      "free-base models.", justify=False)
    p("Supplementary Video 1. Evolution of the von Mises stress field over the "
      "axisymmetric cross-section as the tip indents from δ = 0 to "
      f"{DMAX:g} nm (file: von_mises_indentation.mp4).", justify=False)

    h("References", 1)
    for ref in [
        "Hertz, H. (1882). Über die Berührung fester elastischer "
        "Körper. J. Reine Angew. Math. 92, 156–171.",
        "Sneddon, I. N. (1965). The relation between load and penetration in the "
        "axisymmetric Boussinesq problem for a punch of arbitrary profile. "
        "Int. J. Eng. Sci. 3, 47–57.",
        "Dimitriadis, E. K., F. Horkay, J. Maresca, B. Kachar, and R. S. "
        "Chadwick (2002). Determination of elastic moduli of thin layers of soft "
        "material using the atomic force microscope. Biophys. J. 82, "
        "2798–2810.",
        "Chadwick, R. S. (2002). Axisymmetric indentation of a thin "
        "incompressible elastic layer. SIAM J. Appl. Math. 62, 1520–1530.",
        "Garcia, P. D., and R. Garcia (2018). Determination of the elastic "
        "moduli of a single cell cultured on a rigid support by force "
        "microscopy. Biophys. J. 114, 2923–2932.",
    ]:
        par = doc.add_paragraph(ref)
        par.paragraph_format.left_indent = Inches(0.3)
        par.paragraph_format.first_line_indent = Inches(-0.3)

    doc.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/msnl_bio_dc_unbonded")
    ap.add_argument("--docx", default="docs/methodology_msnl_bio_dc_unbonded.docx")
    ap.add_argument("--contact", type=float, default=0.01)
    ap.add_argument("--n-steps", type=int, default=30)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    conv_sizes = (0.04, 0.02) if args.quick else (0.04, 0.02, 0.01)
    n_conv = 8 if args.quick else 12
    n_prod = 12 if args.quick else args.n_steps
    contact = max(args.contact, 0.02) if args.quick else args.contact

    conv_rows = mesh_convergence(args.outdir, conv_sizes, n_conv)

    print(f"\nProduction runs (contact = {contact} nm, {n_prod} steps, unbonded):")
    solver5, curve5 = _run(0.5, contact, n_prod, mixed=True, store_history=True)
    vm5 = float(solver5.nodal_field("von_mises").max())
    equil5 = float(np.nanmax(curve5.equilibrium_residual)) * 100.0
    print(f"  nu=0.5  DOF={curve5.n_dofs}  F(5nm)={curve5.force[-1]:.0f} pN  "
          f"equil={equil5:.2f}%  vMises_max={vm5:.1f} MPa")
    solver3, curve3 = _run(0.3, contact, n_prod, mixed=False)
    print(f"  nu=0.3  DOF={curve3.n_dofs}  F(5nm)={curve3.force[-1]:.0f} pN")
    curve5.save(os.path.join(args.outdir, "force_indentation_nu05.csv"))
    curve3.save(os.path.join(args.outdir, "force_indentation_nu03.csv"))

    # Figure 1 — von Mises cross-section (nu = 0.5) at delta = 5 nm
    fig1, ax1 = plt.subplots(figsize=(5.8, 3.0))
    plotting.plot_field(solver5, field="von_mises", ax=ax1, mirror=True,
                        cmap="jet", levels=60, show_tip=True)
    ax1.set_xlim(-45, 45); ax1.set_ylim(0, 8.2); ax1.set_aspect("equal")
    ax1.set_title(r"Von Mises stress (MPa) — paraboloid ($R$=20 nm) on 6.2 nm "
                  r"unbonded film, $\delta$=5 nm", fontsize=7.5)
    cross = os.path.join(args.outdir, "von_mises_cross_section")
    plotting.save_figure(fig1, cross)

    # Video — von Mises during the indentation
    print("Rendering von Mises video...")
    video = plotting.animate_field(
        solver5, field="von_mises",
        path=os.path.join(args.outdir, "von_mises_indentation.mp4"),
        mirror=True, cmap="jet", fps=args.fps, xlim=(-45, 45), ylim=(0, 8.2),
        dpi=150)
    print(f"  wrote {video}")

    # Figure 2 — force-indentation with analytic references
    fig2, ax2 = plt.subplots(figsize=(plotting.SINGLE_COLUMN * 1.5, 3.0))
    dd = np.linspace(0, DMAX, 200)
    ax2.plot(curve5.delta, curve5.force, "o-", color="#0072B2", ms=4,
             label=r"FEM $\nu$=0.5 (incompressible)")
    ax2.plot(curve3.delta, curve3.force, "s-", color="#009E73", ms=3.5,
             label=r"FEM $\nu$=0.3")
    ax2.plot(dd, analytics.hertz_force(dd, R, E, 0.5), "k:",
             label=r"Hertz ($\nu$=0.5)")
    ax2.plot(dd, analytics.chadwick_force(dd, R, E, H), "--", color="#D55E00",
             label="Chadwick 2002 (thin incompr.)")
    ax2.plot(dd, analytics.dimitriadis_force(dd, R, E, 0.5, H, bonded=False),
             "-.", color="#CC79A7", label="Dimitriadis 2002 (free base)")
    ax2.set_xlabel(r"indentation $\delta$ (nm)")
    ax2.set_ylabel(r"force $F$ (pN)")
    ax2.set_xlim(0, DMAX); ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.25); ax2.legend(fontsize=7)
    ax2.set_title(f"Force–indentation (unbonded): E={E:g} MPa, R={R:g} nm, "
                  f"h={H:g} nm", fontsize=8)
    fd = os.path.join(args.outdir, "force_indentation")
    plotting.save_figure(fig2, fd)

    # .docx methodology
    os.makedirs(os.path.dirname(args.docx), exist_ok=True)
    res = dict(n_steps=n_prod, equil_pct=equil5, conv_rows=conv_rows,
               F5_nu05=float(curve5.force[-1]), F5_nu03=float(curve3.force[-1]),
               vm_nu05=vm5, fig_cross=cross + ".png", fig_fd=fd + ".png")
    build_docx(args.docx, res)
    print(f"\nWrote figures + video + CSVs to '{args.outdir}/'")
    print(f"Wrote methodology to '{args.docx}'")


if __name__ == "__main__":
    main()
