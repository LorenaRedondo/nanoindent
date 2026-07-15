"""AFM nanoindentation of a 6.2 nm film by an MSNL-BIO-DC tip (rigid paraboloid).

Reproduces, with the open-source solver, the kind of simulation usually run in
commercial FEM software (COMSOL Multiphysics):

    rigid paraboloid  R = 20 nm  (tip E = 170 GPa >> sample -> rigid)
    sample            E = 50 MPa, linear isotropic elastic, thickness h = 6.2 nm
    lateral size      L = 200 nm (no lateral boundary effects)
    bottom            fixed to a rigid substrate (bonded)
    contact           frictionless, indentation up to delta = 2 nm
    mesh              graded: fine at the contact point, coarse at the edges;
                      refined until F(delta_max) converged

Deliverables (written to --outdir):
  * von_mises_cross_section.png/pdf  -- cross-section, von Mises stress (jet)
  * force_indentation.png/pdf        -- F(delta), nu=0.5 vs nu=0.3 + analytics
  * mesh_convergence.png/pdf + .csv  -- F(delta_max) vs element size
  * force_indentation_nu05.csv / _nu03.csv

Poisson's ratio is not specified for the sample; nu = 0.5 (incompressible, the
standard choice for soft hydrated biological material) is the primary case and
uses the locking-free mixed formulation, with nu = 0.3 shown for comparison.

Usage
-----
    python examples/msnl_bio_dc.py [--outdir DIR] [--contact 0.01] [--quick]
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

R, E, L, H, DMAX = 20.0, 50.0, 200.0, 6.2, 2.0
BASE = dict(tip_shape="paraboloid", R=R, E=E, L=L, h=H, delta_max=DMAX,
            mesh_size_edge=4.0, penalty_alpha=600.0)


def _run(nu, mesh_contact, n_steps, mixed):
    cfg = SimulationConfig(
        nu=nu, formulation="mixed" if mixed else "primal",
        n_steps=n_steps, mesh_size_contact=mesh_contact, **BASE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = IndentationSolver(cfg)
        curve = solver.run()
    return solver, curve


def mesh_convergence(outdir, sizes, n_steps):
    """(d) Refine the near-contact element size and tabulate F(delta_max)."""
    print("Mesh convergence (nu=0.5, mixed):")
    rows = []
    prev = None
    for msc in sizes:
        _, c = _run(0.5, msc, n_steps, mixed=True)
        f = float(c.force[-1])
        change = None if prev is None else (f - prev) / prev * 100.0
        rows.append((msc, int(c.n_dofs), f, change))
        print(f"  contact={msc:6.3f} nm  DOF={c.n_dofs:6d}  F(2nm)={f:8.1f} pN"
              + ("" if change is None else f"  ({change:+.2f}%)"))
        prev = f

    with open(os.path.join(outdir, "mesh_convergence.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mesh_size_contact_nm", "n_dofs", "F_delta_max_pN", "rel_change_pct"])
        for msc, dof, f, ch in rows:
            w.writerow([msc, dof, f"{f:.3f}", "" if ch is None else f"{ch:.3f}"])

    plotting.set_publication_style(journal="generic")
    fig, ax = plt.subplots(figsize=(plotting.SINGLE_COLUMN * 1.3, 2.6))
    ax.plot([r[0] for r in rows], [r[2] for r in rows], "o-", color="#0072B2")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("near-contact element size (nm)")
    ax.set_ylabel(r"$F(\delta_\mathrm{max})$ (pN)")
    ax.set_title("Mesh convergence (finer $\\rightarrow$)", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    plotting.save_figure(fig, os.path.join(outdir, "mesh_convergence"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/msnl_bio_dc")
    ap.add_argument("--contact", type=float, default=0.005,
                    help="near-contact element size for the production run (nm)")
    ap.add_argument("--quick", action="store_true",
                    help="coarser/faster settings for a smoke run")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    conv_sizes = (0.04, 0.02) if args.quick else (0.04, 0.02, 0.01, 0.005)
    n_conv = 6 if args.quick else 8
    n_prod = 12 if args.quick else 20
    contact = max(args.contact, 0.02) if args.quick else args.contact

    # (d) convergence
    mesh_convergence(args.outdir, conv_sizes, n_conv)

    # (a) production runs: nu = 0.5 (incompressible, mixed) and nu = 0.3 (primal)
    print(f"\nProduction runs (contact = {contact} nm, {n_prod} steps):")
    solver5, curve5 = _run(0.5, contact, n_prod, mixed=True)
    print(f"  nu=0.5  DOF={curve5.n_dofs}  F(2nm)={curve5.force[-1]:.0f} pN  "
          f"equil={np.nanmax(curve5.equilibrium_residual)*100:.2f}%  "
          f"vMises_max={solver5.nodal_field('von_mises').max():.1f} MPa")
    solver3, curve3 = _run(0.3, contact, n_prod, mixed=False)
    print(f"  nu=0.3  DOF={curve3.n_dofs}  F(2nm)={curve3.force[-1]:.0f} pN  "
          f"equil={np.nanmax(curve3.equilibrium_residual)*100:.2f}%")
    curve5.save(os.path.join(args.outdir, "force_indentation_nu05.csv"))
    curve3.save(os.path.join(args.outdir, "force_indentation_nu03.csv"))

    # Figure 1 — von Mises cross-section (nu = 0.5), zoomed to the contact region
    fig1, ax1 = plt.subplots(figsize=(5.6, 2.8))
    plotting.plot_field(solver5, field="von_mises", ax=ax1, mirror=True,
                        cmap="jet", levels=60, show_tip=True)
    ax1.set_xlim(-35, 35)
    ax1.set_ylim(0, 8.2)
    ax1.set_aspect("equal")
    ax1.set_title(r"Von Mises stress (MPa) — paraboloid ($R$=20 nm) on 6.2 nm "
                  r"film, $\delta$=2 nm", fontsize=7.5)
    plotting.save_figure(fig1, os.path.join(args.outdir, "von_mises_cross_section"))

    # Figure 2 — force-indentation, nu = 0.5 vs 0.3, with analytic references
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
    ax2.set_xlabel(r"indentation $\delta$ (nm)")
    ax2.set_ylabel(r"force $F$ (pN)")
    ax2.set_xlim(0, DMAX)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=7)
    ax2.set_title(f"Force–indentation: E={E:g} MPa, R={R:g} nm, h={H:g} nm",
                  fontsize=8)
    plotting.save_figure(fig2, os.path.join(args.outdir, "force_indentation"))

    print(f"\nWrote figures + CSVs to '{args.outdir}/'.")


if __name__ == "__main__":
    main()
