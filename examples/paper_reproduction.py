"""Reproduce the reference paper's setup: F(delta) for h = 7 and 14 nm.

Runs the FEM indentation for both film thicknesses, overlays the analytic
Hertz and bottom-effect-correction (BEC) models, and saves a CSV + PNG for
each thickness plus a combined comparison figure.

Usage
-----
    python examples/paper_reproduction.py [--outdir DIR] [--quick]
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nanoindent import SimulationConfig, IndentationSolver, analytics


def run_thickness(h: float, quick: bool) -> "ForceCurve":
    cfg = SimulationConfig(
        R=17.0, E=130.0, nu=0.3, h=h, L=100.0,
        delta_max=1.0,
        n_steps=10 if quick else 20,
        mesh_size_contact=0.3 if quick else 0.15,
        mesh_size_edge=6.0,
        penalty_alpha=400.0,
    )
    print(f"\n=== h = {h} nm ===")
    print(cfg.summary())
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        curve = IndentationSolver(cfg, verbose=False).run()
    print(f"  DOFs: {curve.n_dofs}")
    print(f"  F(delta_max) = {curve.force[-1]:.1f} pN")
    print(f"  stiffening F_FEM/F_Hertz at delta_max = "
          f"{curve.stiffening_ratio()[-1]:.3f}")
    print(f"  max equilibrium residual = "
          f"{np.nanmax(curve.equilibrium_residual) * 100:.2f} %")
    print(f"  max interpenetration = "
          f"{np.max(curve.max_penetration / np.maximum(curve.delta, 1e-30)) * 100:.2f} % of delta")
    return curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--quick", action="store_true",
                    help="coarser/faster settings for a smoke run")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    curves = {}
    for h in (7.0, 14.0):
        curve = run_thickness(h, args.quick)
        tag = f"h{int(h)}"
        curve.save(os.path.join(args.outdir, f"{tag}.csv"))
        curve.plot(compare=["hertz", "dimitriadis_bonded", "garcia_bec"],
                   save=os.path.join(args.outdir, f"{tag}.png"))
        curves[h] = curve

    # Combined figure.
    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = {7.0: "tab:red", 14.0: "tab:blue"}
    for h, curve in curves.items():
        ax.plot(curve.delta, curve.force, "o-", color=colors[h], ms=4,
                label=f"FEM h={int(h)} nm")
        dd = np.linspace(0, 1, 200)
        ax.plot(dd, analytics.dimitriadis_force(dd, 17.0, 130.0, 0.3, h),
                "--", color=colors[h], alpha=0.7,
                label=f"Dimitriadis h={int(h)} nm")
    ax.plot(dd, analytics.hertz_force(dd, 17.0, 130.0, 0.3), "k:",
            label="Hertz (half-space)")
    ax.set_xlabel("indentation $\\delta$ (nm)")
    ax.set_ylabel("force $F$ (pN)")
    ax.set_title("Bottom-effect stiffening: h = 7 vs 14 nm (E=130 MPa, $\\nu$=0.3)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    combined = os.path.join(args.outdir, "comparison.png")
    fig.savefig(combined, dpi=150, bbox_inches="tight")
    print(f"\nSaved CSVs and PNGs to '{args.outdir}/' (combined: {combined})")


if __name__ == "__main__":
    main()
