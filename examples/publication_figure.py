"""Generate a journal-ready multi-panel figure from the FEM results.

Produces a two-panel figure (double-column width, vector PDF + 600-dpi PNG):

  (a) force-indentation F(delta) for h = 7 and 14 nm, FEM vs. Hertz and the
      bonded Dimitriadis correction;
  (b) the finite-thickness stiffening ratio F/F_Hertz(delta).

Usage
-----
    python examples/publication_figure.py [--outdir DIR] [--formats pdf png svg]
"""

from __future__ import annotations

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")

from nanoindent import SimulationConfig, IndentationSolver, plotting


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    curves = {}
    for h in (7.0, 14.0):
        cfg = SimulationConfig(R=17.0, E=130.0, nu=0.3, h=h, L=100.0,
                               delta_max=1.0, n_steps=15,
                               mesh_size_contact=0.15, mesh_size_edge=6.0,
                               penalty_alpha=400.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            curves[h] = IndentationSolver(cfg).run()

    plotting.set_publication_style(font_size=8)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(plotting.DOUBLE_COLUMN, 2.8))

    plotting.plot_force_curves(
        curves, compare=["hertz", "dimitriadis_bonded"], ax=axes[0])
    plotting.plot_stiffening_ratio(
        curves, compare=["dimitriadis_bonded"], ax=axes[1])

    axes[0].text(-0.02, 1.03, "(a)", transform=axes[0].transAxes,
                 fontweight="bold", ha="right", va="bottom")
    axes[1].text(-0.02, 1.03, "(b)", transform=axes[1].transAxes,
                 fontweight="bold", ha="right", va="bottom")

    stem = os.path.join(args.outdir, "bottom_effect")
    written = plotting.save_figure(fig, stem, formats=args.formats)
    print("Wrote:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
