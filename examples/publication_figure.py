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
    ap.add_argument("--journal", default="biophysical_journal",
                    help="figure style preset (see plotting.JOURNAL_PRESETS)")
    ap.add_argument("--width", default="double",
                    choices=["single", "onehalf", "double"])
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

    plotting.set_publication_style(journal=args.journal)
    width = plotting.journal_widths(args.journal)[args.width]
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(width, width * 0.42))

    plotting.plot_force_curves(
        curves, compare=["hertz", "dimitriadis_bonded"], ax=axes[0])
    plotting.plot_stiffening_ratio(
        curves, compare=["dimitriadis_bonded"], ax=axes[1])

    plotting.panel_label(axes[0], 0)   # "A" for Cell Press, "(a)" generic
    plotting.panel_label(axes[1], 1)   # "B" / "(b)"

    stem = os.path.join(args.outdir, "bottom_effect")
    written = plotting.save_figure(fig, stem, formats=args.formats)
    print("Wrote:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
