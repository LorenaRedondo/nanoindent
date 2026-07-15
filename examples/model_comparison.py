"""Compare the analytic contact models against the FEM.

Overlays Hertz, Garcia & Garcia 2018, Dimitriadis 2002 (bonded), and
Chadwick 2002 (thin incompressible) against the FEM force-indentation curve,
for one or more film thicknesses. Saves a CSV table of all model forces and a
journal-style comparison figure.

Usage
-----
    python examples/model_comparison.py [--outdir DIR] [--thicknesses 7 14]
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

from nanoindent import (
    SimulationConfig, IndentationSolver, analytics, plotting,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--thicknesses", nargs="+", type=float, default=[7.0, 14.0])
    ap.add_argument("--journal", default="biophysical_journal")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    plotting.set_publication_style(journal=args.journal)
    n = len(args.thicknesses)
    width = plotting.journal_widths(args.journal)["double"]
    fig, axes = plt.subplots(1, n, figsize=(width, width * 0.5 / max(n, 1) * 1.6),
                             squeeze=False)

    for j, h in enumerate(args.thicknesses):
        cfg = SimulationConfig(R=17.0, E=130.0, nu=0.3, h=h, L=100.0,
                               delta_max=1.0, n_steps=12,
                               mesh_size_contact=0.2, mesh_size_edge=6.0,
                               penalty_alpha=400.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            curve = IndentationSolver(cfg).run()

        # Tabulate every model plus the FEM on the FEM's delta grid.
        table = analytics.compare_models(cfg, delta=curve.delta,
                                         models=analytics.BEC_MODELS)
        table["fem"] = curve.force
        csv_path = os.path.join(args.outdir, f"model_comparison_h{int(h)}.csv")
        keys = ["delta"] + analytics.BEC_MODELS + ["fem"]
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(keys)
            for i in range(len(curve.delta)):
                w.writerow([f"{table[k][i]:.6g}" for k in keys])

        ax = axes[0, j]
        plotting.plot_model_comparison(cfg, models=analytics.BEC_MODELS,
                                       fem_curve=curve, ax=ax)
        ax.set_title(f"$h = {h:g}$ nm", fontsize=8)
        plotting.panel_label(ax, j)

        chi = np.sqrt(cfg.R * cfg.delta_max) / h
        print(f"h={h} nm (chi_max={chi:.2f}):  FEM(delta_max)={curve.force[-1]:.0f} pN")
        for k in analytics.BEC_MODELS:
            print(f"    {k:20s} {table[k][-1]:8.0f} pN")

    stem = os.path.join(args.outdir, "model_comparison")
    written = plotting.save_figure(fig, stem, formats=("pdf", "png"))
    print("\nWrote:", ", ".join(written))


if __name__ == "__main__":
    main()
