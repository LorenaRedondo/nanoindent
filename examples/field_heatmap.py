"""Field heatmaps of the FEM indentation (von Mises stress, displacement, ...).

Runs an indentation and renders heatmaps of the recovered fields over the
axisymmetric (r, z) cross-section, mirrored across the axis with the rigid tip
outline overlaid.

Usage
-----
    python examples/field_heatmap.py [--outdir DIR] [--h 7] [--fields von_mises uz]
"""

from __future__ import annotations

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nanoindent import SimulationConfig, IndentationSolver, plotting


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--h", type=float, default=7.0)
    ap.add_argument("--fields", nargs="+",
                    default=["von_mises", "uz", "pressure"])
    ap.add_argument("--deformed", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cfg = SimulationConfig(R=17.0, E=130.0, nu=0.3, h=args.h, L=40.0,
                           delta_max=1.0, n_steps=10, mesh_size_contact=0.15,
                           mesh_size_edge=2.5, penalty_alpha=400.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = IndentationSolver(cfg)
        solver.run()

    plotting.set_publication_style(journal="generic")
    cmaps = {"von_mises": "viridis", "uz": "RdBu_r", "ur": "RdBu_r",
             "displacement": "magma", "pressure": "coolwarm", "sigma_zz": "RdBu_r"}
    n = len(args.fields)
    fig, axes = plt.subplots(n, 1, figsize=(plotting.SINGLE_COLUMN * 1.6, 2.4 * n),
                             squeeze=False)
    for i, field in enumerate(args.fields):
        plotting.plot_field(solver, field=field, ax=axes[i, 0],
                            cmap=cmaps.get(field, "viridis"),
                            deformed=args.deformed)

    stem = os.path.join(args.outdir, f"field_heatmap_h{int(args.h)}")
    written = plotting.save_figure(fig, stem, formats=("png", "pdf"))
    print("Wrote:", ", ".join(written))


if __name__ == "__main__":
    main()
