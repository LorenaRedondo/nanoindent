"""Validate the FEM against analytic Hertz theory in the thick-film limit.

A very thick film (h >> contact radius) should reproduce the Hertz half-space
force-indentation law within a couple of percent. This script runs that case,
prints the ratio F_FEM / F_Hertz, and saves a plot.

Usage
-----
    python examples/hertz_validation.py [--outdir DIR]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cfg = SimulationConfig(
        R=17.0, E=130.0, nu=0.3, h=200.0, L=200.0,
        delta_max=1.0, n_steps=12,
        mesh_size_contact=0.3, mesh_size_edge=12.0,
        penalty_alpha=400.0,
    )
    print(cfg.summary())
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        curve = IndentationSolver(cfg).run()

    fh = analytics.hertz_force(curve.delta, cfg.R, cfg.E, cfg.nu)
    ratio = np.divide(curve.force, fh, out=np.full_like(fh, np.nan), where=fh > 0)

    print("\n delta(nm)   F_FEM(pN)   F_Hertz(pN)   ratio")
    for d, ff, fhh, rr in zip(curve.delta, curve.force, fh, ratio):
        print(f"  {d:6.3f}   {ff:9.2f}   {fhh:10.2f}   {rr:6.3f}")
    good = curve.delta >= 0.5
    print(f"\n mean ratio (delta>=0.5 nm) = {np.nanmean(ratio[good]):.4f}")

    ax = curve.plot(compare=["hertz"])
    ax.figure.savefig(os.path.join(args.outdir, "hertz_validation.png"),
                      dpi=150, bbox_inches="tight")
    print(f" Saved plot to '{args.outdir}/hertz_validation.png'")


if __name__ == "__main__":
    main()
