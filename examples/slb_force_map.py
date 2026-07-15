"""Analyse a JPK force map of a supported lipid bilayer.

Reads a ``.jpk-force-map``, extracts per-curve breakthrough force, bilayer
height, and Young's modulus (Hertz / Dimitriadis / Garcia & Garcia / Chadwick),
prints the distributions, and saves the 4-panel summary figure with a FEM
cross-check.

Usage
-----
    python examples/slb_force_map.py MAP.jpk-force-map [--R 20 --nu 0.5 --h 5 --cap 3]
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")

from nanoindent import read_jpk_map, slb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("map", help="path to a .jpk-force-map file")
    ap.add_argument("--R", type=float, default=20.0, help="tip radius (nm)")
    ap.add_argument("--nu", type=float, default=0.5)
    ap.add_argument("--h", type=float, default=5.0, help="film thickness (nm)")
    ap.add_argument("--cap", type=float, default=3.0, help="fit window δ ≤ (nm)")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    fmap = read_jpk_map(args.map)
    print(f"{len(fmap)} curves ({fmap.n_rows}×{fmap.n_cols}), "
          f"k = {fmap.spring_constant:.3f} N/m, InvOLS = {fmap.inv_ols:.2f} nm/V")

    an = slb.analyze_map(fmap, R=args.R, nu=args.nu,
                         film_thickness=args.h, fit_max_delta=args.cap)
    s = an.summary()
    print(f"\n{'quantity':24s} {'n':>4} {'median':>10} {'mean':>10} {'std':>10}")
    for key, st in s.items():
        print(f"{key:24s} {st['n']:>4} {st['median']:>10.3g} "
              f"{st['mean']:>10.3g} {st['std']:>10.3g}")

    fig = slb.plot_analysis(an, film_thickness=args.h, fit_max_delta=args.cap)
    out = os.path.join(args.outdir, "slb_analysis.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
