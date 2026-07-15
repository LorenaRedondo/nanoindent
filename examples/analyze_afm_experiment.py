"""Analyze a raw AFM force curve: load -> calibrate -> fit modulus.

Demonstrates the ingestion pipeline. A synthetic raw curve (known E = 130 MPa)
is written to CSV to stand in for an instrument export, then loaded, calibrated
to force-indentation, and fit with both the naive Hertz model and the bonded
Dimitriadis correction — showing the bottom-effect artefact on real-shaped data.

For actual experiments, replace the CSV path with your instrument file; the
loader dispatches by extension:
    JPK   .jpk-force / .jpk
    Bruker .spm / .pfc
    Asylum .ibw            (needs `pip install igor2`)
    generic .csv / .txt / .tsv

Usage
-----
    python examples/analyze_afm_experiment.py [--file PATH --k 0.2 --R 17 --h 7 --nu 0.3]
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np

from nanoindent import (
    SimulationConfig, synthetic_raw_curve, load_afm, analyze_experiment,
    fit_modulus,
)


def _write_demo_csv(path: str) -> float:
    cfg = SimulationConfig(R=17.0, E=130.0, nu=0.3, h=7.0, delta_max=1.0)
    raw = synthetic_raw_curve(cfg, spring_constant=0.2, true_E=130.0,
                              source_model="dimitriadis", baseline_nm=3.0,
                              z_contact=50.0, noise_nm=0.02, seed=3)
    np.savetxt(path, np.column_stack([raw.z, raw.deflection]), delimiter=",",
               header="z_nm,deflection_nm", comments="")
    return 0.2   # spring constant used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="raw AFM file (default: demo CSV)")
    ap.add_argument("--k", type=float, default=0.2, help="spring constant (N/m)")
    ap.add_argument("--inv-ols", type=float, default=None, help="nm/V, if deflection in volts")
    ap.add_argument("--R", type=float, default=17.0)
    ap.add_argument("--h", type=float, default=7.0)
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    path = args.file
    if path is None:
        path = os.path.join(args.outdir, "demo_raw_curve.csv")
        args.k = _write_demo_csv(path)
        print(f"(wrote synthetic raw curve to {path}; ground truth E = 130 MPa)\n")

    curve = load_afm(path, spring_constant=args.k, inv_ols=args.inv_ols) \
        if path.lower().endswith((".csv", ".txt", ".tsv")) else load_afm(path)
    if curve.spring_constant is None:
        curve.spring_constant = args.k

    print(f"Loaded {curve.fmt} curve: {len(curve.z)} points, "
          f"k = {curve.spring_constant} N/m")

    delta, force = curve.to_force_indentation(inv_ols=args.inv_ols)
    print(f"Calibrated to force-indentation: "
          f"delta in [{delta.min():.2f}, {delta.max():.2f}] nm, "
          f"F up to {force.max():.0f} pN\n")

    print(f"{'model':>20} {'E (MPa)':>9} {'contact (nm)':>13} {'RMSE (pN)':>10}")
    for model in ("hertz", "dimitriadis"):
        fit = fit_modulus(delta, force, model=model, R=args.R, nu=args.nu,
                          h=args.h, fit_contact_point=True)
        print(f"{model:>20} {fit.E_fit:9.1f} {fit.contact_point:13.3f} "
              f"{fit.rmse_pN:10.1f}")

    print("\nThe Hertz fit over-estimates E on a thin bonded film (bottom "
          "effect); the Dimitriadis fit recovers the true modulus.")


if __name__ == "__main__":
    main()
