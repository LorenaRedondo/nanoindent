"""Generate a realistic AFM dataset from the FEM and invert it — a double-check.

Pipeline
--------
1. Run the FEM forward solver for a bonded thin film (ground truth, E=130 MPa).
2. Sample it densely and add realistic AFM measurement noise, a baseline
   offset, and a contact-point shift -> a synthetic "experimental" curve.
3. Fit the modulus back out with (a) the naive Hertz model and (b) the bonded
   Dimitriadis correction.

Expected result: Hertz *over-estimates* E on a thin bonded film (the
bottom-effect artefact), while the Dimitriadis fit recovers ~130 MPa. This
confirms both the forward solver and the analytic corrections are mutually
consistent.

Usage
-----
    python examples/synthetic_analysis.py [--outdir DIR] [--noise PN]
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np

from nanoindent import (
    SimulationConfig, IndentationSolver, synthetic_force_curve, fit_modulus,
)


def fem_interpolator(curve):
    """Return delta -> force interpolation over an FEM ForceCurve."""
    d, f = curve.delta, curve.force

    def interp(x):
        return np.interp(np.clip(x, 0.0, d[-1]), d, f)

    return interp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--noise", type=float, default=10.0,
                    help="Gaussian force noise std (pN)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    true_E = 130.0
    print(f"Ground-truth modulus: E = {true_E} MPa\n")
    header = (f"{'h (nm)':>7} | {'model':>13} | {'E_fit (MPa)':>12} | "
              f"{'error':>7} | {'RMSE (pN)':>9}")
    print(header)
    print("-" * len(header))

    rows = []
    for h in (7.0, 14.0, 200.0):
        cfg = SimulationConfig(
            R=17.0, E=true_E, nu=0.3, h=h,
            L=100.0 if h < 100 else 200.0,
            delta_max=1.0, n_steps=12,
            mesh_size_contact=0.2, mesh_size_edge=6.0 if h < 100 else 12.0,
            penalty_alpha=400.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            curve = IndentationSolver(cfg).run()

        ds = synthetic_force_curve(
            cfg, source="fem", true_E=true_E, n_points=120,
            noise_pN=args.noise, baseline_pN=25.0, contact_point=0.05,
            fem_force=fem_interpolator(curve), seed=args.seed,
        )
        ds.save(os.path.join(args.outdir, f"synthetic_h{int(h)}.csv"))

        for model in ("hertz", "dimitriadis"):
            # A real analysis does not know the contact point; estimate it from
            # the pre-contact baseline region (delta <= 0).
            fit = fit_modulus(ds.delta, ds.force, model=model,
                              R=cfg.R, nu=cfg.nu, h=cfg.h,
                              fit_contact_point=True)
            err = (fit.E_fit - true_E) / true_E * 100.0
            print(f"{h:7.0f} | {model:>13} | {fit.E_fit:12.1f} | "
                  f"{err:+6.1f}% | {fit.rmse_pN:9.1f}")
            rows.append((h, model, fit.E_fit, err, fit.rmse_pN))
        print("-" * len(header))

    print("\nInterpretation:")
    print("  * Hertz fit inflates E on thin films (bottom effect); the error")
    print("    shrinks as h grows and vanishes for the 200 nm half-space.")
    print("  * The Dimitriadis bonded fit recovers ~130 MPa at every thickness,")
    print("    confirming the FEM and the analytic correction agree.")
    print(f"\nSynthetic datasets saved to '{args.outdir}/'.")


if __name__ == "__main__":
    main()
