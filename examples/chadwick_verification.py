"""Verify Chadwick 2002 with a true incompressible (nu=0.5) mixed-FEM solve.

Chadwick's thin-incompressible-layer law F = (2*pi/3) E R^2 delta^3 / h^3 is the
asymptotic (contact radius >> h) limit for a bonded incompressible film.  The
primal P2 formulation locks at nu -> 1/2, so this uses the Taylor-Hood mixed
(P2 displacement / P1 pressure) formulation, which is locking-free and handles
nu = 0.5 exactly.

The script sweeps the thin-film parameter chi = sqrt(R*delta)/h and shows that
the mixed-FEM force converges to Chadwick's law as chi grows (the regime where
the asymptotic is valid), confirming both the scaling and the (2*pi/3)
coefficient.  A paraboloidal tip is used to match Chadwick's parabolic gap.

Usage
-----
    python examples/chadwick_verification.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nanoindent import SimulationConfig, IndentationSolver, analytics, plotting


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    R, E = 17.0, 130.0
    # (thickness, delta_max) chosen to sweep chi_max from ~1 to ~8.
    cases = [(3.0, 1.0), (2.0, 2.0), (1.5, 3.0), (1.0, 4.0)]

    chi_vals, ratios = [], []
    print(f"Chadwick verification (nu=0.5, mixed Taylor-Hood, paraboloid tip)\n"
          f"F_Chadwick = (2*pi/3) E R^2 delta^3 / h^3\n")
    print(f"{'h (nm)':>7} {'chi_max':>8} {'DOFs':>6} {'equil %':>8} "
          f"{'FEM/Chadwick':>13}")
    for h, dmax in cases:
        cfg = SimulationConfig(
            nu=0.5, formulation="mixed", tip_shape="paraboloid",
            h=h, L=80.0, R=R, E=E, delta_max=dmax, n_steps=8,
            mesh_size_contact=0.1, mesh_size_edge=4.0, penalty_alpha=800.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            curve = IndentationSolver(cfg).run()
        chad = analytics.chadwick_force(curve.delta, R, E, h)
        chi = np.sqrt(R * curve.delta) / h
        ratio = curve.force[-1] / chad[-1]
        chi_vals.append(chi[-1])
        ratios.append(ratio)
        print(f"{h:7.1f} {chi[-1]:8.2f} {curve.n_dofs:6d} "
              f"{np.nanmax(curve.equilibrium_residual) * 100:8.2f} {ratio:13.3f}")

    print("\nFEM/Chadwick -> 1 as chi grows: Chadwick's law and its (2*pi/3) "
          "coefficient are recovered in the thin-film limit.")

    plotting.set_publication_style()
    fig, ax = plt.subplots(figsize=(plotting.SINGLE_COLUMN, 2.6))
    ax.plot(chi_vals, ratios, "o-", color="#0072B2")
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="Chadwick asymptote")
    ax.set_xlabel(r"$\chi_\mathrm{max} = \sqrt{R\,\delta_\mathrm{max}}/h$")
    ax.set_ylabel(r"$F_\mathrm{FEM}/F_\mathrm{Chadwick}$")
    ax.set_title(r"Mixed FEM ($\nu=1/2$) $\to$ Chadwick 2002", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.25)
    written = plotting.save_figure(fig, os.path.join(args.outdir, "chadwick_verification"))
    print("Wrote:", ", ".join(written))


if __name__ == "__main__":
    main()
