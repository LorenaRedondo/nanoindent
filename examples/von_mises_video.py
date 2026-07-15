"""Animate the von Mises stress field as the AFM tip indents the sample.

Runs the indentation with per-step history stored, then renders the field
evolution over the axisymmetric cross-section to a video (MP4 if an ffmpeg
writer is available -- e.g. `pip install imageio-ffmpeg` -- otherwise an
animated GIF).

Usage
-----
    python examples/von_mises_video.py [--outdir DIR] [--field von_mises]
                                       [--n-steps 40] [--fps 12]
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
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--field", default="von_mises",
                    choices=["von_mises", "sigma_zz", "pressure",
                             "displacement", "uz", "ur"])
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--gif", action="store_true", help="force GIF output")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # MSNL-BIO-DC case: paraboloid on a 6.2 nm incompressible film.
    cfg = SimulationConfig(
        tip_shape="paraboloid", R=20.0, E=50.0, nu=0.5, formulation="mixed",
        h=6.2, L=200.0, delta_max=2.0, n_steps=args.n_steps,
        mesh_size_contact=0.02, mesh_size_edge=4.0, penalty_alpha=600.0,
    )
    print(f"Running {args.n_steps} load steps (history stored)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = IndentationSolver(cfg)
        solver.run(store_history=True)

    ext = "gif" if args.gif else "mp4"
    path = os.path.join(args.outdir, f"{args.field}_indentation.{ext}")
    out = plotting.animate_field(
        solver, field=args.field, path=path, mirror=True, cmap="jet",
        fps=args.fps, xlim=(-35, 35), ylim=(0, 8.2), dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
