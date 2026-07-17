"""Animate the real FEM stress field for several distinct insertions.

Renders the actual finite-element solution -- the von Mises stress on the
axisymmetric (r, z) cross-section -- as a rigid tip is driven into the film, for
a sequence of *genuinely different* insertions: each is its own FEM solve with its
own modulus and indentation depth, so the field differs in both magnitude and
extent from one insertion to the next (a soft shallow poke, a stiff deep push, an
interface, ...).

Left panel: the von Mises field (mirrored about the axis) with the tip profile,
on a shared non-linear colour scale so even the soft insertions stay visible.
Right panel: the force-indentation curves accumulating -- completed insertions
stay as light traces, the current one builds in colour -- so several insertions
are visible at once.

Usage
-----
    python examples/fem_insertions_video.py [--fps 10] [--outdir DIR] [--gif]
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import PowerNorm

from nanoindent import SimulationConfig, IndentationSolver


# Each insertion is a distinct solve: (label, E in MPa, indentation depth in nm).
# Varying E *and* depth makes the fields genuinely different, not scalings.
INSERTIONS = [
    ("soft, shallow", 3.0, 0.8),
    ("stiff, deep", 25.0, 1.2),
    ("medium", 10.0, 1.0),
    ("very soft, deep", 1.5, 1.2),
    ("stiff, shallow", 25.0, 0.6),
]

CMAP = "turbo"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=12)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Run a genuine FEM solve per insertion, keeping the per-step field history.
    solves = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for label, E, dmax in INSERTIONS:
            cfg = SimulationConfig(
                R=20.0, E=E, nu=0.5, h=6.0, L=100.0, delta_max=dmax,
                n_steps=args.n_steps, formulation="mixed",
                mesh_size_contact=0.3, mesh_size_edge=12.0,
            )
            print(f"  solving: {label:16s} E={E:5.1f} MPa  delta_max={dmax} nm")
            solver = IndentationSolver(cfg)
            curve = solver.run(store_history=True)
            steps = []
            for st in solver.history:
                vm = solver.nodal_field("von_mises", u=st["u"], p=st["p"])
                steps.append({"delta": st["delta"], "vm": vm})
            solves.append({
                "label": label, "E": E, "dmax": dmax, "solver": solver,
                "curve": curve, "steps": steps,
                "mesh": solver.domain.mesh,
            })

    # Shared scales across all insertions.
    vmax = max(float(np.nanmax(s["vm"])) for sv in solves for s in sv["steps"])
    force_max = max(float(sv["curve"].force[-1]) for sv in solves)
    delta_max_all = max(sv["dmax"] for sv in solves)
    a = np.sqrt(20.0 * delta_max_all)
    xlim = (-5 * a, 5 * a)
    ylim = (0, 6.0 * 1.25)

    fig, (axF, axC) = plt.subplots(1, 2, figsize=(12, 4.0),
                                   gridspec_kw={"width_ratios": [1.5, 1]},
                                   constrained_layout=True)
    # Non-linear (sqrt) colour scale so low-stress soft insertions stay visible
    # under a scale that also spans the stiff peak.
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
    levels = np.linspace(0, vmax, 48)
    sm = ScalarMappable(norm=norm, cmap=CMAP)
    fig.colorbar(sm, ax=axF, fraction=0.046, pad=0.02,
                 label="von Mises stress (MPa)")

    n_steps = len(solves[0]["steps"])
    total = len(solves) * n_steps

    def draw(frame):
        ins, k = divmod(frame, n_steps)
        sv = solves[ins]
        mesh = sv["mesh"]
        r0, z = mesh.p[0], mesh.p[1]
        tris = mesh.t.T
        st = sv["steps"][k]
        delta = st["delta"]

        axF.clear()
        for sgn in (1, -1):
            tri = mtri.Triangulation(sgn * r0, z, tris)
            axF.tricontourf(tri, st["vm"], levels=levels, norm=norm, cmap=CMAP,
                            extend="max")
        rr = np.linspace(xlim[0], xlim[1], 400)
        ztip = sv["solver"].tip.surface_z(np.abs(rr), delta)
        fin = np.isfinite(ztip)
        axF.plot(rr[fin], ztip[fin], color="black", lw=1.4)
        axF.set_xlim(*xlim); axF.set_ylim(*ylim); axF.set_aspect("equal")
        axF.set_xlabel("r (nm)"); axF.set_ylabel("z (nm)")
        axF.set_title(f"insertion {ins + 1}/{len(solves)} — {sv['label']}\n"
                      f"E = {sv['E']:g} MPa,  depth {sv['dmax']:g} nm  (real FEM field)")

        # Force panel: completed insertions stay faint; current one builds.
        axC.clear()
        for j, other in enumerate(solves):
            c = other["curve"]
            if j < ins:
                axC.plot(c.delta, c.force, "-", color="0.75", lw=1.2)
            elif j == ins:
                axC.plot(c.delta[:k + 1], c.force[:k + 1], "-",
                         color="#D55E00", lw=2.6)
                axC.plot([delta], [c.force[k]], "o", color="#D55E00", ms=7)
        axC.set_xlim(0, delta_max_all * 1.02)
        axC.set_ylim(0, force_max * 1.05)
        axC.set_xlabel("indentation delta (nm)")
        axC.set_ylabel("force (pN)")
        axC.set_title("force-indentation (insertions accumulate)")
        axC.text(0.03, 0.95, f"peak so far: {sv['curve'].force[k]:.0f} pN",
                 transform=axC.transAxes, va="top", fontsize=9, color="#D55E00")
        return []

    anim = animation.FuncAnimation(fig, draw, frames=total,
                                   interval=1000 / args.fps, blit=False)

    stem = os.path.join(args.outdir, "fem_insertions_video")
    want_gif = args.gif or not animation.FFMpegWriter.isAvailable()
    if want_gif:
        path = stem + ".gif"; writer = animation.PillowWriter(fps=args.fps)
    else:
        path = stem + ".mp4"; writer = animation.FFMpegWriter(fps=args.fps, bitrate=3000)
    print(f"rendering {total} frames ({len(solves)} insertions) -> {path}")
    anim.save(path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
