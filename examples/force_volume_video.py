"""Animate a force map being acquired: the tip inserting at pixel after pixel.

Rasters a rigid spherical tip across a two-phase sample (stiff domains in a soft
matrix), performing one indentation -- an "insertion" -- at each pixel. Three
panels, animated:

  * left   -- the tip pushing into the sample cross-section at the current pixel,
              the surface dimpling to the current depth, shaded by local stiffness;
  * middle -- the force-indentation curve building up for the current insertion
              (soft pixels give a shallow curve, stiff pixels a steep one);
  * right  -- the modulus map filling in, pixel by pixel, as insertions complete.

All insertions are exact scalings of a single FEM solve (force is proportional to
modulus), so the whole video costs one solve.

Usage
-----
    python examples/force_volume_video.py [--n 8] [--frames-per-insert 5]
                                          [--fps 15] [--outdir DIR] [--gif]
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle

from nanoindent import SimulationConfig
from nanoindent import forcemap as fm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="grid size (n x n insertions)")
    ap.add_argument("--frames-per-insert", type=int, default=5)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--E-soft", type=float, default=2.0)
    ap.add_argument("--E-stiff", type=float, default=20.0)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--gif", action="store_true", help="force GIF output")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    R = 20.0
    cfg = SimulationConfig(R=R, nu=0.5, h=6.0, L=100.0, delta_max=1.5,
                           formulation="mixed", mesh_size_contact=0.35,
                           mesh_size_edge=12.0)
    spec = fm.ForceVolumeSpec(args.n, args.n, pitch_nm=8.0, ramp_freq_hz=2.0,
                              z_amplitude_nm=1.2, points_per_ramp=64)
    print(spec.summary())

    E_true = fm.two_phase_field(args.n, args.n, E_soft=args.E_soft,
                                E_stiff=args.E_stiff, stiff_fraction=0.35, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kernel = fm.geometric_kernel(cfg)

    # Depth schedule for one insertion (approach), and the force along it.
    K = args.frames_per_insert
    delta_ramp = np.linspace(0.0, spec.z_amplitude_nm, K)
    phi_ramp = kernel.phi_of(delta_ramp)                 # pN / MPa
    delta_dense = np.linspace(0.0, spec.z_amplitude_nm, 60)
    phi_dense = kernel.phi_of(delta_dense)

    n_pix = args.n * args.n
    F_stiff_max = args.E_stiff * phi_dense[-1]           # y-limit for force panel

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=args.E_stiff * 1.15)

    fig, (axT, axF, axM) = plt.subplots(1, 3, figsize=(13.5, 4.4))

    # --- static setup: tip panel ---
    a_max = float(np.sqrt(R * spec.z_amplitude_nm))
    xspan = 3.5 * a_max
    axT.set_xlim(-xspan, xspan)
    axT.set_ylim(-2.2 * spec.z_amplitude_nm, R * 0.9)
    axT.set_aspect("equal")
    axT.set_title("insertion (tip into sample)")
    axT.set_xlabel("lateral position (nm)")
    axT.set_ylabel("height (nm)")

    # --- static setup: force panel ---
    axF.plot(delta_dense, args.E_soft * phi_dense, "--", color="0.7", lw=1,
             label=f"soft ({args.E_soft:g} MPa)")
    axF.plot(delta_dense, args.E_stiff * phi_dense, "--", color="0.4", lw=1,
             label=f"stiff ({args.E_stiff:g} MPa)")
    (curve_line,) = axF.plot([], [], "-", color="#D55E00", lw=2.2)
    axF.set_xlim(0, spec.z_amplitude_nm)
    axF.set_ylim(0, F_stiff_max * 1.05)
    axF.set_xlabel("indentation delta (nm)")
    axF.set_ylabel("force (pN)")
    axF.set_title("force-indentation (current insertion)")
    axF.legend(fontsize=8, frameon=False, loc="upper left")

    # --- static setup: map panel ---
    filled = np.full((args.n, args.n), np.nan)
    map_im = axM.imshow(filled, origin="lower", cmap=cmap, norm=norm,
                        interpolation="nearest",
                        extent=[0, args.n, 0, args.n])
    axM.set_title("modulus map filling in")
    axM.set_xlabel("column"); axM.set_ylabel("row")
    fig.colorbar(map_im, ax=axM, fraction=0.046, label="E (MPa)")
    (cursor,) = axM.plot([], [], "s", mfc="none", mec="red", ms=10, mew=2)

    total_frames = n_pix * K
    surf_artists = []

    def draw_insertion(pixel, k):
        # clear previous tip/surface artists
        for art in surf_artists:
            art.remove()
        surf_artists.clear()

        i, j = divmod(pixel, args.n)
        E = float(E_true[i, j])
        depth = delta_ramp[k]
        colour = cmap(norm(E))

        # sample block (shaded by local stiffness), with a dimple under the tip
        x = np.linspace(-xspan, xspan, 220)
        a = np.sqrt(R * depth) if depth > 0 else 0.0
        surface = np.zeros_like(x)
        if depth > 0 and a > 0:
            in_dimple = np.abs(x) < a
            # spherical-cap dimple of central depth `depth`
            surface[in_dimple] = -(depth - (R - np.sqrt(
                np.clip(R ** 2 - x[in_dimple] ** 2, 0, None))))
            surface[in_dimple] = np.minimum(surface[in_dimple], 0.0)
        bottom = -2.2 * spec.z_amplitude_nm
        fillpoly = axT.fill_between(x, surface, bottom, color=colour, alpha=0.85,
                                    linewidth=0)
        (surfline,) = axT.plot(x, surface, color="0.25", lw=1.2)

        # rigid spherical tip: centre so the apex sits at z = depth below surface
        tip = Circle((0.0, R - depth), R, facecolor="0.85", edgecolor="0.3",
                     lw=1.2, zorder=5)
        axT.add_patch(tip)
        label = axT.text(-xspan * 0.95, R * 0.6,
                         f"pixel {pixel + 1}/{n_pix}\nE = {E:.1f} MPa",
                         fontsize=9, va="top")
        surf_artists.extend([fillpoly, surfline, tip, label])

        # force curve so far
        curve_line.set_data(delta_ramp[:k + 1], E * phi_ramp[:k + 1])
        curve_line.set_color(cmap(norm(E)))

        # map: fill completed pixels, highlight current
        filled[i, j] = E
        map_im.set_data(filled)
        cursor.set_data([j + 0.5], [i + 0.5])

    def update(frame):
        pixel, k = divmod(frame, K)
        draw_insertion(pixel, k)
        return surf_artists

    anim = animation.FuncAnimation(fig, update, frames=total_frames,
                                   interval=1000 / args.fps, blit=False)
    fig.tight_layout()

    stem = os.path.join(args.outdir, "force_volume_video")
    want_gif = args.gif or not animation.FFMpegWriter.isAvailable()
    if want_gif:
        path = stem + ".gif"
        writer = animation.PillowWriter(fps=args.fps)
    else:
        path = stem + ".mp4"
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=2600)
    print(f"rendering {total_frames} frames ({n_pix} insertions) -> {path}")
    anim.save(path, writer=writer, dpi=110)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
