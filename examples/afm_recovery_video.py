"""Animate whether a viscoelastic surface recovers between consecutive pokes.

Same material everywhere. The tip does a full approach-retract ("down then up")
at each pixel, steps a few nm laterally, and pokes again -- as an AFM force map
does. On a viscoelastic sample the surface recovers over a relaxation time `tau`;
if the next poke comes too soon (fast scan) or too close (small lateral step), it
lands on material still dented by the previous one, and the measurements overlap.

Top panel: the surface height profile along the scan line as the tip steps
across, so you can watch residual dents persist (or recover) between pokes. The
readout is the residual depression at the current site when its poke begins, as a
fraction of a fresh indentation.

Bottom panel: the recovery-limit map -- residual overlap over lateral pitch ×
ramp frequency -- with this scan's operating point marked. It shows the limits:
above/right (large step, or slow) the next poke sees fresh material; below/left it
does not. The spatial footprint is a real FEM solve; the temporal recovery is a
single-`tau` viscoelastic model (see docs/design.md section 8.4).

Usage
-----
    python examples/afm_recovery_video.py [--pitch 5] [--freq 25] [--tau 0.5]
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

from nanoindent import SimulationConfig
from nanoindent import recovery as rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitch", type=float, default=5.0, help="lateral step (nm)")
    ap.add_argument("--freq", type=float, default=25.0, help="ramp frequency (Hz)")
    ap.add_argument("--tau", type=float, default=0.5, help="recovery time (s)")
    ap.add_argument("--viscous-fraction", type=float, default=1.0,
                    help="beta = 1 - E_inf/E0 (0 = elastic, 1 = fully viscoelastic)")
    ap.add_argument("--n", type=int, default=8, help="number of insertions")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cfg = SimulationConfig(R=20.0, E=5.0, nu=0.5, h=6.0, L=120.0, delta_max=1.2,
                           formulation="mixed", mesh_size_contact=0.3,
                           mesh_size_edge=8.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fp = rec.surface_footprint(cfg)
    a = fp.contact_radius
    print(f"FEM footprint: contact radius a = {a:.2f} nm; "
          f"pitch {args.pitch} nm = {args.pitch / a:.2f} a")

    raster = rec.raster_surface(fp, tau=args.tau, pitch_nm=args.pitch,
                                ramp_freq_hz=args.freq, n_insertions=args.n,
                                viscous_fraction=args.viscous_fraction,
                                steps_per_cycle=48)
    resid_starts = raster.residual_at_starts()
    print(f"residual at each new site when its poke starts (fraction of fresh): "
          f"{np.round(resid_starts * 100, 1)}")

    # Recovery-limit map over pitch x frequency.
    pitches = np.linspace(1.0, 4.0 * a, 60)
    freqs = np.logspace(-0.3, 2.0, 60)
    limit = rec.recovery_limit_map(fp, args.tau, pitches, freqs,
                                   viscous_fraction=args.viscous_fraction)

    # ---- figure ----
    fig, (axS, axL) = plt.subplots(2, 1, figsize=(9, 7.5),
                                   gridspec_kw={"height_ratios": [1, 1.15]},
                                   constrained_layout=True)

    # top: surface profile
    depth = args.tau  # placeholder to avoid lints
    ymin = -1.35 * raster.delta_max
    ymax = 0.9 * raster.delta_max
    (surf_line,) = axS.plot([], [], color="#0072B2", lw=2)
    site_marks = axS.plot(raster.site_x, np.zeros_like(raster.site_x), "v",
                          color="0.6", ms=6)[0]
    (tip_marker,) = axS.plot([], [], "v", color="crimson", ms=13, mfc="crimson")
    resid_text = axS.text(0.02, 0.06, "", transform=axS.transAxes, fontsize=10,
                          va="bottom")
    axS.axhline(0, color="0.8", lw=0.8)
    axS.set_xlim(raster.x[0], raster.x[-1])
    axS.set_ylim(ymin, ymax)
    axS.set_xlabel("lateral position (nm)")
    axS.set_ylabel("surface height (nm)")
    axS.set_title(f"surface recovery — pitch {args.pitch:g} nm ({args.pitch/a:.1f} a), "
                  f"{args.freq:g} Hz,  β = {args.viscous_fraction:g}, τ = {args.tau:g} s")

    # bottom: limit map
    im = axL.pcolormesh(freqs, pitches, 100 * limit, cmap="inferno",
                        shading="auto", vmin=0, vmax=60)
    axL.contour(freqs, pitches, 100 * limit, levels=[10], colors="cyan",
                linewidths=1.6)
    axL.plot([args.freq], [args.pitch], "o", mfc="white", mec="black",
             ms=11, mew=2)
    axL.set_xscale("log")
    axL.axhline(1.5 * a, color="white", ls=":", lw=1)
    axL.text(freqs[1], 1.5 * a + 0.4, "≈1.5 a (footprint edge)", color="white",
             fontsize=8)
    axL.set_xlabel("ramp frequency (Hz)")
    axL.set_ylabel("lateral pitch (nm)")
    axL.set_title("recovery-limit map — residual overlap (%); cyan = 10% boundary")
    fig.colorbar(im, ax=axL, fraction=0.046, label="residual at next poke (%)")

    n_t = len(raster.t)
    # subsample time for a reasonable frame count
    stride = max(1, n_t // (args.n * 24))
    frames = list(range(0, n_t, stride))

    def update(fi):
        it = frames[fi]
        z = -raster.A[it]
        surf_line.set_data(raster.x, z)
        # tip: at its lateral pos, apex at -depth if in contact, else lifted
        tx = raster.tip_x[it]
        if raster.in_contact[it]:
            tipz = -raster.tip_delta[it]
        else:
            tipz = 0.45 * raster.delta_max
        tip_marker.set_data([tx], [tipz])
        # which site are we at, and its residual-at-start
        i = min(int(raster.t[it] * args.freq), args.n - 1)
        resid_text.set_text(
            f"poke {i+1}/{args.n} at x={raster.site_x[i]:.0f} nm   "
            f"residual when it started: {resid_starts[i]*100:.0f}% of fresh")
        return surf_line, tip_marker, resid_text

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=1000 / args.fps, blit=False)

    stem = os.path.join(args.outdir, "afm_recovery_video")
    want_gif = args.gif or not animation.FFMpegWriter.isAvailable()
    if want_gif:
        path = stem + ".gif"; writer = animation.PillowWriter(fps=args.fps)
    else:
        path = stem + ".mp4"; writer = animation.FFMpegWriter(fps=args.fps, bitrate=2800)
    print(f"rendering {len(frames)} frames -> {path}")
    anim.save(path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
