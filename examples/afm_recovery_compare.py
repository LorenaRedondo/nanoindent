"""Recovery between pokes: a viscoelastic material vs a more elastic one.

Same tip, same scan (lateral pitch, ramp frequency), same FEM surface footprint.
Only the material differs -- its viscous fraction β = 1 − E_inf/E0 and recovery
time τ. The viscoelastic material (large β, slow τ, like a soft cell) does not
spring back between pokes and leaves a persistent trough; the more elastic
material (small β, fast τ, like a stiffer gel or a bilayer) recovers cleanly, so
each poke sees fresh material -- the regime you want for an accurate force map.

Top row: surface profiles side by side as the tip steps across.
Bottom: residual-at-next-poke vs lateral pitch for both materials, with the scan
pitch marked -- how much elastic recovery buys you.

Usage
-----
    python examples/afm_recovery_compare.py [--pitch 5] [--freq 25]
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


# (label, viscous fraction beta, recovery time tau s, colour)
VISCOELASTIC = ("viscoelastic (soft cell): β=0.85, τ=0.5 s", 0.85, 0.5, "#D55E00")
ELASTIC = ("more elastic (gel/bilayer): β=0.2, τ=8 ms", 0.20, 0.008, "#0072B2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitch", type=float, default=5.0, help="lateral step (nm)")
    ap.add_argument("--freq", type=float, default=25.0, help="ramp frequency (Hz)")
    ap.add_argument("--n", type=int, default=8)
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

    def run(beta, tau):
        return rec.raster_surface(fp, tau=tau, pitch_nm=args.pitch,
                                  ramp_freq_hz=args.freq, n_insertions=args.n,
                                  viscous_fraction=beta, steps_per_cycle=48)

    r_ve = run(VISCOELASTIC[1], VISCOELASTIC[2])
    r_el = run(ELASTIC[1], ELASTIC[2])
    res_ve = r_ve.residual_at_starts()
    res_el = r_el.residual_at_starts()
    print(f"steady residual at next poke — viscoelastic {res_ve[-1]*100:.0f}%  "
          f"vs elastic {res_el[-1]*100:.0f}%")

    # residual vs pitch curves for both materials (at this frequency)
    pitches = np.linspace(1.0, 4.0 * a, 80)
    rf_ve = [rec.residual_fraction(fp, VISCOELASTIC[2], p, args.freq,
                                   viscous_fraction=VISCOELASTIC[1]) for p in pitches]
    rf_el = [rec.residual_fraction(fp, ELASTIC[2], p, args.freq,
                                   viscous_fraction=ELASTIC[1]) for p in pitches]

    fig = plt.figure(figsize=(11, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.85])
    axV = fig.add_subplot(gs[0, 0])
    axE = fig.add_subplot(gs[0, 1], sharey=axV)
    axR = fig.add_subplot(gs[1, :])

    ymin, ymax = -1.35 * r_ve.delta_max, 0.9 * r_ve.delta_max
    lines = {}
    for ax, raster, meta in ((axV, r_ve, VISCOELASTIC), (axE, r_el, ELASTIC)):
        ax.axhline(0, color="0.85", lw=0.8)
        ax.plot(raster.site_x, np.zeros_like(raster.site_x), "v", color="0.7", ms=5)
        (ln,) = ax.plot([], [], color=meta[3], lw=2)
        (tip,) = ax.plot([], [], "v", color="black", ms=11)
        txt = ax.text(0.03, 0.05, "", transform=ax.transAxes, fontsize=9, va="bottom")
        lines[id(ax)] = (ln, tip, txt, raster)
        ax.set_xlim(raster.x[0], raster.x[-1]); ax.set_ylim(ymin, ymax)
        ax.set_xlabel("lateral position (nm)")
        ax.set_title(meta[0], fontsize=10)
    axV.set_ylabel("surface height (nm)")

    # residual-vs-pitch panel
    axR.plot(pitches / a, 100 * np.array(rf_ve), color=VISCOELASTIC[3], lw=2,
             label="viscoelastic")
    axR.plot(pitches / a, 100 * np.array(rf_el), color=ELASTIC[3], lw=2,
             label="more elastic")
    axR.axvline(args.pitch / a, color="0.5", ls="--", lw=1)
    axR.text(args.pitch / a + 0.05, axR.get_ylim()[1] * 0.8,
             f"scan pitch = {args.pitch/a:.1f} a", fontsize=8, color="0.3")
    axR.axhline(10, color="0.7", ls=":", lw=1)
    axR.set_xlabel("lateral pitch (contact radii)")
    axR.set_ylabel("residual at next poke (%)")
    axR.set_title(f"residual vs pitch at {args.freq:g} Hz — elastic recovery clears the overlap")
    axR.legend(frameon=False)

    n_t = len(r_ve.t)
    stride = max(1, n_t // (args.n * 22))
    frames = list(range(0, n_t, stride))

    def update(fi):
        it = frames[fi]
        arts = []
        for ax, res in ((axV, res_ve), (axE, res_el)):
            ln, tip, txt, raster = lines[id(ax)]
            ln.set_data(raster.x, -raster.A[it])
            tx = raster.tip_x[it]
            tz = -raster.tip_delta[it] if raster.in_contact[it] else 0.45 * raster.delta_max
            tip.set_data([tx], [tz])
            i = min(int(raster.t[it] * args.freq), args.n - 1)
            txt.set_text(f"residual at poke {i+1}: {res[i]*100:.0f}%")
            arts += [ln, tip, txt]
        return arts

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=1000 / args.fps, blit=False)

    stem = os.path.join(args.outdir, "afm_recovery_compare")
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
