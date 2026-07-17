"""Simulate an AFM force map over a heterogeneous sample, and recover it.

Rasters the tip over an ``n x n`` grid at spatial pitch X (nm), ramping a force
curve at frequency Y (Hz) at each pixel, on a two-phase sample (stiff domains in
a soft matrix -- e.g. liquid-ordered lipid domains). One FEM solve builds the
geometric kernel; every pixel is an exact scaling of it by that pixel's modulus
(force is strictly proportional to E). The synthetic curves are then run back
through the ordinary modulus-fit analysis, and the recovered modulus map is
compared to the ground truth.

Two recovery models are shown side by side:
  * ``kernel``      -- fit against the FEM kernel itself: recovers E (machinery check);
  * ``hertz``       -- fit with half-space Hertz: over-estimates on a thin bonded
                       film (the bottom-effect artefact this library exists to quantify).

Usage
-----
    python examples/force_volume.py [--n 24] [--pitch 8] [--freq 2] [--outdir DIR]
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nanoindent import SimulationConfig
from nanoindent import forcemap as fm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="grid size (n x n)")
    ap.add_argument("--pitch", type=float, default=8.0, help="pixel pitch X (nm)")
    ap.add_argument("--freq", type=float, default=2.0, help="ramp frequency Y (Hz)")
    ap.add_argument("--E-soft", type=float, default=2.0, help="soft-phase modulus (MPa)")
    ap.add_argument("--E-stiff", type=float, default=20.0, help="stiff-phase modulus (MPa)")
    ap.add_argument("--noise", type=float, default=0.02, help="deflection noise (nm)")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cfg = SimulationConfig(
        R=20.0, nu=0.5, h=6.0, L=100.0, delta_max=1.5, formulation="mixed",
        mesh_size_contact=0.35, mesh_size_edge=12.0,
    )
    spec = fm.ForceVolumeSpec(
        n_rows=args.n, n_cols=args.n, pitch_nm=args.pitch,
        ramp_freq_hz=args.freq, z_amplitude_nm=1.2, points_per_ramp=96,
    )
    print(spec.summary())

    E_true = fm.two_phase_field(args.n, args.n, E_soft=args.E_soft,
                                E_stiff=args.E_stiff, stiff_fraction=0.35, seed=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vol = fm.simulate_force_volume(spec, E_true, cfg, spring_constant=0.08,
                                       noise_nm=args.noise, seed=3)
        print(f"generated {len(vol)} curves from ONE FEM solve "
              f"(kernel: {len(vol.kernel.delta)} pts to {vol.kernel.delta_max:.2f} nm)")

        E_kernel = vol.recover_modulus_map(model="kernel", fit_max_delta=1.2)
        E_hertz = vol.recover_modulus_map(model="hertz", fit_max_delta=1.2)

    def report(name, Er):
        ok = np.isfinite(Er)
        rel = np.abs(Er[ok] - E_true[ok]) / E_true[ok]
        print(f"  {name:8s}: median rel err {100 * np.median(rel):5.1f}%   "
              f"soft {np.nanmedian(Er[E_true < 10]):5.2f}/{args.E_soft}   "
              f"stiff {np.nanmedian(Er[E_true >= 10]):5.2f}/{args.E_stiff}")

    print("recovered modulus vs ground truth:")
    report("kernel", E_kernel)
    report("hertz", E_hertz)

    # Figure: ground truth, kernel recovery, Hertz recovery (bottom-effect bias).
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    vmax = args.E_stiff * 1.2
    for ax, (title, M, vm) in zip(axes, [
        ("ground truth E (MPa)", E_true, vmax),
        ("recovered: FEM kernel", E_kernel, vmax),
        ("recovered: Hertz (biased high)", E_hertz, None),
    ]):
        im = ax.imshow(M, origin="lower", cmap="viridis", vmin=0,
                       vmax=vm, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("column"); ax.set_ylabel("row")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        f"{args.n}x{args.n} force map, pitch {args.pitch} nm, {args.freq} Hz — "
        f"scan time {spec.total_scan_time_s:.0f} s (elastic)")
    fig.tight_layout()
    out = os.path.join(args.outdir, "force_volume.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
