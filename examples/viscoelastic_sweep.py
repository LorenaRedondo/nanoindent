"""Frequency-dependent modulus of a viscoelastic sample, from the FEM kernel.

The elastic solver gives the geometric kernel phi(delta) = F/E in one solve. The
correspondence principle (Lee & Radok) turns that into a *dynamic* response: for
a linear viscoelastic material with relaxation modulus E(t), the force follows
the hereditary integral F(t) = int E(t-t') d/dt'[phi(delta(t'))] dt'. Driving the
indentation sinusoidally at a range of frequencies recovers the storage and loss
moduli E'(omega), E''(omega).

This script sweeps frequency for a Standard Linear Solid, extracts E'/E'' from
the simulated oscillatory force, and overlays the closed-form complex modulus --
the validation. It also shows a stress-relaxation curve (ramp-and-hold), the
time-domain view of the same material.

Usage
-----
    python examples/viscoelastic_sweep.py [--E0 30 --Einf 10 --tau 1e-3] [--outdir DIR]
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
from nanoindent import viscoelastic as ve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--E0", type=float, default=30.0, help="instantaneous (glassy) modulus (MPa)")
    ap.add_argument("--Einf", type=float, default=10.0, help="equilibrium modulus (MPa)")
    ap.add_argument("--tau", type=float, default=1e-3, help="relaxation time (s)")
    ap.add_argument("--delta0", type=float, default=1.0, help="operating indentation (nm)")
    ap.add_argument("--amp", type=float, default=0.05, help="oscillation amplitude (nm)")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # One FEM solve -> geometric kernel (finite-thickness, exact sphere).
    # Viscoelastic soft matter (cells, gels) is near-incompressible; use the
    # locking-free mixed formulation.
    cfg = SimulationConfig(R=20.0, nu=0.49, h=8.0, L=100.0, delta_max=2.0,
                           formulation="mixed",
                           mesh_size_contact=0.35, mesh_size_edge=12.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kernel = fm.geometric_kernel(cfg)
    print(f"FEM kernel: {len(kernel.delta)} pts to {kernel.delta_max:.2f} nm")

    sls = ve.standard_linear_solid(E0=args.E0, E_inf=args.Einf, tau=args.tau)
    f_corner = 1.0 / (2.0 * np.pi * args.tau)
    print(f"SLS: E0={sls.E0} Einf={sls.E_inf} MPa, tau={args.tau} s "
          f"(corner ~ {f_corner:.0f} Hz)")

    # Frequency sweep around the corner.
    freqs = np.logspace(np.log10(f_corner) - 2.5, np.log10(f_corner) + 2.5, 24)
    omegas = 2.0 * np.pi * freqs
    spec = ve.frequency_sweep(
        sls, kernel.phi_of,
        lambda d0: ve.kernel_slope(kernel.phi_of, d0),
        omegas, delta0=args.delta0, delta_amp=args.amp,
    )
    print(f"recovered E'(w), E''(w) vs analytic: max rel err "
          f"{100 * spec.max_rel_error():.2f}%")

    # Stress relaxation (time domain): ramp fast, then hold.
    t, d, F = ve.ramp_hold_curve(sls, kernel.phi_of, delta_max=args.delta0,
                                 approach_velocity_nm_s=1e4, hold_s=8 * args.tau)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.loglog(spec.freq_hz, spec.storage_exact, "-", color="#0072B2",
               label="E' analytic")
    ax1.loglog(spec.freq_hz, spec.loss_exact, "-", color="#D55E00",
               label="E'' analytic")
    ax1.loglog(spec.freq_hz, spec.storage, "o", color="#0072B2", ms=5,
               mfc="none", label="E' recovered (FEM)")
    ax1.loglog(spec.freq_hz, spec.loss, "s", color="#D55E00", ms=5,
               mfc="none", label="E'' recovered (FEM)")
    ax1.axvline(f_corner, color="0.6", ls=":", lw=1)
    ax1.set_xlabel("frequency (Hz)"); ax1.set_ylabel("modulus (MPa)")
    ax1.set_title("dynamic moduli: FEM+correspondence vs closed form")
    ax1.legend(fontsize=8, frameon=False)

    ax2.plot(t * 1e3, F, "-", color="#009E73")
    ax2.axhline(sls.E0 * kernel.phi_of(np.array([args.delta0]))[0], color="0.6",
                ls=":", lw=1, label="glassy F (E0)")
    ax2.axhline(sls.E_inf * kernel.phi_of(np.array([args.delta0]))[0], color="0.4",
                ls="--", lw=1, label="equilibrium F (Einf)")
    ax2.set_xlabel("time (ms)"); ax2.set_ylabel("force (pN)")
    ax2.set_title("stress relaxation (ramp-and-hold)")
    ax2.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    out = os.path.join(args.outdir, "viscoelastic_sweep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
