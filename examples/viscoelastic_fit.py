"""Recover a sample's viscoelastic parameters from measured force curves.

The inverse of the forward model: given AFM data, fit the Standard-Linear-Solid
parameters E0 (glassy), E_inf (equilibrium), tau (relaxation time). Two standard
protocols, each a closed-form separation of *geometry* (the FEM kernel phi) from
*material* (the relaxation function):

  * stress relaxation -- ramp to a depth, hold, fit the force decay
    F(t) = phi*(E_inf + (E0-E_inf) exp(-t/tau));
  * frequency sweep -- oscillate, fit the storage/loss moduli E'(w), E''(w).

This runs a synthetic round-trip: build data from known parameters, add realistic
noise, fit it back, and overlay the recovered model -- the same round-trip test
the rest of the library uses to trust a result. On real data you would pass your
measured hold curve (with phi from a tip/geometry calibration) or your measured
E'/E'' spectrum to the same two functions.

Usage
-----
    python examples/viscoelastic_fit.py [--E0 45 --Einf 12 --tau 0.03]
                                        [--noise-pN 15] [--outdir DIR]
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nanoindent import SimulationConfig, forcemap as fm, viscoelastic as ve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--E0", type=float, default=45.0, help="true glassy modulus (MPa)")
    ap.add_argument("--Einf", type=float, default=12.0, help="true equilibrium modulus (MPa)")
    ap.add_argument("--tau", type=float, default=0.03, help="true relaxation time (s)")
    ap.add_argument("--delta", type=float, default=1.2, help="hold indentation (nm)")
    ap.add_argument("--noise-pN", type=float, default=15.0, help="force noise (pN)")
    ap.add_argument("--noise-MPa", type=float, default=0.6, help="E'/E'' noise (MPa)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cfg = SimulationConfig(R=20.0, nu=0.5, h=6.0, L=100.0, delta_max=1.5,
                           formulation="mixed", mesh_size_contact=0.35,
                           mesh_size_edge=12.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kernel = fm.geometric_kernel(cfg)
    rng = np.random.default_rng(args.seed)

    true = ve.standard_linear_solid(E0=args.E0, E_inf=args.Einf, tau=args.tau)
    phi = float(kernel.phi_of(np.array([args.delta]))[0])
    print(f"TRUE:  E0={args.E0} MPa  E_inf={args.Einf} MPa  tau={args.tau*1e3:.0f} ms  "
          f"(beta={1-args.Einf/args.E0:.2f})\n")

    # ---- Method 1: stress relaxation ----
    t, d, F = ve.ramp_hold_curve(true, kernel.phi_of, delta_max=args.delta,
                                 approach_velocity_nm_s=2e4, hold_s=8 * args.tau,
                                 n_ramp=60, n_hold=300)
    hold = t >= t[d.argmax()]
    th, Fh = t[hold] - t[hold][0], F[hold]
    Fh_noisy = Fh + rng.normal(0, args.noise_pN, Fh.size)
    fit_r = ve.fit_relaxation(th, Fh_noisy, phi)
    print("stress relaxation ->", fit_r.summary())

    # ---- Method 2: frequency sweep ----
    freqs = np.logspace(-0.5, 3.0, 18)
    omegas = 2 * np.pi * freqs
    spec = ve.frequency_sweep(true, kernel.phi_of,
                              lambda d0: ve.kernel_slope(kernel.phi_of, d0),
                              omegas, delta0=1.0, delta_amp=0.05)
    Ep = spec.storage + rng.normal(0, args.noise_MPa, spec.storage.size)
    Epp = spec.loss + rng.normal(0, args.noise_MPa, spec.loss.size)
    fit_d = ve.fit_dynamic_spectrum(omegas, Ep, Epp)
    print("frequency sweep   ->", fit_d.summary())

    # ---- figure ----
    fig, (axR, axD) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    axR.plot(th * 1e3, Fh_noisy, "o", ms=3, color="0.6", label="measured (noisy)")
    Efit = fit_r.model.E_relax(th)
    axR.plot(th * 1e3, phi * Efit, "-", color="#D55E00", lw=2.2, label="SLS fit")
    axR.set_xlabel("time in hold (ms)")
    axR.set_ylabel("force (pN)")
    axR.set_title("stress relaxation")
    axR.text(0.5, 0.9, f"fit: E0={fit_r.E0:.1f}, E∞={fit_r.E_inf:.1f} MPa, "
             f"τ={fit_r.tau*1e3:.0f} ms\ntrue: {args.E0:.0f}, {args.Einf:.0f} MPa, "
             f"{args.tau*1e3:.0f} ms", transform=axR.transAxes, fontsize=8, va="top")
    axR.legend(fontsize=8, frameon=False, loc="upper right")

    axD.loglog(freqs, Ep, "o", ms=4, color="#0072B2", mfc="none", label="E' measured")
    axD.loglog(freqs, Epp, "s", ms=4, color="#D55E00", mfc="none", label="E'' measured")
    wfit = 2 * np.pi * freqs
    axD.loglog(freqs, fit_d.model.storage(wfit), "-", color="#0072B2", lw=2, label="E' fit")
    axD.loglog(freqs, fit_d.model.loss(wfit), "-", color="#D55E00", lw=2, label="E'' fit")
    axD.axvline(1.0 / (2 * np.pi * fit_d.tau), color="0.6", ls=":", lw=1)
    axD.set_xlabel("frequency (Hz)")
    axD.set_ylabel("modulus (MPa)")
    axD.set_title("frequency sweep")
    axD.legend(fontsize=8, frameon=False, loc="lower right")

    out = os.path.join(args.outdir, "viscoelastic_fit.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
