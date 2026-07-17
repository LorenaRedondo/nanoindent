"""How viscoelastic force curves differ from an AFM point of view.

An AFM does not measure a modulus; it measures a **force-indentation curve** and
you fit a modulus to it. On a viscoelastic sample that fit is rate-dependent and
the approach and retract disagree — two things an elastic Hertz fit silently gets
wrong. This script makes both visible for a set of materials that share the *same*
equilibrium modulus `E_inf` and differ only in viscous fraction β = 1 − E_inf/E0
and recovery time τ.

Panel A — force vs indentation, one full approach-retract at the scan rate:
  * apparent stiffness: the faster relaxation is outrun, the more the curve rides
    up toward the glassy `E0` — so identical materials read as very different
    moduli under a naive Hertz fit;
  * hysteresis: approach (solid) and retract (dashed) enclose a loop whose area is
    the energy dissipated in the cycle, and the retract ends below zero (the
    viscoelastic "adhesion"-like pull still felt at zero indentation).

Panel B — apparent modulus vs loading rate: a Hertz fit to the approach branch,
swept over rate. Every material starts at its shared `E_inf` when pushed slowly
and climbs toward its own glassy `E0` when pushed fast. This is the AFM artifact:
the number you report depends on how fast you pushed.

The geometric kernel is one real FEM solve; the time response is the Lee-Radok
correspondence (see docs/design.md section 8.2). The **retract** branch uses the
loading-theory convolution, not Ting's receding-contact correction, so the loop is
qualitatively right but the retract branch is approximate (docs/design.md 8.4).

Usage
-----
    python examples/afm_force_curves.py [--freq 25] [--E-inf 10] [--outdir DIR]
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

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# (label, viscous fraction beta, recovery time tau (s), colour)
MATERIALS = [
    ("elastic (β=0.2, τ=8 ms)", 0.20, 0.008, "#0072B2"),
    ("balanced (β=0.55, τ=80 ms)", 0.55, 0.080, "#009E73"),
    ("viscoelastic (β=0.85, τ=0.5 s)", 0.85, 0.500, "#D55E00"),
]


def down_up(delta_max, contact_time, n=500):
    """Triangular approach-retract indentation history (t, delta)."""
    t = np.linspace(0.0, contact_time, n)
    delta = delta_max * (1.0 - np.abs(2.0 * t / contact_time - 1.0))
    return t, delta


def apparent_modulus(kernel, sls, delta_max, contact_time):
    """Hertz-style fit of the modulus to the approach branch (closed form)."""
    t, delta = down_up(delta_max, contact_time)
    F = ve.hereditary_force(t, delta, sls, kernel.phi_of)
    up = t <= contact_time / 2.0
    phi = kernel.phi_of(delta[up])
    denom = float(np.dot(phi, phi))
    return float(np.dot(F[up], phi) / denom) if denom > 0 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=float, default=25.0, help="ramp frequency (Hz)")
    ap.add_argument("--E-inf", type=float, default=10.0,
                    help="shared equilibrium modulus (MPa)")
    ap.add_argument("--delta-max", type=float, default=1.2, help="peak indentation (nm)")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cfg = SimulationConfig(R=20.0, nu=0.5, h=6.0, L=100.0, delta_max=1.5,
                           formulation="mixed", mesh_size_contact=0.35,
                           mesh_size_edge=12.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kernel = fm.geometric_kernel(cfg)

    contact_time = 0.5 / args.freq
    t, delta = down_up(args.delta_max, contact_time)
    up = t <= contact_time / 2.0
    F_eq = args.E_inf * kernel.phi_of(delta)   # shared equilibrium curve

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    # ---- Panel A: force-indentation, approach + retract ----
    axA.plot(delta[up], F_eq[up], ":", color="0.5", lw=1.4,
             label=f"equilibrium E_inf = {args.E_inf:g} MPa")
    print(f"scan {args.freq:g} Hz  (contact time {contact_time*1e3:.0f} ms);  "
          f"all share E_inf = {args.E_inf:g} MPa\n")
    for label, beta, tau, color in MATERIALS:
        E0 = args.E_inf / (1.0 - beta)
        sls = ve.standard_linear_solid(E0=E0, E_inf=args.E_inf, tau=tau)
        F = ve.hereditary_force(t, delta, sls, kernel.phi_of)
        axA.plot(delta[up], F[up], "-", color=color, lw=2.2, label=label)
        axA.plot(delta[~up], F[~up], "--", color=color, lw=1.6)
        loop = abs(_trapz(F, delta))
        E_app = apparent_modulus(kernel, sls, args.delta_max, contact_time)
        print(f"  {label:34s} E0={E0:5.1f}  F_peak={F.max():6.0f} pN  "
              f"Hertz-fit E={E_app:5.1f} MPa ({E_app/args.E_inf:.1f}x E_inf)  "
              f"loop={loop:5.0f} pN·nm")
    axA.axhline(0, color="0.85", lw=0.8)
    axA.set_xlabel("indentation δ (nm)")
    axA.set_ylabel("force (pN)")
    axA.set_title(f"force-indentation at {args.freq:g} Hz — solid approach, dashed retract")
    axA.legend(fontsize=8, frameon=False, loc="upper left")

    # ---- Panel B: apparent modulus vs loading rate ----
    freqs = np.logspace(-1.5, 3.0, 40)
    for label, beta, tau, color in MATERIALS:
        E0 = args.E_inf / (1.0 - beta)
        sls = ve.standard_linear_solid(E0=E0, E_inf=args.E_inf, tau=tau)
        Eapp = [apparent_modulus(kernel, sls, args.delta_max, 0.5 / f) for f in freqs]
        axB.semilogx(freqs, Eapp, "-", color=color, lw=2.2, label=label.split(" (")[0])
        axB.axhline(E0, color=color, ls=":", lw=0.9)
    axB.axhline(args.E_inf, color="0.5", ls="--", lw=1)
    axB.axvline(args.freq, color="0.6", ls="-", lw=0.8)
    axB.text(args.freq * 1.1, args.E_inf * 1.05, "scan rate", fontsize=8, color="0.4")
    axB.set_xlabel("ramp frequency (Hz)")
    axB.set_ylabel("apparent (Hertz-fit) modulus (MPa)")
    axB.set_title("measured modulus depends on how fast you push")
    axB.legend(fontsize=8, frameon=False, loc="upper left")

    out = os.path.join(args.outdir, "afm_force_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
