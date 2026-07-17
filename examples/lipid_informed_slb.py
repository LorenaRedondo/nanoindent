"""Configure an SLB analysis from a lipid species name, then score the prediction.

Bridges the LIPIVERSE lipid atlas to this library (see :mod:`nanoindent.lipids`):

1. **configure** — fetch the composition-derived bilayer prediction for a species
   (``"PC 34:1"``) and use its thickness as the film thickness ``h`` for the
   finite-thickness contact models, instead of hand-setting ``--h 5``;
2. **validate** — put the prediction next to what the AFM force map actually
   measured (thickness, breakthrough force, kappa_A), which is the direction that
   can calibrate LIPIVERSE's heuristic coefficients against data.

The thickness is a **prediction**, not geometry: it is a prior to test. The
config's ``label`` carries that provenance into every result and figure, and the
Young's modulus is never predicted — it comes from the fit, or from ``--E``.

Usage
-----
    # offline, no LIPIVERSE service needed (built-in species library):
    python examples/lipid_informed_slb.py --species "PC 34:1" --offline

    # against a running LIPIVERSE (`npm run api` in the lipids repo):
    python examples/lipid_informed_slb.py --species "PC 34:1"

    # with a real force map, to compare prediction against measurement:
    python examples/lipid_informed_slb.py --species "PC 34:1" --map MAP.jpk-force-map
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")

from nanoindent import lipids, slb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default="PC 34:1", help="lipid shorthand, e.g. 'PC 34:1'")
    ap.add_argument("--map", help="optional .jpk-force-map to compare against")
    ap.add_argument("--url", default=None, help="LIPIVERSE base URL (default $LIPIVERSE_URL)")
    ap.add_argument("--offline", action="store_true",
                    help="use the built-in species library instead of the service")
    ap.add_argument("--R", type=float, default=20.0, help="tip radius (nm)")
    ap.add_argument("--temperature", type=float, default=25.0, help="experiment temp (deg C)")
    ap.add_argument("--E", type=float, default=None,
                    help="Young's modulus (MPa) for the FEM; default: fitted from --map")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    transport = lipids.MockLipiverseTransport() if args.offline else None

    # -- 1. the lipid prior ------------------------------------------------
    client = lipids.LipiverseClient(args.url, transport=transport)
    try:
        pred = client.get_species(args.species)
    except lipids.NotAMembraneLipid as exc:
        raise SystemExit(f"{exc}\nPick a bilayer-forming class (PC, PE, PS, SM, ...).")
    except lipids.LipiverseUnavailable as exc:
        raise SystemExit(f"{exc}\nRe-run with --offline to use the built-in library.")

    phase = pred.phase_at(args.temperature)
    source = "built-in offline library" if args.offline else (pred.source_url or "?")
    print(f"LIPIVERSE  {pred.summary()}")
    print(f"  source     {source}")
    print(f"  phase      {phase} at {args.temperature:g} C (Tm = {pred.tm_c:g} C)")
    print(f"  geometry   {pred.curvature}, area/lipid {pred.area_per_lipid_a2:g} A^2")
    print(f"  provenance {pred.provenance} — {pred.note}")

    # -- 2. measure, if a map was given ------------------------------------
    analysis = None
    if args.map:
        from nanoindent import read_jpk_map

        fmap = read_jpk_map(args.map)
        print(f"\n{len(fmap)} curves, k = {fmap.spring_constant:.3f} N/m")
        # the predicted thickness sets h for the finite-thickness models
        analysis = slb.analyze_map(fmap, R=args.R, nu=0.5,
                                   film_thickness=pred.thickness_nm,
                                   fit_max_delta=pred.thickness_nm)

    # -- 3. the FEM config, with provenance in the label -------------------
    E = args.E
    if E is None and analysis is not None:
        import numpy as np

        fitted = analysis.moduli("dimitriadis")
        E = float(np.median(fitted)) if fitted.size else None
    if E is None:
        E = 15.0
        print("\nNo --E and no fit available; using E = 15 MPa as a stated assumption.")

    cfg = pred.to_simulation_config(E=E, R=args.R, temperature_c=args.temperature)
    print(f"\nSimulationConfig  {cfg.summary()}")
    print(f"  label  {cfg.label}")

    # -- 4. predicted vs measured -----------------------------------------
    if analysis is not None:
        cmp = pred.compare_to_slb(analysis)
        print(f"\npredicted vs measured (height definition: {cmp['height_definition']})")
        print(f"{'quantity':22s} {'predicted':>10} {'measured':>10} {'n':>4} {'meas/pred':>10}")
        for key in ("thickness_nm", "breakthrough_nn", "kappa_A_mN_per_m"):
            e = cmp[key]
            print(f"{key:22s} {e['predicted']:>10.3g} {e['measured_median']:>10.3g} "
                  f"{e['n']:>4} {e['ratio']:>10.2f}")

        os.makedirs(args.outdir, exist_ok=True)
        fig = slb.plot_analysis(analysis, film_thickness=pred.thickness_nm,
                                fit_max_delta=pred.thickness_nm)
        out = os.path.join(args.outdir, f"slb_{pred.species_id}.png")
        fig.savefig(out, dpi=200)
        print(f"\nwrote {out}")
    else:
        print("\nNo --map given: pass a force map to score the prediction "
              "against measurement.")


if __name__ == "__main__":
    main()
