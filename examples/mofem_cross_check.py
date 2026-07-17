"""Cross-check the FEM against MoFEM's axisymmetric Hertz benchmark.

MoFEM (https://mofem.eng.gla.ac.uk, MIT) ships an axisymmetric contact atom test
-- ``hertz_mfront_2d_axisymm_flat`` in tutorial ADV-1 -- that is the same problem
this library solves: a rigid sphere, given as an exact-sphere signed distance
function, indenting a flat linear-elastic axisymmetric body. It reaches that
answer by a different route: augmented-Lagrangian contact with an H(div) traction
field, where this library uses penalty contact with an active-set Newton solver.

This script runs our solver at MoFEM's configuration and sweeps the contact mesh.

What it is *not*: validation against MoFEM. MoFEM is not ground truth -- it carries
its own discretisation error, which is exactly the argument in design.md section
6.2. Both codes are measured here against the same analytic ruler (Hertz), and the
useful output is the *spread*, not the agreement. See design.md section 6.3.

Usage
-----
    python examples/mofem_cross_check.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np

from nanoindent import SimulationConfig, IndentationSolver, analytics

# MoFEM ADV-1 `hertz_mfront_2d_axisymm_flat`, from its CMakeLists/atom test and
# the mesh bounding box of mesh_files/hertz_2d_4p.h5m. Units are arbitrary --
# linear elasticity is scale-invariant, so this is our nm/MPa/pN solver run on
# their numbers, not a unit conversion.
MOFEM_R = 100.0     # indenter radius (sdf_hertz_2d_axisymm_sphere.py)
MOFEM_DELTA = 0.01  # indentation depth
MOFEM_E = 1e3       # mf_param_10_0
MOFEM_NU = 0.4      # mf_param_10_1
MOFEM_L = 50.0      # mesh radial extent
MOFEM_H = 100.0     # mesh depth

# MoFEM's own result, from `mpirun -np 4 contact_2d ... -atom_test 5` on
# MoFEM 0.15.5 / v0.15.0 image. Recorded rather than re-run: reproducing it needs
# their Docker image and an AVX-free rebuild of libBehaviour.so (design.md 6.3).
MOFEM_FORCE = 15.82581
MOFEM_ACTIVE_AREA = 3.162873

# Contact-zone element sizes to sweep. The Hertz contact radius here is
# a = sqrt(R*delta) = 1.0, so these span roughly a/20 to a/140.
MESH_SIZES = (0.050, 0.030, 0.020, 0.015, 0.010, 0.007)


def run_at(mesh_size_contact: float, penalty_alpha: float = 400.0):
    """Solve MoFEM's configuration at one contact-mesh size."""
    cfg = SimulationConfig(
        R=MOFEM_R, E=MOFEM_E, nu=MOFEM_NU, h=MOFEM_H, L=MOFEM_L,
        delta_max=MOFEM_DELTA, n_steps=6,
        mesh_size_contact=mesh_size_contact,
        mesh_size_edge=5.0,
        penalty_alpha=penalty_alpha,
    )
    with warnings.catch_warnings():
        # The equilibrium check compares contact-pressure and bottom-sigma_zz
        # integrals. MoFEM's mesh is a roller-supported half-space and ours is
        # bonded; with a = 1 << h = 100 the bottom BC is irrelevant to F by
        # Saint-Venant, but the bottom measure is noisy this far from contact.
        warnings.simplefilter("ignore")
        curve = IndentationSolver(cfg).run()
    return curve.force[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    f_hertz = float(analytics.hertz_force(
        np.array([MOFEM_DELTA]), MOFEM_R, MOFEM_E, MOFEM_NU)[0])

    print(f" MoFEM ADV-1 axisymmetric configuration:")
    print(f"   R = {MOFEM_R}, delta = {MOFEM_DELTA}, E = {MOFEM_E}, nu = {MOFEM_NU}")
    print(f"   delta/R = {MOFEM_DELTA / MOFEM_R:.1e}, a_Hertz = {(MOFEM_R * MOFEM_DELTA) ** 0.5:.3f}")
    print(f"   F_Hertz (analytic) = {f_hertz:.5f}\n")

    print(" mesh_size_contact     F_FEM      F/F_Hertz")
    ratios = []
    for mc in MESH_SIZES:
        f = run_at(mc)
        r = f / f_hertz
        ratios.append(r)
        print(f"   {mc:.3f}            {f:9.5f}     {r:.5f}")

    lo, hi = min(ratios), max(ratios)
    band = 100.0 * (hi - lo) / 2.0
    print(f"\n ours   : F/F_Hertz in [{lo:.5f}, {hi:.5f}]  -> +/-{band:.2f}% scatter")
    print(f" MoFEM  : F/F_Hertz  = {MOFEM_FORCE / f_hertz:.5f}  ({100 * (MOFEM_FORCE / f_hertz - 1):+.2f}%)")

    # The sweep does not settle -- it oscillates. That is the finding: the
    # active set changes discretely as elements enter and leave contact, so F
    # carries a floor of discretisation noise that refinement does not remove.
    monotonic = all(a >= b for a, b in zip(ratios, ratios[1:])) or \
                all(a <= b for a, b in zip(ratios, ratios[1:]))
    print(f" mesh sweep monotonic: {monotonic}")

    mofem_inside = lo <= MOFEM_FORCE / f_hertz <= hi
    print(f" MoFEM ratio inside our scatter band: {mofem_inside}")

    # Penalty is the free parameter of the formulation (design.md section 6);
    # if F moves when alpha does, the penalty is setting the answer.
    f400 = run_at(0.030, penalty_alpha=400.0)
    f1000 = run_at(0.030, penalty_alpha=1000.0)
    print(f"\n penalty_alpha 400 -> 1000 at mesh 0.030: "
          f"{f400 / f_hertz:.5f} -> {f1000 / f_hertz:.5f} "
          f"({100 * abs(f1000 - f400) / f400:.2f}% change)")

    out = os.path.join(args.outdir, "mofem_cross_check.csv")
    with open(out, "w") as fh:
        fh.write("mesh_size_contact,F_FEM,F_over_F_hertz\n")
        for mc, r in zip(MESH_SIZES, ratios):
            fh.write(f"{mc},{r * f_hertz:.6f},{r:.6f}\n")
    print(f"\n Saved '{out}'")


if __name__ == "__main__":
    main()
