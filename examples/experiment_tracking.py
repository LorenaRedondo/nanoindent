"""Experiment tracking: timestamps, run manifests, and cross-run comparison.

Each control run is recorded with a run id, wall-clock start/end timestamps, and
MLflow-shaped params / metrics / tags. A directory of runs is then indexed into a
comparison table with no external dependency (:class:`ExperimentIndex`), and can
optionally be exported to MLflow.

    python examples/experiment_tracking.py
"""

import os
import tempfile

from nanoindent import ExperimentIndex
from nanoindent.control import run_from_config


def main():
    root = tempfile.mkdtemp(prefix="nanoindent_runs_")

    # sweep a couple of settings — each is a recorded, comparable run
    for i, (npart, D) in enumerate([(3, 150.0), (5, 250.0), (4, 200.0)]):
        run_from_config({
            "source": {"type": "simulated", "frames": 40, "scan_px": 64,
                       "n_particles": npart, "D_nm2_per_s": D, "seed": i},
            "policy": ["recenter_on_brightest", "ramp_on_event"],
            "safety": {"max_scan_offset_px": 24.0, "pause_after_lost_frames": 5},
            "outdir": os.path.join(root, f"run_{i}"),
            "tags": {"sample": "SLB-demo", "operator": "afm",
                     "n_particles": npart},
        })

    # index every run manifest under the root into one comparison table
    idx = ExperimentIndex(root)
    print(f"{len(idx)} runs under {root}\n")

    cols = ["run_id", "started_at", "tags.n_particles",
            "params.D_nm2_per_s", "metrics.diffusion_final_nm2_s",
            "metrics.real_time_factor", "metrics.guard_clipped_bounds"]
    print(" | ".join(c.split(".")[-1] for c in cols))
    for r in idx.rows():
        print(" | ".join(f"{r.get(c, '')}"[:22] for c in cols))

    csv = os.path.join(root, "comparison.csv")
    idx.to_csv(csv)
    print(f"\nwrote {csv}")

    best = idx.best("diffusion_final_nm2_s", maximize=True)
    if best:
        print(f"highest recovered diffusion: run {best['run_id']} "
              f"({best['metrics.diffusion_final_nm2_s']:.0f} nm²/s)")

    print("\nTo push these to MLflow (needs `pip install \"nanoindent[tracking]\"`):")
    print(f"    nanoindent-runs {root} --mlflow --experiment afm-control")


if __name__ == "__main__":
    main()
