"""Record a control run, then replay it from disk.

The recorder writes a run to a directory; :func:`nanoindent.load_run` reads it
back into a :class:`RecordedRun` you can re-plot, turn back into an
:class:`~nanoindent.video.AFMVideo`, or animate — without re-running the loop.
The dashboard's *Run browser* page does the same interactively.

    python examples/replay_run.py
"""

import os
import tempfile

import matplotlib
matplotlib.use("Agg")

from nanoindent import load_run
from nanoindent.control import run_from_config


def main():
    work = tempfile.mkdtemp(prefix="nanoindent_replay_")
    run_dir = os.path.join(work, "run")

    # record a run (frames kept so we can replay them)
    run_from_config({
        "source": {"type": "simulated", "frames": 80, "scan_px": 64,
                   "n_particles": 4, "D_nm2_per_s": 200.0, "seed": 1},
        "policy": ["recenter_on_brightest", "ramp_on_event"],
        "safety": {"max_scan_offset_px": 24.0, "pause_after_lost_frames": 5},
        "save_frames": True, "outdir": run_dir, "tags": {"sample": "SLB-demo"},
    })
    print(f"recorded run to {run_dir}")

    # replay it from disk — no re-running of the control loop
    run = load_run(run_dir)
    print(f"run {run.run_id}: {run.n_frames} frames, {len(run.commands)} commands, "
          f"{len(run.events)} events")
    print(f"metrics: real-time factor {run.metrics.get('real_time_factor'):.2f}×, "
          f"final diffusion {run.metrics.get('diffusion_final_nm2_s', float('nan')):.0f} nm²/s")

    fig = run.plot_summary()
    fig.savefig(os.path.join(work, "replay_summary.png"), dpi=140,
                bbox_inches="tight")
    out = run.animate(os.path.join(work, "replay.gif"), fps=15)
    print(f"wrote replay_summary.png and {os.path.basename(out)} under {work}")


if __name__ == "__main__":
    main()
