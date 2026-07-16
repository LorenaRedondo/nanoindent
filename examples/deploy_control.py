"""Deployable, vendor-agnostic control run with the safety envelope + recorder.

Shows the pieces that make the loop usable on a real microscope today without
any SDK:

  * a **directory source** — process frames as the instrument writes them to a
    folder (here we pre-write a synthetic movie, then batch-read it);
  * a **safety envelope** — rate limiting, scan-bounds clip, setpoint clamp,
    auto-pause on lost tracking;
  * a **run recorder** — frames, per-frame analysis, commands, events, timings
    and a manifest, all to disk.

Everything is also exposed as a CLI:

    python -m nanoindent.control --source directory --dir OUT --pattern "frame_*.npy" \\
        --pixel-size 2.0 --dt 0.05 --outdir runs/live \\
        --max-scan-offset-px 24 --pause-after-lost 5

    python examples/deploy_control.py
"""

import os
import tempfile

import numpy as np

from nanoindent import (DirectoryWatchSource, RunRecorder, SafetyEnvelope,
                        SafetyLimits, StreamingAnalyzer, run_control_loop,
                        recenter_on_brightest, combine_policies, ramp_on_event)
from nanoindent.video import synthetic_video


def main():
    work = tempfile.mkdtemp(prefix="nanoindent_deploy_")
    frames_dir = os.path.join(work, "scan_out")
    os.makedirs(frames_dir)

    # stand in for a microscope writing frames to a folder
    vid = synthetic_video(n_frames=80, size=64, n_particles=4,
                          pixel_size_nm=2.0, frame_interval_s=0.05, seed=1)
    for i, f in enumerate(vid.frames):
        np.save(os.path.join(frames_dir, f"frame_{i:03d}.npy"), f)
    print(f"wrote {vid.n_frames} frames to {frames_dir}")

    # deployable source: read frames from disk (no SDK)
    source = DirectoryWatchSource(frames_dir, pattern="frame_*.npy",
                                  pixel_size_nm=2.0, frame_interval_s=0.05)

    # safety envelope
    envelope = SafetyEnvelope(SafetyLimits(
        min_frames_between_commands=2,     # at most one command every 2 frames
        max_scan_offset_px=24.0,           # keep the scan within ±24 px of start
        setpoint_range=(0.1, 1.0),         # clamp any setpoint change
        pause_after_lost_frames=5))        # pause if no feature for 5 frames
    policy = envelope.guard(combine_policies(recenter_on_brightest(),
                                             ramp_on_event()))

    # recorder
    recorder = RunRecorder(os.path.join(work, "run"), save_frames=True)

    an = StreamingAnalyzer(source.pixel_size_nm, source.frame_interval_s,
                           max_step_px=6.0, event_k=3.0)
    report = run_control_loop(recorder.wrap(source), an, policy,
                              on_state=recorder.on_state)
    manifest = recorder.save(report, meta={"instrument": "directory-sim",
                                           "safety": envelope.counts})

    print(f"processed {report.n_frames} frames, "
          f"real-time factor {report.real_time_factor():.2f}×")
    print(f"safety: {envelope.counts}")
    print(f"recorded to {manifest['outdir']}: "
          f"{sorted(os.listdir(manifest['outdir']))}")


if __name__ == "__main__":
    main()
