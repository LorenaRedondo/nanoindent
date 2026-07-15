"""Pseudo-realtime streaming analysis + simulated closed-loop control.

Demonstrates the full loop against a *reactive simulated instrument* (no
hardware): frames stream in, the online analyzer tracks particles as they
arrive, and a policy keeps the brightest feature centred by re-centering the
scan window — with per-frame timing showing the real-time headroom.

Swap ``SimulatedInstrument`` for a thin vendor adapter exposing the same
``__iter__`` / ``send`` contract to drive a real microscope. The fast tip-
protection feedback must remain on the instrument's own controller; this loop
is supervisory only.

    python examples/streaming_control.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")

from nanoindent import (
    SimulatedInstrument, StreamingAnalyzer, run_control_loop,
    recenter_on_brightest, ramp_on_event, combine_policies, plot_control_run,
)


def rms_offset(inst):
    return float(np.sqrt(np.mean(np.sum(inst.target_offset_px() ** 2, axis=1))))


def main():
    kw = dict(scan_px=64, pixel_size_nm=2.0, frame_interval_s=0.05,
              n_particles=4, D_nm2_per_s=200.0, max_frames=150, seed=1)

    # ---- closed loop: follow the brightest feature ----------------------
    inst = SimulatedInstrument(**kw)
    an = StreamingAnalyzer(inst.pixel_size_nm, inst.frame_interval_s, max_step_px=6.0)
    policy = combine_policies(recenter_on_brightest(), ramp_on_event())
    rep = run_control_loop(inst, an, policy, apply=True)

    # ---- open-loop baseline (same seed, no actuation) -------------------
    inst0 = SimulatedInstrument(**kw)
    an0 = StreamingAnalyzer(inst0.pixel_size_nm, inst0.frame_interval_s, max_step_px=6.0)
    run_control_loop(inst0, an0, recenter_on_brightest(), apply=False)

    period_ms = inst.frame_interval_s * 1e3
    print(f"acquisition: {inst.scan_px}x{inst.scan_px} px @ {period_ms:.0f} ms/frame "
          f"({1/inst.frame_interval_s:.0f} fps)")
    print(f"analysis   : median {np.median(rep.proc_ms[1:]):.2f} ms/frame "
          f"(real-time factor {rep.real_time_factor():.2f}×, "
          f"keeps up: {rep.kept_up()})")
    print(f"commands   : {len(rep.commands)} issued, "
          f"{len(inst.ramp_events)} ramps triggered, {len(rep.events)} events")
    print(f"following  : target RMS offset {rms_offset(inst):.1f} px (closed) "
          f"vs {rms_offset(inst0):.1f} px (open) — "
          f"{rms_offset(inst0)/rms_offset(inst):.1f}x tighter")

    fig = plot_control_run(rep, inst)
    fig.savefig("streaming_control.png", dpi=150, bbox_inches="tight")
    print("wrote streaming_control.png")


if __name__ == "__main__":
    main()
