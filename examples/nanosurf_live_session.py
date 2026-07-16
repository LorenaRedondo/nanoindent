"""Nanosurf — a realistic live-style control session (no hardware).

Runs the full Nanosurf path against the in-process ``MockNanosurfSPM``, which
mirrors the real ``nanosurf.SPM()`` object tree, so it maps onto what a real
session would do:

  1. connect and **read capabilities** — pixel size / frame interval derived
     from the scan geometry (ImageWidth / Points / Scantime / Lines);
  2. **engage** (approach) and set feedback setpoint / gains / operating mode;
  3. **scan** while the online analyzer tracks features and detects events;
  4. on each event, run a **force–distance spectroscopy** and keep the curve;
  5. **withdraw** and disconnect.

To drive a real microscope, replace ``MockNanosurfSPM`` with the real package::

    import nanosurf
    from nanoindent.adapters import NanosurfAdapter
    ad = NanosurfAdapter(nanosurf.SPM(), record_ramps=True)   # CoreAFM/DriveAFM/…

The property/method names track the Nanosurf scripting reference; the
frame-buffer read is marked VERIFY (event-driven on real hardware). Fast tip-
protection stays on the instrument controller; this loop is supervisory only.

    python examples/nanosurf_live_session.py
"""

import numpy as np

from nanoindent import (StreamingAnalyzer, run_control_loop,
                        recenter_on_brightest, combine_policies, ramp_on_event)
from nanoindent.adapters import BackendSimulator, NanosurfAdapter, MockNanosurfSPM


def main():
    backend = BackendSimulator(scan_px=64, pixel_size_nm=2.0,
                               frame_interval_s=0.05, n_particles=4,
                               D_nm2_per_s=200.0, max_frames=120, seed=1)
    spm = MockNanosurfSPM(backend, model="DriveAFM")   # swap for nanosurf.SPM()

    with NanosurfAdapter(spm, max_frames=120, record_ramps=True) as ad:
        caps = ad.capabilities
        print(f"controller: {caps.model}, {caps.points}×{caps.lines} px, "
              f"{caps.pixel_size_nm:.1f} nm/px, {1/caps.frame_interval_s:.0f} fps, "
              f"mode {caps.operating_mode}")
        ad.engage()
        print(f"approach: {ad.approach_status}")
        ad.set_setpoint(0.3)
        ad.set_feedback_gains(p=0.5, i=120.0)

        an = StreamingAnalyzer(caps.pixel_size_nm, caps.frame_interval_s,
                               max_step_px=6.0, event_k=3.0)
        policy = combine_policies(recenter_on_brightest(), ramp_on_event())
        rep = run_control_loop(ad, an, policy)

        ad.withdraw()
        print(f"approach: {ad.approach_status}")

    print(f"scanned {rep.n_frames} frames, "
          f"median analysis {np.median(rep.proc_ms[1:]):.2f} ms/frame "
          f"(real-time factor {rep.real_time_factor():.2f}×)")
    print(f"{len(rep.events)} events → {len(ad.force_curves)} force–distance "
          f"curves recorded")
    if ad.force_curves:
        peaks = [fc.max_force_pN for fc in ad.force_curves]
        print(f"  force-curve peak forces: {np.min(peaks):.0f}–{np.max(peaks):.0f} pN")


if __name__ == "__main__":
    main()
