"""Asylum (Igor Pro) — a realistic live-style control session (no hardware).

Runs the full Asylum path against a loopback ``IgorListenerServer`` over TCP,
which emulates the Igor-side listener (``docs/asylum_igor_listener.ipf``):

  1. connect and read **capabilities** — pixel size / frame interval derived
     from the Asylum scan globals (ScanSize / PointsLines / ScanRate);
  2. **engage** (Approach) and set feedback setpoint / gains;
  3. **scan** while the online analyzer tracks features and detects events;
  4. on each event, run a **single-force spectroscopy** and keep the F–d curve;
  5. **withdraw** and disconnect.

To drive a real microscope, run the reference listener inside your Asylum Igor
experiment and point a SocketIgorBridge at it::

    from nanoindent.adapters import AsylumAdapter, SocketIgorBridge
    ad = AsylumAdapter(SocketIgorBridge("192.168.1.20", 55070), record_ramps=True)

The Igor function names are placeholders marked VERIFY — substitute the exact
Asylum calls (see the .ipf). Fast tip-protection stays on the instrument
controller; this loop is supervisory only.

    python examples/asylum_live_session.py
"""

import numpy as np

from nanoindent import (StreamingAnalyzer, run_control_loop,
                        recenter_on_brightest, combine_policies, ramp_on_event)
from nanoindent.adapters import (BackendSimulator, AsylumAdapter,
                                 SocketIgorBridge, IgorListenerServer)


def main():
    backend = BackendSimulator(scan_px=64, pixel_size_nm=2.0,
                               frame_interval_s=0.05, n_particles=4,
                               D_nm2_per_s=200.0, max_frames=120, seed=1)
    server = IgorListenerServer(backend, mode="AC").start()   # stands in for Igor

    bridge = SocketIgorBridge("127.0.0.1", server.port, timeout=10.0)
    with AsylumAdapter(bridge, max_frames=120, record_ramps=True) as ad:
        caps = ad.capabilities
        print(f"scan: {caps.points}×{caps.lines} px over {caps.scan_size_nm:.0f} nm "
              f"→ {caps.pixel_size_nm:.1f} nm/px, {1/caps.frame_interval_s:.0f} fps, "
              f"mode {caps.mode}")
        ad.engage()
        ad.set_setpoint(0.3)
        ad.set_feedback_gains(p=0.5, i=120.0)

        an = StreamingAnalyzer(caps.pixel_size_nm, caps.frame_interval_s,
                               max_step_px=6.0, event_k=3.0)
        policy = combine_policies(recenter_on_brightest(), ramp_on_event())
        rep = run_control_loop(ad, an, policy)

        ad.withdraw()

    server.join(timeout=5.0)

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
