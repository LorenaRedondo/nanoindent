"""JPK / Bruker BioAFM — a realistic live-style control session (no hardware).

Runs the full deepened JPK path against the faithful loopback
``JPKInstrumentServer`` (which emulates the handshake, push streaming, and force
spectroscopy), so it mirrors what a real session would do:

  1. connect and **handshake** — the adapter reads pixel size / frame interval
     from the instrument (not from hand-set constants);
  2. **engage** and set the feedback setpoint / gains;
  3. **stream** frames while the online analyzer tracks features and detects
     events;
  4. on each event, trigger a **force–distance ramp** and keep the returned
     curve;
  5. **withdraw** and disconnect.

To drive a real microscope, replace ``JPKInstrumentServer`` + its loopback with
a real host/port::

    from nanoindent.adapters import JPKAdapter, SocketJPKConnection
    ad = JPKAdapter(SocketJPKConnection("192.168.1.10", 55055),
                    stream=True, record_ramps=True)

The wire grammar is a documented-shape placeholder (see jpk.py, marked VERIFY) —
confirm it against your JPK scripting reference. Fast tip-protection stays on the
instrument controller; this loop is supervisory only.

    python examples/jpk_live_session.py
"""

import numpy as np

from nanoindent import (StreamingAnalyzer, run_control_loop,
                        recenter_on_brightest, combine_policies, ramp_on_event)
from nanoindent.adapters import (BackendSimulator, JPKAdapter,
                                 SocketJPKConnection, JPKInstrumentServer)


def main():
    # a stand-in instrument (loopback); swap for a real host/port to go live
    backend = BackendSimulator(scan_px=64, pixel_size_nm=2.0,
                               frame_interval_s=0.05, n_particles=4,
                               D_nm2_per_s=200.0, max_frames=120, seed=1)
    server = JPKInstrumentServer(backend).start()

    conn = SocketJPKConnection("127.0.0.1", server.port, timeout=10.0)
    # pixel size / frame interval left unset — the handshake fills them in.
    # The context manager keeps one connection open across setup → loop →
    # teardown (the control loop no longer closes a caller-opened connection).
    with JPKAdapter(conn, max_frames=120, stream=True, record_ramps=True) as ad:
        caps = ad.capabilities
        print(f"handshake: pixel {caps.pixel_size_nm} nm/px, "
              f"{1/caps.frame_interval_s:.0f} fps, channels {caps.channels}")
        print(f"ping: {ad.ping()}")
        ad.engage()
        ad.set_setpoint(0.3)
        ad.set_feedback_gains(p=0.5, i=120.0)
        ad.set_line_rate(50.0)

        # ---- stream + analyse + event-triggered force ramps -------------
        an = StreamingAnalyzer(caps.pixel_size_nm, caps.frame_interval_s,
                               max_step_px=6.0, event_k=3.0)
        policy = combine_policies(recenter_on_brightest(), ramp_on_event())
        rep = run_control_loop(ad, an, policy)

        ad.withdraw()

    print(f"streamed {rep.n_frames} frames, "
          f"median analysis {np.median(rep.proc_ms[1:]):.2f} ms/frame "
          f"(real-time factor {rep.real_time_factor():.2f}×)")
    print(f"{len(rep.events)} events → {len(ad.force_ramps)} force–distance "
          f"curves recorded")
    if ad.force_ramps:
        peaks = [fr.max_force_pN for fr in ad.force_ramps]
        print(f"  force-ramp peak forces: "
              f"{np.min(peaks):.0f}–{np.max(peaks):.0f} pN")

    server.join(timeout=5.0)


if __name__ == "__main__":
    main()
