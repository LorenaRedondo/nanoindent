"""Vendor FrameSource adapter scaffolds — Nanosurf, JPK/Bruker, Asylum.

Shows the same closed-loop control running through each Tier-2 vendor adapter,
driven by an in-process mock backend (shared simulated physics) so no hardware
is needed. Because every adapter satisfies the same ``FrameSource`` contract and
translates the same generic commands, all three produce the identical result —
which is exactly what makes the translation layer trustworthy before it meets a
real instrument.

To go live, construct the adapter with a real connection instead of a mock:

    NanosurfAdapter()                                   # uses the nanosurf package
    JPKAdapter(host="192.168.1.10", port=55055)         # real SPM socket server
    AsylumAdapter(bridge=my_igor_socket_bridge)         # Igor listener bridge

The transport specifics (marked ``VERIFY`` in each module) must be confirmed
against the vendor SDK. Fast tip-protection stays on the instrument controller;
this loop is supervisory only.

    python examples/instrument_adapters.py
"""

import numpy as np

from nanoindent import (StreamingAnalyzer, run_control_loop,
                        recenter_on_brightest, combine_policies, ramp_on_event)
from nanoindent.adapters import (BackendSimulator, NanosurfAdapter, MockNanosurfSPM,
                                 JPKAdapter, MockJPKConnection,
                                 AsylumAdapter, MockIgorBridge)

BK = dict(scan_px=64, pixel_size_nm=2.0, frame_interval_s=0.05, n_particles=4,
          D_nm2_per_s=200.0, max_frames=120, seed=1)


def rms(backend):
    o = backend.instrument.target_offset_px()
    return float(np.sqrt(np.mean(np.sum(o ** 2, axis=1))))


def build(kind, backend):
    px, dt, n = backend.pixel_size_nm, backend.frame_interval_s, BK["max_frames"]
    if kind == "Nanosurf":
        return NanosurfAdapter(MockNanosurfSPM(backend), pixel_size_nm=px,
                               frame_interval_s=dt, max_frames=n)
    if kind == "JPK/Bruker":
        return JPKAdapter(MockJPKConnection(backend), pixel_size_nm=px,
                          frame_interval_s=dt, max_frames=n)
    return AsylumAdapter(MockIgorBridge(backend), pixel_size_nm=px,
                         frame_interval_s=dt, max_frames=n)


def main():
    print(f"{'vendor':12s} {'frames':>7s} {'commands':>9s} {'target RMS':>11s}")
    for kind in ("Nanosurf", "JPK/Bruker", "Asylum"):
        b = BackendSimulator(**BK)
        ad = build(kind, b)
        an = StreamingAnalyzer(ad.pixel_size_nm, ad.frame_interval_s, max_step_px=6.0)
        rep = run_control_loop(
            ad, an, combine_policies(recenter_on_brightest(), ramp_on_event()))
        print(f"{kind:12s} {rep.n_frames:7d} {len(ad.command_log):9d} "
              f"{rms(b):9.1f} px")
    print("\nAll three adapters translate the same commands and close the same "
          "loop.\nSwap the mock connection for a real one to drive hardware.")


if __name__ == "__main__":
    main()
