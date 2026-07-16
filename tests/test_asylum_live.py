"""Deepened Asylum (Igor Pro) adapter: capability read-back, engage/withdraw,
feedback, force spectroscopy, and the socket bridge to a loopback Igor listener.

Exercised against the in-process ``MockIgorBridge`` and, for the wire path,
``SocketIgorBridge`` talking to a loopback ``IgorListenerServer`` (which mirrors
what ``docs/asylum_igor_listener.ipf`` does in Igor).
"""

import numpy as np
import pytest

from nanoindent import (
    StreamingAnalyzer, run_control_loop, recenter_on_brightest,
    combine_policies, ramp_on_event, Command,
)
from nanoindent.adapters import (
    BackendSimulator, AsylumAdapter, MockIgorBridge, SocketIgorBridge,
    IgorListenerServer, AsylumCapabilities, AsylumForceCurve,
)

BK = dict(scan_px=48, pixel_size_nm=2.0, frame_interval_s=0.05, n_particles=3,
          D_nm2_per_s=200.0, max_frames=50, seed=1)
NF = BK["max_frames"]


# --------------------------------------------------------------------------
# capabilities
# --------------------------------------------------------------------------
def test_capabilities_from_igor_globals():
    b = BackendSimulator(**BK)
    ad = AsylumAdapter(MockIgorBridge(b, mode="AC"), pixel_size_nm=99.0,
                       frame_interval_s=9.9, max_frames=NF)
    list(ad)
    caps = ad.capabilities
    assert isinstance(caps, AsylumCapabilities)
    assert caps.points == 48 and caps.mode == "AC"
    assert caps.pixel_size_nm == pytest.approx(2.0)       # scan_size / points
    assert caps.frame_interval_s == pytest.approx(0.05)   # lines / scan_rate
    assert ad.pixel_size_nm == pytest.approx(2.0)         # auto-calibrated


def test_requires_bridge():
    ad = AsylumAdapter(None)
    with pytest.raises(RuntimeError):
        list(ad)


# --------------------------------------------------------------------------
# igor command emission
# --------------------------------------------------------------------------
def test_emits_asylum_igor_calls():
    b = BackendSimulator(**BK)
    br = MockIgorBridge(b)
    ad = AsylumAdapter(br, max_frames=1)
    ad._ensure_connected(); ad._last_shape = (48, 48)
    ad._apply_command(Command("recenter", {"dy_px": 3.0, "dx_px": -2.0}))
    ad._apply_command(Command("set_setpoint", {"value": 0.3}))
    assert any(c.startswith("SetScanOffset") for c in br.history)
    assert any("SetpointDefl" in c for c in br.history)


def test_engage_withdraw_gains_emit_igor():
    b = BackendSimulator(**BK)
    br = MockIgorBridge(b)
    ad = AsylumAdapter(br, max_frames=1)
    ad._ensure_connected()
    ad.engage(); ad.withdraw(); ad.set_feedback_gains(0.5, 120.0)
    ad.set_operating_mode("Contact")
    assert any(c.startswith("Approach") for c in br.history)
    assert any(c.startswith("Withdraw") for c in br.history)
    assert any("ProportionalGain" in c for c in br.history)
    assert any("IntegralGain" in c for c in br.history)
    assert any("ImagingMode" in c for c in br.history)


def test_engage_withdraw_via_command_kinds():
    b = BackendSimulator(**BK)
    br = MockIgorBridge(b)
    ad = AsylumAdapter(br, max_frames=1)
    ad._ensure_connected()
    ad._apply_command(Command("engage"))
    ad._apply_command(Command("set_gains", {"p": 1.0, "i": 200.0}))
    ad._apply_command(Command("withdraw"))
    assert any(c.startswith("Approach") for c in br.history)
    assert any(c.startswith("Withdraw") for c in br.history)


# --------------------------------------------------------------------------
# force spectroscopy
# --------------------------------------------------------------------------
def test_force_curve_returns_curve():
    b = BackendSimulator(**BK)
    ad = AsylumAdapter(MockIgorBridge(b), max_frames=1)
    fc = ad.force_curve(10.0, 5.0)
    assert isinstance(fc, AsylumForceCurve)
    assert fc.max_force_pN > 0 and np.all(np.diff(fc.z_nm) > 0)
    assert len(ad.force_curves) == 1


def test_record_ramps_on_event():
    b = BackendSimulator(**dict(BK, seed=5))
    ad = AsylumAdapter(MockIgorBridge(b), max_frames=NF, record_ramps=True)
    an = StreamingAnalyzer(2.0, 0.05, event_k=3.0)
    rep = run_control_loop(ad, an, combine_policies(
        recenter_on_brightest(), ramp_on_event()))
    n_ramp_cmds = sum(1 for c in ad.command_log if c.kind == "trigger_ramp")
    assert len(ad.force_curves) == n_ramp_cmds
    for fc in ad.force_curves:
        assert fc.max_force_pN > 0


# --------------------------------------------------------------------------
# closed loop
# --------------------------------------------------------------------------
def test_closed_loop_tight_following_in_process():
    b = BackendSimulator(**BK)
    ad = AsylumAdapter(MockIgorBridge(b), max_frames=NF)
    an = StreamingAnalyzer(2.0, 0.05, max_step_px=6.0)
    rep = run_control_loop(ad, an, recenter_on_brightest())
    off = np.sqrt(np.mean(np.sum(b.instrument.target_offset_px() ** 2, axis=1)))
    assert rep.n_frames == NF and off < BK["scan_px"] / 8


# --------------------------------------------------------------------------
# socket bridge → loopback Igor listener (real TCP)
# --------------------------------------------------------------------------
def test_socket_bridge_closed_loop_over_igor_listener():
    b = BackendSimulator(**BK)
    srv = IgorListenerServer(b).start()
    try:
        conn = SocketIgorBridge("127.0.0.1", srv.port, timeout=10)
        ad = AsylumAdapter(conn, max_frames=NF, record_ramps=True)
        an = StreamingAnalyzer(2.0, 0.05, max_step_px=6.0, event_k=3.0)
        rep = run_control_loop(ad, an, combine_policies(
            recenter_on_brightest(), ramp_on_event()))
    finally:
        srv.join(timeout=10)
    off = np.sqrt(np.mean(np.sum(b.instrument.target_offset_px() ** 2, axis=1)))
    assert rep.n_frames == NF                    # frames over the IMG framing
    assert off < BK["scan_px"] / 8               # loop closed over the wire
    assert ad.pixel_size_nm == pytest.approx(2.0)   # CAPS handshake
    for fc in ad.force_curves:                   # FORCE round trip
        assert fc.max_force_pN > 0


def test_socket_bridge_exec_and_caps():
    b = BackendSimulator(**BK)
    srv = IgorListenerServer(b, mode="Contact").start()
    try:
        conn = SocketIgorBridge("127.0.0.1", srv.port, timeout=10)
        assert conn.execute("Approach()") == "OK"
        caps = conn.capabilities()
        assert caps.mode == "Contact" and caps.points == 48
        conn.close()
    finally:
        srv.join(timeout=10)
