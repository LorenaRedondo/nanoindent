"""Deepened JPK/Bruker adapter: handshake, streaming, force spectroscopy, control.

Exercised against both the in-process ``MockJPKConnection`` and the faithful
``JPKInstrumentServer`` over a real TCP loopback. Pull mode keeps the strictly-
ordered closed loop (tight feature-following); streaming mode is validated for
high-throughput data flow, capability negotiation, and event-triggered force
ramps (recentering carries frame latency under push streaming, by construction).
"""

import threading

import numpy as np
import pytest

from nanoindent import (
    StreamingAnalyzer, run_control_loop, recenter_on_brightest,
    combine_policies, ramp_on_event, Command,
)
from nanoindent.adapters import (
    BackendSimulator, JPKAdapter, MockJPKConnection, SocketJPKConnection,
    JPKInstrumentServer, JPKCapabilities, JPKForceRamp, JPKError,
)

BK = dict(scan_px=48, pixel_size_nm=2.0, frame_interval_s=0.05, n_particles=3,
          D_nm2_per_s=200.0, max_frames=60, seed=1)
NF = BK["max_frames"]


# --------------------------------------------------------------------------
# handshake / capabilities
# --------------------------------------------------------------------------
def test_mock_handshake_reports_capabilities():
    b = BackendSimulator(**BK)
    caps = MockJPKConnection(b).handshake()
    assert isinstance(caps, JPKCapabilities)
    assert caps.pixel_size_nm == pytest.approx(2.0)
    assert caps.frame_interval_s == pytest.approx(0.05)
    assert caps.pixels == 48 and caps.channels == ["height"]


def test_adapter_auto_calibrates_from_handshake():
    b = BackendSimulator(**BK)
    # constructor values are deliberately wrong; handshake should override them
    ad = JPKAdapter(MockJPKConnection(b), pixel_size_nm=99.0,
                    frame_interval_s=9.9, max_frames=NF, auto_calibrate=True)
    list(ad)                                   # triggers _connect + handshake
    assert ad.pixel_size_nm == pytest.approx(2.0)
    assert ad.frame_interval_s == pytest.approx(0.05)
    assert ad.capabilities is not None and ad.capabilities.pixels == 48


def test_auto_calibrate_can_be_disabled():
    b = BackendSimulator(**BK)
    ad = JPKAdapter(MockJPKConnection(b), pixel_size_nm=7.0,
                    frame_interval_s=0.2, max_frames=1, auto_calibrate=False)
    list(ad)
    assert ad.pixel_size_nm == pytest.approx(7.0)


# --------------------------------------------------------------------------
# force spectroscopy round trip
# --------------------------------------------------------------------------
def test_mock_force_ramp_returns_curve():
    b = BackendSimulator(**BK)
    fr = MockJPKConnection(b).force_ramp(3.0, -2.0)
    assert isinstance(fr, JPKForceRamp)
    assert fr.z_nm.shape == fr.force_pN.shape and fr.z_nm.size > 1
    assert fr.max_force_pN > 0
    assert np.all(np.diff(fr.z_nm) > 0)        # monotone approach


def test_record_ramps_on_event():
    b = BackendSimulator(**dict(BK, seed=5))
    ad = JPKAdapter(MockJPKConnection(b), max_frames=NF, record_ramps=True)
    an = StreamingAnalyzer(2.0, 0.05, event_k=3.0)
    rep = run_control_loop(ad, an, combine_policies(
        recenter_on_brightest(), ramp_on_event()))
    n_ramp_cmds = sum(1 for c in ad.command_log if c.kind == "trigger_ramp")
    # each ramp command produced a recorded F-d curve (not just a logged verb)
    assert len(ad.force_ramps) == n_ramp_cmds
    if ad.force_ramps:
        assert ad.force_ramps[0].max_force_pN > 0


def test_explicit_force_ramp_via_adapter():
    b = BackendSimulator(**BK)
    ad = JPKAdapter(MockJPKConnection(b), max_frames=1)
    fr = ad.force_ramp(10.0, 5.0)
    assert fr.max_force_pN > 0 and len(ad.force_ramps) == 1


# --------------------------------------------------------------------------
# direct instrument control
# --------------------------------------------------------------------------
def test_direct_control_commands_reach_instrument():
    b = BackendSimulator(**BK)
    conn = MockJPKConnection(b)
    ad = JPKAdapter(conn, max_frames=1)
    ad.engage(); ad.set_feedback_gains(0.5, 120.0); ad.set_line_rate(50.0)
    ad.set_setpoint(0.3); ad.move_to(100.0, 50.0); ad.withdraw()
    assert any(s.startswith("ENGAGE") for s in conn.sent)
    assert any(s.startswith("SET-GAINS 0.5 120") for s in conn.sent)
    assert any(s.startswith("MOVE-TO") for s in conn.sent)
    assert any(s.startswith("WITHDRAW") for s in conn.sent)


def test_command_kinds_engage_withdraw_gains():
    b = BackendSimulator(**BK)
    conn = MockJPKConnection(b)
    ad = JPKAdapter(conn, max_frames=1)
    ad._ensure_connected()
    ad._apply_command(Command("engage"))
    ad._apply_command(Command("set_gains", {"p": 1.0, "i": 200.0}))
    ad._apply_command(Command("withdraw"))
    assert "ENGAGE" in conn.sent and "WITHDRAW" in conn.sent
    assert any(s.startswith("SET-GAINS 1 200") for s in conn.sent)


# --------------------------------------------------------------------------
# pull mode over TCP: strictly-ordered closed loop still tight
# --------------------------------------------------------------------------
def test_pull_mode_closed_loop_tight():
    b = BackendSimulator(**BK)
    srv = JPKInstrumentServer(b).start()
    try:
        ad = JPKAdapter(SocketJPKConnection("127.0.0.1", srv.port), max_frames=NF)
        an = StreamingAnalyzer(2.0, 0.05, max_step_px=6.0)
        rep = run_control_loop(ad, an, recenter_on_brightest())
    finally:
        srv.join(timeout=10)
    off = np.sqrt(np.mean(np.sum(b.instrument.target_offset_px() ** 2, axis=1)))
    assert rep.n_frames == NF
    assert off < BK["scan_px"] / 8             # feature held near centre
    assert ad.capabilities is not None         # handshake happened over TCP


# --------------------------------------------------------------------------
# streaming mode over TCP: data flow + event-triggered ramps
# --------------------------------------------------------------------------
def test_streaming_mode_data_flow_and_ramps():
    b = BackendSimulator(**dict(BK, seed=4))
    srv = JPKInstrumentServer(b).start()
    try:
        conn = SocketJPKConnection("127.0.0.1", srv.port, timeout=10)
        ad = JPKAdapter(conn, max_frames=NF, stream=True, record_ramps=True)
        an = StreamingAnalyzer(2.0, 0.05, event_k=3.0)
        rep = run_control_loop(ad, an, combine_policies(
            recenter_on_brightest(), ramp_on_event()))
    finally:
        srv.join(timeout=10)
    # all pushed frames were received and analysed
    assert rep.n_frames == NF
    assert ad.pixel_size_nm == pytest.approx(2.0)   # handshake over the stream
    # any event ramps completed their round trip and returned real curves
    for fr in ad.force_ramps:
        assert fr.max_force_pN > 0


def test_ping_over_tcp():
    b = BackendSimulator(**BK)
    srv = JPKInstrumentServer(b).start()
    try:
        ad = JPKAdapter(SocketJPKConnection("127.0.0.1", srv.port), max_frames=1)
        ad._ensure_connected()
        assert ad.ping() is True
    finally:
        ad._disconnect()
        srv.join(timeout=10)


def test_socket_error_reply_raises():
    b = BackendSimulator(**BK)
    srv = JPKInstrumentServer(b).start()
    try:
        conn = SocketJPKConnection("127.0.0.1", srv.port, timeout=10)
        with pytest.raises(JPKError):
            conn.request("BOGUS-VERB 1 2")
        conn.close()
    finally:
        srv.join(timeout=10)
