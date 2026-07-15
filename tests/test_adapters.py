"""Vendor FrameSource adapter scaffolds (Nanosurf / JPK / Asylum).

Each adapter is exercised end-to-end against an in-process mock backend that
shares the reactive simulated physics, so a passing closed-loop test proves the
adapter's command translation and streaming are correct — only the real
transport (COM / socket / Igor bridge) remains to be verified against hardware.
The JPK socket path is additionally exercised over a real TCP loopback.
"""

import threading

import numpy as np
import pytest

from nanoindent import (
    StreamingAnalyzer, run_control_loop, recenter_on_brightest,
    combine_policies, ramp_on_event, Command,
)
from nanoindent.adapters import (
    BackendSimulator, InstrumentAdapter,
    NanosurfAdapter, MockNanosurfSPM,
    JPKAdapter, MockJPKConnection, SocketJPKConnection, serve_backend,
    handle_one_client,
    AsylumAdapter, MockIgorBridge,
)

BK = dict(scan_px=64, pixel_size_nm=2.0, frame_interval_s=0.05, n_particles=4,
          D_nm2_per_s=200.0, max_frames=80, seed=1)
NF = BK["max_frames"]


def _rms_offset(backend):
    o = backend.instrument.target_offset_px()
    return float(np.sqrt(np.mean(np.sum(o ** 2, axis=1))))


def _make(kind, backend):
    px, dt = backend.pixel_size_nm, backend.frame_interval_s
    if kind == "nanosurf":
        return NanosurfAdapter(MockNanosurfSPM(backend), pixel_size_nm=px,
                               frame_interval_s=dt, max_frames=NF)
    if kind == "jpk":
        return JPKAdapter(MockJPKConnection(backend), pixel_size_nm=px,
                          frame_interval_s=dt, max_frames=NF)
    if kind == "asylum":
        return AsylumAdapter(MockIgorBridge(backend), pixel_size_nm=px,
                             frame_interval_s=dt, max_frames=NF)
    raise ValueError(kind)


ALL = ["nanosurf", "jpk", "asylum"]


# --------------------------------------------------------------------------
# contract + streaming
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ALL)
def test_adapter_is_frame_source(kind):
    b = BackendSimulator(**BK)
    ad = _make(kind, b)
    assert isinstance(ad, InstrumentAdapter)
    assert ad.pixel_size_nm == b.pixel_size_nm
    assert ad.frame_interval_s == b.frame_interval_s
    frames = list(ad)
    assert len(frames) == NF
    assert frames[0].shape == (BK["scan_px"], BK["scan_px"])


@pytest.mark.parametrize("kind", ALL)
def test_adapter_closes_the_loop(kind):
    # closed loop keeps the tracked feature centred, just like the raw simulator
    b = BackendSimulator(**BK)
    ad = _make(kind, b)
    an = StreamingAnalyzer(ad.pixel_size_nm, ad.frame_interval_s, max_step_px=6.0)
    rep = run_control_loop(ad, an, recenter_on_brightest())
    assert rep.n_frames == NF
    assert len(ad.command_log) > 0
    assert _rms_offset(b) < BK["scan_px"] / 8       # feature held near centre


@pytest.mark.parametrize("kind", ALL)
def test_adapter_open_loop_drifts(kind):
    # same seed, no actuation -> the feature wanders far from centre
    b = BackendSimulator(**BK)
    ad = _make(kind, b)
    an = StreamingAnalyzer(ad.pixel_size_nm, ad.frame_interval_s, max_step_px=6.0)
    run_control_loop(ad, an, recenter_on_brightest(), apply=False)
    assert _rms_offset(b) > BK["scan_px"] / 6


# --------------------------------------------------------------------------
# command translation reaches the backend
# --------------------------------------------------------------------------
def test_nanosurf_translates_all_commands():
    b = BackendSimulator(**BK)
    spm = MockNanosurfSPM(b)
    ad = NanosurfAdapter(spm, pixel_size_nm=b.pixel_size_nm,
                         frame_interval_s=b.frame_interval_s, max_frames=NF)
    ad._last_shape = (64, 64)
    ad.send(Command("set_setpoint", {"value": 0.7}))
    assert spm.application.ZController.SetPoint == pytest.approx(0.7)
    ad.send(Command("recenter", {"dy_px": 3.0, "dx_px": -2.0}))
    assert spm.application.Scan.CenterPosY != 0.0     # centre moved (metres)
    ad.send(Command("trigger_ramp", {"y_px": 40.0, "x_px": 20.0}))
    assert len(b.instrument.ramp_events) == 1
    ad.send(Command("pause"))
    assert b.instrument.paused is True


def test_jpk_emits_protocol_verbs():
    b = BackendSimulator(**BK)
    conn = MockJPKConnection(b)
    ad = JPKAdapter(conn, pixel_size_nm=b.pixel_size_nm,
                    frame_interval_s=b.frame_interval_s, max_frames=NF)
    ad._last_shape = (64, 64)
    ad.send(Command("recenter", {"dy_px": 2.0, "dx_px": 1.0}))
    ad.send(Command("set_setpoint", {"value": 0.5}))
    ad.send(Command("pause"))
    assert any(s.startswith("MOVE-BY") for s in conn.sent)
    assert any(s.startswith("SET-SETPOINT") for s in conn.sent)
    assert "PAUSE" in conn.sent


def test_asylum_emits_igor_calls():
    b = BackendSimulator(**BK)
    br = MockIgorBridge(b)
    ad = AsylumAdapter(br, pixel_size_nm=b.pixel_size_nm,
                       frame_interval_s=b.frame_interval_s, max_frames=NF)
    ad._last_shape = (64, 64)
    ad.send(Command("recenter", {"dy_px": 2.0, "dx_px": 1.0}))
    ad.send(Command("set_setpoint", {"value": 0.5}))
    assert any(c.startswith("SetScanOffset") for c in br.history)
    assert any("SetpointDefl" in c for c in br.history)


def test_asylum_requires_bridge():
    ad = AsylumAdapter(None)
    with pytest.raises(RuntimeError):
        list(ad)


# --------------------------------------------------------------------------
# ramps logged consistently
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ALL)
def test_adapter_ramp_on_event(kind):
    b = BackendSimulator(**dict(BK, seed=5))
    ad = _make(kind, b)
    an = StreamingAnalyzer(ad.pixel_size_nm, ad.frame_interval_s, event_k=3.0)
    rep = run_control_loop(ad, an, combine_policies(
        recenter_on_brightest(), ramp_on_event()))
    n_ramp_cmds = sum(1 for c in ad.command_log if c.kind == "trigger_ramp")
    assert len(b.instrument.ramp_events) == n_ramp_cmds


# --------------------------------------------------------------------------
# context manager + disconnect
# --------------------------------------------------------------------------
def test_adapter_context_manager_disconnects():
    b = BackendSimulator(**BK)
    spm = MockNanosurfSPM(b)
    with NanosurfAdapter(spm, pixel_size_nm=b.pixel_size_nm,
                         frame_interval_s=b.frame_interval_s) as ad:
        assert ad._connected
    assert spm.connected is False


# --------------------------------------------------------------------------
# JPK over a real TCP loopback socket
# --------------------------------------------------------------------------
def test_jpk_socket_loopback_closes_loop():
    b = BackendSimulator(**BK)
    srv, port = serve_backend(b)
    t = threading.Thread(target=handle_one_client, args=(srv, b), daemon=True)
    t.start()
    try:
        conn = SocketJPKConnection("127.0.0.1", port, timeout=10.0)
        ad = JPKAdapter(conn, pixel_size_nm=b.pixel_size_nm,
                        frame_interval_s=b.frame_interval_s, max_frames=NF)
        an = StreamingAnalyzer(ad.pixel_size_nm, ad.frame_interval_s, max_step_px=6.0)
        rep = run_control_loop(ad, an, recenter_on_brightest())
    finally:
        t.join(timeout=10.0)
    assert rep.n_frames == NF
    assert _rms_offset(b) < BK["scan_px"] / 8         # loop closed over the wire
