"""Deepened Nanosurf adapter: capability read-back, approach/engage, feedback,
operating mode, and force–distance spectroscopy.

All against the in-process ``MockNanosurfSPM``, which mirrors the real
``nanosurf.SPM()`` object tree (application.Scan / ZController / OperatingMode /
Approach / Spec), so the exact calls the adapter would issue on hardware are
exercised.
"""

import numpy as np
import pytest

from nanoindent import (
    StreamingAnalyzer, run_control_loop, recenter_on_brightest,
    combine_policies, ramp_on_event, Command,
)
from nanoindent.adapters import (
    BackendSimulator, NanosurfAdapter, MockNanosurfSPM, NanosurfCapabilities,
    NanosurfForceCurve,
)

BK = dict(scan_px=48, pixel_size_nm=2.0, frame_interval_s=0.05, n_particles=3,
          D_nm2_per_s=200.0, max_frames=60, seed=1)
NF = BK["max_frames"]


# --------------------------------------------------------------------------
# capabilities / auto-calibration
# --------------------------------------------------------------------------
def test_reads_capabilities_from_geometry():
    b = BackendSimulator(**BK)
    ad = NanosurfAdapter(MockNanosurfSPM(b, model="DriveAFM"),
                         pixel_size_nm=99.0, frame_interval_s=9.9, max_frames=NF)
    list(ad)                                        # connect + read capabilities
    caps = ad.capabilities
    assert isinstance(caps, NanosurfCapabilities)
    assert caps.model == "DriveAFM"
    assert caps.pixel_size_nm == pytest.approx(2.0)     # ImageWidth / Points
    assert caps.frame_interval_s == pytest.approx(0.05)  # Scantime / Lines
    assert caps.points == 48 and caps.operating_mode == "DynamicAFM"
    # auto-calibrated, overriding the wrong constructor values
    assert ad.pixel_size_nm == pytest.approx(2.0)


def test_auto_calibrate_can_be_disabled():
    b = BackendSimulator(**BK)
    ad = NanosurfAdapter(MockNanosurfSPM(b), pixel_size_nm=7.0,
                         frame_interval_s=0.2, max_frames=1, auto_calibrate=False)
    list(ad)
    assert ad.pixel_size_nm == pytest.approx(7.0)


# --------------------------------------------------------------------------
# approach / engage state machine
# --------------------------------------------------------------------------
def test_engage_withdraw_drive_approach_status():
    b = BackendSimulator(**BK)
    spm = MockNanosurfSPM(b)
    ad = NanosurfAdapter(spm, max_frames=1)
    ad._ensure_connected()
    assert ad.approach_status == "Idle"
    ad.engage()
    assert ad.approach_status == "Engaged" == spm.application.Approach.Status
    ad.withdraw()
    assert ad.approach_status == "Retracted"


def test_engage_withdraw_via_command_kinds():
    b = BackendSimulator(**BK)
    spm = MockNanosurfSPM(b)
    ad = NanosurfAdapter(spm, max_frames=1)
    ad._ensure_connected()
    ad._apply_command(Command("engage"))
    assert spm.application.Approach.Status == "Engaged"
    ad._apply_command(Command("withdraw"))
    assert spm.application.Approach.Status == "Retracted"


# --------------------------------------------------------------------------
# feedback + operating mode
# --------------------------------------------------------------------------
def test_setpoint_and_gains_reach_zcontroller():
    b = BackendSimulator(**BK)
    spm = MockNanosurfSPM(b)
    ad = NanosurfAdapter(spm, max_frames=1)
    ad.set_setpoint(0.3)
    ad.set_feedback_gains(0.5, 120.0)
    zc = spm.application.ZController
    assert zc.SetPoint == pytest.approx(0.3)
    assert zc.PGain == pytest.approx(0.5) and zc.IGain == pytest.approx(120.0)
    ad._apply_command(Command("set_gains", {"p": 1.0, "i": 200.0}))
    assert zc.PGain == pytest.approx(1.0)


def test_set_operating_mode():
    b = BackendSimulator(**BK)
    spm = MockNanosurfSPM(b)
    ad = NanosurfAdapter(spm, max_frames=1)
    ad.set_operating_mode("StaticAFM")
    assert spm.application.OperatingMode.OperatingMode == "StaticAFM"


# --------------------------------------------------------------------------
# scan centre in metres round-trips to a window move
# --------------------------------------------------------------------------
def test_recenter_sets_scan_centre_metres():
    b = BackendSimulator(**BK)
    spm = MockNanosurfSPM(b)
    ad = NanosurfAdapter(spm, max_frames=1)
    ad._ensure_connected(); ad._last_shape = (48, 48)
    ad._apply_command(Command("recenter", {"dy_px": 3.0, "dx_px": -2.0}))
    # 3 px * 2 nm/px = 6 nm -> 6e-9 m
    assert spm.application.Scan.CenterPosY == pytest.approx(6e-9)
    assert spm.application.Scan.CenterPosX == pytest.approx(-4e-9)


# --------------------------------------------------------------------------
# force–distance spectroscopy
# --------------------------------------------------------------------------
def test_force_curve_returns_curve():
    b = BackendSimulator(**BK)
    ad = NanosurfAdapter(MockNanosurfSPM(b), max_frames=1)
    fc = ad.force_curve(10.0, 5.0)
    assert isinstance(fc, NanosurfForceCurve)
    assert fc.z_nm.shape == fc.force_pN.shape and fc.max_force_pN > 0
    assert np.all(np.diff(fc.z_nm) > 0) and len(ad.force_curves) == 1


def test_record_ramps_on_event():
    b = BackendSimulator(**dict(BK, seed=5))
    ad = NanosurfAdapter(MockNanosurfSPM(b), max_frames=NF, record_ramps=True)
    an = StreamingAnalyzer(2.0, 0.05, event_k=3.0)
    rep = run_control_loop(ad, an, combine_policies(
        recenter_on_brightest(), ramp_on_event()))
    n_ramp_cmds = sum(1 for c in ad.command_log if c.kind == "trigger_ramp")
    assert len(ad.force_curves) == n_ramp_cmds
    for fc in ad.force_curves:
        assert fc.max_force_pN > 0


# --------------------------------------------------------------------------
# closed loop + lifecycle
# --------------------------------------------------------------------------
def test_closed_loop_tight_following():
    b = BackendSimulator(**BK)
    ad = NanosurfAdapter(MockNanosurfSPM(b), max_frames=NF)
    an = StreamingAnalyzer(2.0, 0.05, max_step_px=6.0)
    rep = run_control_loop(ad, an, recenter_on_brightest())
    off = np.sqrt(np.mean(np.sum(b.instrument.target_offset_px() ** 2, axis=1)))
    assert rep.n_frames == NF
    assert off < BK["scan_px"] / 8


def test_session_context_manager():
    b = BackendSimulator(**BK)
    spm = MockNanosurfSPM(b)
    with NanosurfAdapter(spm, max_frames=NF) as ad:
        assert ad._connected and spm.connected
        ad.engage()
        run_control_loop(ad, StreamingAnalyzer(2.0, 0.05), recenter_on_brightest())
        ad.withdraw()
    assert spm.connected is False                   # disconnected on exit
