"""Safety / robustness envelope."""

import time

import numpy as np
import pytest

from nanoindent import (SafetyEnvelope, SafetyLimits, StreamingAnalyzer,
                        run_control_loop, recenter_on_brightest, combine_policies,
                        ramp_on_event, SimulatedInstrument)
from nanoindent.streaming import AnalysisState, Command


def _state(frame_index=0, n_particles=1, detections=None):
    det = np.array([[10.0, 10.0]]) if detections is None else detections
    return AnalysisState(
        frame_index=frame_index, time_s=frame_index * 0.05, frame_shape=(48, 48),
        pixel_size_nm=2.0, detections=det,
        detection_heights=np.ones(len(det)) if len(det) else np.empty(0),
        tracks=[], mean_height_nm=0.0, rms_roughness_nm=0.0, coverage=0.0,
        drift_px=(0.0, 0.0), events=[], diffusion_nm2_s=None)


# --------------------------------------------------------------------------
# rate limiting
# --------------------------------------------------------------------------
def test_rate_limit_throttles_commands():
    env = SafetyEnvelope(SafetyLimits(min_frames_between_commands=3))
    always = lambda s: [Command("recenter", {"dy_px": 1.0, "dx_px": 0.0})]
    guarded = env.guard(always)
    emitted = [len(guarded(_state(frame_index=i))) for i in range(6)]
    # frames 0,3 pass; 1,2,4,5 throttled
    assert emitted == [1, 0, 0, 1, 0, 0]
    assert env.counts["throttled"] == 4


# --------------------------------------------------------------------------
# setpoint clamp
# --------------------------------------------------------------------------
def test_setpoint_clamped_to_range():
    env = SafetyEnvelope(SafetyLimits(setpoint_range=(0.2, 0.8)))
    guarded = env.guard(lambda s: [Command("set_setpoint", {"value": 5.0})])
    out = guarded(_state())
    assert out[0].params["value"] == pytest.approx(0.8)
    assert env.counts["clamped_setpoint"] == 1


# --------------------------------------------------------------------------
# scan-bounds clip
# --------------------------------------------------------------------------
def test_scan_bounds_clip_accumulated_offset():
    env = SafetyEnvelope(SafetyLimits(max_scan_offset_px=5.0))
    guarded = env.guard(lambda s: [Command("recenter", {"dy_px": 10.0, "dx_px": 0.0})])
    out = guarded(_state(frame_index=0))
    # first move clipped to the 5 px bound
    assert out[0].params["dy_px"] == pytest.approx(5.0)
    # subsequent moves in the same direction are clipped to zero (already at bound)
    out2 = guarded(_state(frame_index=1))
    assert out2[0].params["dy_px"] == pytest.approx(0.0, abs=1e-9)
    assert env.counts["clipped_bounds"] >= 1


# --------------------------------------------------------------------------
# auto-pause on lost tracking
# --------------------------------------------------------------------------
def test_auto_pause_and_resume_on_lost_tracking():
    env = SafetyEnvelope(SafetyLimits(pause_after_lost_frames=3))
    move = lambda s: [Command("recenter", {"dy_px": 1.0, "dx_px": 0.0})]
    guarded = env.guard(move)
    # 3 empty frames -> pause on the 3rd; normal command suppressed while paused
    for i in range(2):
        assert not env.paused and guarded(_state(frame_index=i, n_particles=1, detections=np.empty((0, 2))))[0].kind == "recenter"
    out = guarded(_state(frame_index=2, n_particles=1, detections=np.empty((0, 2))))
    assert env.paused and any(c.kind == "pause" for c in out)
    # while paused, no recenter is emitted
    assert all(c.kind != "recenter" for c in guarded(_state(frame_index=3, detections=np.empty((0, 2)))))
    # tracking returns -> resume
    out = guarded(_state(frame_index=4, n_particles=1))
    assert not env.paused and any(c.kind == "resume" for c in out)
    assert env.counts["auto_pauses"] == 1 and env.counts["auto_resumes"] == 1


# --------------------------------------------------------------------------
# frame-gap watchdog
# --------------------------------------------------------------------------
def test_watchdog_pauses_on_frame_gap():
    env = SafetyEnvelope(SafetyLimits(max_frame_gap_s=0.05))
    guarded = env.guard(lambda s: [])
    guarded(_state(frame_index=0))          # establishes the clock
    time.sleep(0.12)                        # a stall longer than the limit
    out = guarded(_state(frame_index=1))
    assert env.counts["watchdog_trips"] == 1 and any(c.kind == "pause" for c in out)


# --------------------------------------------------------------------------
# end-to-end through the control loop
# --------------------------------------------------------------------------
def test_guarded_policy_runs_in_loop():
    inst = SimulatedInstrument(scan_px=48, max_frames=40, n_particles=3, seed=1)
    an = StreamingAnalyzer(inst.pixel_size_nm, inst.frame_interval_s, event_k=3.0)
    env = SafetyEnvelope(SafetyLimits(min_frames_between_commands=2,
                                      max_scan_offset_px=12.0,
                                      setpoint_range=(0.1, 1.0)))
    policy = env.guard(combine_policies(recenter_on_brightest(), ramp_on_event()))
    rep = run_control_loop(inst, an, policy)
    assert rep.n_frames == 40
    # the rate limiter and bounds guard actually fired
    assert env.counts["throttled"] > 0
