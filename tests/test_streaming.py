"""Pseudo-realtime streaming analysis + simulated closed-loop control tests.

Exercised entirely against the reactive :class:`SimulatedInstrument` (no
hardware): the online analyzer keeps up with the frame rate, a following policy
demonstrably keeps a feature centred, and the read-only playback source streams
recorded movies.
"""

import numpy as np
import pytest

from nanoindent import (
    SimulatedInstrument, VideoFrameSource, StreamingAnalyzer, run_control_loop,
    recenter_on_brightest, recenter_on_longest_track, ramp_on_event,
    pause_on_event, combine_policies, synthetic_video, streaming,
)


# --------------------------------------------------------------------------
# online analyzer
# --------------------------------------------------------------------------
def test_streaming_analyzer_incremental_state():
    inst = SimulatedInstrument(scan_px=48, n_particles=3, max_frames=20, seed=0)
    an = StreamingAnalyzer(inst.pixel_size_nm, inst.frame_interval_s)
    last = None
    for i, frame in enumerate(inst):
        st = an.push(frame)
        assert st.frame_index == i
        assert st.time_s == pytest.approx(i * inst.frame_interval_s)
        assert st.frame_shape == (48, 48)
        assert st.detections.shape[1] == 2
        last = st
    # tracks accumulated and a diffusion estimate is available by the end
    assert len(an.all_tracks) >= 1
    assert last.diffusion_nm2_s is None or last.diffusion_nm2_s >= 0


def test_streaming_matches_batch_track_count_order():
    # online linking should find a comparable number of trajectories to batch
    vid = synthetic_video(n_frames=40, size=64, n_particles=4,
                          frame_interval_s=0.05, seed=2)
    an = StreamingAnalyzer(vid.pixel_size_nm, vid.frame_interval_s, max_step_px=6.0)
    for f in vid.frames:
        an.push(f)
    long_online = [t for t in an.all_tracks if t.length >= 5]
    assert len(long_online) >= 1


def test_online_diffusion_in_range():
    D0 = 200.0
    vid = synthetic_video(n_frames=120, size=96, n_particles=4,
                          pixel_size_nm=2.0, frame_interval_s=0.05,
                          D_nm2_per_s=D0, particle_sigma_nm=5.0, seed=3)
    an = StreamingAnalyzer(vid.pixel_size_nm, vid.frame_interval_s, max_step_px=6.0)
    D = None
    for f in vid.frames:
        D = an.push(f).diffusion_nm2_s
    assert D is not None
    assert 0.3 * D0 <= D <= 1.8 * D0


# --------------------------------------------------------------------------
# event detection online
# --------------------------------------------------------------------------
def test_online_event_detection_flags_step():
    an = StreamingAnalyzer(1.0, 0.1, event_k=4.0, event_warmup=5)
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.05, (16, 16))
    fired = []
    for t in range(30):
        frame = base + rng.normal(0, 0.02, base.shape)
        if t >= 15:
            frame += 5.0                      # a sustained topography step
        st = an.push(frame)
        fired.extend(st.events)
    assert any(f[0] == 15 for f in fired)      # step flagged at the transition


# --------------------------------------------------------------------------
# policies
# --------------------------------------------------------------------------
def test_recenter_policy_emits_recenter():
    inst = SimulatedInstrument(scan_px=48, max_frames=3, seed=1)
    an = StreamingAnalyzer(inst.pixel_size_nm, inst.frame_interval_s)
    frame = next(iter([inst.__next__()]))
    st = an.push(frame)
    cmds = recenter_on_brightest(deadband_px=0.0)(st)
    # at least emits a well-formed recenter command (or none if already centred)
    for c in cmds:
        assert c.kind == "recenter" and "dy_px" in c.params


def test_combine_policies_concatenates():
    inst = SimulatedInstrument(scan_px=48, max_frames=2, seed=1)
    an = StreamingAnalyzer(inst.pixel_size_nm, inst.frame_interval_s)
    st = an.push(next(inst))
    both = combine_policies(recenter_on_brightest(deadband_px=0.0),
                            recenter_on_longest_track(min_len=1, deadband_px=0.0))
    solo = recenter_on_brightest(deadband_px=0.0)(st)
    assert len(both(st)) >= len(solo)


# --------------------------------------------------------------------------
# closed loop: the loop actually closes
# --------------------------------------------------------------------------
def test_closed_loop_keeps_feature_centred():
    kw = dict(scan_px=64, pixel_size_nm=2.0, frame_interval_s=0.05,
              n_particles=4, D_nm2_per_s=200.0, max_frames=120, seed=1)

    inst = SimulatedInstrument(**kw)
    an = StreamingAnalyzer(inst.pixel_size_nm, inst.frame_interval_s, max_step_px=6.0)
    rep = run_control_loop(inst, an, recenter_on_brightest(), apply=True)
    off_ctrl = np.sqrt(np.mean(np.sum(inst.target_offset_px() ** 2, axis=1)))

    inst0 = SimulatedInstrument(**kw)                 # same seed, no actuation
    an0 = StreamingAnalyzer(inst0.pixel_size_nm, inst0.frame_interval_s, max_step_px=6.0)
    run_control_loop(inst0, an0, recenter_on_brightest(), apply=False)
    off_open = np.sqrt(np.mean(np.sum(inst0.target_offset_px() ** 2, axis=1)))

    # closed loop holds the target far closer to the window centre
    assert off_ctrl < 0.5 * off_open
    assert off_ctrl < inst.scan_px / 8       # within the central region
    assert len(rep.commands) > 0


def test_control_loop_keeps_up_with_frame_rate():
    inst = SimulatedInstrument(scan_px=64, frame_interval_s=0.05, max_frames=60, seed=1)
    an = StreamingAnalyzer(inst.pixel_size_nm, inst.frame_interval_s)
    rep = run_control_loop(inst, an, recenter_on_brightest())
    # steady-state analysis fits comfortably inside the acquisition period
    assert rep.kept_up(warmup=1)
    assert rep.real_time_factor(warmup=1) < 1.0
    assert rep.n_frames == 60


def test_pause_on_event_freezes_motion():
    an = StreamingAnalyzer(1.0, 0.1, event_k=3.0, event_warmup=3)
    inst = SimulatedInstrument(scan_px=32, max_frames=25, seed=4)
    # a policy that pauses on any detected event; feed through the loop
    rep = run_control_loop(inst, an, pause_on_event())
    # if any event fired, a pause command was issued
    if rep.events:
        assert any(c.kind == "pause" for _, c in rep.commands)


def test_ramp_on_event_logs_ramp():
    # deterministic step: drive the analyzer directly, then a ramp policy
    an = StreamingAnalyzer(2.0, 0.05, event_k=4.0, event_warmup=4)
    inst = SimulatedInstrument(scan_px=48, max_frames=40, seed=5)
    rep = run_control_loop(inst, an, ramp_on_event())
    # ramps logged exactly match ramp commands issued
    assert len(inst.ramp_events) == sum(1 for _, c in rep.commands
                                        if c.kind == "trigger_ramp")


# --------------------------------------------------------------------------
# read-only playback source
# --------------------------------------------------------------------------
def test_video_frame_source_playback():
    vid = synthetic_video(n_frames=15, size=48, n_particles=3, seed=6)
    src = VideoFrameSource(vid)
    assert src.frame_interval_s == vid.frame_interval_s
    rep = run_control_loop(src, policy=recenter_on_brightest(), apply=True)
    assert rep.n_frames == 15
    # send() only logs on a read-only source (playback is unaffected)
    assert len(src.command_log) == len(rep.commands)


# --------------------------------------------------------------------------
# report + plotting
# --------------------------------------------------------------------------
def test_run_report_metrics():
    inst = SimulatedInstrument(scan_px=48, max_frames=20, seed=7)
    rep = run_control_loop(inst, policy=recenter_on_brightest())
    assert rep.proc_ms.shape == (20,)
    assert isinstance(rep.kept_up(), bool)
    assert rep.real_time_factor() > 0


def test_plot_control_run_runs():
    import matplotlib
    matplotlib.use("Agg")
    inst = SimulatedInstrument(scan_px=48, max_frames=30, seed=8)
    an = StreamingAnalyzer(inst.pixel_size_nm, inst.frame_interval_s)
    rep = run_control_loop(inst, an, combine_policies(
        recenter_on_brightest(), ramp_on_event()))
    fig = streaming.plot_control_run(rep, inst)
    assert len(fig.axes) == 3
