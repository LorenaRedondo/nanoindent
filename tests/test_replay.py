"""Replay a recorded run from disk."""

import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from nanoindent import load_run, RecordedRun
from nanoindent.control import run_from_config


def _record(tmp_path, save_frames=True, seed=1):
    return run_from_config({
        "source": {"type": "simulated", "frames": 30, "scan_px": 48, "seed": seed},
        "policy": ["recenter_on_brightest", "ramp_on_event"],
        "safety": {"max_scan_offset_px": 20.0},
        "save_frames": save_frames, "outdir": str(tmp_path / "run"),
        "tags": {"sample": "SLB"},
    })


def test_load_run_reconstructs_everything(tmp_path):
    _record(tmp_path)
    run = load_run(str(tmp_path / "run"))
    assert isinstance(run, RecordedRun)
    assert run.n_frames == 30
    assert run.frames is not None and run.frames.shape == (30, 48, 48)
    assert len(run.commands) > 0
    assert run.params["source_type"] == "simulated"
    assert run.metrics["n_frames"] == 30
    assert run.tags["sample"] == "SLB"
    # stats columns coerced to numbers
    assert isinstance(run.stats["frame"][0], int)
    assert run.stats["acq_time_s"] == run.stats["acq_time_s"]   # present


def test_plot_summary_runs(tmp_path):
    _record(tmp_path)
    fig = load_run(str(tmp_path / "run")).plot_summary()
    assert len(fig.axes) == 4


def test_to_video_and_animate(tmp_path):
    _record(tmp_path)
    run = load_run(str(tmp_path / "run"))
    vid = run.to_video()
    assert vid.shape == (30, 48, 48) and vid.pixel_size_nm == pytest.approx(2.0)
    out = run.animate(str(tmp_path / "replay.gif"), fps=8)
    assert out.endswith(".gif") and os.path.getsize(out) > 0


def test_to_video_without_frames_raises(tmp_path):
    _record(tmp_path, save_frames=False)
    run = load_run(str(tmp_path / "run"))
    assert run.frames is None
    with pytest.raises(ValueError):
        run.to_video()
    # summary still works without frames
    assert len(run.plot_summary().axes) == 4
