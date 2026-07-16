"""Run recorder + config-driven control CLI."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from nanoindent import (RunRecorder, StreamingAnalyzer, run_control_loop,
                        recenter_on_brightest, combine_policies, ramp_on_event,
                        SimulatedInstrument)
from nanoindent.control import run_from_config, build_policy, build_safety


# --------------------------------------------------------------------------
# recorder
# --------------------------------------------------------------------------
def test_recorder_writes_all_artifacts(tmp_path):
    inst = SimulatedInstrument(scan_px=48, max_frames=30, n_particles=3, seed=1)
    an = StreamingAnalyzer(inst.pixel_size_nm, inst.frame_interval_s, event_k=3.0)
    rec = RunRecorder(str(tmp_path / "run"), save_frames=True)
    rep = run_control_loop(rec.wrap(inst), an,
                           combine_policies(recenter_on_brightest(), ramp_on_event()),
                           on_state=rec.on_state)
    manifest = rec.save(rep, meta={"instrument": "sim"})

    out = str(tmp_path / "run")
    for f in ("stats.csv", "commands.jsonl", "events.jsonl", "timings.csv",
              "frames.npz", "manifest.json"):
        assert os.path.exists(os.path.join(out, f))
    # stats has one row per frame (+ header)
    assert len(open(os.path.join(out, "stats.csv")).read().splitlines()) == 31
    # commands.jsonl parses and matches the report
    cmds = [json.loads(l) for l in open(os.path.join(out, "commands.jsonl"))]
    assert len(cmds) == len(rep.commands)
    assert manifest["counts"]["frames"] == 30
    frames = np.load(os.path.join(out, "frames.npz"))["frames"]
    assert frames.shape == (30, 48, 48)


def test_recorder_without_frames(tmp_path):
    inst = SimulatedInstrument(scan_px=32, max_frames=10, seed=1)
    an = StreamingAnalyzer(inst.pixel_size_nm, inst.frame_interval_s)
    rec = RunRecorder(str(tmp_path / "r"))
    rep = run_control_loop(inst, an, recenter_on_brightest(), on_state=rec.on_state)
    rec.save(rep)
    assert not os.path.exists(str(tmp_path / "r" / "frames.npz"))
    assert os.path.exists(str(tmp_path / "r" / "stats.csv"))


# --------------------------------------------------------------------------
# config builders
# --------------------------------------------------------------------------
def test_build_policy_and_safety():
    p = build_policy(["recenter_on_brightest", "ramp_on_event"])
    assert callable(p)
    with pytest.raises(ValueError):
        build_policy(["nonsense"])
    env = build_safety({"max_scan_offset_px": 10.0, "pause_after_lost_frames": 5})
    assert env is not None and env.limits.max_scan_offset_px == 10.0
    assert build_safety({}) is None


def test_run_from_config_simulated(tmp_path):
    manifest = run_from_config({
        "source": {"type": "simulated", "frames": 25, "scan_px": 48},
        "policy": ["recenter_on_brightest", "ramp_on_event"],
        "outdir": str(tmp_path / "cfg"),
        "safety": {"max_scan_offset_px": 20.0, "pause_after_lost_frames": 6},
    })
    assert manifest["counts"]["frames"] == 25
    assert "safety_counts" in manifest["meta"]
    assert os.path.exists(str(tmp_path / "cfg" / "manifest.json"))


def test_run_from_config_directory(tmp_path):
    from nanoindent.video import synthetic_video
    d = tmp_path / "frames"
    d.mkdir()
    vid = synthetic_video(n_frames=8, size=32, seed=1)
    for i, f in enumerate(vid.frames):
        np.save(str(d / f"f_{i:03d}.npy"), f)
    manifest = run_from_config({
        "source": {"type": "directory", "directory": str(d), "pattern": "f_*.npy",
                   "pixel_size_nm": 2.0, "frame_interval_s": 0.05},
        "policy": ["recenter_on_brightest"], "outdir": str(tmp_path / "out"),
    })
    assert manifest["counts"]["frames"] == 8


# --------------------------------------------------------------------------
# CLI entry point (subprocess)
# --------------------------------------------------------------------------
def test_cli_runs_simulated(tmp_path):
    out = str(tmp_path / "clirun")
    r = subprocess.run(
        [sys.executable, "-m", "nanoindent.control", "--source", "simulated",
         "--frames", "20", "--policy", "recenter_on_brightest,ramp_on_event",
         "--outdir", out, "--max-scan-offset-px", "18", "--pause-after-lost", "5"],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    manifest = json.loads(r.stdout)
    assert manifest["counts"]["frames"] == 20
    assert os.path.exists(os.path.join(out, "stats.csv"))
