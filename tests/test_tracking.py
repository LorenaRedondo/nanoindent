"""Experiment tracking: timestamps, MLflow-shaped manifests, index, MLflow log."""

import os
import sys
import types

import numpy as np
import pytest

from nanoindent import (RunRecorder, StreamingAnalyzer, run_control_loop,
                        recenter_on_brightest, combine_policies, ramp_on_event,
                        SimulatedInstrument, DirectoryWatchSource, ExperimentIndex,
                        load_manifest, make_run_id)
from nanoindent.sources import read_frame_timestamp
from nanoindent.control import run_from_config
from nanoindent.video import synthetic_video


def _write_frames(d, n, ext="npy"):
    vid = synthetic_video(n_frames=n, size=32, seed=1)
    for i, f in enumerate(vid.frames):
        np.save(os.path.join(d, f"frame_{i:03d}.{ext}"), f)
    return vid


# --------------------------------------------------------------------------
# timestamps
# --------------------------------------------------------------------------
def test_run_id_sortable_and_unique():
    a, b = make_run_id(), make_run_id()
    assert a != b and len(a.split("_")) == 2


def test_read_frame_timestamp_mtime(tmp_path):
    _write_frames(str(tmp_path), 1)
    ts = read_frame_timestamp(str(tmp_path / "frame_000.npy"), "mtime")
    assert isinstance(ts, float) and ts > 0
    assert read_frame_timestamp(str(tmp_path / "frame_000.npy"), "none") is None


def test_directory_source_captures_frame_times(tmp_path):
    _write_frames(str(tmp_path), 5)
    src = DirectoryWatchSource(str(tmp_path), pattern="frame_*.npy",
                               timestamp="mtime")
    frames = list(src)
    assert len(src.frame_times) == 5 == len(src.frame_paths)
    assert all(isinstance(t, float) for t in src.frame_times)


def test_recorder_stats_has_timestamps(tmp_path):
    _write_frames(str(tmp_path / "f"), 6) if (tmp_path / "f").mkdir() or True else None
    src = DirectoryWatchSource(str(tmp_path / "f"), pattern="frame_*.npy",
                               pixel_size_nm=2.0, frame_interval_s=0.05)
    rec = RunRecorder(str(tmp_path / "run"))
    an = StreamingAnalyzer(2.0, 0.05)
    rep = run_control_loop(rec.wrap(src), an, recenter_on_brightest(),
                           on_state=rec.on_state)
    rec.save(rep)
    header = open(str(tmp_path / "run" / "stats.csv")).readline().strip().split(",")
    assert "wall_time_s" in header and "acq_time_s" in header
    # the acquisition timestamps (file mtimes) made it into the CSV
    rows = open(str(tmp_path / "run" / "stats.csv")).read().splitlines()[1:]
    acq_idx = header.index("acq_time_s")
    assert all(r.split(",")[acq_idx] for r in rows)


# --------------------------------------------------------------------------
# MLflow-shaped manifest
# --------------------------------------------------------------------------
def test_manifest_has_provenance(tmp_path):
    inst = SimulatedInstrument(scan_px=32, max_frames=15, seed=1)
    an = StreamingAnalyzer(inst.pixel_size_nm, inst.frame_interval_s)
    rec = RunRecorder(str(tmp_path / "run"))
    rep = run_control_loop(inst, an, recenter_on_brightest(), on_state=rec.on_state)
    manifest = rec.save(rep, params={"R_nm": 20}, metrics={"custom": 1.5},
                        tags={"sample": "SLB"})
    for key in ("run_id", "started_at", "ended_at", "duration_s", "params",
                "metrics", "tags"):
        assert key in manifest
    assert manifest["params"]["R_nm"] == 20
    assert manifest["metrics"]["custom"] == 1.5
    assert manifest["metrics"]["n_frames"] == 15      # auto metric
    assert manifest["tags"]["sample"] == "SLB"


def test_run_from_config_records_params_and_tags(tmp_path):
    manifest = run_from_config({
        "source": {"type": "simulated", "frames": 20, "scan_px": 48, "seed": 1},
        "policy": ["recenter_on_brightest"], "outdir": str(tmp_path / "r"),
        "safety": {"max_scan_offset_px": 15.0},
        "tags": {"sample": "SLB", "operator": "afm"},
    })
    assert manifest["params"]["source_type"] == "simulated"
    assert manifest["params"]["safety_max_scan_offset_px"] == 15.0
    assert manifest["tags"]["sample"] == "SLB"
    assert "guard_clipped_bounds" in manifest["metrics"]


# --------------------------------------------------------------------------
# experiment index
# --------------------------------------------------------------------------
def _make_runs(root, n=3):
    for s in range(n):
        run_from_config({
            "source": {"type": "simulated", "frames": 15, "scan_px": 48, "seed": s},
            "policy": ["recenter_on_brightest"],
            "outdir": os.path.join(root, f"run_{s}"),
            "tags": {"batch": "A"}})


def test_experiment_index_and_csv(tmp_path):
    root = str(tmp_path / "runs")
    _make_runs(root, 3)
    idx = ExperimentIndex(root)
    assert len(idx) == 3
    rows = idx.rows()
    assert all("run_id" in r and "metrics.n_frames" in r for r in rows)
    csv = str(tmp_path / "cmp.csv")
    idx.to_csv(csv)
    lines = open(csv).read().splitlines()
    assert len(lines) == 4                            # header + 3 runs
    assert "params.source_type" in lines[0]


def test_experiment_index_best(tmp_path):
    root = str(tmp_path / "runs")
    _make_runs(root, 3)
    idx = ExperimentIndex(root)
    best = idx.best("n_frames", maximize=True)
    assert best is not None and best["metrics.n_frames"] == 15


def test_index_empty_directory(tmp_path):
    idx = ExperimentIndex(str(tmp_path))
    assert len(idx) == 0 and idx.rows() == []


# --------------------------------------------------------------------------
# optional MLflow export (stubbed — no real mlflow needed)
# --------------------------------------------------------------------------
def test_log_run_to_mlflow_with_stub(tmp_path, monkeypatch):
    inst = SimulatedInstrument(scan_px=32, max_frames=12, seed=1)
    an = StreamingAnalyzer(inst.pixel_size_nm, inst.frame_interval_s)
    rec = RunRecorder(str(tmp_path / "run"))
    rep = run_control_loop(inst, an, recenter_on_brightest(), on_state=rec.on_state)
    manifest = rec.save(rep, params={"R_nm": 20}, metrics={"custom": 2.0},
                        tags={"s": "x"})

    captured = {}
    stub = types.ModuleType("mlflow")

    class _Run:
        class info:
            run_id = "stub"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    stub.set_tracking_uri = lambda u: captured.__setitem__("uri", u)
    stub.set_experiment = lambda e: captured.__setitem__("exp", e)
    stub.start_run = lambda run_name=None: _Run()
    stub.log_params = lambda p: captured.__setitem__("params", p)
    stub.log_metrics = lambda m: captured.__setitem__("metrics", m)
    stub.set_tags = lambda t: captured.__setitem__("tags", t)
    stub.log_artifacts = lambda d: captured.__setitem__("artifacts", d)
    monkeypatch.setitem(sys.modules, "mlflow", stub)

    from nanoindent.tracking import log_run_to_mlflow
    rid = log_run_to_mlflow(manifest, experiment="afm")
    assert rid == "stub"
    assert captured["exp"] == "afm"
    assert captured["params"]["R_nm"] == 20
    assert captured["metrics"]["custom"] == 2.0
    assert captured["tags"]["s"] == "x"
    assert captured["artifacts"] == manifest["outdir"]


def test_log_run_to_mlflow_missing_dependency(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)   # simulate not installed
    with pytest.raises(ImportError):
        from nanoindent.tracking import log_run_to_mlflow
        log_run_to_mlflow({"run_id": "x", "params": {}, "metrics": {}, "tags": {}})
