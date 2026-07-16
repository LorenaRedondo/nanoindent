"""Deployable frame sources: directory watch + reconnecting wrapper."""

import os
import threading
import time

import numpy as np
import pytest

from nanoindent import (DirectoryWatchSource, RetryFrameSource, read_frame_file,
                        SimulatedInstrument, run_control_loop, StreamingAnalyzer,
                        recenter_on_brightest)
from nanoindent.streaming import Command
from nanoindent.video import synthetic_video


def _write_frames(d, n, size=32, prefix="frame_", ext="npy"):
    vid = synthetic_video(n_frames=n, size=size, seed=1)
    for i, f in enumerate(vid.frames):
        p = os.path.join(d, f"{prefix}{i:03d}.{ext}")
        if ext == "npy":
            np.save(p, f)
        elif ext == "csv":
            np.savetxt(p, f, delimiter=",")
    return vid


# --------------------------------------------------------------------------
# per-file readers
# --------------------------------------------------------------------------
def test_read_frame_file_npy_and_csv(tmp_path):
    _write_frames(str(tmp_path), 1, ext="npy")
    _write_frames(str(tmp_path), 1, ext="csv")
    fn = read_frame_file(str(tmp_path / "frame_000.npy"))
    fc = read_frame_file(str(tmp_path / "frame_000.csv"))
    assert fn.shape == (32, 32) and fc.shape == (32, 32)
    assert read_frame_file(str(tmp_path / "missing.npy")) is None


# --------------------------------------------------------------------------
# batch mode
# --------------------------------------------------------------------------
def test_directory_batch_reads_existing(tmp_path):
    _write_frames(str(tmp_path), 6)
    src = DirectoryWatchSource(str(tmp_path), pattern="frame_*.npy",
                               pixel_size_nm=2.0, frame_interval_s=0.05)
    frames = list(src)
    assert len(frames) == 6 and frames[0].shape == (32, 32)
    assert src.pixel_size_nm == 2.0


def test_directory_max_frames_caps(tmp_path):
    _write_frames(str(tmp_path), 10)
    src = DirectoryWatchSource(str(tmp_path), pattern="frame_*.npy", max_frames=4)
    assert len(list(src)) == 4


def test_directory_drives_control_loop(tmp_path):
    _write_frames(str(tmp_path), 12, size=48)
    src = DirectoryWatchSource(str(tmp_path), pattern="frame_*.npy",
                               pixel_size_nm=2.0, frame_interval_s=0.05)
    an = StreamingAnalyzer(2.0, 0.05)
    rep = run_control_loop(src, an, recenter_on_brightest())
    assert rep.n_frames == 12


def test_watch_requires_termination(tmp_path):
    with pytest.raises(ValueError):
        DirectoryWatchSource(str(tmp_path), watch=True)


def test_send_writes_command_files(tmp_path):
    _write_frames(str(tmp_path), 1)
    outbox = str(tmp_path / "cmds")
    src = DirectoryWatchSource(str(tmp_path), pattern="frame_*.npy",
                               command_dir=outbox)
    src.send(Command("recenter", {"dy_px": 1.0, "dx_px": 2.0}))
    files = os.listdir(outbox)
    assert len(files) == 1
    assert "recenter" in open(os.path.join(outbox, files[0])).read()


# --------------------------------------------------------------------------
# watch mode (new files appear)
# --------------------------------------------------------------------------
def test_watch_mode_picks_up_new_files(tmp_path):
    d = str(tmp_path)
    vid = synthetic_video(n_frames=6, size=32, seed=1)

    def writer():
        for i, f in enumerate(vid.frames):
            time.sleep(0.02)
            np.save(os.path.join(d, f"frame_{i:03d}.npy"), f)

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    src = DirectoryWatchSource(d, pattern="frame_*.npy", watch=True,
                               poll_interval_s=0.01, idle_timeout_s=0.5)
    frames = list(src)
    t.join(timeout=5)
    assert len(frames) == 6


# --------------------------------------------------------------------------
# reconnecting source
# --------------------------------------------------------------------------
def test_retry_source_reconnects_on_construct_failure():
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return SimulatedInstrument(scan_px=32, max_frames=5, seed=2)

    rs = RetryFrameSource(factory, retries=5)
    frames = list(rs)
    assert calls["n"] == 3 and len(frames) == 5


def test_retry_source_gives_up():
    def factory():
        raise ConnectionError("always down")
    with pytest.raises(RuntimeError):
        list(RetryFrameSource(factory, retries=2))
