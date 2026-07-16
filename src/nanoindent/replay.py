"""Replay a recorded control run from a :class:`~nanoindent.recording.RunRecorder`
directory.

:func:`load_run` reconstructs a :class:`RecordedRun` from the artefacts on disk
(``manifest.json``, ``stats.csv``, ``commands.jsonl``, ``events.jsonl``,
``timings.csv``, and ``frames.npz`` if saved), so a finished session can be
re-examined, re-plotted, re-analysed, or animated — the read side of the recorder.

    from nanoindent.replay import load_run
    run = load_run("runs/session1")
    run.plot_summary()                 # per-frame stats, commands, events, timings
    vid = run.to_video()               # frames back as an AFMVideo for re-analysis
    run.animate("runs/session1/replay.mp4")
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Dict, List, Optional

import numpy as np

from .tracking import load_manifest


def _read_csv(path: str) -> Dict[str, list]:
    """Read a CSV into ``{column: [values]}`` with numeric coercion."""
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        lines = fh.read().splitlines()
    if not lines:
        return {}
    cols = lines[0].split(",")
    out: Dict[str, list] = {c: [] for c in cols}
    for line in lines[1:]:
        parts = line.split(",")
        for c, v in zip(cols, parts):
            out[c].append(_num(v))
    return out


def _num(v: str):
    if v == "":
        return None
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f
    except ValueError:
        return v


def _read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


@dataclass
class RecordedRun:
    """A recorded run reconstructed from disk."""

    outdir: str
    manifest: dict
    stats: Dict[str, list]
    commands: List[dict]
    events: List[dict]
    timings: Dict[str, list]
    frames: Optional[np.ndarray]

    # -- convenience accessors --------------------------------------------
    @property
    def run_id(self) -> str:
        return self.manifest.get("run_id", "")

    @property
    def params(self) -> dict:
        return self.manifest.get("params", {})

    @property
    def metrics(self) -> dict:
        return self.manifest.get("metrics", {})

    @property
    def tags(self) -> dict:
        return self.manifest.get("tags", {})

    @property
    def n_frames(self) -> int:
        return len(self.stats.get("frame", []))

    @property
    def pixel_size_nm(self) -> float:
        return float(self.params.get("pixel_size_nm") or 1.0)

    @property
    def frame_interval_s(self) -> float:
        return float(self.params.get("frame_interval_s") or 0.1)

    # -- reconstruction ----------------------------------------------------
    def to_video(self):
        """Return the recorded frames as an :class:`~nanoindent.video.AFMVideo`."""
        if self.frames is None:
            raise ValueError("this run was recorded without frames "
                             "(save_frames=False)")
        from .video import AFMVideo
        return AFMVideo(self.frames, pixel_size_nm=self.pixel_size_nm,
                        frame_interval_s=self.frame_interval_s,
                        metadata={"run_id": self.run_id}, source=self.outdir)

    def animate(self, path: str = "replay.mp4", **kwargs):
        """Animate the recorded frames (requires ``save_frames`` at record time)."""
        from .video import animate_video
        return animate_video(self.to_video(), path=path, **kwargs)

    # -- visualisation -----------------------------------------------------
    def plot_summary(self):
        """Four-panel replay: height/roughness, coverage/particles, command &
        event timeline, and per-frame timing."""
        import matplotlib.pyplot as plt

        s = self.stats
        t = s.get("time_s", list(range(self.n_frames)))
        fig, axs = plt.subplots(2, 2, figsize=(11, 6.5))

        ax = axs[0, 0]
        if s.get("mean_height_nm"):
            ax.plot(t, s["mean_height_nm"], label="mean height", color="#0072B2")
        if s.get("rms_roughness_nm"):
            ax.plot(t, s["rms_roughness_nm"], label="roughness", color="#D55E00")
        ax.set_xlabel("time (s)"); ax.set_ylabel("nm"); ax.legend(fontsize=8)
        ax.set_title("height / roughness", fontsize=9); ax.grid(alpha=0.25)

        ax = axs[0, 1]
        if s.get("coverage"):
            ax.plot(t, s["coverage"], label="coverage", color="#009E73")
        if s.get("n_particles"):
            ax.plot(t, s["n_particles"], label="particles", color="#CC79A7")
        ax.set_xlabel("time (s)"); ax.legend(fontsize=8)
        ax.set_title("coverage / particle count", fontsize=9); ax.grid(alpha=0.25)

        ax = axs[1, 0]
        colors = {"recenter": "#0072B2", "trigger_ramp": "crimson",
                  "pause": "orange", "resume": "#009E73"}
        for c in self.commands:
            ax.vlines(c["frame"], 0, 1, color=colors.get(c["kind"], "gray"), lw=0.7)
        for e in self.events:
            ax.plot(e["frame"], 1.05, "v", color="crimson", ms=6)
        ax.set_ylim(0, 1.2); ax.set_yticks([]); ax.set_xlabel("frame")
        ax.set_title(f"{len(self.commands)} commands, {len(self.events)} events",
                     fontsize=9)

        ax = axs[1, 1]
        if self.timings.get("proc_ms"):
            ax.plot(self.timings["frame"], self.timings["proc_ms"], ".-", ms=3,
                    color="#0072B2")
        ax.set_xlabel("frame"); ax.set_ylabel("analysis time (ms)")
        ax.set_title("per-frame timing", fontsize=9); ax.grid(alpha=0.25)
        ax.set_ylim(bottom=0)

        fig.suptitle(f"run {self.run_id}", fontsize=10)
        fig.tight_layout()
        return fig


def load_run(outdir: str) -> RecordedRun:
    """Load a recorded run from a :class:`RunRecorder` output directory."""
    manifest = load_manifest(outdir)
    frames = None
    fp = os.path.join(outdir, "frames.npz")
    if os.path.exists(fp):
        frames = np.load(fp)["frames"]
    return RecordedRun(
        outdir=outdir, manifest=manifest,
        stats=_read_csv(os.path.join(outdir, "stats.csv")),
        commands=_read_jsonl(os.path.join(outdir, "commands.jsonl")),
        events=_read_jsonl(os.path.join(outdir, "events.jsonl")),
        timings=_read_csv(os.path.join(outdir, "timings.csv")),
        frames=frames)
