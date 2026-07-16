"""Record a control-loop run to disk.

:class:`RunRecorder` captures per-frame analysis, the commands issued, detected
events, per-frame timings and a run manifest — everything needed to audit or
replay a session. Optionally it tees the raw frames to an ``.npz`` stack.

Usage
-----
    rec = RunRecorder("runs/session1", save_frames=True)
    source = rec.wrap(source)                       # tee frames (if save_frames)
    report = run_control_loop(source, analyzer, policy, on_state=rec.on_state)
    rec.save(report, meta={"instrument": "sim"})    # writes all artefacts
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np

from .streaming import AnalysisState, RunReport


class RecordingSource:
    """Wrap a ``FrameSource`` to tee each frame into a recorder."""

    def __init__(self, source, recorder: "RunRecorder"):
        self._source = source
        self._rec = recorder
        self.pixel_size_nm = getattr(source, "pixel_size_nm", 1.0)
        self.frame_interval_s = getattr(source, "frame_interval_s", 0.1)

    def __iter__(self):
        for frame in self._source:
            if self._rec.save_frames:
                self._rec._frames.append(np.asarray(frame, dtype=float))
            yield frame

    def send(self, command):
        if hasattr(self._source, "send"):
            self._source.send(command)


class RunRecorder:
    """Collect per-frame analysis, commands, events, timings, and (optionally)
    frames, and write them under ``outdir``."""

    def __init__(self, outdir: str, *, save_frames: bool = False):
        self.outdir = outdir
        self.save_frames = save_frames
        self._stats: List[dict] = []
        self._frames: List[np.ndarray] = []
        os.makedirs(outdir, exist_ok=True)

    # -- collection --------------------------------------------------------
    def wrap(self, source):
        """Return a :class:`RecordingSource` teeing frames from ``source``."""
        return RecordingSource(source, self)

    def on_state(self, state: AnalysisState) -> None:
        """Per-frame callback for :func:`run_control_loop`'s ``on_state``."""
        self._stats.append({
            "frame": state.frame_index, "time_s": state.time_s,
            "n_particles": state.n_particles,
            "mean_height_nm": state.mean_height_nm,
            "rms_roughness_nm": state.rms_roughness_nm,
            "coverage": state.coverage,
            "drift_dy_px": state.drift_px[0], "drift_dx_px": state.drift_px[1],
            "diffusion_nm2_s": state.diffusion_nm2_s,
            "n_tracks": len(state.tracks), "n_events": len(state.events),
        })

    # -- output ------------------------------------------------------------
    def save(self, report: Optional[RunReport] = None, meta: Optional[dict] = None):
        """Write every collected artefact under ``outdir`` and return a manifest."""
        self._write_stats_csv()
        counts = {"frames": len(self._stats)}
        if report is not None:
            counts["commands"] = self._write_commands(report)
            counts["events"] = self._write_events(report)
            self._write_timings(report)
        if self.save_frames and self._frames:
            np.savez_compressed(os.path.join(self.outdir, "frames.npz"),
                                frames=np.stack(self._frames))
            counts["saved_frames"] = len(self._frames)
        manifest = {"outdir": self.outdir, "counts": counts, "meta": meta or {}}
        if report is not None:
            manifest["real_time_factor"] = report.real_time_factor()
            manifest["kept_up"] = report.kept_up()
        with open(os.path.join(self.outdir, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)
        return manifest

    def _write_stats_csv(self) -> None:
        path = os.path.join(self.outdir, "stats.csv")
        cols = ["frame", "time_s", "n_particles", "mean_height_nm",
                "rms_roughness_nm", "coverage", "drift_dy_px", "drift_dx_px",
                "diffusion_nm2_s", "n_tracks", "n_events"]
        with open(path, "w") as fh:
            fh.write(",".join(cols) + "\n")
            for r in self._stats:
                fh.write(",".join("" if r[c] is None else f"{r[c]}" for c in cols) + "\n")

    def _write_commands(self, report: RunReport) -> int:
        path = os.path.join(self.outdir, "commands.jsonl")
        with open(path, "w") as fh:
            for frame_index, cmd in report.commands:
                fh.write(json.dumps({"frame": frame_index, "kind": cmd.kind,
                                     "params": cmd.params, "reason": cmd.reason}) + "\n")
        return len(report.commands)

    def _write_events(self, report: RunReport) -> int:
        path = os.path.join(self.outdir, "events.jsonl")
        with open(path, "w") as fh:
            for frame_index, t, delta in report.events:
                fh.write(json.dumps({"frame": frame_index, "time_s": t,
                                     "delta_nm": delta}) + "\n")
        return len(report.events)

    def _write_timings(self, report: RunReport) -> None:
        path = os.path.join(self.outdir, "timings.csv")
        with open(path, "w") as fh:
            fh.write("frame,proc_ms\n")
            for i, ms in enumerate(report.proc_ms):
                fh.write(f"{i},{ms:.4f}\n")
