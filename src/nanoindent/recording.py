"""Record a control-loop run to disk (MLflow-shaped, with real timestamps).

:class:`RunRecorder` captures per-frame analysis, the commands issued, detected
events, per-frame timings, and — for provenance and cross-run comparison — a
**run id**, wall-clock **start/end timestamps**, and **params / metrics / tags**.
The ``manifest.json`` it writes is deliberately MLflow-shaped, so a directory of
runs can be indexed and compared (see :mod:`nanoindent.tracking`) or exported to
MLflow.

Usage
-----
    rec = RunRecorder("runs/session1", save_frames=True)
    source = rec.wrap(source)                       # tee frames + acquisition times
    report = run_control_loop(source, analyzer, policy, on_state=rec.on_state)
    rec.save(report, params={"R_nm": 20, "policy": "recenter"},
             metrics={"n_events": 3}, tags={"instrument": "sim"})
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import time
import uuid
from typing import List, Optional

import numpy as np

from .streaming import AnalysisState, RunReport


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def make_run_id() -> str:
    """A sortable, unique run id: ``YYYYmmddTHHMMSS_<8 hex>``."""
    return datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]


class RecordingSource:
    """Wrap a ``FrameSource`` to tee each frame (and its acquisition time)."""

    def __init__(self, source, recorder: "RunRecorder"):
        self._source = source
        self._rec = recorder
        self.pixel_size_nm = getattr(source, "pixel_size_nm", 1.0)
        self.frame_interval_s = getattr(source, "frame_interval_s", 0.1)

    def __iter__(self):
        for frame in self._source:
            self._rec._wall_times.append(time.time())
            src_times = getattr(self._source, "frame_times", None)
            self._rec._acq_times.append(src_times[-1] if src_times else None)
            if self._rec.save_frames:
                self._rec._frames.append(np.asarray(frame, dtype=float))
            yield frame

    def send(self, command):
        if hasattr(self._source, "send"):
            self._source.send(command)


class RunRecorder:
    """Collect per-frame analysis, commands, events, timings, frames, and
    provenance (run id, timestamps, params/metrics/tags), and write them under
    ``outdir``."""

    def __init__(self, outdir: str, *, save_frames: bool = False,
                 run_id: Optional[str] = None):
        self.outdir = outdir
        self.save_frames = save_frames
        self.run_id = run_id or make_run_id()
        self.started_at = _now_iso()
        self._start_perf = time.perf_counter()
        self._stats: List[dict] = []
        self._frames: List[np.ndarray] = []
        self._wall_times: List[float] = []
        self._acq_times: List[Optional[float]] = []
        os.makedirs(outdir, exist_ok=True)

    # -- collection --------------------------------------------------------
    def wrap(self, source):
        """Return a :class:`RecordingSource` teeing frames from ``source``."""
        return RecordingSource(source, self)

    def on_state(self, state: AnalysisState) -> None:
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
    def save(self, report: Optional[RunReport] = None, *,
             params: Optional[dict] = None, metrics: Optional[dict] = None,
             tags: Optional[dict] = None, meta: Optional[dict] = None) -> dict:
        """Write every artefact and an MLflow-shaped ``manifest.json``."""
        ended_at = _now_iso()
        duration = time.perf_counter() - self._start_perf
        self._write_stats_csv()

        counts = {"frames": len(self._stats)}
        auto_metrics = {"n_frames": len(self._stats)}
        if report is not None:
            counts["commands"] = self._write_commands(report)
            counts["events"] = self._write_events(report)
            self._write_timings(report)
            auto_metrics.update(
                n_commands=len(report.commands), n_events=len(report.events),
                real_time_factor=report.real_time_factor(),
                kept_up=int(report.kept_up()))
        # final diffusion estimate if the analyzer produced one
        diffs = [r["diffusion_nm2_s"] for r in self._stats
                 if r["diffusion_nm2_s"] is not None]
        if diffs:
            auto_metrics["diffusion_final_nm2_s"] = float(diffs[-1])
        if self.save_frames and self._frames:
            np.savez_compressed(os.path.join(self.outdir, "frames.npz"),
                                frames=np.stack(self._frames))
            counts["saved_frames"] = len(self._frames)

        merged_metrics = {**auto_metrics, **(metrics or {})}
        manifest = {
            "run_id": self.run_id,
            "started_at": self.started_at, "ended_at": ended_at,
            "duration_s": round(duration, 4),
            "params": params or {}, "metrics": merged_metrics,
            "tags": tags or {}, "counts": counts, "meta": meta or {},
            "outdir": self.outdir,
        }
        with open(os.path.join(self.outdir, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2, default=str)
        return manifest

    def _write_stats_csv(self) -> None:
        path = os.path.join(self.outdir, "stats.csv")
        cols = ["frame", "time_s", "wall_time_s", "acq_time_s", "n_particles",
                "mean_height_nm", "rms_roughness_nm", "coverage",
                "drift_dy_px", "drift_dx_px", "diffusion_nm2_s", "n_tracks",
                "n_events"]
        with open(path, "w") as fh:
            fh.write(",".join(cols) + "\n")
            for i, r in enumerate(self._stats):
                row = dict(r)
                row["wall_time_s"] = (self._wall_times[i]
                                      if i < len(self._wall_times) else None)
                row["acq_time_s"] = (self._acq_times[i]
                                     if i < len(self._acq_times) else None)
                fh.write(",".join("" if row.get(c) is None else f"{row[c]}"
                                  for c in cols) + "\n")

    def _write_commands(self, report: RunReport) -> int:
        with open(os.path.join(self.outdir, "commands.jsonl"), "w") as fh:
            for frame_index, cmd in report.commands:
                fh.write(json.dumps({"frame": frame_index, "kind": cmd.kind,
                                     "params": cmd.params, "reason": cmd.reason}) + "\n")
        return len(report.commands)

    def _write_events(self, report: RunReport) -> int:
        with open(os.path.join(self.outdir, "events.jsonl"), "w") as fh:
            for frame_index, t, delta in report.events:
                fh.write(json.dumps({"frame": frame_index, "time_s": t,
                                     "delta_nm": delta}) + "\n")
        return len(report.events)

    def _write_timings(self, report: RunReport) -> None:
        with open(os.path.join(self.outdir, "timings.csv"), "w") as fh:
            fh.write("frame,proc_ms\n")
            for i, ms in enumerate(report.proc_ms):
                fh.write(f"{i},{ms:.4f}\n")
