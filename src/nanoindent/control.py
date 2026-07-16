"""Config-driven control-loop runner (CLI).

Wires a frame source, the online analyzer, a policy, the safety envelope, and a
run recorder together and runs the supervisory loop — the deployable entry point
that needs no code, just a config or CLI flags.

    python -m nanoindent.control --source simulated --frames 100 \\
        --policy recenter_on_brightest,ramp_on_event --outdir runs/demo \\
        --max-scan-offset-px 24 --pause-after-lost 5

    python -m nanoindent.control --source directory --dir ./scan_out \\
        --pattern "frame_*.npy" --pixel-size 2.0 --dt 0.05 --outdir runs/live

Or programmatically via :func:`run_from_config`.
"""

from __future__ import annotations

import argparse
import json
from typing import Optional

import numpy as np

from . import streaming
from .streaming import StreamingAnalyzer, run_control_loop, combine_policies
from .safety import SafetyEnvelope, SafetyLimits
from .recording import RunRecorder

_POLICIES = {
    "recenter_on_brightest": streaming.recenter_on_brightest,
    "recenter_on_longest_track": streaming.recenter_on_longest_track,
    "ramp_on_event": streaming.ramp_on_event,
    "pause_on_event": streaming.pause_on_event,
}


def build_source(cfg: dict):
    """Construct a frame source from a config dict (``type`` + params)."""
    kind = cfg.get("type", "simulated")
    if kind == "simulated":
        return streaming.SimulatedInstrument(
            scan_px=cfg.get("scan_px", 64),
            pixel_size_nm=cfg.get("pixel_size_nm", 2.0),
            frame_interval_s=cfg.get("frame_interval_s", 0.05),
            n_particles=cfg.get("n_particles", 4),
            D_nm2_per_s=cfg.get("D_nm2_per_s", 200.0),
            max_frames=cfg.get("frames", 100), seed=cfg.get("seed", 0))
    if kind == "directory":
        from .sources import DirectoryWatchSource
        return DirectoryWatchSource(
            cfg["directory"], pattern=cfg.get("pattern", "*.npy"),
            pixel_size_nm=cfg.get("pixel_size_nm", 1.0),
            frame_interval_s=cfg.get("frame_interval_s", 0.1),
            watch=cfg.get("watch", False), max_frames=cfg.get("frames"),
            idle_timeout_s=cfg.get("idle_timeout_s"),
            stop_file=cfg.get("stop_file"),
            command_dir=cfg.get("command_dir"))
    raise ValueError(f"unknown source type {kind!r}")


def build_policy(names):
    ps = []
    for n in names:
        if n not in _POLICIES:
            raise ValueError(f"unknown policy {n!r}; known: {list(_POLICIES)}")
        ps.append(_POLICIES[n]())
    return combine_policies(*ps) if len(ps) != 1 else ps[0]


def build_safety(cfg: dict) -> Optional[SafetyEnvelope]:
    keys = ("min_frames_between_commands", "max_scan_offset_px",
            "setpoint_range", "pause_after_lost_frames", "max_frame_gap_s")
    if not any(cfg.get(k) is not None for k in keys):
        return None
    return SafetyEnvelope(SafetyLimits(
        min_frames_between_commands=cfg.get("min_frames_between_commands", 0),
        max_scan_offset_px=cfg.get("max_scan_offset_px"),
        setpoint_range=(tuple(cfg["setpoint_range"])
                        if cfg.get("setpoint_range") else None),
        pause_after_lost_frames=cfg.get("pause_after_lost_frames"),
        max_frame_gap_s=cfg.get("max_frame_gap_s")))


def run_from_config(cfg: dict) -> dict:
    """Run the loop described by ``cfg`` and return the recorder manifest.

    The manifest records **params** (the resolved config), **metrics** (run
    outcomes), and **tags** (``cfg["tags"]``), so runs can be indexed/compared
    (see :mod:`nanoindent.tracking`). If ``cfg["mlflow"]`` is set, the run is also
    logged to MLflow.
    """
    source_cfg = cfg.get("source", {})
    policy_names = cfg.get("policy", ["recenter_on_brightest"])
    safety_cfg = cfg.get("safety", {})
    source = build_source(source_cfg)
    analyzer = StreamingAnalyzer(
        getattr(source, "pixel_size_nm", 2.0),
        getattr(source, "frame_interval_s", 0.05),
        max_step_px=cfg.get("max_step_px", 6.0),
        event_k=cfg.get("event_k", 4.0))
    policy = build_policy(policy_names)
    envelope = build_safety(safety_cfg)
    if envelope is not None:
        policy = envelope.guard(policy)

    recorder = RunRecorder(cfg.get("outdir", "run"),
                           save_frames=cfg.get("save_frames", False))
    src = recorder.wrap(source)
    report = run_control_loop(src, analyzer, policy,
                              apply=cfg.get("apply", True),
                              on_state=recorder.on_state)

    # provenance: params (what was run) + tags (how to group) for comparison
    params = {"source_type": source_cfg.get("type", "simulated"),
              "policy": ",".join(policy_names),
              "pixel_size_nm": getattr(source, "pixel_size_nm", None),
              "frame_interval_s": getattr(source, "frame_interval_s", None),
              "max_step_px": cfg.get("max_step_px", 6.0),
              "event_k": cfg.get("event_k", 4.0)}
    for k in ("scan_px", "n_particles", "D_nm2_per_s", "directory", "pattern"):
        if k in source_cfg:
            params[k] = source_cfg[k]
    for k, v in safety_cfg.items():
        if v is not None:
            params[f"safety_{k}"] = v
    metrics = {}
    if envelope is not None:
        metrics.update({f"guard_{k}": v for k, v in envelope.counts.items()})
    tags = dict(cfg.get("tags", {}))
    tags.setdefault("source", source_cfg.get("type", "simulated"))

    meta = {"policy": policy_names}
    if envelope is not None:
        meta["safety_counts"] = envelope.counts
    manifest = recorder.save(report, params=params, metrics=metrics,
                             tags=tags, meta=meta)

    if cfg.get("mlflow"):
        from .tracking import log_run_to_mlflow
        log_run_to_mlflow(manifest, experiment=cfg.get("experiment", "nanoindent"),
                          tracking_uri=cfg.get("tracking_uri"))
    return manifest


def _args_to_config(a) -> dict:
    source = {"type": a.source, "frames": a.frames,
              "pixel_size_nm": a.pixel_size, "frame_interval_s": a.dt}
    if a.source == "directory":
        source.update(directory=a.dir, pattern=a.pattern, watch=a.watch,
                      idle_timeout_s=a.idle_timeout, command_dir=a.command_dir)
    safety = {"min_frames_between_commands": a.min_frames_between_commands,
              "max_scan_offset_px": a.max_scan_offset_px,
              "pause_after_lost_frames": a.pause_after_lost,
              "max_frame_gap_s": a.max_frame_gap_s}
    if a.setpoint_range:
        safety["setpoint_range"] = a.setpoint_range
    tags = dict(t.split("=", 1) for t in a.tag) if a.tag else {}
    return {"source": source, "policy": a.policy.split(","), "outdir": a.outdir,
            "save_frames": a.save_frames, "safety": safety,
            "max_step_px": a.max_step_px, "event_k": a.event_k, "tags": tags,
            "mlflow": a.mlflow, "experiment": a.experiment,
            "tracking_uri": a.tracking_uri}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="nanoindent supervisory control loop")
    ap.add_argument("--config", help="JSON config file (overrides CLI flags)")
    ap.add_argument("--source", default="simulated",
                    choices=["simulated", "directory"])
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--pixel-size", type=float, default=2.0, dest="pixel_size")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--dir", default=".", help="directory source: folder to watch")
    ap.add_argument("--pattern", default="*.npy")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--idle-timeout", type=float, default=None, dest="idle_timeout")
    ap.add_argument("--command-dir", default=None, dest="command_dir")
    ap.add_argument("--policy", default="recenter_on_brightest",
                    help="comma-separated: " + ",".join(_POLICIES))
    ap.add_argument("--outdir", default="run")
    ap.add_argument("--save-frames", action="store_true", dest="save_frames")
    ap.add_argument("--max-step-px", type=float, default=6.0, dest="max_step_px")
    ap.add_argument("--event-k", type=float, default=4.0, dest="event_k")
    # safety
    ap.add_argument("--min-frames-between-commands", type=int, default=0,
                    dest="min_frames_between_commands")
    ap.add_argument("--max-scan-offset-px", type=float, default=None,
                    dest="max_scan_offset_px")
    ap.add_argument("--setpoint-range", type=float, nargs=2, default=None,
                    dest="setpoint_range", metavar=("LO", "HI"))
    ap.add_argument("--pause-after-lost", type=int, default=None,
                    dest="pause_after_lost")
    ap.add_argument("--max-frame-gap-s", type=float, default=None,
                    dest="max_frame_gap_s")
    # experiment tracking
    ap.add_argument("--tag", action="append", metavar="KEY=VALUE",
                    help="run tag for grouping/comparison (repeatable)")
    ap.add_argument("--mlflow", action="store_true",
                    help="also log the run to MLflow (needs mlflow installed)")
    ap.add_argument("--experiment", default="nanoindent")
    ap.add_argument("--tracking-uri", default=None, dest="tracking_uri")
    a = ap.parse_args(argv)

    cfg = json.load(open(a.config)) if a.config else _args_to_config(a)
    manifest = run_from_config(cfg)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
