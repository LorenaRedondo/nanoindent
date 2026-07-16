"""Experiment tracking: index and compare runs, and optionally log to MLflow.

Every :class:`~nanoindent.recording.RunRecorder` writes an MLflow-shaped
``manifest.json`` (run id, timestamps, params, metrics, tags). This module turns
a directory of such runs into a comparison table with **no external dependency**
(:class:`ExperimentIndex`), and offers an **optional** MLflow export
(:func:`log_run_to_mlflow`) for those who want the MLflow UI / tracking server.

    from nanoindent.tracking import ExperimentIndex
    idx = ExperimentIndex("runs")
    idx.to_csv("runs/comparison.csv")        # one row per run, params+metrics flattened
    rows = idx.rows()                        # list of dicts (or idx.to_dataframe())

    # optional, needs `pip install "nanoindent[tracking]"`
    from nanoindent.tracking import log_run_to_mlflow
    log_run_to_mlflow("runs/session1", experiment="afm-control")
"""

from __future__ import annotations

import glob
import json
import os
from typing import List, Optional


def load_manifest(path: str) -> dict:
    """Load a run ``manifest.json`` (accepts the file or its directory)."""
    if os.path.isdir(path):
        path = os.path.join(path, "manifest.json")
    with open(path) as fh:
        return json.load(fh)


def _flatten(manifest: dict) -> dict:
    """Flatten a manifest into one comparison row (``params.*`` / ``metrics.*``)."""
    row = {
        "run_id": manifest.get("run_id"),
        "started_at": manifest.get("started_at"),
        "ended_at": manifest.get("ended_at"),
        "duration_s": manifest.get("duration_s"),
        "outdir": manifest.get("outdir"),
    }
    for section in ("params", "metrics", "tags"):
        for k, v in (manifest.get(section) or {}).items():
            row[f"{section}.{k}"] = v
    return row


class ExperimentIndex:
    """Index every ``manifest.json`` under a root directory for comparison."""

    def __init__(self, root: str, pattern: str = "**/manifest.json"):
        self.root = root
        self.pattern = pattern
        self._manifests: List[dict] = []
        self.reload()

    def reload(self) -> "ExperimentIndex":
        paths = glob.glob(os.path.join(self.root, self.pattern), recursive=True)
        # also match a manifest directly in root
        direct = os.path.join(self.root, "manifest.json")
        if os.path.exists(direct) and direct not in paths:
            paths.append(direct)
        mans = []
        for p in sorted(paths):
            try:
                mans.append(load_manifest(p))
            except (OSError, ValueError):
                continue
        # newest first by started_at
        self._manifests = sorted(mans, key=lambda m: m.get("started_at") or "",
                                 reverse=True)
        return self

    def __len__(self) -> int:
        return len(self._manifests)

    def manifests(self) -> List[dict]:
        return list(self._manifests)

    def rows(self) -> List[dict]:
        """One flattened dict per run (params/metrics/tags as ``section.key``)."""
        return [_flatten(m) for m in self._manifests]

    def columns(self) -> List[str]:
        cols: List[str] = []
        for r in self.rows():
            for k in r:
                if k not in cols:
                    cols.append(k)
        return cols

    def to_csv(self, path: str) -> str:
        rows, cols = self.rows(), self.columns()
        with open(path, "w") as fh:
            fh.write(",".join(cols) + "\n")
            for r in rows:
                fh.write(",".join(_csv_cell(r.get(c)) for c in cols) + "\n")
        return path

    def to_dataframe(self):
        """Return a pandas DataFrame (requires pandas)."""
        import pandas as pd
        return pd.DataFrame(self.rows())

    def best(self, metric: str, maximize: bool = True) -> Optional[dict]:
        """The run with the best ``metrics.<metric>`` (returns its flat row)."""
        key = metric if metric.startswith("metrics.") else f"metrics.{metric}"
        scored = [r for r in self.rows() if isinstance(r.get(key), (int, float))]
        if not scored:
            return None
        return (max if maximize else min)(scored, key=lambda r: r[key])


def _csv_cell(v) -> str:
    if v is None:
        return ""
    s = str(v)
    return '"' + s.replace('"', '""') + '"' if ("," in s or '"' in s) else s


# ==========================================================================
# optional MLflow export
# ==========================================================================
def log_run_to_mlflow(run, *, experiment: str = "nanoindent",
                      tracking_uri: Optional[str] = None,
                      log_artifacts: bool = True):
    """Log a recorded run to MLflow (optional; needs ``mlflow`` installed).

    ``run`` is a manifest dict, a ``manifest.json`` path, or a run directory.
    Params, metrics and tags are logged; if ``log_artifacts`` and the run
    directory is available, its files (stats.csv, figures, …) are uploaded.
    """
    try:
        import mlflow
    except ImportError as exc:                # pragma: no cover - env dependent
        raise ImportError(
            "log_run_to_mlflow needs MLflow: pip install \"nanoindent[tracking]\" "
            "(or pip install mlflow)") from exc

    manifest = run if isinstance(run, dict) else load_manifest(run)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=manifest.get("run_id")) as active:
        params = {k: v for k, v in (manifest.get("params") or {}).items()}
        metrics = {k: float(v) for k, v in (manifest.get("metrics") or {}).items()
                   if isinstance(v, (int, float))}
        tags = {k: str(v) for k, v in (manifest.get("tags") or {}).items()}
        tags.setdefault("run_id", manifest.get("run_id", ""))
        if manifest.get("started_at"):
            tags["started_at"] = manifest["started_at"]
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)
        outdir = manifest.get("outdir")
        if log_artifacts and outdir and os.path.isdir(outdir):
            mlflow.log_artifacts(outdir)
        return active.info.run_id


# ==========================================================================
# CLI: nanoindent-runs — index a directory of runs into a comparison table
# ==========================================================================
def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Index/compare nanoindent run directories")
    ap.add_argument("root", help="directory containing run folders")
    ap.add_argument("--csv", help="write the comparison table to this CSV")
    ap.add_argument("--best", help="print the run maximizing this metric")
    ap.add_argument("--mlflow", action="store_true",
                    help="log every run to MLflow (needs mlflow installed)")
    ap.add_argument("--experiment", default="nanoindent")
    a = ap.parse_args(argv)

    idx = ExperimentIndex(a.root)
    print(f"{len(idx)} run(s) under {a.root}")
    if a.csv:
        idx.to_csv(a.csv)
        print(f"wrote {a.csv}")
    else:
        cols = idx.columns()
        print(",".join(cols))
        for r in idx.rows():
            print(",".join(_csv_cell(r.get(c)) for c in cols))
    if a.best:
        b = idx.best(a.best)
        print(f"best {a.best}: {b['run_id'] if b else 'n/a'}")
    if a.mlflow:
        for m in idx.manifests():
            log_run_to_mlflow(m, experiment=a.experiment)
        print(f"logged {len(idx)} run(s) to MLflow experiment {a.experiment!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
