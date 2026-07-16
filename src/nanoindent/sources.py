"""Deployable, vendor-agnostic frame sources.

* :class:`DirectoryWatchSource` — a :class:`~nanoindent.streaming.FrameSource`
  that reads frames from files as an instrument writes them to a directory. This
  is the **works-on-any-instrument-today** path: no SDK, no wire protocol — point
  it at the microscope's output folder and process frames as they land (or batch
  an existing folder).
* :class:`RetryFrameSource` — wraps a source factory with connect/stream retry,
  so a flaky socket source reconnects instead of aborting the run.

Both satisfy the ``FrameSource`` contract (``pixel_size_nm`` / ``frame_interval_s``
/ ``__iter__`` / ``send``), so they drop straight into
:func:`nanoindent.streaming.run_control_loop`.
"""

from __future__ import annotations

import glob
import os
import time
from typing import Callable, List, Optional

import numpy as np

from .streaming import Command


# ==========================================================================
# per-file frame readers
# ==========================================================================
def read_frame_file(path: str) -> Optional[np.ndarray]:
    """Read a single 2-D height frame from a file (dispatch by extension).

    Supports ``.npy`` / ``.npz`` / ``.tif`` / ``.tiff`` / ``.csv`` / ``.txt``.
    Returns ``None`` if the file cannot be read yet (e.g. mid-write).
    """
    p = path.lower()
    try:
        if p.endswith(".npy"):
            arr = np.load(path)
        elif p.endswith(".npz"):
            d = np.load(path)
            key = "frame" if "frame" in d else list(d.keys())[0]
            arr = d[key]
        elif p.endswith((".tif", ".tiff")):
            from PIL import Image
            with Image.open(path) as im:
                arr = np.asarray(im, dtype=float)
        elif p.endswith(".csv"):
            arr = np.loadtxt(path, delimiter=",")
        elif p.endswith(".txt"):
            arr = np.loadtxt(path)
        else:
            raise ValueError(f"unsupported frame file type: {path!r}")
    except (OSError, ValueError, EOFError):
        return None
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 3:                      # a stack in one file: take first plane
        arr = arr[0]
    return arr if arr.ndim == 2 else None


def read_frame_timestamp(path: str, mode: str = "mtime") -> Optional[float]:
    """Acquisition time of a frame file, as a Unix timestamp.

    ``mode="mtime"`` uses the file's modification time (what the instrument
    stamps when it writes the frame). ``mode="metadata"`` reads an embedded
    timestamp — the TIFF ``DateTime`` tag — falling back to mtime. ``"none"``
    returns ``None``.
    """
    if mode == "none":
        return None
    if mode == "metadata" and path.lower().endswith((".tif", ".tiff")):
        try:
            from PIL import Image
            from datetime import datetime
            with Image.open(path) as im:
                dt = (im.tag_v2.get(306) if hasattr(im, "tag_v2") else None) \
                    or (im.info.get("DateTime") if hasattr(im, "info") else None)
            if dt:
                return datetime.strptime(dt, "%Y:%m:%d %H:%M:%S").timestamp()
        except Exception:                  # noqa: BLE001 — fall back to mtime
            pass
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


# ==========================================================================
# directory watch source
# ==========================================================================
class DirectoryWatchSource:
    """Yield frames from files appearing in a directory.

    Parameters
    ----------
    directory, pattern:
        Folder and glob (e.g. ``"*.npy"``, ``"scan_*.tif"``) to match frames.
    pixel_size_nm, frame_interval_s:
        Calibration carried on the ``FrameSource`` contract.
    watch:
        ``False`` (default) processes the files already present once and stops
        (batch). ``True`` polls for new files until a termination condition is
        met — set at least one of ``max_frames`` / ``idle_timeout_s`` /
        ``stop_file`` to bound it.
    sort_by:
        ``"name"`` (default) or ``"mtime"`` ordering of files.
    command_dir:
        If set, :meth:`send` writes each command to a text file here (for
        instruments driven by a watched command folder); otherwise commands are
        only logged.
    """

    def __init__(self, directory: str, *, pattern: str = "*.npy",
                 pixel_size_nm: float = 1.0, frame_interval_s: float = 0.1,
                 watch: bool = False, poll_interval_s: float = 0.2,
                 max_frames: Optional[int] = None,
                 idle_timeout_s: Optional[float] = None,
                 stop_file: Optional[str] = None, sort_by: str = "name",
                 command_dir: Optional[str] = None, timestamp: str = "mtime"):
        self.directory = directory
        self.pattern = pattern
        self.pixel_size_nm = float(pixel_size_nm)
        self.frame_interval_s = float(frame_interval_s)
        self.watch = watch
        self.poll_interval_s = poll_interval_s
        self.max_frames = max_frames
        self.idle_timeout_s = idle_timeout_s
        self.stop_file = stop_file
        self.sort_by = sort_by
        self.command_dir = command_dir
        self.timestamp = timestamp
        self.command_log: List[Command] = []
        # acquisition timestamps / paths, populated as frames are yielded
        self.frame_times: List[Optional[float]] = []
        self.frame_paths: List[str] = []
        if watch and max_frames is None and idle_timeout_s is None and stop_file is None:
            raise ValueError("watch=True needs a termination condition: set "
                             "max_frames, idle_timeout_s, or stop_file")

    def _list(self) -> List[str]:
        files = glob.glob(os.path.join(self.directory, self.pattern))
        key = os.path.getmtime if self.sort_by == "mtime" else (lambda f: f)
        return sorted(files, key=key)

    def __iter__(self):
        seen: set = set()
        i = 0
        last_new = time.perf_counter()
        while True:
            for f in self._list():
                if f in seen:
                    continue
                seen.add(f)
                frame = read_frame_file(f)
                if frame is None:
                    seen.discard(f)               # retry a mid-write file later
                    continue
                last_new = time.perf_counter()
                self.frame_paths.append(f)
                self.frame_times.append(read_frame_timestamp(f, self.timestamp))
                yield frame
                i += 1
                if self.max_frames is not None and i >= self.max_frames:
                    return
            if not self.watch:
                return
            if self.stop_file and os.path.exists(self.stop_file):
                return
            if self.idle_timeout_s is not None and \
                    time.perf_counter() - last_new > self.idle_timeout_s:
                return
            time.sleep(self.poll_interval_s)

    def send(self, command: Command) -> None:
        self.command_log.append(command)
        if self.command_dir:
            os.makedirs(self.command_dir, exist_ok=True)
            path = os.path.join(self.command_dir, f"cmd_{len(self.command_log):06d}.txt")
            params = " ".join(f"{k}={v}" for k, v in command.params.items())
            with open(path, "w") as fh:
                fh.write(f"{command.kind} {params}\n")


# ==========================================================================
# reconnecting source wrapper
# ==========================================================================
class RetryFrameSource:
    """Wrap a source *factory* with connect/stream retry (reconnect handling).

    ``factory()`` must return a fresh ``FrameSource``. If construction fails, it
    is retried up to ``retries`` times with exponential backoff; if the frame
    stream raises mid-run, the source is rebuilt and iteration continues
    (best-effort — frames already consumed are not replayed).
    """

    def __init__(self, factory: Callable[[], object], *, retries: int = 3,
                 backoff_s: float = 0.0, pixel_size_nm: float = 1.0,
                 frame_interval_s: float = 0.1):
        self._factory = factory
        self.retries = retries
        self.backoff_s = backoff_s
        self.pixel_size_nm = float(pixel_size_nm)
        self.frame_interval_s = float(frame_interval_s)
        self._src: Optional[object] = None
        self.reconnects = 0

    def _connect(self) -> None:
        last: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                self._src = self._factory()
                self.pixel_size_nm = getattr(self._src, "pixel_size_nm", self.pixel_size_nm)
                self.frame_interval_s = getattr(self._src, "frame_interval_s",
                                                self.frame_interval_s)
                return
            except Exception as exc:                # noqa: BLE001 (retry any failure)
                last = exc
                if self.backoff_s:
                    time.sleep(self.backoff_s * (2 ** attempt))
        raise RuntimeError(f"could not connect after {self.retries + 1} attempts: {last}")

    def __iter__(self):
        if self._src is None:
            self._connect()
        it = iter(self._src)
        while True:
            try:
                frame = next(it)
            except StopIteration:
                return
            except Exception:                       # noqa: BLE001 — reconnect + resume
                self.reconnects += 1
                self._connect()
                it = iter(self._src)
                continue
            yield frame

    def send(self, command: Command) -> None:
        if self._src is not None and hasattr(self._src, "send"):
            try:
                self._src.send(command)
            except Exception:                       # noqa: BLE001 — best effort
                pass
