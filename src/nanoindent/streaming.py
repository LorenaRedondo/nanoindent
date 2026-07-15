"""Pseudo-realtime streaming analysis and (simulated) closed-loop control.

This module turns the batch HS-AFM analysis of :mod:`nanoindent.video` into an
*online* pipeline suitable for reacting to frames as they arrive, and provides a
reactive **simulated instrument** so the whole feedback loop can be exercised
and tested without hardware.

Architecture (three seams)
--------------------------
``FrameSource`` (read)  →  :class:`StreamingAnalyzer` (analyse)  →  a *policy*
(decide) → ``FrameSource.send`` (write / actuate).

* :class:`StreamingAnalyzer` — incremental particle linking, rolling statistics,
  online drift and step/event detection, and a running diffusion estimate. One
  :meth:`~StreamingAnalyzer.push` per frame returns an :class:`AnalysisState`.
* :class:`SimulatedInstrument` — a ``FrameSource`` whose scan window *reacts* to
  commands (re-centering, ramps, pause), so a policy that follows a feature can
  be shown to actually keep it in view.
* Policies are plain callables ``AnalysisState -> list[Command]``; a few useful
  ones are provided (:func:`recenter_on_brightest`, :func:`ramp_on_event`,
  :func:`pause_on_event`).
* :func:`run_control_loop` wires a source, analyzer and policy together and
  returns a :class:`RunReport` with per-frame timings (the real-time headroom),
  the commands issued, and the events seen.

Safety
------
This is a **supervisory** loop only. It runs at frame rate (tens of Hz) and
issues high-level commands. The fast z-feedback that protects the tip runs at
kHz–MHz on the instrument's own real-time controller and **must stay there** —
never route tip protection through this Python loop. Swap
:class:`SimulatedInstrument` for a thin vendor adapter implementing the same
``__iter__`` / ``send`` contract to drive a real microscope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Sequence, Tuple
import time

import numpy as np

from .video import detect_particles, _phase_shift, diffusion_coefficient


# ==========================================================================
# commands (supervisory instructions to the instrument)
# ==========================================================================
@dataclass
class Command:
    """A high-level, instrument-agnostic supervisory instruction.

    Positions are expressed in *image* space (window pixels) so policies stay
    instrument-agnostic; the ``FrameSource`` translates to its own units.
    """

    kind: str
    params: dict = field(default_factory=dict)
    reason: str = ""


def recenter(dy_px: float, dx_px: float, reason: str = "") -> Command:
    """Shift the scan window by ``(dy, dx)`` window-pixels."""
    return Command("recenter", {"dy_px": float(dy_px), "dx_px": float(dx_px)}, reason)


def trigger_ramp(y_px: float, x_px: float, reason: str = "") -> Command:
    """Request a force–distance ramp at window-pixel ``(y, x)``."""
    return Command("trigger_ramp", {"y_px": float(y_px), "x_px": float(x_px)}, reason)


def set_setpoint(value: float, reason: str = "") -> Command:
    return Command("set_setpoint", {"value": float(value)}, reason)


def pause(reason: str = "") -> Command:
    return Command("pause", {}, reason)


def resume(reason: str = "") -> Command:
    return Command("resume", {}, reason)


# ==========================================================================
# online analysis
# ==========================================================================
@dataclass
class OnlineTrack:
    """A live particle trajectory being extended frame by frame."""

    id: int
    points: List[Tuple[float, float, float]]   # (frame, y, x)  px
    last_seen: int

    @property
    def length(self) -> int:
        return len(self.points)

    @property
    def last_yx(self) -> Tuple[float, float]:
        return (self.points[-1][1], self.points[-1][2])

    def as_array(self) -> np.ndarray:
        return np.array(self.points, dtype=float)


@dataclass
class AnalysisState:
    """The live findings after one frame."""

    frame_index: int
    time_s: float
    frame_shape: Tuple[int, int]
    pixel_size_nm: float
    detections: np.ndarray                # (n, 2) centroids (y, x) px
    detection_heights: np.ndarray         # (n,) height at each centroid (nm)
    tracks: List[OnlineTrack]             # active tracks
    mean_height_nm: float
    rms_roughness_nm: float
    coverage: float
    drift_px: Tuple[float, float]         # (dy, dx) vs previous frame
    events: List[Tuple[int, float, float]]   # NEW this frame: (frame, time, delta)
    diffusion_nm2_s: Optional[float]      # ensemble-median D so far

    @property
    def n_particles(self) -> int:
        return len(self.detections)

    def longest_track(self) -> Optional[OnlineTrack]:
        return max(self.tracks, key=lambda t: t.length) if self.tracks else None

    def brightest_detection(self) -> Optional[Tuple[float, float]]:
        if len(self.detections) == 0:
            return None
        j = int(np.argmax(self.detection_heights))
        return (float(self.detections[j, 0]), float(self.detections[j, 1]))


class StreamingAnalyzer:
    """Incremental HS-AFM analysis: push one frame, get an :class:`AnalysisState`.

    Keeps active/retired tracks, a rolling mean-height series for online event
    detection, and a running ensemble diffusion estimate. The per-frame cost is
    dominated by :func:`~nanoindent.video.detect_particles` and one FFT for
    drift — both far below typical HS-AFM frame periods (see ``examples``).
    """

    def __init__(self, pixel_size_nm: float = 1.0, frame_interval_s: float = 0.1,
                 *, threshold: Optional[float] = None, max_step_px: float = 8.0,
                 min_size: int = 2, max_gap: int = 2, min_track_len: int = 5,
                 event_k: float = 4.0, event_warmup: int = 5,
                 estimate_drift: bool = True):
        self.pixel_size_nm = pixel_size_nm
        self.frame_interval_s = frame_interval_s
        self.threshold = threshold
        self.max_step_px = max_step_px
        self.min_size = min_size
        self.max_gap = max_gap
        self.min_track_len = min_track_len
        self.event_k = event_k
        self.event_warmup = event_warmup
        self.estimate_drift = estimate_drift

        self._tracks: List[OnlineTrack] = []
        self._retired: List[OnlineTrack] = []
        self._next_id = 0
        self.frame_index = -1
        self._prev_frame: Optional[np.ndarray] = None
        self._mean_series: List[float] = []
        self.events: List[Tuple[int, float, float]] = []

    # ---- internals -------------------------------------------------------
    def _link(self, cents: np.ndarray) -> None:
        fi = self.frame_index
        assigned: set = set()
        # long tracks claim their nearest detection first (identity stability)
        for tr in sorted(self._tracks, key=lambda t: -t.length):
            if len(cents) == 0:
                break
            d = np.hypot(cents[:, 0] - tr.last_yx[0], cents[:, 1] - tr.last_yx[1])
            for j in np.argsort(d):
                j = int(j)
                if j in assigned:
                    continue
                if d[j] <= self.max_step_px:
                    tr.points.append((fi, float(cents[j, 0]), float(cents[j, 1])))
                    tr.last_seen = fi
                    assigned.add(j)
                break
        for j, c in enumerate(cents):
            if j not in assigned:
                self._tracks.append(
                    OnlineTrack(self._next_id, [(fi, float(c[0]), float(c[1]))], fi))
                self._next_id += 1
        still: List[OnlineTrack] = []
        for tr in self._tracks:
            (still if fi - tr.last_seen <= self.max_gap else self._retired).append(tr)
        self._tracks = still

    def _detect_events(self, mean_height: float, t: float) -> list:
        self._mean_series.append(mean_height)
        s = self._mean_series
        if len(s) < max(2, self.event_warmup):
            return []
        diffs = np.diff(s)
        mad = np.median(np.abs(diffs - np.median(diffs)))
        thr = self.event_k * 1.4826 * mad if mad > 0 else np.inf
        delta = s[-1] - s[-2]
        if abs(delta) >= thr:
            ev = (self.frame_index, t, float(delta))
            self.events.append(ev)
            return [ev]
        return []

    def _ensemble_D(self) -> Optional[float]:
        Ds = []
        for tr in self._tracks + self._retired:
            if tr.length >= self.min_track_len:
                D = diffusion_coefficient(tr.as_array(), self.pixel_size_nm,
                                          self.frame_interval_s)
                if D is not None and D >= 0:
                    Ds.append(D)
        return float(np.median(Ds)) if Ds else None

    # ---- public ----------------------------------------------------------
    def push(self, frame: np.ndarray) -> AnalysisState:
        frame = np.asarray(frame, dtype=float)
        self.frame_index += 1
        t = self.frame_index * self.frame_interval_s

        cents = detect_particles(frame, self.threshold, self.min_size)
        if len(cents):
            yi = np.clip(np.round(cents[:, 0]).astype(int), 0, frame.shape[0] - 1)
            xi = np.clip(np.round(cents[:, 1]).astype(int), 0, frame.shape[1] - 1)
            heights = frame[yi, xi]
        else:
            heights = np.empty(0)
        self._link(cents)

        if self.estimate_drift and self._prev_frame is not None:
            drift = _phase_shift(self._prev_frame, frame)
        else:
            drift = (0.0, 0.0)
        self._prev_frame = frame

        m, rms = float(frame.mean()), float(frame.std())
        cov_thr = self.threshold if self.threshold is not None else m + rms
        cov = float((frame > cov_thr).mean())
        new_events = self._detect_events(m, t)
        D = self._ensemble_D()

        return AnalysisState(
            frame_index=self.frame_index, time_s=t, frame_shape=frame.shape,
            pixel_size_nm=self.pixel_size_nm, detections=cents,
            detection_heights=heights, tracks=list(self._tracks),
            mean_height_nm=m, rms_roughness_nm=rms, coverage=cov,
            drift_px=drift, events=new_events, diffusion_nm2_s=D)

    @property
    def all_tracks(self) -> List[OnlineTrack]:
        return self._tracks + self._retired


# ==========================================================================
# policies (AnalysisState -> list[Command])
# ==========================================================================
Policy = Callable[[AnalysisState], List[Command]]


def recenter_on_brightest(deadband_px: float = 0.5) -> Policy:
    """Keep the brightest feature centred in the scan window."""
    def policy(state: AnalysisState) -> List[Command]:
        b = state.brightest_detection()
        if b is None:
            return []
        cy, cx = state.frame_shape[0] / 2, state.frame_shape[1] / 2
        dy, dx = b[0] - cy, b[1] - cx
        if abs(dy) < deadband_px and abs(dx) < deadband_px:
            return []
        return [recenter(dy, dx, reason="centre brightest feature")]
    return policy


def recenter_on_longest_track(min_len: int = 3, deadband_px: float = 0.5) -> Policy:
    """Keep the longest-lived track centred (identity-stable following)."""
    def policy(state: AnalysisState) -> List[Command]:
        tr = state.longest_track()
        if tr is None or tr.length < min_len:
            return []
        cy, cx = state.frame_shape[0] / 2, state.frame_shape[1] / 2
        dy, dx = tr.last_yx[0] - cy, tr.last_yx[1] - cx
        if abs(dy) < deadband_px and abs(dx) < deadband_px:
            return []
        return [recenter(dy, dx, reason=f"centre track {tr.id}")]
    return policy


def ramp_on_event() -> Policy:
    """Trigger a force–distance ramp at the window centre on each new event."""
    def policy(state: AnalysisState) -> List[Command]:
        cy, cx = state.frame_shape[0] / 2, state.frame_shape[1] / 2
        return [trigger_ramp(cy, cx, reason=f"event Δ={d:.2f} nm")
                for (_, _, d) in state.events]
    return policy


def pause_on_event() -> Policy:
    """Pause acquisition when a step-like event is detected."""
    def policy(state: AnalysisState) -> List[Command]:
        return [pause(reason=f"event Δ={state.events[0][2]:.2f} nm")] if state.events else []
    return policy


def combine_policies(*policies: Policy) -> Policy:
    """Run several policies and concatenate their commands."""
    def policy(state: AnalysisState) -> List[Command]:
        out: List[Command] = []
        for p in policies:
            out.extend(p(state))
        return out
    return policy


# ==========================================================================
# frame sources
# ==========================================================================
class FrameSource(Protocol):
    """The instrument seam: iterate frames, optionally accept commands.

    A real adapter wraps the vendor SDK; ``pixel_size_nm`` / ``frame_interval_s``
    carry calibration, ``__iter__`` yields ``(H, W)`` height frames as they are
    acquired, and ``send`` applies a supervisory :class:`Command` (a no-op for a
    read-only source).
    """

    pixel_size_nm: float
    frame_interval_s: float

    def __iter__(self): ...

    def send(self, command: Command) -> None: ...


class VideoFrameSource:
    """Play a recorded :class:`~nanoindent.video.AFMVideo` back frame by frame.

    A read-only source (``send`` logs commands but has no effect) — use it to
    develop and benchmark the streaming pipeline against real recorded movies.
    """

    def __init__(self, video):
        self.video = video
        self.pixel_size_nm = float(np.mean(video.pixel_size_nm))
        self.frame_interval_s = video.frame_interval_s
        self.command_log: List[Tuple[int, Command]] = []
        self._i = -1

    def __iter__(self):
        for i in range(self.video.n_frames):
            self._i = i
            yield self.video.frames[i]

    def send(self, command: Command) -> None:
        self.command_log.append((self._i, command))


class SimulatedInstrument:
    """A reactive ``FrameSource``: diffusing particles imaged through a movable
    scan window that responds to :class:`Command` s.

    Particles undergo Brownian motion in a field larger than the scan window;
    each frame renders the current window. ``send(recenter(...))`` moves the
    window, so a following policy can be shown to keep a feature in view. Ramps,
    setpoint changes and pause/resume are logged (and pause freezes motion).
    Particle 0 is the brightest ("feature of interest").
    """

    def __init__(self, scan_px: int = 64, pixel_size_nm: float = 2.0,
                 frame_interval_s: float = 0.05, n_particles: int = 4,
                 D_nm2_per_s: float = 200.0, field_factor: float = 3.0,
                 particle_sigma_nm: float = 5.0, particle_height: float = 3.0,
                 target_boost: float = 2.0, noise_nm: float = 0.15,
                 max_frames: int = 200, seed: int = 0):
        self.scan_px = scan_px
        self.pixel_size_nm = pixel_size_nm
        self.frame_interval_s = frame_interval_s
        self.particle_sigma_nm = particle_sigma_nm
        self.particle_height = particle_height
        self.target_boost = target_boost
        self.noise_nm = noise_nm
        self.max_frames = max_frames

        self._rng = np.random.default_rng(seed)
        self.field_nm = scan_px * pixel_size_nm * field_factor
        self._pos = self._rng.uniform(0.25, 0.75, (n_particles, 2)) * self.field_nm
        self._pos[0] = [self.field_nm / 2, self.field_nm / 2]     # target starts centred
        self._step = np.sqrt(2 * D_nm2_per_s * frame_interval_s)
        self.center = np.array([self.field_nm / 2, self.field_nm / 2])
        self.setpoint = 1.0
        self.paused = False

        self.command_log: List[Tuple[int, Command]] = []
        self.ramp_events: List[Tuple[int, float, float]] = []
        self.center_history: List[np.ndarray] = []
        self.target_history: List[np.ndarray] = []
        self.metadata = {"true_D_nm2_per_s": D_nm2_per_s, "n_particles": n_particles}
        self._frame = -1

    def _render(self) -> np.ndarray:
        n, px = self.scan_px, self.pixel_size_nm
        y0 = self.center[0] - n / 2 * px
        x0 = self.center[1] - n / 2 * px
        yy, xx = np.mgrid[0:n, 0:n]
        Y, X = y0 + yy * px, x0 + xx * px
        img = self._rng.normal(0, self.noise_nm, (n, n))
        for k, (py, pxp) in enumerate(self._pos):
            amp = self.particle_height * (self.target_boost if k == 0 else 1.0)
            img += amp * np.exp(-(((X - pxp) ** 2 + (Y - py) ** 2)
                                  / (2 * self.particle_sigma_nm ** 2)))
        return img

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        self._frame += 1
        if self._frame >= self.max_frames:
            raise StopIteration
        if not self.paused:
            self._pos += self._rng.normal(0, self._step, self._pos.shape)
            self._pos = np.clip(self._pos, 0, self.field_nm)
        self.center_history.append(self.center.copy())
        self.target_history.append(self._pos[0].copy())
        return self._render()

    def send(self, command: Command) -> None:
        self.command_log.append((self._frame, command))
        p = command.params
        if command.kind == "recenter" and not self.paused:
            self.center = np.clip(
                self.center + np.array([p["dy_px"], p["dx_px"]]) * self.pixel_size_nm,
                0, self.field_nm)
        elif command.kind == "trigger_ramp":
            y = self.center[0] + (p["y_px"] - self.scan_px / 2) * self.pixel_size_nm
            x = self.center[1] + (p["x_px"] - self.scan_px / 2) * self.pixel_size_nm
            self.ramp_events.append((self._frame, float(y), float(x)))
        elif command.kind == "set_setpoint":
            self.setpoint = p["value"]
        elif command.kind == "pause":
            self.paused = True
        elif command.kind == "resume":
            self.paused = False

    # diagnostics ----------------------------------------------------------
    def target_offset_px(self) -> np.ndarray:
        """RMS window offset of the target particle over the run (pixels)."""
        c = np.array(self.center_history)
        t = np.array(self.target_history)
        return (t - c) / self.pixel_size_nm


# ==========================================================================
# control loop
# ==========================================================================
@dataclass
class RunReport:
    """Outcome of a control-loop run: timings, commands, events."""

    n_frames: int
    frame_interval_s: float
    proc_ms: np.ndarray
    commands: List[Tuple[int, Command]]
    events: List[Tuple[int, float, float]]
    states: List[AnalysisState]

    def kept_up(self, warmup: int = 1) -> bool:
        """True if every steady-state frame was analysed within its acquisition
        period. The first ``warmup`` frames are excluded — they carry one-time
        import/JIT costs (scipy, FFT planning) that do not recur."""
        steady = self.proc_ms[warmup:]
        return bool(steady.size) and float(steady.max()) < self.frame_interval_s * 1e3

    def real_time_factor(self, warmup: int = 1) -> float:
        """Median steady-state processing time as a fraction of the frame period
        (< 1 means spare headroom)."""
        steady = self.proc_ms[warmup:]
        med = float(np.median(steady)) if steady.size else float(np.median(self.proc_ms))
        return med / (self.frame_interval_s * 1e3)


def run_control_loop(source, analyzer: Optional[StreamingAnalyzer] = None,
                     policy: Optional[Policy] = None, *,
                     max_frames: Optional[int] = None, apply: bool = True,
                     on_state: Optional[Callable[[AnalysisState], None]] = None,
                     capture_states: bool = False) -> RunReport:
    """Stream frames from ``source`` through ``analyzer`` and ``policy``.

    For each frame: analyse it, ask the policy for commands, time the work, and
    (if ``apply``) send the commands back to the source. Returns a
    :class:`RunReport`. Set ``apply=False`` for an open-loop (read-only) run.
    """
    if analyzer is None:
        analyzer = StreamingAnalyzer(getattr(source, "pixel_size_nm", 1.0),
                                     getattr(source, "frame_interval_s", 0.1))
    proc: List[float] = []
    cmds: List[Tuple[int, Command]] = []
    events: List[Tuple[int, float, float]] = []
    states: List[AnalysisState] = []

    for i, frame in enumerate(source):
        if max_frames is not None and i >= max_frames:
            break
        t0 = time.perf_counter()
        state = analyzer.push(frame)
        out = policy(state) if policy else []
        proc.append((time.perf_counter() - t0) * 1e3)
        for c in out:
            cmds.append((state.frame_index, c))
            if apply and hasattr(source, "send"):
                source.send(c)
        events.extend(state.events)
        if on_state:
            on_state(state)
        if capture_states:
            states.append(state)

    return RunReport(len(proc), analyzer.frame_interval_s, np.array(proc),
                     cmds, events, states)


# ==========================================================================
# visualisation
# ==========================================================================
def plot_control_run(report: RunReport, instrument: Optional[SimulatedInstrument] = None):
    """Three-panel run summary: per-frame timing, feature-following, timeline."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))

    # 1) processing time vs the real-time budget
    fr = np.arange(report.n_frames)
    axs[0].plot(fr, report.proc_ms, ".-", ms=3, color="#0072B2", label="analysis")
    budget = report.frame_interval_s * 1e3
    axs[0].axhline(budget, color="crimson", ls="--", label=f"frame period {budget:.0f} ms")
    axs[0].set_xlabel("frame"); axs[0].set_ylabel("processing time (ms)")
    axs[0].set_title(f"real-time factor {report.real_time_factor():.2f}×", fontsize=9)
    axs[0].legend(fontsize=7); axs[0].grid(alpha=0.25); axs[0].set_ylim(bottom=0)

    # 2) feature following (needs the simulated instrument's ground truth)
    if instrument is not None:
        off = instrument.target_offset_px()
        rms = float(np.sqrt(np.mean(np.sum(off ** 2, axis=1))))
        axs[1].plot(off[:, 1], off[:, 0], ".-", ms=3, color="#009E73")
        axs[1].scatter([0], [0], marker="+", s=120, color="crimson")
        half = instrument.scan_px / 2
        axs[1].set_xlim(-half, half); axs[1].set_ylim(-half, half)
        axs[1].set_aspect("equal")
        axs[1].set_xlabel("x offset (px)"); axs[1].set_ylabel("y offset (px)")
        axs[1].set_title(f"target vs window centre (RMS {rms:.1f} px)", fontsize=9)
        axs[1].grid(alpha=0.25)
    else:
        axs[1].axis("off")

    # 3) command / event timeline
    for fi, c in report.commands:
        color = {"recenter": "#0072B2", "trigger_ramp": "crimson",
                 "pause": "orange"}.get(c.kind, "gray")
        axs[2].vlines(fi, 0, 1, color=color, lw=0.8)
    for fi, _, _ in report.events:
        axs[2].plot(fi, 1.05, "v", color="crimson", ms=6)
    axs[2].set_ylim(0, 1.2); axs[2].set_yticks([])
    axs[2].set_xlabel("frame")
    axs[2].set_title(f"{len(report.commands)} commands, {len(report.events)} events",
                     fontsize=9)
    fig.tight_layout()
    return fig
