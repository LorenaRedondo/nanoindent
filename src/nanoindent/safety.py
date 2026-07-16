"""Safety / robustness envelope for the supervisory control loop.

A :class:`SafetyEnvelope` wraps a *policy* and enforces guards before commands
reach the instrument:

* **rate limiting** — no more than one actuating command every N frames;
* **scan-bounds guard** — clip re-centering so the accumulated scan offset never
  leaves an allowed radius (the loop can't walk the scan off the sample);
* **setpoint guard** — clamp ``set_setpoint`` to a safe range;
* **auto-pause on lost tracking** — pause after K frames with no feature, resume
  on recovery;
* **frame-gap watchdog** — pause if the time between frames exceeds a limit
  (a stalled acquisition).

This is *supervisory* only — it governs the high-level command stream. The fast
tip-protection z-feedback stays on the instrument's own real-time controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable, List, Optional, Tuple

from .streaming import AnalysisState, Command, pause, resume, recenter, set_setpoint

Policy = Callable[[AnalysisState], List[Command]]


@dataclass
class SafetyLimits:
    """Guard thresholds; leave a field ``None`` to disable that guard."""

    min_frames_between_commands: int = 0
    max_scan_offset_px: Optional[float] = None
    setpoint_range: Optional[Tuple[float, float]] = None
    pause_after_lost_frames: Optional[int] = None
    resume_on_recovery: bool = True
    max_frame_gap_s: Optional[float] = None


class SafetyEnvelope:
    """Wrap a policy with the guards in :class:`SafetyLimits`.

    ``counts`` tallies how often each guard fired (for observability), and
    :attr:`paused` reflects the auto-pause state.
    """

    def __init__(self, limits: SafetyLimits):
        self.limits = limits
        self._last_cmd_frame = -10 ** 9
        self._cum_dy = 0.0
        self._cum_dx = 0.0
        self._lost = 0
        self._paused = False
        self._last_time: Optional[float] = None
        self.counts = {"throttled": 0, "clamped_setpoint": 0,
                       "clipped_bounds": 0, "auto_pauses": 0,
                       "auto_resumes": 0, "watchdog_trips": 0}

    @property
    def paused(self) -> bool:
        return self._paused

    def guard(self, policy: Policy) -> Policy:
        """Return a policy that applies the safety guards to ``policy``'s output."""
        def guarded(state: AnalysisState) -> List[Command]:
            safety = self._safety_commands(state)
            if self._paused:
                return safety                       # suppress normal actuation
            return safety + self._filter(state, policy(state))
        return guarded

    # -- injected safety commands -----------------------------------------
    def _safety_commands(self, state: AnalysisState) -> List[Command]:
        out: List[Command] = []
        L = self.limits

        # frame-gap watchdog (wall clock)
        if L.max_frame_gap_s is not None:
            now = time.perf_counter()
            if self._last_time is not None and now - self._last_time > L.max_frame_gap_s:
                self.counts["watchdog_trips"] += 1
                if not self._paused:
                    self._paused = True
                    out.append(pause(reason="watchdog: frame gap"))
            self._last_time = now

        # auto-pause / resume on lost tracking
        if L.pause_after_lost_frames is not None:
            if state.n_particles == 0:
                self._lost += 1
                if not self._paused and self._lost >= L.pause_after_lost_frames:
                    self._paused = True
                    self.counts["auto_pauses"] += 1
                    out.append(pause(reason="lost tracking"))
            else:
                if self._paused and L.resume_on_recovery:
                    self._paused = False
                    self.counts["auto_resumes"] += 1
                    out.append(resume(reason="tracking recovered"))
                self._lost = 0
        return out

    # -- filter / clamp outgoing commands ---------------------------------
    def _filter(self, state: AnalysisState, cmds: List[Command]) -> List[Command]:
        L = self.limits
        actuating = {"recenter", "trigger_ramp", "set_setpoint"}
        out: List[Command] = []
        for c in cmds:
            if c.kind in actuating and L.min_frames_between_commands > 0 and \
                    state.frame_index - self._last_cmd_frame < L.min_frames_between_commands:
                self.counts["throttled"] += 1
                continue
            if c.kind == "set_setpoint" and L.setpoint_range is not None:
                lo, hi = L.setpoint_range
                v = c.params["value"]
                if v < lo or v > hi:
                    self.counts["clamped_setpoint"] += 1
                    c = set_setpoint(min(max(v, lo), hi), reason="clamped to range")
            if c.kind == "recenter" and L.max_scan_offset_px is not None:
                dy, dx = c.params["dy_px"], c.params["dx_px"]
                ny, nx = self._cum_dy + dy, self._cum_dx + dx
                r = (ny ** 2 + nx ** 2) ** 0.5
                if r > L.max_scan_offset_px:
                    scale = L.max_scan_offset_px / r if r > 0 else 0.0
                    ny, nx = ny * scale, nx * scale
                    dy, dx = ny - self._cum_dy, nx - self._cum_dx
                    self.counts["clipped_bounds"] += 1
                    c = recenter(dy, dx, reason="clipped to scan bounds")
                self._cum_dy += dy
                self._cum_dx += dx
            out.append(c)
            if c.kind in actuating:
                self._last_cmd_frame = state.frame_index
        return out
