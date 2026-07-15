"""Shared physics backend for the vendor mock emulators.

Each vendor mock (``MockNanosurfSPM``, ``MockJPKConnection``, ``MockIgorBridge``)
adapts its own API idiom onto the primitive operations of this
:class:`BackendSimulator`, which wraps the reactive
:class:`nanoindent.streaming.SimulatedInstrument`. That way all three adapters
drive the *same* validated physics, so a passing closed-loop test proves the
adapter's command translation is correct — the only thing left to swap in is the
real transport.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..streaming import (SimulatedInstrument, recenter, set_setpoint,
                         trigger_ramp, pause, resume)


class BackendSimulator:
    """A :class:`SimulatedInstrument` exposed through instrument-style primitives.

    Accepts the same keyword arguments as :class:`SimulatedInstrument`.
    """

    def __init__(self, **kwargs):
        self.instrument = SimulatedInstrument(**kwargs)
        self._frames = iter(self.instrument)

    @property
    def pixel_size_nm(self) -> float:
        return self.instrument.pixel_size_nm

    @property
    def frame_interval_s(self) -> float:
        return self.instrument.frame_interval_s

    @property
    def scan_px(self) -> int:
        return self.instrument.scan_px

    # ---- primitive operations the vendor mocks call ---------------------
    def acquire(self) -> Optional[np.ndarray]:
        """Next frame, or ``None`` when the run is exhausted."""
        try:
            return next(self._frames)
        except StopIteration:
            return None

    def move_center_nm(self, dy_nm: float, dx_nm: float) -> None:
        px = self.instrument.pixel_size_nm
        self.instrument.send(recenter(dy_nm / px, dx_nm / px))

    def set_setpoint(self, value: float) -> None:
        self.instrument.send(set_setpoint(value))

    def ramp_offset_nm(self, dy_nm: float, dx_nm: float) -> None:
        px, n = self.instrument.pixel_size_nm, self.instrument.scan_px
        self.instrument.send(trigger_ramp(dy_nm / px + n / 2, dx_nm / px + n / 2))

    def pause(self) -> None:
        self.instrument.send(pause())

    def resume(self) -> None:
        self.instrument.send(resume())
