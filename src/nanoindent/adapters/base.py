"""Base class for real-instrument ``FrameSource`` adapters.

An adapter wraps a vendor's control interface and exposes the same
``__iter__`` / ``send`` contract as :class:`nanoindent.streaming.FrameSource`,
so it drops straight into :func:`nanoindent.streaming.run_control_loop`. The
base class handles the lifecycle (connect → stream frames → disconnect),
command logging, and the pixel↔nm conversions policies need; subclasses
implement only the three transport hooks.

Verification status
-------------------
The concrete vendor adapters (:mod:`.nanosurf`, :mod:`.jpk`, :mod:`.asylum`)
are **scaffolds**: their command translation and streaming logic are exercised
end-to-end against in-process **mock** backends (see :mod:`.emulation`), but the
real vendor calls / wire protocols are reconstructed from documentation and
**must be verified against the actual SDK and instrument**. Every place that
needs hardware confirmation is marked ``VERIFY``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from ..streaming import Command


class InstrumentAdapter(ABC):
    """A ``FrameSource`` backed by a real (or mocked) instrument.

    Parameters
    ----------
    pixel_size_nm, frame_interval_s:
        Calibration carried on the ``FrameSource`` contract.
    max_frames:
        Stop after this many frames (``None`` = until the source is exhausted).

    Subclasses implement :meth:`_connect`, :meth:`_acquire_frame` and
    :meth:`_apply_command` (and optionally :meth:`_disconnect`).
    """

    def __init__(self, pixel_size_nm: float, frame_interval_s: float, *,
                 max_frames: Optional[int] = None):
        self.pixel_size_nm = float(pixel_size_nm)
        self.frame_interval_s = float(frame_interval_s)
        self.max_frames = max_frames
        self.command_log: List[Command] = []
        self._connected = False
        self._last_shape: Tuple[int, int] = (0, 0)

    # ---- transport hooks (implemented by subclasses) --------------------
    @abstractmethod
    def _connect(self) -> None:
        """Open the vendor connection; set up any handles."""

    @abstractmethod
    def _acquire_frame(self) -> Optional[np.ndarray]:
        """Return the next height frame (nm), or ``None`` at end of stream."""

    @abstractmethod
    def _apply_command(self, command: Command) -> None:
        """Translate and apply one supervisory command to the instrument."""

    def _disconnect(self) -> None:
        """Close the vendor connection (default: nothing)."""

    # ---- helpers for command translation --------------------------------
    def _recenter_offset_nm(self, command: Command) -> Tuple[float, float]:
        """A ``recenter`` command's window-pixel shift as an ``(dy, dx)`` in nm."""
        return (command.params["dy_px"] * self.pixel_size_nm,
                command.params["dx_px"] * self.pixel_size_nm)

    def _point_offset_nm(self, y_px: float, x_px: float) -> Tuple[float, float]:
        """A window-pixel point as an ``(dy, dx)`` offset from the frame centre (nm)."""
        n, m = self._last_shape if self._last_shape != (0, 0) else (0, 0)
        return ((y_px - n / 2) * self.pixel_size_nm,
                (x_px - m / 2) * self.pixel_size_nm)

    # ---- FrameSource contract -------------------------------------------
    def _ensure_connected(self) -> None:
        if not self._connected:
            self._connect()
            self._connected = True

    def __iter__(self):
        self._ensure_connected()
        i = 0
        try:
            while self.max_frames is None or i < self.max_frames:
                frame = self._acquire_frame()
                if frame is None:
                    break
                frame = np.asarray(frame, dtype=float)
                self._last_shape = frame.shape
                yield frame
                i += 1
        finally:
            self._disconnect()
            self._connected = False

    def send(self, command: Command) -> None:
        self.command_log.append(command)
        self._ensure_connected()
        self._apply_command(command)

    # ---- context manager -------------------------------------------------
    def __enter__(self):
        self._ensure_connected()
        return self

    def __exit__(self, *exc):
        if self._connected:
            self._disconnect()
            self._connected = False
        return False
