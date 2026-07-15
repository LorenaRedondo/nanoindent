"""Nanosurf adapter (scaffold).

Nanosurf ships an official ``nanosurf`` Python package (a COM/studio API), so
the adapter is in-process Python calls — the most Python-native of the Tier-2
vendors. Real usage looks roughly like::

    import nanosurf
    spm = nanosurf.SPM()                    # connect to the running app
    app = spm.application
    app.Scan.CenterPosX / .CenterPosY       # metres
    app.ZController.SetPoint                 # setpoint
    # frame data via the scan buffer / a captured-frame callback

The exact property/method names below are reconstructed from documentation and
marked ``VERIFY`` — confirm them against the installed ``nanosurf`` package. The
adapter is validated end-to-end against :class:`MockNanosurfSPM`, which mimics
this surface on top of the shared physics backend.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..streaming import Command
from .base import InstrumentAdapter
from .emulation import BackendSimulator


# ==========================================================================
# mock SPM (mimics the nanosurf.SPM() object surface for tests / demos)
# ==========================================================================
class _MockScan:
    def __init__(self, backend: BackendSimulator):
        self._b = backend
        self.CenterPosX = 0.0          # metres (VERIFY: real API units)
        self.CenterPosY = 0.0
        self.scanning = True

    def capture_frame(self) -> Optional[np.ndarray]:
        """VERIFY: real API delivers frames via a scan buffer / frame-ready event."""
        return self._b.acquire()

    def move_center(self, dy_nm: float, dx_nm: float) -> None:
        # VERIFY: real code sets absolute CenterPosX/Y in metres, e.g.
        #   self.CenterPosY += dy_nm * 1e-9
        self.CenterPosY += dy_nm * 1e-9
        self.CenterPosX += dx_nm * 1e-9
        self._b.move_center_nm(dy_nm, dx_nm)

    def spectroscopy_at(self, dy_nm: float, dx_nm: float) -> None:
        """VERIFY: real API triggers a force-distance spectroscopy at a point."""
        self._b.ramp_offset_nm(dy_nm, dx_nm)

    def stop(self) -> None:
        self.scanning = False
        self._b.pause()

    def start(self) -> None:
        self.scanning = True
        self._b.resume()


class _MockZController:
    def __init__(self, backend: BackendSimulator):
        self._b = backend
        self._setpoint = 1.0

    @property
    def SetPoint(self) -> float:
        return self._setpoint

    @SetPoint.setter
    def SetPoint(self, value: float) -> None:
        self._setpoint = value
        self._b.set_setpoint(value)


class _MockApplication:
    def __init__(self, backend: BackendSimulator):
        self.Scan = _MockScan(backend)
        self.ZController = _MockZController(backend)


class MockNanosurfSPM:
    """In-process stand-in for ``nanosurf.SPM()`` backed by simulated physics."""

    def __init__(self, backend: Optional[BackendSimulator] = None, **backend_kwargs):
        self.backend = backend or BackendSimulator(**backend_kwargs)
        self.application = _MockApplication(self.backend)
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False


# ==========================================================================
# adapter
# ==========================================================================
class NanosurfAdapter(InstrumentAdapter):
    """Drive a Nanosurf SPM as a :class:`~nanoindent.streaming.FrameSource`.

    Pass ``spm=MockNanosurfSPM(...)`` to run without hardware; leave it ``None``
    to connect to a real instrument via the ``nanosurf`` package.
    """

    def __init__(self, spm: Optional[object] = None, *,
                 pixel_size_nm: float = 2.0, frame_interval_s: float = 0.05,
                 max_frames: Optional[int] = None, owns_connection: bool = True):
        super().__init__(pixel_size_nm, frame_interval_s, max_frames=max_frames)
        self._spm = spm
        self._app = None
        self._owns = owns_connection

    def _connect(self) -> None:
        if self._spm is None:
            import nanosurf                       # VERIFY: real connection
            self._spm = nanosurf.SPM()
        if hasattr(self._spm, "connect"):
            self._spm.connect()
        self._app = self._spm.application

    def _acquire_frame(self) -> Optional[np.ndarray]:
        return self._app.Scan.capture_frame()

    def _apply_command(self, command: Command) -> None:
        scan = self._app.Scan
        if command.kind == "recenter":
            dy, dx = self._recenter_offset_nm(command)
            scan.move_center(dy, dx)
        elif command.kind == "set_setpoint":
            self._app.ZController.SetPoint = command.params["value"]
        elif command.kind == "trigger_ramp":
            dy, dx = self._point_offset_nm(command.params["y_px"],
                                           command.params["x_px"])
            scan.spectroscopy_at(dy, dx)
        elif command.kind == "pause":
            scan.stop()
        elif command.kind == "resume":
            scan.start()

    def _disconnect(self) -> None:
        if self._owns and self._spm is not None and hasattr(self._spm, "disconnect"):
            self._spm.disconnect()
