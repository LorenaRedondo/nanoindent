"""Nanosurf adapter.

Nanosurf ships an official ``nanosurf`` Python package (a COM/studio API), so
the adapter is in-process Python calls — the most Python-native of the Tier-2
vendors. Real usage follows the documented object tree::

    import nanosurf
    spm = nanosurf.SPM()                       # or CoreAFM() / DriveAFM() / C3000()
    app = spm.application
    app.Scan.ImageWidth / .Points / .Lines     # geometry (metres / px)
    app.Scan.CenterPosX / .CenterPosY          # scan centre (metres)
    app.Scan.Scantime                          # per-frame time (s)
    app.Scan.Start() / .Stop()
    app.ZController.SetPoint / .PGain / .IGain
    app.OperatingMode.OperatingMode
    app.Approach.StartApproach() / .StartWithdraw()
    app.Spec  (spectroscopy: force–distance curves)

This module models that tree so the exact calls the adapter would make against
the real package are exercised end-to-end against :class:`MockNanosurfSPM`. The
property/method names track the Nanosurf scripting reference; anything not
independently confirmed is marked ``VERIFY`` (notably the frame-buffer read,
which is event-driven on real hardware). Being in-process, there is no wire
protocol — the adapter reads calibration from the instrument on connect
(``auto_calibrate``), drives approach/withdraw, feedback, operating mode, and
force spectroscopy, exactly as a real session would.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..streaming import Command
from .base import InstrumentAdapter
from .emulation import BackendSimulator


# ==========================================================================
# types
# ==========================================================================
@dataclass
class NanosurfCapabilities:
    """Instrument geometry / mode read from the controller on connect."""

    model: str = ""
    pixel_size_nm: float = 1.0
    frame_interval_s: float = 0.1
    points: int = 0
    lines: int = 0
    operating_mode: str = ""


@dataclass
class NanosurfForceCurve:
    """A force–distance curve from a spectroscopy (``Spec``) run."""

    z_nm: np.ndarray
    force_pN: np.ndarray
    x_nm: float = 0.0
    y_nm: float = 0.0

    @property
    def max_force_pN(self) -> float:
        return float(np.max(self.force_pN)) if self.force_pN.size else 0.0


def _synthetic_force_curve(dy_nm: float, dx_nm: float, n: int = 128,
                           contact_nm: float = 8.0, R: float = 20.0,
                           E: float = 50.0) -> NanosurfForceCurve:
    """A Hertz-shaped approach curve (contact then ~5 nm indentation), mock only."""
    from .. import analytics
    z = np.linspace(0.0, 13.0, n)
    delta = np.clip(z - contact_nm, 0.0, None)
    F = analytics.hertz_force(delta, R, E, 0.5)
    return NanosurfForceCurve(z_nm=z, force_pN=F, x_nm=dx_nm, y_nm=dy_nm)


# ==========================================================================
# mock SPM — mirrors the real nanosurf.SPM() object tree
# ==========================================================================
class _MockScan:
    """``application.Scan`` — geometry, scan centre, start/stop, frame buffer."""

    def __init__(self, backend: BackendSimulator):
        self._b = backend
        px_m = backend.pixel_size_nm * 1e-9
        self.Points = backend.scan_px
        self.Lines = backend.scan_px
        self.ImageWidth = backend.scan_px * px_m          # metres
        self.ImageHeight = backend.scan_px * px_m
        self.Scantime = backend.scan_px * backend.frame_interval_s   # s / frame
        self.Rotation = 0.0
        self.IsScanning = True
        self._cx = 0.0
        self._cy = 0.0

    # scan centre — absolute, in metres (real API); setting it moves the window
    @property
    def CenterPosX(self) -> float:
        return self._cx

    @CenterPosX.setter
    def CenterPosX(self, value: float) -> None:
        dx_nm = (value - self._cx) * 1e9
        self._cx = value
        if dx_nm:
            self._b.move_center_nm(0.0, dx_nm)

    @property
    def CenterPosY(self) -> float:
        return self._cy

    @CenterPosY.setter
    def CenterPosY(self, value: float) -> None:
        dy_nm = (value - self._cy) * 1e9
        self._cy = value
        if dy_nm:
            self._b.move_center_nm(dy_nm, 0.0)

    def Start(self) -> None:
        self.IsScanning = True
        self._b.resume()

    def Stop(self) -> None:
        self.IsScanning = False
        self._b.pause()

    def get_frame(self) -> Optional[np.ndarray]:
        """VERIFY: on real hardware frames arrive via a scan data buffer /
        line- or frame-ready event, not a blocking pull."""
        return self._b.acquire()


class _MockZController:
    """``application.ZController`` — feedback setpoint and gains."""

    def __init__(self, backend: BackendSimulator):
        self._b = backend
        self._setpoint = 1.0
        self.PGain = 1000.0
        self.IGain = 1000.0
        self.Enabled = True

    @property
    def SetPoint(self) -> float:
        return self._setpoint

    @SetPoint.setter
    def SetPoint(self, value: float) -> None:
        self._setpoint = value
        self._b.set_setpoint(value)


class _MockOperatingMode:
    def __init__(self):
        self.OperatingMode = "DynamicAFM"          # VERIFY: real enum values
        self.VibratingAmpl = 1.0


class _MockApproach:
    """``application.Approach`` — engage / withdraw state machine."""

    def __init__(self):
        self.Status = "Idle"

    def StartApproach(self) -> None:
        self.Status = "Engaged"

    def StartWithdraw(self) -> None:
        self.Status = "Retracted"

    StartRetract = StartWithdraw


class _MockSpec:
    """``application.Spec`` — force–distance spectroscopy."""

    def __init__(self, backend: BackendSimulator):
        self._b = backend

    def force_distance(self, dy_nm: float, dx_nm: float) -> NanosurfForceCurve:
        self._b.ramp_offset_nm(dy_nm, dx_nm)
        return _synthetic_force_curve(dy_nm, dx_nm)


class _MockApplication:
    def __init__(self, backend: BackendSimulator):
        self.Scan = _MockScan(backend)
        self.ZController = _MockZController(backend)
        self.OperatingMode = _MockOperatingMode()
        self.Approach = _MockApproach()
        self.Spec = _MockSpec(backend)


class MockNanosurfSPM:
    """In-process stand-in for ``nanosurf.SPM()`` backed by simulated physics."""

    def __init__(self, backend: Optional[BackendSimulator] = None, *,
                 model: str = "CoreAFM", **backend_kwargs):
        self.backend = backend or BackendSimulator(**backend_kwargs)
        self.application = _MockApplication(self.backend)
        self.model = model
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

    Parameters
    ----------
    auto_calibrate:
        Read pixel size / frame interval from the scan geometry on connect.
    record_ramps:
        On a ``trigger_ramp`` command, run a force–distance spectroscopy and keep
        the :class:`NanosurfForceCurve` in :attr:`force_curves`.
    """

    def __init__(self, spm: Optional[object] = None, *,
                 pixel_size_nm: float = 2.0, frame_interval_s: float = 0.05,
                 max_frames: Optional[int] = None, owns_connection: bool = True,
                 auto_calibrate: bool = True, record_ramps: bool = False):
        super().__init__(pixel_size_nm, frame_interval_s, max_frames=max_frames)
        self._spm = spm
        self._app = None
        self._owns = owns_connection
        self._auto_calibrate = auto_calibrate
        self._record_ramps = record_ramps
        self.capabilities: Optional[NanosurfCapabilities] = None
        self.force_curves: List[NanosurfForceCurve] = []

    # -- lifecycle ---------------------------------------------------------
    def _connect(self) -> None:
        if self._spm is None:
            import nanosurf                       # VERIFY: real connection
            self._spm = nanosurf.SPM()
        if hasattr(self._spm, "connect"):
            self._spm.connect()
        self._app = self._spm.application
        self.capabilities = self._read_capabilities()
        if self._auto_calibrate and self.capabilities:
            if self.capabilities.pixel_size_nm:
                self.pixel_size_nm = self.capabilities.pixel_size_nm
            if self.capabilities.frame_interval_s:
                self.frame_interval_s = self.capabilities.frame_interval_s

    def _read_capabilities(self) -> NanosurfCapabilities:
        scan = self._app.Scan
        px = (scan.ImageWidth / scan.Points) * 1e9 if scan.Points else self.pixel_size_nm
        dt = (scan.Scantime / scan.Lines) if scan.Lines else self.frame_interval_s
        return NanosurfCapabilities(
            model=getattr(self._spm, "model", ""), pixel_size_nm=px,
            frame_interval_s=dt, points=scan.Points, lines=scan.Lines,
            operating_mode=getattr(self._app.OperatingMode, "OperatingMode", ""))

    def _acquire_frame(self) -> Optional[np.ndarray]:
        return self._app.Scan.get_frame()

    def _apply_command(self, command: Command) -> None:
        app, k = self._app, command.kind
        if k == "recenter":
            dy, dx = self._recenter_offset_nm(command)
            app.Scan.CenterPosY = app.Scan.CenterPosY + dy * 1e-9   # metres
            app.Scan.CenterPosX = app.Scan.CenterPosX + dx * 1e-9
        elif k == "set_setpoint":
            app.ZController.SetPoint = command.params["value"]
        elif k == "trigger_ramp":
            dy, dx = self._point_offset_nm(command.params["y_px"],
                                           command.params["x_px"])
            curve = app.Spec.force_distance(dy, dx)
            if self._record_ramps:
                self.force_curves.append(curve)
        elif k == "pause":
            app.Scan.Stop()
        elif k == "resume":
            app.Scan.Start()
        elif k == "engage":
            app.Approach.StartApproach()
        elif k == "withdraw":
            app.Approach.StartWithdraw()
        elif k == "set_gains":
            app.ZController.PGain = command.params["p"]
            app.ZController.IGain = command.params["i"]

    def _disconnect(self) -> None:
        if self._owns and self._spm is not None and hasattr(self._spm, "disconnect"):
            self._spm.disconnect()

    # -- direct instrument control (outside the loop) ----------------------
    def engage(self) -> None:
        self._ensure_connected(); self._app.Approach.StartApproach()

    def withdraw(self) -> None:
        self._ensure_connected(); self._app.Approach.StartWithdraw()

    def set_setpoint(self, value: float) -> None:
        self._ensure_connected(); self._app.ZController.SetPoint = value

    def set_feedback_gains(self, p: float, i: float) -> None:
        self._ensure_connected()
        self._app.ZController.PGain = p
        self._app.ZController.IGain = i

    def set_operating_mode(self, mode: str) -> None:
        self._ensure_connected()
        self._app.OperatingMode.OperatingMode = mode

    def force_curve(self, y_nm: float = 0.0, x_nm: float = 0.0) -> NanosurfForceCurve:
        self._ensure_connected()
        curve = self._app.Spec.force_distance(y_nm, x_nm)
        self.force_curves.append(curve)
        return curve

    @property
    def approach_status(self) -> str:
        return self._app.Approach.Status if self._app else "Idle"
