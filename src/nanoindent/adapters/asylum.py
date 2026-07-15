"""Asylum Research adapter (scaffold).

Asylum's controller runs on **Igor Pro**, so there is no native Python API: the
adapter talks to an **Igor-side listener** that receives command strings and
executes Igor functions (setting global variables / calling procedures), and
returns frame data. Transport is typically a socket or a watched command file.

The command grammar below is a documented-shape **placeholder** (``VERIFY``):
Igor calls such as ``PV("SetpointDefl", v)``, a scan-offset setter, and a
force-ramp trigger. The adapter is validated against :class:`MockIgorBridge`,
which maps these strings onto the shared physics backend; a real deployment
swaps in a socket/file bridge to the running Igor experiment.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..streaming import Command
from .base import InstrumentAdapter
from .emulation import BackendSimulator


# ==========================================================================
# mock Igor bridge (maps Igor command strings onto the physics backend)
# ==========================================================================
class MockIgorBridge:
    """In-process stand-in for a socket/file bridge to an Igor listener."""

    def __init__(self, backend: Optional[BackendSimulator] = None, **backend_kwargs):
        self.backend = backend or BackendSimulator(**backend_kwargs)
        self.history: list = []

    def execute(self, cmd: str) -> str:
        """Run one Igor command string; returns ``"OK"`` or an error."""
        self.history.append(cmd)
        name, *args = cmd.replace("(", " ").replace(")", " ").replace(",", " ").split()
        b = self.backend
        if name == "SetScanOffset":            # SetScanOffset(dy_nm, dx_nm)
            b.move_center_nm(float(args[0]), float(args[1])); return "OK"
        if name == "PV" and args and args[0].strip('"') == "SetpointDefl":
            b.set_setpoint(float(args[1])); return "OK"
        if name == "DoForceRamp":              # DoForceRamp(dy_nm, dx_nm)
            b.ramp_offset_nm(float(args[0]), float(args[1])); return "OK"
        if name == "StopScan":
            b.pause(); return "OK"
        if name == "StartScan":
            b.resume(); return "OK"
        return f"ERR unknown Igor call {name}"

    def read_frame(self) -> Optional[np.ndarray]:
        """VERIFY: real bridge returns the latest scan frame from Igor waves."""
        return self.backend.acquire()

    def close(self) -> None:
        pass


# ==========================================================================
# adapter
# ==========================================================================
class AsylumAdapter(InstrumentAdapter):
    """Drive an Asylum (Igor Pro) AFM via an Igor-side listener bridge.

    Pass ``bridge=MockIgorBridge(...)`` to run without hardware; a real
    deployment supplies a socket/file bridge speaking the same ``execute`` /
    ``read_frame`` contract.
    """

    def __init__(self, bridge: Optional[object] = None, *,
                 pixel_size_nm: float = 2.0, frame_interval_s: float = 0.05,
                 max_frames: Optional[int] = None):
        super().__init__(pixel_size_nm, frame_interval_s, max_frames=max_frames)
        self._bridge = bridge

    def _connect(self) -> None:
        if self._bridge is None:
            raise RuntimeError(
                "AsylumAdapter needs an Igor bridge. Provide bridge=MockIgorBridge() "
                "for testing, or a socket/file bridge to the running Igor experiment.")

    def _acquire_frame(self) -> Optional[np.ndarray]:
        return self._bridge.read_frame()

    def _apply_command(self, command: Command) -> None:
        br = self._bridge
        if command.kind == "recenter":
            dy, dx = self._recenter_offset_nm(command)
            br.execute(f"SetScanOffset({dy:.5f}, {dx:.5f})")
        elif command.kind == "set_setpoint":
            br.execute(f'PV("SetpointDefl", {command.params["value"]:.6g})')
        elif command.kind == "trigger_ramp":
            dy, dx = self._point_offset_nm(command.params["y_px"],
                                           command.params["x_px"])
            br.execute(f"DoForceRamp({dy:.5f}, {dx:.5f})")
        elif command.kind == "pause":
            br.execute("StopScan()")
        elif command.kind == "resume":
            br.execute("StartScan()")

    def _disconnect(self) -> None:
        if self._bridge is not None:
            self._bridge.close()
