"""Asylum Research (Igor Pro) adapter.

Asylum's AFM software (MFP-3D, Cypher) runs inside **Igor Pro**, so there is no
native Python API. Control from outside goes through an **Igor-side listener**: a
background task in the Asylum experiment that receives commands and runs Igor
code (``Execute/Q``), sets Asylum global parameters (``PV(...)``), and returns
wave data (frames, force curves). The adapter speaks *Igor command strings*; a
:class:`IgorBridge` handles transport.

Two bridges are provided against one command grammar: :class:`MockIgorBridge`
(in-process, for tests) and :class:`SocketIgorBridge` (TCP to a real Igor
listener). :class:`IgorListenerServer` is a faithful loopback mock so the socket
path is testable without hardware, and ``docs/asylum_igor_listener.ipf`` is a
reference Igor procedure implementing the listener contract.

.. warning::
   The Asylum Igor function / parameter names (``SetScanOffset``,
   ``PV("SetpointDefl", …)``, ``Approach``/``Withdraw``, ``StartFrame`` …) are a
   documented-shape **placeholder**, marked ``VERIFY`` — confirm them against
   the Asylum programmer's guide for your software version. The bridge/transport
   and the adapter logic are exercised end to end; only the exact Igor calls
   need substituting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import socket
import threading
from typing import List, Optional

import numpy as np

from ..streaming import Command
from .base import InstrumentAdapter
from .emulation import BackendSimulator


# ==========================================================================
# types
# ==========================================================================
@dataclass
class AsylumCapabilities:
    """Scan geometry / mode read from Igor globals."""

    scan_size_nm: float = 0.0
    points: int = 0
    lines: int = 0
    scan_rate_hz: float = 0.0
    mode: str = ""

    @property
    def pixel_size_nm(self) -> float:
        return self.scan_size_nm / self.points if self.points else 0.0

    @property
    def frame_interval_s(self) -> float:
        return self.lines / self.scan_rate_hz if self.scan_rate_hz else 0.0


@dataclass
class AsylumForceCurve:
    """A force–distance curve from a single-force spectroscopy."""

    z_nm: np.ndarray
    force_pN: np.ndarray
    x_nm: float = 0.0
    y_nm: float = 0.0

    @property
    def max_force_pN(self) -> float:
        return float(np.max(self.force_pN)) if self.force_pN.size else 0.0


def _synthetic_force_curve(dy_nm: float, dx_nm: float, n: int = 128,
                           contact_nm: float = 8.0, R: float = 20.0,
                           E: float = 50.0) -> AsylumForceCurve:
    from .. import analytics
    z = np.linspace(0.0, 13.0, n)
    delta = np.clip(z - contact_nm, 0.0, None)
    F = analytics.hertz_force(delta, R, E, 0.5)
    return AsylumForceCurve(z_nm=z, force_pN=F, x_nm=dx_nm, y_nm=dy_nm)


def _exec_igor(backend: BackendSimulator, cmd: str) -> str:
    """Drive the physics backend from an Igor command string (mock listener).

    Real Igor runs these with ``Execute/Q``; here we parse the placeholder
    calls. Returns ``OK`` or an ``ERR`` reply.
    """
    name, *args = cmd.replace("(", " ").replace(")", " ").replace(",", " ").split()
    if name == "SetScanOffset":                    # SetScanOffset(dy_nm, dx_nm)
        backend.move_center_nm(float(args[0]), float(args[1])); return "OK"
    if name == "SetScanOffsetAbs":
        return "OK"
    if name == "PV" and args:
        param = args[0].strip('"')
        if param == "SetpointDefl":
            backend.set_setpoint(float(args[1])); return "OK"
        return "OK"                                # IntegralGain / ImagingMode / …
    if name in ("DoForceRamp", "DoSingleForce"):
        backend.ramp_offset_nm(float(args[0]), float(args[1])); return "OK"
    if name in ("StopScan", "StopFrame"):
        backend.pause(); return "OK"
    if name in ("StartScan", "StartFrame"):
        backend.resume(); return "OK"
    if name in ("Approach", "Withdraw", "SetFeedbackGains"):
        return "OK"
    return f"ERR unknown Igor call {name}"


def _caps_string(backend: BackendSimulator, mode: str) -> str:
    size = backend.scan_px * backend.pixel_size_nm
    rate = backend.scan_px / backend.frame_interval_s if backend.frame_interval_s else 0.0
    return (f"CAPS scan_size_nm={size} points={backend.scan_px} "
            f"lines={backend.scan_px} scan_rate_hz={rate} mode={mode}")


def _parse_caps(reply: str) -> AsylumCapabilities:
    d = {}
    if reply.startswith("CAPS"):
        for tok in reply.split()[1:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                d[k] = v
    return AsylumCapabilities(
        scan_size_nm=float(d.get("scan_size_nm", 0.0)),
        points=int(d.get("points", 0)), lines=int(d.get("lines", 0)),
        scan_rate_hz=float(d.get("scan_rate_hz", 0.0)), mode=d.get("mode", ""))


# ==========================================================================
# in-process bridge (no sockets)
# ==========================================================================
class MockIgorBridge:
    """In-process stand-in for a socket/file bridge to an Igor listener."""

    def __init__(self, backend: Optional[BackendSimulator] = None, *,
                 mode: str = "AC", **backend_kwargs):
        self.backend = backend or BackendSimulator(**backend_kwargs)
        self.mode = mode
        self.history: list = []

    def execute(self, cmd: str) -> str:
        """Run one Igor command string; returns ``"OK"`` or an error."""
        self.history.append(cmd)
        return _exec_igor(self.backend, cmd)

    def capabilities(self) -> AsylumCapabilities:
        return _parse_caps(_caps_string(self.backend, self.mode))

    def read_frame(self) -> Optional[np.ndarray]:
        """VERIFY: real bridge returns the latest scan frame from Igor waves."""
        return self.backend.acquire()

    def force_curve(self, dy_nm: float, dx_nm: float) -> AsylumForceCurve:
        self.history.append(f"DoSingleForce({dy_nm:.5f}, {dx_nm:.5f})")
        self.backend.ramp_offset_nm(dy_nm, dx_nm)
        return _synthetic_force_curve(dy_nm, dx_nm)

    def close(self) -> None:
        pass


# ==========================================================================
# real TCP bridge to an Igor listener
# ==========================================================================
class SocketIgorBridge:
    """TCP client to an Igor-side listener (see ``docs/asylum_igor_listener.ipf``).

    Protocol (text, newline-terminated)::

        EXEC <igor command>   -> OK | ERR <msg>
        CAPS                  -> CAPS scan_size_nm=.. points=.. lines=.. …
        FRAME                 -> IMG h w itemsize\\n<f8 bytes> | NONE
        FORCE <dy> <dx>       -> FD n\\n<2n f8: z_nm,F_pN>
        CLOSE
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 55070,
                 timeout: float = 10.0):
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._rf = self._sock.makefile("rb")

    def _line(self, text: str) -> str:
        self._sock.sendall((text + "\n").encode())
        return self._rf.readline().decode().strip()

    def execute(self, cmd: str) -> str:
        reply = self._line(f"EXEC {cmd}")
        if reply.startswith("ERR"):
            raise RuntimeError(f"Igor error: {reply}")
        return reply

    def capabilities(self) -> AsylumCapabilities:
        return _parse_caps(self._line("CAPS"))

    def read_frame(self) -> Optional[np.ndarray]:
        self._sock.sendall(b"FRAME\n")
        header = self._rf.readline().decode().strip()
        if header in ("NONE", ""):
            return None
        _, h, w, itemsize = header.split()
        h, w, itemsize = int(h), int(w), int(itemsize)
        raw = self._rf.read(h * w * itemsize)
        return np.frombuffer(raw, dtype="<f8").reshape(h, w)

    def force_curve(self, dy_nm: float, dx_nm: float) -> AsylumForceCurve:
        self._sock.sendall(f"FORCE {dy_nm:.5f} {dx_nm:.5f}\n".encode())
        header = self._rf.readline().decode().strip()
        n = int(header.split()[1])
        raw = self._rf.read(n * 2 * 8)
        inter = np.frombuffer(raw, dtype="<f8")
        return AsylumForceCurve(z_nm=inter[0::2].copy(), force_pN=inter[1::2].copy(),
                                x_nm=dx_nm, y_nm=dy_nm)

    def close(self) -> None:
        try:
            self._sock.sendall(b"CLOSE\n")
        except OSError:
            pass
        try:
            self._rf.close()
        finally:
            self._sock.close()


# ==========================================================================
# adapter
# ==========================================================================
class AsylumAdapter(InstrumentAdapter):
    """Drive an Asylum (Igor Pro) AFM via an Igor-side listener bridge.

    Pass ``bridge=MockIgorBridge(...)`` to run without hardware, or a
    :class:`SocketIgorBridge` to a real Igor listener. ``record_ramps=True``
    stores the F–d curve from each event-triggered single-force run in
    :attr:`force_curves`.
    """

    def __init__(self, bridge: Optional[object] = None, *,
                 pixel_size_nm: float = 2.0, frame_interval_s: float = 0.05,
                 max_frames: Optional[int] = None, auto_calibrate: bool = True,
                 record_ramps: bool = False):
        super().__init__(pixel_size_nm, frame_interval_s, max_frames=max_frames)
        self._bridge = bridge
        self._auto_calibrate = auto_calibrate
        self._record_ramps = record_ramps
        self.capabilities: Optional[AsylumCapabilities] = None
        self.force_curves: List[AsylumForceCurve] = []

    # -- lifecycle ---------------------------------------------------------
    def _connect(self) -> None:
        if self._bridge is None:
            raise RuntimeError(
                "AsylumAdapter needs an Igor bridge. Provide bridge=MockIgorBridge() "
                "for testing, or a SocketIgorBridge to the running Igor experiment.")
        if hasattr(self._bridge, "capabilities"):
            caps = self._bridge.capabilities()
            self.capabilities = caps
            if self._auto_calibrate:
                if caps.pixel_size_nm:
                    self.pixel_size_nm = caps.pixel_size_nm
                if caps.frame_interval_s:
                    self.frame_interval_s = caps.frame_interval_s

    def _acquire_frame(self) -> Optional[np.ndarray]:
        return self._bridge.read_frame()

    def _apply_command(self, command: Command) -> None:
        br, k = self._bridge, command.kind
        if k == "recenter":
            dy, dx = self._recenter_offset_nm(command)
            br.execute(f"SetScanOffset({dy:.5f}, {dx:.5f})")
        elif k == "set_setpoint":
            br.execute(f'PV("SetpointDefl", {command.params["value"]:.6g})')
        elif k == "trigger_ramp":
            dy, dx = self._point_offset_nm(command.params["y_px"],
                                           command.params["x_px"])
            if self._record_ramps and hasattr(br, "force_curve"):
                self.force_curves.append(br.force_curve(dy, dx))
            else:
                br.execute(f"DoSingleForce({dy:.5f}, {dx:.5f})")
        elif k == "pause":
            br.execute("StopFrame()")
        elif k == "resume":
            br.execute("StartFrame()")
        elif k == "engage":
            br.execute("Approach()")
        elif k == "withdraw":
            br.execute("Withdraw()")
        elif k == "set_gains":
            br.execute(f'PV("ProportionalGain", {command.params["p"]:.6g})')
            br.execute(f'PV("IntegralGain", {command.params["i"]:.6g})')

    def _disconnect(self) -> None:
        if self._bridge is not None:
            self._bridge.close()

    # -- direct instrument control -----------------------------------------
    def engage(self) -> None:
        self._ensure_connected(); self._bridge.execute("Approach()")

    def withdraw(self) -> None:
        self._ensure_connected(); self._bridge.execute("Withdraw()")

    def set_setpoint(self, value: float) -> None:
        self._ensure_connected()
        self._bridge.execute(f'PV("SetpointDefl", {value:.6g})')

    def set_feedback_gains(self, p: float, i: float) -> None:
        self._ensure_connected()
        self._bridge.execute(f'PV("ProportionalGain", {p:.6g})')
        self._bridge.execute(f'PV("IntegralGain", {i:.6g})')

    def set_operating_mode(self, mode: str) -> None:
        self._ensure_connected()
        self._bridge.execute(f'PV("ImagingMode", "{mode}")')

    def force_curve(self, y_nm: float = 0.0, x_nm: float = 0.0) -> AsylumForceCurve:
        self._ensure_connected()
        fc = self._bridge.force_curve(y_nm, x_nm)
        self.force_curves.append(fc)
        return fc


# ==========================================================================
# faithful loopback Igor listener (TCP) — for tests without hardware
# ==========================================================================
class IgorListenerServer:
    """A loopback TCP server emulating the Igor listener for tests.

    Implements the :class:`SocketIgorBridge` protocol against a
    :class:`BackendSimulator`: ``EXEC`` runs the Igor placeholder command,
    ``CAPS`` reports geometry, ``FRAME`` pulls a frame, ``FORCE`` returns a
    force curve. Mirrors what ``docs/asylum_igor_listener.ipf`` does in Igor.
    """

    def __init__(self, backend: BackendSimulator, host: str = "127.0.0.1",
                 port: int = 0, mode: str = "AC"):
        self.backend = backend
        self.mode = mode
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(1)
        self.port = self._srv.getsockname()[1]
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "IgorListenerServer":
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _serve(self) -> None:
        self._srv.settimeout(10.0)
        conn, _ = self._srv.accept()
        conn.settimeout(10.0)
        rf = conn.makefile("rb")
        b = self.backend
        try:
            while True:
                line = rf.readline()
                if not line:
                    break
                s = line.decode().strip()
                verb, _, rest = s.partition(" ")
                if verb == "CLOSE":
                    break
                elif verb == "EXEC":
                    conn.sendall((_exec_igor(b, rest) + "\n").encode())
                elif verb == "CAPS":
                    conn.sendall((_caps_string(b, self.mode) + "\n").encode())
                elif verb == "FRAME":
                    frame = b.acquire()
                    if frame is None:
                        conn.sendall(b"NONE\n")
                    else:
                        a = np.asarray(frame, dtype="<f8")
                        h, w = a.shape
                        conn.sendall(f"IMG {h} {w} {a.itemsize}\n".encode())
                        conn.sendall(a.tobytes())
                elif verb == "FORCE":
                    dy, dx = (float(v) for v in rest.split())
                    b.ramp_offset_nm(dy, dx)
                    fc = _synthetic_force_curve(dy, dx)
                    inter = np.empty(fc.z_nm.size * 2, dtype="<f8")
                    inter[0::2] = fc.z_nm; inter[1::2] = fc.force_pN
                    conn.sendall(f"FD {fc.z_nm.size}\n".encode())
                    conn.sendall(inter.tobytes())
                else:
                    conn.sendall(f"ERR unknown {s!r}\n".encode())
        finally:
            rf.close(); conn.close(); self._srv.close()
