"""JPK / Bruker BioAFM adapter (scaffold).

JPK's SPM control (NanoWizard / ForceRobot / the NanoRacer HS-AFM) exposes a
**TCP socket scripting interface**: the client sends text commands and receives
replies / data. The exact verb grammar is proprietary and version-specific — the
line protocol below is a documented-shape **placeholder**, marked ``VERIFY``:

    control (text):   "<VERB> <args>\\n"  ->  "OK\\n" | "ERR <msg>\\n"
    frame  (binary):  "ACQUIRE\\n" -> "FRAME <h> <w> <itemsize>\\n" <raw f8 bytes>
                                      | "END\\n"  (stream exhausted)

Two connections are provided against the same protocol: :class:`SocketJPKConnection`
(a real TCP client) and :class:`MockJPKConnection` (in-process, no sockets). A
tiny :func:`serve_backend` loopback server lets the socket path be exercised in
tests. Swap the verb strings / framing for JPK's real grammar when connecting to
hardware.
"""

from __future__ import annotations

import socket
import struct
from typing import Optional

import numpy as np

from ..streaming import Command
from .base import InstrumentAdapter
from .emulation import BackendSimulator


# ==========================================================================
# connections (both speak the same placeholder protocol)
# ==========================================================================
class MockJPKConnection:
    """In-process connection: parses protocol verbs, calls the physics backend."""

    def __init__(self, backend: Optional[BackendSimulator] = None, **backend_kwargs):
        self.backend = backend or BackendSimulator(**backend_kwargs)
        self.sent: list = []

    def request(self, line: str) -> str:
        self.sent.append(line)
        verb, *args = line.split()
        b = self.backend
        if verb == "MOVE-BY":
            b.move_center_nm(float(args[0]), float(args[1])); return "OK"
        if verb == "SET-SETPOINT":
            b.set_setpoint(float(args[0])); return "OK"
        if verb == "RAMP":
            b.ramp_offset_nm(float(args[0]), float(args[1])); return "OK"
        if verb == "PAUSE":
            b.pause(); return "OK"
        if verb == "RESUME":
            b.resume(); return "OK"
        return f"ERR unknown verb {verb}"

    def acquire_frame(self) -> Optional[np.ndarray]:
        return self.backend.acquire()

    def close(self) -> None:
        pass


class SocketJPKConnection:
    """Real TCP client for the placeholder protocol (``VERIFY`` against JPK)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 55055,
                 timeout: float = 10.0):
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._rf = self._sock.makefile("rb")

    def request(self, line: str) -> str:
        self._sock.sendall((line + "\n").encode())
        reply = self._rf.readline().decode().strip()
        if reply.startswith("ERR"):
            raise RuntimeError(f"instrument error: {reply}")
        return reply

    def acquire_frame(self) -> Optional[np.ndarray]:
        self._sock.sendall(b"ACQUIRE\n")
        header = self._rf.readline().decode().strip()
        if header == "END":
            return None
        tag, h, w, itemsize = header.split()
        h, w, itemsize = int(h), int(w), int(itemsize)
        raw = self._rf.read(h * w * itemsize)
        return np.frombuffer(raw, dtype="<f8").reshape(h, w)

    def close(self) -> None:
        try:
            self._sock.sendall(b"CLOSE\n")
        except OSError:
            pass
        self._rf.close()
        self._sock.close()


# ==========================================================================
# adapter
# ==========================================================================
class JPKAdapter(InstrumentAdapter):
    """Drive a JPK/Bruker SPM over the socket protocol.

    Pass ``connection=MockJPKConnection(...)`` (or a :class:`SocketJPKConnection`
    to a real server) for hardware-free testing; otherwise a socket connection is
    opened to ``host``/``port`` on first use.
    """

    def __init__(self, connection: Optional[object] = None, *,
                 host: str = "127.0.0.1", port: int = 55055,
                 pixel_size_nm: float = 2.0, frame_interval_s: float = 0.05,
                 max_frames: Optional[int] = None):
        super().__init__(pixel_size_nm, frame_interval_s, max_frames=max_frames)
        self._conn = connection
        self._host, self._port = host, port

    def _connect(self) -> None:
        if self._conn is None:
            self._conn = SocketJPKConnection(self._host, self._port)

    def _acquire_frame(self) -> Optional[np.ndarray]:
        return self._conn.acquire_frame()

    def _apply_command(self, command: Command) -> None:
        if command.kind == "recenter":
            dy, dx = self._recenter_offset_nm(command)
            self._conn.request(f"MOVE-BY {dy:.5f} {dx:.5f}")
        elif command.kind == "set_setpoint":
            self._conn.request(f"SET-SETPOINT {command.params['value']:.6g}")
        elif command.kind == "trigger_ramp":
            dy, dx = self._point_offset_nm(command.params["y_px"],
                                           command.params["x_px"])
            self._conn.request(f"RAMP {dy:.5f} {dx:.5f}")
        elif command.kind == "pause":
            self._conn.request("PAUSE")
        elif command.kind == "resume":
            self._conn.request("RESUME")

    def _disconnect(self) -> None:
        if self._conn is not None:
            self._conn.close()


# ==========================================================================
# loopback server (exercises the socket path without hardware)
# ==========================================================================
def serve_backend(backend: BackendSimulator, host: str = "127.0.0.1",
                  port: int = 0) -> "tuple[socket.socket, int]":
    """Bind a one-client server serving ``backend`` over the JPK placeholder
    protocol. Returns ``(server_socket, port)``; call :func:`handle_one_client`
    on it (typically in a background thread) to serve a single connection."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    return srv, srv.getsockname()[1]


def handle_one_client(srv: socket.socket, backend: BackendSimulator,
                      timeout: float = 10.0) -> None:
    """Accept one client and serve requests until it closes / stream ends."""
    srv.settimeout(timeout)
    conn, _ = srv.accept()
    conn.settimeout(timeout)
    rf = conn.makefile("rb")
    try:
        while True:
            line = rf.readline()
            if not line:
                break
            cmd = line.decode().strip()
            if cmd == "CLOSE":
                break
            if cmd == "ACQUIRE":
                frame = backend.acquire()
                if frame is None:
                    conn.sendall(b"END\n")
                    continue
                arr = np.asarray(frame, dtype="<f8")
                h, w = arr.shape
                conn.sendall(f"FRAME {h} {w} {arr.itemsize}\n".encode())
                conn.sendall(arr.tobytes())
                continue
            # control verb -> reuse the mock parser for the reply
            reply = MockJPKConnection(backend).request(cmd)
            conn.sendall((reply + "\n").encode())
    finally:
        rf.close()
        conn.close()
        srv.close()
