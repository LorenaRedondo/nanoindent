"""JPK / Bruker BioAFM adapter (version-aware).

JPK's SPM control (NanoWizard / ForceRobot / the NanoRacer HS-AFM) exposes a
**TCP socket scripting interface**: the client sends text commands and receives
replies and binary data. The concrete verb grammar is proprietary and has
changed across software generations, so the wire strings live in a swappable
:class:`~nanoindent.adapters.jpk_dialects.JPKDialect` (see ``jpk_dialects.py``) —
every string is a documented-shape placeholder marked ``VERIFY``. The adapter
auto-detects the software version from the handshake and selects the matching
dialect; filling in a version's real grammar is a localized, one-line-per-command
edit in ``jpk_dialects.py``.

This module models a realistic live connection: handshake / capability
negotiation, push streaming, force spectroscopy, control (engage/withdraw,
setpoint, gains, line rate, moves), keepalive, structured errors and timeouts.
:class:`SocketJPKConnection` (real TCP) and :class:`MockJPKConnection`
(in-process) implement the same surface; :class:`JPKInstrumentServer` is a
faithful, dialect-aware loopback mock so the whole path is testable without
hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import queue
import socket
import threading
from typing import List, Optional

import numpy as np

from ..streaming import Command
from .base import InstrumentAdapter
from .emulation import BackendSimulator
from .jpk_dialects import (JPKDialect, REFERENCE, get_dialect, select_dialect,
                           parse_caps)


# ==========================================================================
# protocol types
# ==========================================================================
class JPKError(RuntimeError):
    """An error reply (``ERR …``) or protocol failure from the instrument."""


@dataclass
class JPKCapabilities:
    """Instrument calibration + software version reported by the handshake."""

    pixel_size_nm: float = 1.0
    frame_interval_s: float = 0.1
    pixels: int = 0
    channels: List[str] = field(default_factory=lambda: ["height"])
    version: str = ""
    raw: str = ""


@dataclass
class JPKForceRamp:
    """A force–distance curve returned by a ``FORCE-RAMP`` request."""

    z_nm: np.ndarray
    force_pN: np.ndarray
    y_nm: float = 0.0
    x_nm: float = 0.0

    @property
    def max_force_pN(self) -> float:
        return float(np.max(self.force_pN)) if self.force_pN.size else 0.0


def _synthetic_force_ramp(dy_nm: float, dx_nm: float, n: int = 128,
                          contact_nm: float = 8.0, R: float = 20.0,
                          E: float = 50.0) -> JPKForceRamp:
    """A Hertz-shaped approach curve (contact then ~5 nm indentation), mock only."""
    from .. import analytics
    z = np.linspace(0.0, 13.0, n)
    delta = np.clip(z - contact_nm, 0.0, None)
    F = analytics.hertz_force(delta, R, E, 0.5)
    return JPKForceRamp(z_nm=z, force_pN=F, y_nm=dy_nm, x_nm=dx_nm)


def _caps_from_dict(d: dict, raw: str) -> JPKCapabilities:
    ch = d.get("channels", "height").split(",")
    return JPKCapabilities(
        pixel_size_nm=float(d.get("pixel_size_nm", 1.0)),
        frame_interval_s=float(d.get("frame_interval_s", 0.1)),
        pixels=int(d.get("pixels", 0)), channels=ch,
        version=d.get("version", ""), raw=raw)


# ---- binary framing (prefix comes from the dialect) ----------------------
def _encode_frame(arr: np.ndarray, frame_token: str = "FRAME") -> bytes:
    a = np.asarray(arr, dtype="<f8")
    h, w = a.shape
    return f"{frame_token} {h} {w} {a.itemsize}\n".encode() + a.tobytes()


def _encode_fd(ramp: JPKForceRamp, fd_token: str = "FD") -> bytes:
    inter = np.empty(ramp.z_nm.size * 2, dtype="<f8")
    inter[0::2] = ramp.z_nm
    inter[1::2] = ramp.force_pN
    return f"{fd_token} {ramp.z_nm.size}\n".encode() + inter.tobytes()


def _read_exact(rf, nbytes: int) -> bytes:
    buf = rf.read(nbytes)
    if buf is None or len(buf) != nbytes:
        raise JPKError("short read on binary payload")
    return buf


def _caps_line(dialect: JPKDialect, backend: BackendSimulator,
               version: str) -> str:
    return (f"{dialect.tokens['caps']} pixel_size_nm={backend.pixel_size_nm} "
            f"frame_interval_s={backend.frame_interval_s} "
            f"pixels={backend.scan_px} channels=height version={version}")


# ==========================================================================
# in-process connection (no sockets)
# ==========================================================================
class MockJPKConnection:
    """In-process connection: decodes the dialect and drives the physics backend.

    Mirrors :class:`SocketJPKConnection` so ``JPKAdapter`` (streaming, handshake,
    force-ramp recording, any dialect) can be exercised without a socket.
    """

    def __init__(self, backend: Optional[BackendSimulator] = None, *,
                 dialect: JPKDialect = REFERENCE, version: str = "reference",
                 **backend_kwargs):
        self.backend = backend or BackendSimulator(**backend_kwargs)
        self.dialect = dialect
        self.version = version
        self.sent: list = []

    # control ---------------------------------------------------------------
    def request(self, line: str) -> str:
        self.sent.append(line)
        op, args = self.dialect.parse_command(line)
        b, tok = self.backend, self.dialect.tokens
        if op == "hello":
            return _caps_line(self.dialect, b, self.version)
        if op == "ping":
            return tok["pong"]
        if op == "move_by":
            b.move_center_nm(float(args[0]), float(args[1])); return tok["ok"]
        if op == "set_setpoint":
            b.set_setpoint(float(args[0])); return tok["ok"]
        if op == "ramp" or op == "force_ramp":
            b.ramp_offset_nm(float(args[0]), float(args[1])); return tok["ok"]
        if op == "pause":
            b.pause(); return tok["ok"]
        if op == "resume":
            b.resume(); return tok["ok"]
        if op in ("move_to", "set_gains", "set_line_rate", "engage", "withdraw"):
            return tok["ok"]
        return f"{tok['error']} unknown command {line!r}"

    def handshake(self, client: str = "nanoindent/1") -> JPKCapabilities:
        raw = self.request(self.dialect.encode("hello", client=client))
        return _caps_from_dict(parse_caps(raw, self.dialect.tokens["caps"]), raw)

    # data ------------------------------------------------------------------
    def acquire_frame(self) -> Optional[np.ndarray]:
        return self.backend.acquire()

    # streaming (mock: pull under the hood) ---------------------------------
    def start_stream(self) -> None:
        self.sent.append(self.dialect.encode("stream_on"))

    def read_frame(self, timeout: float = 10.0) -> Optional[np.ndarray]:
        return self.backend.acquire()

    def stop_stream(self) -> None:
        self.sent.append(self.dialect.encode("stream_off"))

    # force spectroscopy ----------------------------------------------------
    def force_ramp(self, dy_nm: float, dx_nm: float) -> JPKForceRamp:
        self.sent.append(self.dialect.encode("force_ramp", dy=dy_nm, dx=dx_nm))
        self.backend.ramp_offset_nm(dy_nm, dx_nm)
        return _synthetic_force_ramp(dy_nm, dx_nm)

    def close(self) -> None:
        pass


# ==========================================================================
# real TCP connection
# ==========================================================================
class SocketJPKConnection:
    """TCP client for the JPK scripting protocol (dialect-driven; ``VERIFY``).

    Synchronous by default (``request`` / ``acquire_frame`` / ``force_ramp`` read
    inline). After :meth:`start_stream`, a background reader demultiplexes pushed
    frames, force-curve data, and command replies off the one socket.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 55055,
                 timeout: float = 10.0, dialect: JPKDialect = REFERENCE):
        self._host, self._port, self._timeout = host, port, timeout
        self.dialect = dialect
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._rf = self._sock.makefile("rb")
        self._wlock = threading.Lock()
        self._streaming = False
        self._reader: Optional[threading.Thread] = None
        self._frames: "queue.Queue" = queue.Queue()
        self._replies: "queue.Queue" = queue.Queue()
        self._fds: "queue.Queue" = queue.Queue()
        self._reader_err: Optional[Exception] = None

    def _send(self, line: str) -> None:
        with self._wlock:
            self._sock.sendall((line + "\n").encode())

    # -- synchronous mode ---------------------------------------------------
    def request(self, line: str) -> str:
        if self._streaming:
            self._send(line)
            reply = self._replies.get(timeout=self._timeout)
        else:
            self._send(line)
            reply = self._rf.readline().decode().strip()
        if self.dialect.is_error(reply):
            raise JPKError(f"instrument error: {reply}")
        return reply

    def handshake(self, client: str = "nanoindent/1") -> JPKCapabilities:
        raw = self.request(self.dialect.encode("hello", client=client))
        return _caps_from_dict(parse_caps(raw, self.dialect.tokens["caps"]), raw)

    def ping(self) -> bool:
        return self.dialect.is_pong(self.request(self.dialect.encode("ping")))

    def acquire_frame(self) -> Optional[np.ndarray]:
        self._send(self.dialect.encode("acquire"))
        return self._read_frame_payload(self._rf.readline().decode().strip())

    def _read_frame_payload(self, header: str) -> Optional[np.ndarray]:
        tok = self.dialect.tokens
        if header in (tok["end"], tok["stream_end"], ""):
            return None
        parts = header.split()
        if parts[0] != tok["frame"]:
            raise JPKError(f"expected {tok['frame']}, got {header!r}")
        h, w, itemsize = int(parts[1]), int(parts[2]), int(parts[3])
        raw = _read_exact(self._rf, h * w * itemsize)
        return np.frombuffer(raw, dtype="<f8").reshape(h, w)

    def force_ramp(self, dy_nm: float, dx_nm: float) -> JPKForceRamp:
        line = self.dialect.encode("force_ramp", dy=dy_nm, dx=dx_nm)
        if self._streaming:
            self._send(line)
            return self._fds.get(timeout=self._timeout)
        self._send(line)
        return self._read_fd_payload(self._rf.readline().decode().strip(),
                                     dy_nm, dx_nm)

    def _read_fd_payload(self, header: str, y=0.0, x=0.0) -> JPKForceRamp:
        parts = header.split()
        if parts[0] != self.dialect.tokens["fd"]:
            raise JPKError(f"expected {self.dialect.tokens['fd']}, got {header!r}")
        n = int(parts[1])
        raw = _read_exact(self._rf, n * 2 * 8)
        inter = np.frombuffer(raw, dtype="<f8")
        return JPKForceRamp(z_nm=inter[0::2].copy(), force_pN=inter[1::2].copy(),
                            y_nm=y, x_nm=x)

    # -- streaming mode -----------------------------------------------------
    def start_stream(self) -> None:
        if self._streaming:
            return
        self._streaming = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._send(self.dialect.encode("stream_on"))
        try:
            self._replies.get(timeout=self._timeout)      # consume STREAM ON ack
        except queue.Empty:
            pass

    def _read_loop(self) -> None:
        tok = self.dialect.tokens
        try:
            while self._streaming:
                header = self._rf.readline().decode().strip()
                if header == "":
                    break
                tag = header.split()[0]
                if tag == tok["frame"]:
                    self._frames.put(self._read_frame_payload(header))
                elif tag == tok["fd"]:
                    self._fds.put(self._read_fd_payload(header))
                elif tag in (tok["stream_end"], tok["end"]):
                    self._frames.put(None)
                else:
                    self._replies.put(header)
        except Exception as exc:
            self._reader_err = exc
            self._frames.put(None)

    def read_frame(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        if not self._streaming:
            return self.acquire_frame()
        try:
            frame = self._frames.get(timeout=timeout or self._timeout)
        except queue.Empty:
            return None
        if self._reader_err is not None:
            raise JPKError(f"stream reader failed: {self._reader_err}")
        return frame

    def stop_stream(self) -> None:
        if not self._streaming:
            return
        try:
            self._send(self.dialect.encode("stream_off"))
        except OSError:
            pass
        self._streaming = False

    def close(self) -> None:
        self._streaming = False
        try:
            with self._wlock:
                self._sock.sendall((self.dialect.encode("close") + "\n").encode())
        except OSError:
            pass
        try:
            self._rf.close()
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass
        if self._reader is not None:
            self._reader.join(timeout=2.0)


# ==========================================================================
# adapter
# ==========================================================================
class JPKAdapter(InstrumentAdapter):
    """Drive a JPK/Bruker SPM as a :class:`~nanoindent.streaming.FrameSource`.

    ``dialect`` selects the command grammar: ``"auto"`` (default) detects the
    software version from the handshake and picks the matching profile; a name
    (``"jpk_spm_4plus"``, …) or a :class:`JPKDialect` forces one. See
    :meth:`JPKAdapter.__init__` for ``stream`` / ``auto_calibrate`` /
    ``record_ramps``.
    """

    def __init__(self, connection: Optional[object] = None, *,
                 host: str = "127.0.0.1", port: int = 55055,
                 pixel_size_nm: float = 2.0, frame_interval_s: float = 0.05,
                 max_frames: Optional[int] = None, stream: bool = False,
                 auto_calibrate: bool = True, record_ramps: bool = False,
                 dialect="auto", client_name: str = "nanoindent/1"):
        super().__init__(pixel_size_nm, frame_interval_s, max_frames=max_frames)
        self._conn = connection
        self._host, self._port = host, port
        self._stream = stream
        self._auto_calibrate = auto_calibrate
        self._record_ramps = record_ramps
        self._client_name = client_name
        self._dialect_spec = dialect
        self._auto_dialect = (dialect == "auto")
        self.dialect: JPKDialect = get_dialect(dialect)
        self.capabilities: Optional[JPKCapabilities] = None
        self.force_ramps: List[JPKForceRamp] = []

    # -- lifecycle ---------------------------------------------------------
    def _connect(self) -> None:
        if self._conn is None:
            self._conn = SocketJPKConnection(self._host, self._port,
                                             dialect=self.dialect)
        elif getattr(self._conn, "dialect", None) is not None and \
                not self._auto_dialect:
            self._conn.dialect = self.dialect
        if hasattr(self._conn, "handshake"):
            caps = self._conn.handshake(self._client_name)
            self.capabilities = caps
            if self._auto_dialect and caps.version:
                self.dialect = select_dialect(caps.version)
                if hasattr(self._conn, "dialect"):
                    self._conn.dialect = self.dialect
            if self._auto_calibrate:
                if caps.pixel_size_nm:
                    self.pixel_size_nm = caps.pixel_size_nm
                if caps.frame_interval_s:
                    self.frame_interval_s = caps.frame_interval_s
        if self._stream and hasattr(self._conn, "start_stream"):
            self._conn.start_stream()

    def _acquire_frame(self) -> Optional[np.ndarray]:
        if self._stream and hasattr(self._conn, "read_frame"):
            return self._conn.read_frame()
        return self._conn.acquire_frame()

    def _cmd(self, op: str, **kw) -> str:
        return self._conn.request(self.dialect.encode(op, **kw))

    def _apply_command(self, command: Command) -> None:
        k = command.kind
        if k == "recenter":
            dy, dx = self._recenter_offset_nm(command)
            self._cmd("move_by", dy=dy, dx=dx)
        elif k == "set_setpoint":
            self._cmd("set_setpoint", v=command.params["value"])
        elif k == "trigger_ramp":
            dy, dx = self._point_offset_nm(command.params["y_px"],
                                           command.params["x_px"])
            if self._record_ramps and hasattr(self._conn, "force_ramp"):
                self.force_ramps.append(self._conn.force_ramp(dy, dx))
            else:
                self._cmd("force_ramp", dy=dy, dx=dx)
        elif k == "pause":
            self._cmd("pause")
        elif k == "resume":
            self._cmd("resume")
        elif k == "engage":
            self._cmd("engage")
        elif k == "withdraw":
            self._cmd("withdraw")
        elif k == "set_gains":
            self._cmd("set_gains", p=command.params["p"], i=command.params["i"])

    def _disconnect(self) -> None:
        if self._conn is not None:
            self._conn.close()

    # -- direct instrument control (outside the loop) ----------------------
    def engage(self) -> None:
        self._ensure_connected(); self._cmd("engage")

    def withdraw(self) -> None:
        self._ensure_connected(); self._cmd("withdraw")

    def set_setpoint(self, value: float) -> None:
        self._ensure_connected(); self._cmd("set_setpoint", v=value)

    def set_feedback_gains(self, p: float, i: float) -> None:
        self._ensure_connected(); self._cmd("set_gains", p=p, i=i)

    def set_line_rate(self, hz: float) -> None:
        self._ensure_connected(); self._cmd("set_line_rate", hz=hz)

    def move_to(self, y_nm: float, x_nm: float) -> None:
        self._ensure_connected(); self._cmd("move_to", y=y_nm, x=x_nm)

    def force_ramp(self, y_nm: float = 0.0, x_nm: float = 0.0) -> JPKForceRamp:
        self._ensure_connected()
        fr = self._conn.force_ramp(y_nm, x_nm)
        self.force_ramps.append(fr)
        return fr

    def ping(self) -> bool:
        self._ensure_connected()
        return self.dialect.is_pong(self._cmd("ping"))


# ==========================================================================
# faithful loopback mock instrument (dialect-aware)
# ==========================================================================
class JPKInstrumentServer:
    """A loopback TCP server emulating the JPK protocol for a given dialect.

    Serves one client per :meth:`start` from a :class:`BackendSimulator`,
    supporting the handshake (reporting ``version``), control verbs, pull
    acquisition, push streaming, and force ramps. Use it to test the socket path
    — including a non-default dialect — end to end.
    """

    def __init__(self, backend: BackendSimulator, host: str = "127.0.0.1",
                 port: int = 0, push_interval_s: float = 0.0,
                 dialect: JPKDialect = REFERENCE, version: str = "reference"):
        self.backend = backend
        self.push_interval_s = push_interval_s
        self.dialect = dialect
        self.version = version
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(1)
        self.port = self._srv.getsockname()[1]
        self._thread: Optional[threading.Thread] = None
        self._pusher: Optional[threading.Thread] = None
        self._streaming = False
        self._wlock = threading.Lock()

    def start(self) -> "JPKInstrumentServer":
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _send(self, conn, data: bytes) -> None:
        with self._wlock:
            conn.sendall(data)

    def _line(self, conn, text: str) -> None:
        self._send(conn, (text + "\n").encode())

    def _serve(self) -> None:
        self._srv.settimeout(10.0)
        conn, _ = self._srv.accept()
        conn.settimeout(10.0)
        rf = conn.makefile("rb")
        b, d, tok = self.backend, self.dialect, self.dialect.tokens
        try:
            while True:
                line = rf.readline()
                if not line:
                    break
                op, args = d.parse_command(line.decode().strip())
                if op == "close":
                    break
                elif op == "hello":
                    self._line(conn, _caps_line(d, b, self.version))
                elif op == "ping":
                    self._line(conn, tok["pong"])
                elif op == "acquire":
                    frame = b.acquire()
                    self._send(conn, (tok["end"] + "\n").encode() if frame is None
                               else _encode_frame(frame, tok["frame"]))
                elif op == "stream_on":
                    self._line(conn, tok["ok"]); self._start_push(conn)
                elif op == "stream_off":
                    self._streaming = False
                    if self._pusher:
                        self._pusher.join(timeout=5.0)
                    self._line(conn, tok["ok"]); self._line(conn, tok["stream_end"])
                elif op == "force_ramp":
                    dy, dx = float(args[0]), float(args[1])
                    b.ramp_offset_nm(dy, dx)
                    self._send(conn, _encode_fd(_synthetic_force_ramp(dy, dx),
                                                tok["fd"]))
                elif op == "move_by":
                    b.move_center_nm(float(args[0]), float(args[1]))
                    self._line(conn, tok["ok"])
                elif op == "set_setpoint":
                    b.set_setpoint(float(args[0])); self._line(conn, tok["ok"])
                elif op == "pause":
                    b.pause(); self._line(conn, tok["ok"])
                elif op == "resume":
                    b.resume(); self._line(conn, tok["ok"])
                elif op in ("move_to", "set_gains", "set_line_rate", "engage",
                            "withdraw"):
                    self._line(conn, tok["ok"])
                else:
                    self._line(conn, f"{tok['error']} unknown {line!r}")
        finally:
            self._streaming = False
            if self._pusher:
                self._pusher.join(timeout=5.0)
            rf.close(); conn.close(); self._srv.close()

    def _start_push(self, conn) -> None:
        self._streaming = True
        tok = self.dialect.tokens

        def push():
            import time
            while self._streaming:
                frame = self.backend.acquire()
                if frame is None:
                    try:
                        self._line(conn, tok["stream_end"])
                    except OSError:
                        pass
                    self._streaming = False
                    break
                try:
                    self._send(conn, _encode_frame(frame, tok["frame"]))
                except OSError:
                    break
                if self.push_interval_s:
                    time.sleep(self.push_interval_s)

        self._pusher = threading.Thread(target=push, daemon=True)
        self._pusher.start()


# ==========================================================================
# simple pull loopback server (kept for the basic socket test)
# ==========================================================================
def serve_backend(backend: BackendSimulator, host: str = "127.0.0.1",
                  port: int = 0) -> "tuple[socket.socket, int]":
    """Bind a one-client pull server (REFERENCE dialect). Returns
    ``(server_socket, port)``; call :func:`handle_one_client` on it."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    return srv, srv.getsockname()[1]


def handle_one_client(srv: socket.socket, backend: BackendSimulator,
                      timeout: float = 10.0) -> None:
    """Accept one client and serve pull requests + control verbs (REFERENCE)."""
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
                conn.sendall(b"END\n" if frame is None else _encode_frame(frame))
                continue
            reply = MockJPKConnection(backend).request(cmd)
            conn.sendall((reply + "\n").encode())
    finally:
        rf.close()
        conn.close()
        srv.close()
