"""JPK / Bruker BioAFM adapter.

JPK's SPM control (NanoWizard / ForceRobot / the NanoRacer HS-AFM) exposes a
**TCP socket scripting interface**: the client sends text commands and receives
replies and binary data. The concrete verb grammar is proprietary and
version-specific, so the line protocol here is a **documented-shape placeholder**
— every wire detail is marked ``VERIFY`` and must be confirmed against the JPK
scripting reference for your software version before a live run.

This module models a realistic live connection, not just a pull loop:

* **Handshake / capabilities** — the client greets the instrument and reads back
  its calibration (pixel size, frame interval, channels), so the adapter is
  configured *by the microscope* rather than by hand.
* **Push streaming** — high-speed AFM pushes frames continuously; the streaming
  client runs a background reader that demultiplexes frames, force-curve data,
  and command replies off one socket.
* **Force spectroscopy** — a ``FORCE-RAMP`` round trip returns the F–distance
  curve, so a policy can *act on the analysis* (drop a ramp on a detected event)
  and get the measurement back.
* **Control + robustness** — engage/withdraw, setpoint, feedback gains, line
  rate, absolute moves, keepalive ping, timeouts and structured errors.

Protocol (placeholder — ``VERIFY``)
-----------------------------------
Text control, newline-terminated, replies newline-terminated::

    HELLO <client>          -> CAPS pixel_size_nm=.. frame_interval_s=.. pixels=.. channels=..
    PING                    -> PONG
    MOVE-BY <dy> <dx>       -> OK        (relative, nm)
    MOVE-TO <y> <x>         -> OK        (absolute, nm)
    SET-SETPOINT <v>        -> OK
    SET-GAINS <p> <i>       -> OK
    SET-LINE-RATE <hz>      -> OK
    ENGAGE / WITHDRAW       -> OK
    PAUSE / RESUME          -> OK
    ACQUIRE                 -> FRAME <h> <w> <itemsize>\\n<bytes>  |  END
    STREAM ON / OFF         -> OK   (ON: server pushes FRAME…; ends with STREAM-END)
    FORCE-RAMP <dy> <dx>    -> FD <n>\\n<2n float64: z_nm,F_pN interleaved>
    CLOSE                   -> (server closes)

:class:`SocketJPKConnection` (real TCP) and :class:`MockJPKConnection`
(in-process) implement the same surface; :class:`JPKInstrumentServer` is a
faithful loopback mock (handshake, push streaming, force ramps) so the whole
path is testable without hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import queue
import socket
import threading
from typing import List, Optional, Tuple

import numpy as np

from ..streaming import Command
from .base import InstrumentAdapter
from .emulation import BackendSimulator


# ==========================================================================
# protocol types
# ==========================================================================
class JPKError(RuntimeError):
    """An error reply (``ERR …``) or protocol failure from the instrument."""


@dataclass
class JPKCapabilities:
    """Instrument calibration reported by the handshake."""

    pixel_size_nm: float = 1.0
    frame_interval_s: float = 0.1
    pixels: int = 0
    channels: List[str] = field(default_factory=lambda: ["height"])
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
    """A Hertz-shaped approach curve (contact then ~5 nm indentation), for the
    mock instrument."""
    from .. import analytics
    z = np.linspace(0.0, 13.0, n)                 # ~5 nm indentation past contact
    delta = np.clip(z - contact_nm, 0.0, None)
    F = analytics.hertz_force(delta, R, E, 0.5)
    return JPKForceRamp(z_nm=z, force_pN=F, y_nm=dy_nm, x_nm=dx_nm)


# ---- binary framing helpers (shared by client + server) ------------------
def _encode_frame(arr: np.ndarray) -> bytes:
    a = np.asarray(arr, dtype="<f8")
    h, w = a.shape
    return f"FRAME {h} {w} {a.itemsize}\n".encode() + a.tobytes()


def _encode_fd(ramp: JPKForceRamp) -> bytes:
    inter = np.empty(ramp.z_nm.size * 2, dtype="<f8")
    inter[0::2] = ramp.z_nm
    inter[1::2] = ramp.force_pN
    return f"FD {ramp.z_nm.size}\n".encode() + inter.tobytes()


def _read_exact(rf, nbytes: int) -> bytes:
    buf = rf.read(nbytes)
    if buf is None or len(buf) != nbytes:
        raise JPKError("short read on binary payload")
    return buf


# ==========================================================================
# in-process connection (no sockets)
# ==========================================================================
class MockJPKConnection:
    """In-process connection: parses protocol verbs, drives the physics backend.

    Mirrors :class:`SocketJPKConnection` so ``JPKAdapter`` (including streaming,
    handshake, and force-ramp recording) can be exercised without a socket.
    """

    def __init__(self, backend: Optional[BackendSimulator] = None, **backend_kwargs):
        self.backend = backend or BackendSimulator(**backend_kwargs)
        self.sent: list = []

    # control ---------------------------------------------------------------
    def request(self, line: str) -> str:
        self.sent.append(line)
        verb, *args = line.split()
        b = self.backend
        if verb == "HELLO":
            return (f"CAPS pixel_size_nm={b.pixel_size_nm} "
                    f"frame_interval_s={b.frame_interval_s} "
                    f"pixels={b.scan_px} channels=height")
        if verb == "PING":
            return "PONG"
        if verb == "MOVE-BY":
            b.move_center_nm(float(args[0]), float(args[1])); return "OK"
        if verb == "MOVE-TO":
            return "OK"                       # absolute move (mock: no-op move)
        if verb == "SET-SETPOINT":
            b.set_setpoint(float(args[0])); return "OK"
        if verb in ("SET-GAINS", "SET-LINE-RATE", "SET-PIXELS", "ENGAGE",
                    "WITHDRAW"):
            return "OK"
        if verb == "RAMP":
            b.ramp_offset_nm(float(args[0]), float(args[1])); return "OK"
        if verb == "PAUSE":
            b.pause(); return "OK"
        if verb == "RESUME":
            b.resume(); return "OK"
        return f"ERR unknown verb {verb}"

    def handshake(self, client: str = "nanoindent/1") -> JPKCapabilities:
        raw = self.request(f"HELLO {client}")
        return _parse_caps(raw)

    # data ------------------------------------------------------------------
    def acquire_frame(self) -> Optional[np.ndarray]:
        return self.backend.acquire()

    # streaming (mock: pull under the hood) ---------------------------------
    def start_stream(self) -> None:
        self.sent.append("STREAM ON")

    def read_frame(self, timeout: float = 10.0) -> Optional[np.ndarray]:
        return self.backend.acquire()

    def stop_stream(self) -> None:
        self.sent.append("STREAM OFF")

    # force spectroscopy ----------------------------------------------------
    def force_ramp(self, dy_nm: float, dx_nm: float) -> JPKForceRamp:
        self.sent.append(f"FORCE-RAMP {dy_nm:.5f} {dx_nm:.5f}")
        self.backend.ramp_offset_nm(dy_nm, dx_nm)
        return _synthetic_force_ramp(dy_nm, dx_nm)

    def close(self) -> None:
        pass


def _parse_caps(line: str) -> JPKCapabilities:
    caps = JPKCapabilities(raw=line)
    if not line.startswith("CAPS"):
        return caps
    for tok in line.split()[1:]:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if k == "pixel_size_nm":
            caps.pixel_size_nm = float(v)
        elif k == "frame_interval_s":
            caps.frame_interval_s = float(v)
        elif k == "pixels":
            caps.pixels = int(v)
        elif k == "channels":
            caps.channels = v.split(",")
    return caps


# ==========================================================================
# real TCP connection
# ==========================================================================
class SocketJPKConnection:
    """TCP client for the JPK scripting protocol (``VERIFY`` against hardware).

    Two modes share one socket:

    * **synchronous** (default) — ``request`` / ``acquire_frame`` /
      ``force_ramp`` read the reply inline. Best for setup and pull acquisition.
    * **streaming** — after :meth:`start_stream`, a background reader thread
      demultiplexes pushed frames, force-curve data, and command replies into
      queues, so :meth:`read_frame`, :meth:`request`, and :meth:`force_ramp`
      work concurrently while frames stream in.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 55055,
                 timeout: float = 10.0):
        self._host, self._port, self._timeout = host, port, timeout
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._rf = self._sock.makefile("rb")
        self._wlock = threading.Lock()
        self._streaming = False
        self._reader: Optional[threading.Thread] = None
        self._frames: "queue.Queue" = queue.Queue()
        self._replies: "queue.Queue" = queue.Queue()
        self._fds: "queue.Queue" = queue.Queue()
        self._reader_err: Optional[Exception] = None

    # -- low-level send -----------------------------------------------------
    def _send(self, line: str) -> None:
        with self._wlock:
            self._sock.sendall((line + "\n").encode())

    # -- synchronous mode ---------------------------------------------------
    def request(self, line: str) -> str:
        if self._streaming:
            self._send(line)
            reply = self._replies.get(timeout=self._timeout)
            if reply.startswith("ERR"):
                raise JPKError(reply)
            return reply
        self._send(line)
        reply = self._rf.readline().decode().strip()
        if reply.startswith("ERR"):
            raise JPKError(f"instrument error: {reply}")
        return reply

    def handshake(self, client: str = "nanoindent/1") -> JPKCapabilities:
        return _parse_caps(self.request(f"HELLO {client}"))

    def ping(self) -> bool:
        return self.request("PING") == "PONG"

    def acquire_frame(self) -> Optional[np.ndarray]:
        """Pull one frame (synchronous mode only)."""
        self._send("ACQUIRE")
        header = self._rf.readline().decode().strip()
        return self._read_frame_payload(header)

    def _read_frame_payload(self, header: str) -> Optional[np.ndarray]:
        if header in ("END", "STREAM-END", ""):
            return None
        parts = header.split()
        if parts[0] != "FRAME":
            raise JPKError(f"expected FRAME, got {header!r}")
        h, w, itemsize = int(parts[1]), int(parts[2]), int(parts[3])
        raw = _read_exact(self._rf, h * w * itemsize)
        return np.frombuffer(raw, dtype="<f8").reshape(h, w)

    def force_ramp(self, dy_nm: float, dx_nm: float) -> JPKForceRamp:
        if self._streaming:
            self._send(f"FORCE-RAMP {dy_nm:.5f} {dx_nm:.5f}")
            return self._fds.get(timeout=self._timeout)
        self._send(f"FORCE-RAMP {dy_nm:.5f} {dx_nm:.5f}")
        header = self._rf.readline().decode().strip()
        return self._read_fd_payload(header, dy_nm, dx_nm)

    def _read_fd_payload(self, header: str, y=0.0, x=0.0) -> JPKForceRamp:
        parts = header.split()
        if parts[0] != "FD":
            raise JPKError(f"expected FD, got {header!r}")
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
        self._send("STREAM ON")
        # consume the STREAM ON acknowledgement so the reply queue stays aligned
        # with subsequent control requests
        try:
            self._replies.get(timeout=self._timeout)
        except queue.Empty:
            pass

    def _read_loop(self) -> None:
        try:
            while self._streaming:
                header = self._rf.readline().decode().strip()
                if header == "":
                    break
                tag = header.split()[0]
                if tag == "FRAME":
                    self._frames.put(self._read_frame_payload(header))
                elif tag == "FD":
                    self._fds.put(self._read_fd_payload(header))
                elif tag in ("STREAM-END", "END"):
                    self._frames.put(None)
                else:                                   # OK / PONG / CAPS / ERR
                    self._replies.put(header)
        except Exception as exc:                        # surface to read_frame
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
            self._send("STREAM OFF")           # best-effort graceful stop
        except OSError:
            pass
        self._streaming = False

    def close(self) -> None:
        self._streaming = False
        try:
            with self._wlock:
                self._sock.sendall(b"CLOSE\n")
        except OSError:
            pass
        # closing the socket unblocks the reader thread's readline()
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

    Parameters
    ----------
    connection:
        A :class:`MockJPKConnection` (hardware-free) or :class:`SocketJPKConnection`.
        If ``None``, a socket connection to ``host``/``port`` is opened on first use.
    stream:
        Use push streaming (background reader) instead of pull acquisition.
    auto_calibrate:
        Read pixel size / frame interval from the instrument handshake and adopt
        them (overriding the constructor values).
    record_ramps:
        On a ``trigger_ramp`` command, perform a ``FORCE-RAMP`` round trip and
        append the returned :class:`JPKForceRamp` to :attr:`force_ramps`
        (instead of just logging the ramp). This is the "act on the analysis,
        get the curve back" path.
    """

    def __init__(self, connection: Optional[object] = None, *,
                 host: str = "127.0.0.1", port: int = 55055,
                 pixel_size_nm: float = 2.0, frame_interval_s: float = 0.05,
                 max_frames: Optional[int] = None, stream: bool = False,
                 auto_calibrate: bool = True, record_ramps: bool = False,
                 client_name: str = "nanoindent/1"):
        super().__init__(pixel_size_nm, frame_interval_s, max_frames=max_frames)
        self._conn = connection
        self._host, self._port = host, port
        self._stream = stream
        self._auto_calibrate = auto_calibrate
        self._record_ramps = record_ramps
        self._client_name = client_name
        self.capabilities: Optional[JPKCapabilities] = None
        self.force_ramps: List[JPKForceRamp] = []

    # -- lifecycle ---------------------------------------------------------
    def _connect(self) -> None:
        if self._conn is None:
            self._conn = SocketJPKConnection(self._host, self._port)
        if self._auto_calibrate and hasattr(self._conn, "handshake"):
            caps = self._conn.handshake(self._client_name)
            self.capabilities = caps
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

    def _apply_command(self, command: Command) -> None:
        c, k = self._conn, command.kind
        if k == "recenter":
            dy, dx = self._recenter_offset_nm(command)
            c.request(f"MOVE-BY {dy:.5f} {dx:.5f}")
        elif k == "set_setpoint":
            c.request(f"SET-SETPOINT {command.params['value']:.6g}")
        elif k == "trigger_ramp":
            dy, dx = self._point_offset_nm(command.params["y_px"],
                                           command.params["x_px"])
            if self._record_ramps and hasattr(c, "force_ramp"):
                self.force_ramps.append(c.force_ramp(dy, dx))
            else:
                c.request(f"RAMP {dy:.5f} {dx:.5f}")
        elif k == "pause":
            c.request("PAUSE")
        elif k == "resume":
            c.request("RESUME")
        elif k == "engage":
            c.request("ENGAGE")
        elif k == "withdraw":
            c.request("WITHDRAW")
        elif k == "set_gains":
            c.request(f"SET-GAINS {command.params['p']:.6g} "
                      f"{command.params['i']:.6g}")

    def _disconnect(self) -> None:
        if self._conn is not None:
            self._conn.close()

    # -- direct instrument control (outside the loop) ----------------------
    def engage(self) -> None:
        self._ensure_connected(); self._conn.request("ENGAGE")

    def withdraw(self) -> None:
        self._ensure_connected(); self._conn.request("WITHDRAW")

    def set_setpoint(self, value: float) -> None:
        self._ensure_connected(); self._conn.request(f"SET-SETPOINT {value:.6g}")

    def set_feedback_gains(self, p: float, i: float) -> None:
        self._ensure_connected(); self._conn.request(f"SET-GAINS {p:.6g} {i:.6g}")

    def set_line_rate(self, hz: float) -> None:
        self._ensure_connected(); self._conn.request(f"SET-LINE-RATE {hz:.6g}")

    def move_to(self, y_nm: float, x_nm: float) -> None:
        self._ensure_connected(); self._conn.request(f"MOVE-TO {y_nm:.5f} {x_nm:.5f}")

    def force_ramp(self, y_nm: float = 0.0, x_nm: float = 0.0) -> JPKForceRamp:
        self._ensure_connected()
        fr = self._conn.force_ramp(y_nm, x_nm)
        self.force_ramps.append(fr)
        return fr

    def ping(self) -> bool:
        self._ensure_connected()
        return self._conn.request("PING") == "PONG"


# ==========================================================================
# faithful loopback mock instrument (handshake + streaming + force ramps)
# ==========================================================================
class JPKInstrumentServer:
    """A loopback TCP server emulating the JPK scripting protocol.

    Serves one client per :meth:`start` from a :class:`BackendSimulator`,
    supporting the handshake, control verbs, pull acquisition, push streaming
    (``STREAM ON`` → frames pushed until the backend is exhausted or ``STREAM
    OFF``), and force ramps. Use it to test the socket path end-to-end.
    """

    def __init__(self, backend: BackendSimulator, host: str = "127.0.0.1",
                 port: int = 0, push_interval_s: float = 0.0):
        self.backend = backend
        self.push_interval_s = push_interval_s
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
                parts = line.decode().strip().split()
                if not parts:
                    continue
                verb = parts[0]
                if verb == "CLOSE":
                    break
                elif verb == "HELLO":
                    self._send(conn, (f"CAPS pixel_size_nm={b.pixel_size_nm} "
                                      f"frame_interval_s={b.frame_interval_s} "
                                      f"pixels={b.scan_px} channels=height\n").encode())
                elif verb == "PING":
                    self._send(conn, b"PONG\n")
                elif verb == "ACQUIRE":
                    frame = b.acquire()
                    self._send(conn, b"END\n" if frame is None
                               else _encode_frame(frame))
                elif verb == "STREAM":
                    if parts[1:2] == ["ON"]:
                        self._send(conn, b"OK\n")
                        self._start_push(conn)
                    else:
                        self._streaming = False
                        if self._pusher:
                            self._pusher.join(timeout=5.0)
                        self._send(conn, b"OK\n")
                        self._send(conn, b"STREAM-END\n")
                elif verb == "FORCE-RAMP":
                    dy = float(parts[1]); dx = float(parts[2])
                    b.ramp_offset_nm(dy, dx)
                    self._send(conn, _encode_fd(_synthetic_force_ramp(dy, dx)))
                elif verb == "MOVE-BY":
                    b.move_center_nm(float(parts[1]), float(parts[2]))
                    self._send(conn, b"OK\n")
                elif verb == "SET-SETPOINT":
                    b.set_setpoint(float(parts[1])); self._send(conn, b"OK\n")
                elif verb == "PAUSE":
                    b.pause(); self._send(conn, b"OK\n")
                elif verb == "RESUME":
                    b.resume(); self._send(conn, b"OK\n")
                elif verb in ("MOVE-TO", "SET-GAINS", "SET-LINE-RATE",
                              "SET-PIXELS", "ENGAGE", "WITHDRAW"):
                    self._send(conn, b"OK\n")
                else:
                    self._send(conn, f"ERR unknown verb {verb}\n".encode())
        finally:
            self._streaming = False
            if self._pusher:
                self._pusher.join(timeout=5.0)
            rf.close(); conn.close(); self._srv.close()

    def _start_push(self, conn) -> None:
        self._streaming = True

        def push():
            import time
            while self._streaming:
                frame = self.backend.acquire()
                if frame is None:
                    try:
                        self._send(conn, b"STREAM-END\n")
                    except OSError:
                        pass
                    self._streaming = False
                    break
                try:
                    self._send(conn, _encode_frame(frame))
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
    """Bind a one-client pull server. Returns ``(server_socket, port)``; call
    :func:`handle_one_client` on it (typically in a background thread)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    return srv, srv.getsockname()[1]


def handle_one_client(srv: socket.socket, backend: BackendSimulator,
                      timeout: float = 10.0) -> None:
    """Accept one client and serve pull requests + control verbs until it
    closes. Frames via ``ACQUIRE``; control verbs answered by the mock parser."""
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
