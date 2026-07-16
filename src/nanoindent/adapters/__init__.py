"""Vendor ``FrameSource`` adapters for real (HS-)AFM instruments.

Each adapter exposes the :class:`nanoindent.streaming.FrameSource` contract, so
it plugs straight into :func:`nanoindent.streaming.run_control_loop`. All three
are **scaffolds** validated against in-process mock backends (shared physics via
:class:`BackendSimulator`); the real vendor calls / wire protocols are marked
``VERIFY`` and must be confirmed against the SDK and instrument.

* :class:`NanosurfAdapter` — official ``nanosurf`` Python package (in-process).
* :class:`JPKAdapter` — JPK/Bruker BioAFM TCP socket protocol.
* :class:`AsylumAdapter` — Asylum Research via an Igor Pro listener bridge.
"""

from .base import InstrumentAdapter
from .emulation import BackendSimulator
from .nanosurf import (NanosurfAdapter, MockNanosurfSPM, NanosurfCapabilities,
                       NanosurfForceCurve)
from .jpk import (
    JPKAdapter, MockJPKConnection, SocketJPKConnection, JPKInstrumentServer,
    JPKCapabilities, JPKForceRamp, JPKError, serve_backend, handle_one_client,
)
from .jpk_dialects import (
    JPKDialect, REFERENCE, JPK_SPM_LEGACY, JPK_SPM_4PLUS, BRUKER_BIOAFM,
    DIALECTS, select_dialect, get_dialect,
)
from .asylum import AsylumAdapter, MockIgorBridge

__all__ = [
    "InstrumentAdapter",
    "BackendSimulator",
    "NanosurfAdapter",
    "MockNanosurfSPM",
    "NanosurfCapabilities",
    "NanosurfForceCurve",
    "JPKAdapter",
    "MockJPKConnection",
    "SocketJPKConnection",
    "JPKInstrumentServer",
    "JPKCapabilities",
    "JPKForceRamp",
    "JPKError",
    "JPKDialect",
    "REFERENCE",
    "JPK_SPM_LEGACY",
    "JPK_SPM_4PLUS",
    "BRUKER_BIOAFM",
    "DIALECTS",
    "select_dialect",
    "get_dialect",
    "serve_backend",
    "handle_one_client",
    "AsylumAdapter",
    "MockIgorBridge",
]
