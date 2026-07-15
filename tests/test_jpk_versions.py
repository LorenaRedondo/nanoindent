"""Version-aware JPK dialects: selection, auto-detection, and round-trip.

The shipped per-version dialects are placeholders that currently share the
REFERENCE grammar (their real verbs must come from each version's manual), so a
*fictional distinct* dialect (``_DEMO``, different verbs and tokens) is used to
prove the dialect machinery drives an arbitrary grammar end-to-end.
"""

import numpy as np
import pytest

from nanoindent import (
    StreamingAnalyzer, run_control_loop, recenter_on_brightest, combine_policies,
    ramp_on_event,
)
from nanoindent.adapters import (
    BackendSimulator, JPKAdapter, MockJPKConnection, SocketJPKConnection,
    JPKInstrumentServer, JPKDialect, REFERENCE, JPK_SPM_LEGACY, JPK_SPM_4PLUS,
    BRUKER_BIOAFM, DIALECTS, select_dialect, get_dialect,
)

BK = dict(scan_px=48, pixel_size_nm=2.0, frame_interval_s=0.05, n_particles=3,
          D_nm2_per_s=200.0, max_frames=50, seed=1)
NF = BK["max_frames"]


# a fictional dialect with a completely different grammar + response tokens
_DEMO = JPKDialect(
    name="demo", versions="fictional test grammar", verified=False,
    templates={
        "hello": "greet {client}", "move_by": "mv {dy:.5f} {dx:.5f}",
        "move_to": "mvabs {y:.5f} {x:.5f}", "set_setpoint": "sp {v:.6g}",
        "set_gains": "gains {p:.6g} {i:.6g}", "set_line_rate": "lrate {hz:.6g}",
        "engage": "tip-down", "withdraw": "tip-up", "pause": "hold",
        "resume": "go", "acquire": "grab", "stream_on": "stream start",
        "stream_off": "stream stop", "force_ramp": "ramp {dy:.5f} {dx:.5f}",
        "ping": "hb", "close": "bye",
    },
    tokens={"ok": "ACK", "pong": "BEAT", "error": "FAIL", "caps": "INFO",
            "frame": "IMG", "fd": "CURVE", "stream_end": "STREAM-DONE",
            "end": "DONE"})


# --------------------------------------------------------------------------
# registry + selection
# --------------------------------------------------------------------------
def test_dialects_registered():
    for name in ("reference", "jpk_spm_legacy", "jpk_spm_4plus", "bruker_bioafm"):
        assert name in DIALECTS
    # only the reference/loopback grammar is claimed verified
    assert REFERENCE.verified
    assert not JPK_SPM_LEGACY.verified and not BRUKER_BIOAFM.verified


@pytest.mark.parametrize("version,expected", [
    ("NanoWizard 3.2 / JPK SPM 3.4", JPK_SPM_LEGACY),
    ("ForceRobot 300", JPK_SPM_LEGACY),
    ("NanoWizard 4.2 (JPK SPM 5.1)", JPK_SPM_4PLUS),
    ("Sense ULTRA Speed", JPK_SPM_4PLUS),
    ("Bruker BioAFM 2.1", BRUKER_BIOAFM),
    ("NanoRacer HS-AFM", BRUKER_BIOAFM),
    ("something unknown", REFERENCE),
    ("", REFERENCE),
])
def test_select_dialect_maps_versions(version, expected):
    assert select_dialect(version) is expected


def test_get_dialect_resolves_spec():
    assert get_dialect("auto") is REFERENCE
    assert get_dialect(None) is REFERENCE
    assert get_dialect("bruker_bioafm") is BRUKER_BIOAFM
    assert get_dialect(_DEMO) is _DEMO
    with pytest.raises(ValueError):
        get_dialect("no-such-dialect")


# --------------------------------------------------------------------------
# generic encode / parse round-trip
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dialect", [REFERENCE, _DEMO])
def test_encode_parse_round_trip(dialect):
    assert dialect.parse_command(dialect.encode("move_by", dy=1.5, dx=-2.0))[0] == "move_by"
    assert dialect.parse_command(dialect.encode("stream_on"))[0] == "stream_on"
    assert dialect.parse_command(dialect.encode("stream_off"))[0] == "stream_off"
    assert dialect.parse_command(dialect.encode("engage"))[0] == "engage"
    op, args = dialect.parse_command(dialect.encode("set_gains", p=0.5, i=120.0))
    assert op == "set_gains" and len(args) == 2


# --------------------------------------------------------------------------
# version auto-detection (no sockets)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("version,expected_name", [
    ("NanoWizard 4.2 (JPK SPM 5.1)", "jpk_spm_4plus"),
    ("Bruker BioAFM 2.1", "bruker_bioafm"),
    ("NanoWizard 3.0", "jpk_spm_legacy"),
])
def test_adapter_auto_selects_dialect_from_handshake(version, expected_name):
    b = BackendSimulator(**BK)
    conn = MockJPKConnection(b, version=version)
    ad = JPKAdapter(conn, max_frames=1, dialect="auto")
    list(ad)                                       # connect + handshake
    assert ad.capabilities.version == version
    assert ad.dialect.name == expected_name


def test_explicit_dialect_overrides_auto():
    b = BackendSimulator(**BK)
    conn = MockJPKConnection(b, version="Bruker BioAFM 2.1", dialect=JPK_SPM_LEGACY)
    ad = JPKAdapter(conn, max_frames=1, dialect="jpk_spm_legacy")
    list(ad)
    assert ad.dialect.name == "jpk_spm_legacy"     # not overridden by the version


# --------------------------------------------------------------------------
# a genuinely different grammar drives the whole path
# --------------------------------------------------------------------------
def test_demo_dialect_emits_its_own_verbs():
    b = BackendSimulator(**BK)
    conn = MockJPKConnection(b, dialect=_DEMO)
    ad = JPKAdapter(conn, max_frames=1, dialect=_DEMO)
    ad._ensure_connected()
    ad.engage(); ad.set_feedback_gains(0.5, 120.0)
    assert "tip-down" in conn.sent
    assert any(s.startswith("gains 0.5 120") for s in conn.sent)
    assert any(s.startswith("greet") for s in conn.sent)   # handshake verb


def test_demo_dialect_closed_loop_over_tcp():
    b = BackendSimulator(**BK)
    srv = JPKInstrumentServer(b, dialect=_DEMO, version="demo").start()
    try:
        conn = SocketJPKConnection("127.0.0.1", srv.port, timeout=10, dialect=_DEMO)
        ad = JPKAdapter(conn, max_frames=NF, dialect=_DEMO, record_ramps=True)
        an = StreamingAnalyzer(2.0, 0.05, max_step_px=6.0, event_k=3.0)
        rep = run_control_loop(ad, an, combine_policies(
            recenter_on_brightest(), ramp_on_event()))
    finally:
        srv.join(timeout=10)
    off = np.sqrt(np.mean(np.sum(b.instrument.target_offset_px() ** 2, axis=1)))
    assert rep.n_frames == NF                      # frames decoded via the IMG token
    assert off < BK["scan_px"] / 8                 # closed loop over the alt grammar
    for fr in ad.force_ramps:                      # CURVE-token force data decoded
        assert fr.max_force_pN > 0


def test_registered_version_dialect_round_trips_over_tcp():
    # a shipped placeholder dialect (shares REFERENCE grammar) works end to end
    b = BackendSimulator(**BK)
    srv = JPKInstrumentServer(b, dialect=BRUKER_BIOAFM,
                              version="Bruker BioAFM 2.1").start()
    try:
        conn = SocketJPKConnection("127.0.0.1", srv.port, timeout=10,
                                   dialect=BRUKER_BIOAFM)
        ad = JPKAdapter(conn, max_frames=NF, dialect="auto")
        an = StreamingAnalyzer(2.0, 0.05, max_step_px=6.0)
        rep = run_control_loop(ad, an, recenter_on_brightest())
    finally:
        srv.join(timeout=10)
    assert rep.n_frames == NF
    assert ad.dialect.name == "bruker_bioafm"      # auto-selected from version
