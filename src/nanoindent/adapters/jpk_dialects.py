"""Version-aware command dialects for the JPK / Bruker scripting protocol.

JPK's scripting grammar is proprietary and has changed across software
generations (JPK SPM on NanoWizard 1–3, the 4.x–6.x line on NanoWizard 4 /
Sense / ULTRA Speed, and the post-2018 Bruker BioAFM software). Rather than bake
one grammar into the adapter, each generation is a swappable :class:`JPKDialect`
— a table of command *templates* and response *tokens* plus a generic
encode/parse. The adapter auto-detects the version from the handshake and selects
the matching dialect.

.. warning::
   The concrete verb strings below are **placeholders**, ``verified=False``. The
   real per-version grammar must be copied from that version's JPK/Bruker
   scripting reference — this module gives each version a named slot and a single
   place to paste it. The mapping in :func:`select_dialect` is heuristic. Only a
   dialect confirmed against real hardware should be marked ``verified=True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple


# Canonical operations every dialect must template. Keys are stable; the
# right-hand strings are the *wire* form (VERIFY per version).
REFERENCE_TEMPLATES: Dict[str, str] = {
    "hello": "HELLO {client}",
    "move_by": "MOVE-BY {dy:.5f} {dx:.5f}",
    "move_to": "MOVE-TO {y:.5f} {x:.5f}",
    "set_setpoint": "SET-SETPOINT {v:.6g}",
    "set_gains": "SET-GAINS {p:.6g} {i:.6g}",
    "set_line_rate": "SET-LINE-RATE {hz:.6g}",
    "engage": "ENGAGE",
    "withdraw": "WITHDRAW",
    "pause": "PAUSE",
    "resume": "RESUME",
    "acquire": "ACQUIRE",
    "stream_on": "STREAM ON",
    "stream_off": "STREAM OFF",
    "force_ramp": "FORCE-RAMP {dy:.5f} {dx:.5f}",
    "ping": "PING",
    "close": "CLOSE",
}

REFERENCE_TOKENS: Dict[str, str] = {
    "ok": "OK", "pong": "PONG", "error": "ERR", "caps": "CAPS",
    "frame": "FRAME", "fd": "FD", "stream_end": "STREAM-END", "end": "END",
}


@dataclass(frozen=True)
class JPKDialect:
    """A version-specific command grammar (templates + response tokens)."""

    name: str
    versions: str = ""
    verified: bool = False
    templates: Dict[str, str] = field(default_factory=lambda: dict(REFERENCE_TEMPLATES))
    tokens: Dict[str, str] = field(default_factory=lambda: dict(REFERENCE_TOKENS))

    # ---- encode ----------------------------------------------------------
    def encode(self, op: str, **kw) -> str:
        return self.templates[op].format(**kw)

    def verb(self, op: str) -> str:
        """The literal command word(s) of ``op`` (the template up to the first
        placeholder)."""
        t = self.templates[op]
        i = t.find("{")
        return (t[:i] if i >= 0 else t).strip()

    # ---- decode (generic, longest-verb match) ----------------------------
    def parse_command(self, line: str) -> Tuple[Optional[str], List[str]]:
        """Map a received wire line back to ``(op, args)`` (args as strings)."""
        s = line.strip()
        best: Optional[str] = None
        for op in self.templates:
            v = self.verb(op)
            if s == v or s.startswith(v + " "):
                if best is None or len(v) > len(self.verb(best)):
                    best = op
        if best is None:
            return None, []
        rest = s[len(self.verb(best)):].strip()
        return best, (rest.split() if rest else [])

    # ---- response tokens -------------------------------------------------
    def is_ok(self, reply: str) -> bool:
        return reply.strip() == self.tokens["ok"]

    def is_error(self, reply: str) -> bool:
        return reply.startswith(self.tokens["error"])

    def is_pong(self, reply: str) -> bool:
        return reply.strip() == self.tokens["pong"]


def parse_caps(reply: str, caps_token: str = "CAPS"):
    """Parse a capabilities reply ``CAPS k=v k=v … version=<free text>`` into a
    dict. ``version`` is taken last and may contain spaces (to end of line)."""
    out: Dict[str, str] = {}
    if not reply.startswith(caps_token):
        return out
    body = reply[len(caps_token):].strip()
    idx = body.find("version=")
    if idx >= 0:
        out["version"] = body[idx + len("version="):].strip()
        body = body[:idx].strip()
    for tok in body.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


# ==========================================================================
# dialect registry — one slot per JPK/Bruker software generation
# ==========================================================================
#: Default / development grammar (matches the loopback mock server).
REFERENCE = JPKDialect(name="reference", versions="loopback / development",
                       verified=True)

#: JPK SPM on NanoWizard 1–3 / ForceRobot / CellHesion (SPM ≤ 3.x era).
#: PLACEHOLDER — paste the grammar from the JPK SPM (≤3.x) scripting reference.
JPK_SPM_LEGACY = replace(REFERENCE, name="jpk_spm_legacy",
                         versions="NanoWizard 1–3, ForceRobot, CellHesion (JPK SPM ≤3.x)",
                         verified=False)

#: JPK SPM 4.x–6.x on NanoWizard 4 / Sense / ULTRA Speed.
#: PLACEHOLDER — paste the grammar from the JPK SPM 4–6 scripting reference.
JPK_SPM_4PLUS = replace(REFERENCE, name="jpk_spm_4plus",
                        versions="NanoWizard 4 / Sense / ULTRA Speed (JPK SPM 4.x–6.x)",
                        verified=False)

#: Post-2018 Bruker BioAFM software (incl. NanoRacer HS-AFM).
#: PLACEHOLDER — paste the grammar from the Bruker BioAFM scripting reference.
BRUKER_BIOAFM = replace(REFERENCE, name="bruker_bioafm",
                        versions="Bruker BioAFM software, NanoRacer (post-2018)",
                        verified=False)

DIALECTS: Dict[str, JPKDialect] = {
    d.name: d for d in (REFERENCE, JPK_SPM_LEGACY, JPK_SPM_4PLUS, BRUKER_BIOAFM)
}


def select_dialect(version: str) -> JPKDialect:
    """Heuristically map an instrument version string to a dialect (``VERIFY``).

    Falls back to :data:`REFERENCE` for unknown / empty versions.
    """
    v = (version or "").lower()
    if any(k in v for k in ("bruker", "bioafm", "nanoracer")):
        return BRUKER_BIOAFM
    if any(k in v for k in ("nanowizard 4", "nanowizard4", "sense", "ultra",
                            "spm 4", "spm 5", "spm 6", "spm4", "spm5", "spm6")):
        return JPK_SPM_4PLUS
    if any(k in v for k in ("nanowizard", "forcerobot", "cellhesion",
                            "spm 1", "spm 2", "spm 3")):
        return JPK_SPM_LEGACY
    return REFERENCE


def get_dialect(spec) -> JPKDialect:
    """Resolve a dialect from a name, a :class:`JPKDialect`, or ``None``/'auto'."""
    if spec is None or spec == "auto":
        return REFERENCE
    if isinstance(spec, JPKDialect):
        return spec
    if spec in DIALECTS:
        return DIALECTS[spec]
    raise ValueError(f"unknown JPK dialect {spec!r}; known: {list(DIALECTS)}")
