"""LIPIVERSE adapter — composition-derived membrane properties for SLB runs.

Bridges the LIPIVERSE lipid atlas (a TypeScript service exposing
a REST API) to this library, so a supported-lipid-bilayer (SLB) experiment can be
configured from a **lipid species name** (``"PC 34:1"``) instead of a hand-set
film thickness.

Both projects already describe the same bilayer observables, in the same units::

    LIPIVERSE (predicted from composition)     nanoindent (measured from AFM)
    -------------------------------------      ------------------------------
    membrane.thicknessNm        (nm)     <->   slb height_step / height_contact_substrate
    membrane.breakthroughNn     (nN)     <->   slb breakthrough_force  (nN)
    membrane.compressibilityKa  (mN/m)   <->   slb kappa_A             (mN/m)
    membrane.tmC                (deg C)  ->    phase state -> expected stiffness
    membrane.packing / .bilayer          ->    is this species even lamellar?

so this module supports both directions:

* **configure** — :meth:`MembranePrediction.to_simulation_config` and
  ``thickness_nm`` feed :mod:`nanoindent.slb` / :class:`~nanoindent.config.SimulationConfig`;
* **validate** — :meth:`MembranePrediction.compare_to_slb` puts the prediction next
  to a measured :class:`~nanoindent.slb.SLBAnalysis`, which is the direction that
  can actually calibrate LIPIVERSE's coefficients against data.

Provenance
----------
**LIPIVERSE's membrane numbers are heuristic predictions, not measurements.** They
come from hand-tuned composition formulas (Tanford chain geometry for thickness;
linear fits in carbons/double-bonds per chain for breakthrough force and kappa_A)
and LIPIVERSE itself tags them ``predicted``. This library is otherwise built on
validated physics with no magic numbers in solver code, so the tag is carried all
the way through rather than dropped at the boundary:

* every :class:`MembranePrediction` carries :attr:`~MembranePrediction.provenance`
  (``"predicted"``) and :attr:`~MembranePrediction.note`;
* :meth:`~MembranePrediction.to_simulation_config` stamps the predicted-``h`` origin
  into the config ``label``, so it travels into results and figures;
* ``E`` is **never** inferred — LIPIVERSE does not predict a Young's modulus, which
  is the input the FEM most needs, so callers must pass it explicitly.

Treat a predicted ``h`` as a prior to test, not as geometry to trust.

Dependencies & offline use
--------------------------
Transport is stdlib :mod:`urllib` only — no new dependencies, and nothing here is
imported at package-import time cost. The network is never required: inject
:class:`MockLipiverseTransport` (a small offline species library) or any
``Callable[[str], dict]``, which is also how the tests stay hermetic.

VERIFY
------
Assumptions reconstructed from the LIPIVERSE source that should be confirmed
against experiment before being relied on quantitatively:

* ``thicknessNm`` (``2*lc*0.065 + 0.9 - 0.35*dpc``) includes a headgroup term, so
  it is compared against the SLB **contact->substrate** height (definition A) by
  default rather than the breakthrough step (definition B). Both are reported.
* Bilayers are treated as near-incompressible (``nu = 0.5``, mixed formulation).
  LIPIVERSE supplies no Poisson ratio.
* ``phase37`` in the API is hardcoded at 37 C; this module ignores it and
  recomputes the phase from ``tmC`` at the caller's actual temperature.

Two limits of the prior, observed directly from LIPIVERSE 0.1.0's output and
worth knowing before trusting a predicted number:

* **The nanomechanics are headgroup-blind.** ``breakthroughNn`` and
  ``compressibilityKa`` are functions of carbons/double-bonds per chain only, so
  PC 34:1, SM 34:1 and PE 34:1 all return F_break = 2 nN and kappa_A = 277 mN/m.
  Real SM and PC bilayers differ substantially in both. Cross-class comparisons
  of these two fields carry no information.
* **Predicted thickness is nearly flat.** PC 36:2 / PC 34:1 / PC 32:0 span
  3.71 / 3.72 / 3.73 nm, whereas a measured gel-phase DPPC bilayer is markedly
  thicker than fluid DOPC. The prior discriminates far less than reality does,
  so a close predicted-vs-measured ratio is weak evidence and a poor one is
  expected. ``thicknessNm`` varies mainly through double-bonds per chain.

Both are reasons to run :meth:`MembranePrediction.compare_to_slb` and calibrate,
rather than to consume the prediction as an input.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

__all__ = [
    "LIPIVERSE_DEFAULT_URL",
    "REQUIRED_MEMBRANE_FIELDS",
    "LipiverseError",
    "LipiverseUnavailable",
    "UnknownSpecies",
    "NotAMembraneLipid",
    "MembranePrediction",
    "LipiverseClient",
    "MockLipiverseTransport",
    "fetch_membrane",
]

LIPIVERSE_DEFAULT_URL = "http://localhost:8787"
"""Default LIPIVERSE REST base URL (``npm run api`` in the lipids repo).

Override per call, or globally with the ``LIPIVERSE_URL`` environment variable.
"""

_DEFAULT_TIMEOUT_S = 5.0

REQUIRED_MEMBRANE_FIELDS = (
    "thicknessNm",
    "areaPerLipid",
    "packing",
    "tmC",
    "breakthroughNn",
    "compressibilityKa",
)
"""Numeric fields a LIPIVERSE ``membrane`` block must carry for this adapter to parse it.

This is the **cross-repo contract**. LIPIVERSE ships from a separate repository, so
nothing in this repo's CI sees it change; ``tests/test_lipiverse_contract.py`` pins
this tuple against a frozen golden record and, opt-in, against a live service.
:meth:`MembranePrediction.from_record` raises :class:`LipiverseError` if any is
missing or non-numeric — a loud failure at the boundary rather than a wrong number
downstream.
"""


# ==========================================================================
# errors
# ==========================================================================
class LipiverseError(Exception):
    """Base class for every LIPIVERSE adapter failure."""


class LipiverseUnavailable(LipiverseError):
    """The LIPIVERSE service could not be reached (not running, network, timeout).

    Distinct from :class:`UnknownSpecies`: the question was never answered, as
    opposed to answered with "no such species".
    """


class UnknownSpecies(LipiverseError):
    """LIPIVERSE has no record for the requested species id."""


class NotAMembraneLipid(LipiverseError):
    """The species exists but forms no bilayer (``membrane`` is null).

    Neutral storage lipids (TG, DG, CE, ...) have no headgroup area in LIPIVERSE's
    model, so there is no thickness to hand a film simulation.
    """


# ==========================================================================
# species ids
# ==========================================================================
def normalize_species_id(species: str) -> str:
    """Normalize a species name to LIPIVERSE's id form (``"PC 34:1"`` -> ``"PC-34-1"``).

    Accepts the shorthand (``"PC 34:1"``), the id (``"PC-34-1"``), and the loose
    spellings LIPIVERSE's own parser takes (``"PC34:1"``).
    """
    s = str(species).strip()
    if not s:
        raise UnknownSpecies("empty species id")
    # LIPIVERSE: /^([A-Za-z]+)[- ]?(\d{1,2})[:\-](\d{1,2})$/
    m = re.match(r"^([A-Za-z]+)[- ]?(\d{1,2})[:\-](\d{1,2})$", s)
    if not m:
        raise UnknownSpecies(
            f"cannot parse species {species!r}; expected e.g. 'PC 34:1' or 'PC-34-1'"
        )
    return f"{m.group(1).upper()}-{int(m.group(2))}-{int(m.group(3))}"


# ==========================================================================
# the prediction record
# ==========================================================================
@dataclass(frozen=True)
class MembranePrediction:
    """Composition-derived bilayer properties for one lipid species.

    Every numeric field here is **predicted** (see the module docstring); the
    provenance tag rides along so downstream results can keep saying so.
    """

    species_id: str                  # "PC-34-1"
    shorthand: str                   # "PC 34:1"
    formula: str
    exact_mass: float

    # -- structure ---------------------------------------------------------
    thickness_nm: float
    """Bilayer thickness (nm), Tanford chain geometry + headgroup term."""

    area_per_lipid_a2: float
    """Area per lipid (A^2)."""

    packing: float
    """Shape/packing parameter P = v / (a0 * lc)."""

    curvature: str
    """Preferred phase geometry (micellar / lamellar / inverted hexagonal)."""

    bilayer: bool
    """True when ``packing`` sits in the lamellar window (0.55 <= P <= 1.05)."""

    # -- thermotropic ------------------------------------------------------
    tm_c: float
    """Main gel->fluid transition temperature (deg C)."""

    # -- nanomechanics (the SLB observables) -------------------------------
    breakthrough_nn: float
    """Predicted breakthrough (yield) force (nN)."""

    kappa_a_mn_per_m: float
    """Predicted area-compressibility modulus kappa_A (mN/m)."""

    # -- bookkeeping -------------------------------------------------------
    provenance: str = "predicted"
    note: str = ""
    source_url: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    # ---- phase ----------------------------------------------------------
    def is_fluid_at(self, temperature_c: float) -> bool:
        """Whether the bilayer is fluid (liquid-crystalline) at ``temperature_c``.

        Recomputed from :attr:`tm_c`; the API's ``phase37`` field is ignored
        because it is hardcoded at 37 C while AFM is usually run at room
        temperature.
        """
        return float(temperature_c) > self.tm_c

    def phase_at(self, temperature_c: float) -> str:
        """``"fluid (liquid-crystalline)"`` or ``"gel (ordered)"`` at ``temperature_c``."""
        return ("fluid (liquid-crystalline)" if self.is_fluid_at(temperature_c)
                else "gel (ordered)")

    # ---- simulation config ----------------------------------------------
    def to_simulation_config(
        self,
        E: float,
        *,
        R: float = 20.0,
        delta_max: Optional[float] = None,
        temperature_c: float = 25.0,
        nu: float = 0.5,
        tip_shape: str = "paraboloid",
        substrate: str = "bonded",
        n_steps: int = 12,
        **overrides: Any,
    ):
        """Build a :class:`~nanoindent.config.SimulationConfig` for this bilayer.

        The film thickness ``h`` is the **predicted** :attr:`thickness_nm`, and that
        origin is stamped into the config ``label`` so it survives into results.

        Parameters
        ----------
        E:
            Young's modulus (MPa). **Required** — LIPIVERSE predicts no modulus,
            so there is nothing honest to default to. Fit it from experiment (see
            :func:`nanoindent.slb.analyze_map`) or state it as an assumption.
        R:
            Tip radius (nm).
        delta_max:
            Maximum indentation (nm). Defaults to ~80% of the bilayer thickness,
            keeping the finite-thickness contact models in their valid range and
            staying below breakthrough.
        temperature_c:
            Experiment temperature, used only to report the phase in the label.
        nu:
            Poisson ratio. Defaults to 0.5 (near-incompressible bilayer, VERIFY);
            ``>= 0.45`` automatically selects the locking-free mixed formulation.
        substrate:
            ``"bonded"`` matches an SLB on mica/glass.
        **overrides:
            Any other :class:`~nanoindent.config.SimulationConfig` field.

        Raises
        ------
        NotAMembraneLipid
            If this species' packing parameter puts it outside the lamellar window.
        """
        from .config import SimulationConfig

        if not self.bilayer:
            raise NotAMembraneLipid(
                f"{self.shorthand} is not lamellar (packing={self.packing}, "
                f"{self.curvature}); a film simulation would be meaningless"
            )

        delta = float(delta_max) if delta_max is not None else round(0.8 * self.thickness_nm, 3)
        phase = self.phase_at(temperature_c)
        label = (
            f"{self.shorthand} @ {temperature_c:g} C, {phase} | "
            f"h={self.thickness_nm} nm (LIPIVERSE, predicted) | E={E:g} MPa (caller)"
        )

        cfg_kwargs: Dict[str, Any] = dict(
            tip_shape=tip_shape,
            R=float(R),
            E=float(E),
            nu=float(nu),
            h=float(self.thickness_nm),
            substrate=substrate,
            L=max(20.0, 10.0 * math.sqrt(float(R) * delta)),
            delta_max=delta,
            n_steps=int(n_steps),
            mesh_size_contact=max(0.02, delta / 50.0),
            mesh_size_edge=4.0,
            penalty_alpha=600.0,
            formulation="mixed" if float(nu) >= 0.45 else "primal",
            label=label,
        )
        cfg_kwargs.update(overrides)
        with warnings.catch_warnings():
            # a deliberate nu=0.5 + mixed formulation is not a locking risk
            if cfg_kwargs.get("formulation") == "mixed":
                warnings.simplefilter("ignore")
            return SimulationConfig(**cfg_kwargs)

    # ---- validation against measurement ---------------------------------
    def compare_to_slb(self, analysis, *, height_definition: str = "contact_substrate") -> dict:
        """Put this prediction next to a measured :class:`~nanoindent.slb.SLBAnalysis`.

        Returns one entry per shared observable with the predicted value, the
        measured median, the number of curves behind it, and the ratio
        ``measured / predicted``. Missing measurements come back as ``nan``
        rather than being silently dropped.

        Parameters
        ----------
        height_definition:
            Which SLB height to compare thickness against — ``"contact_substrate"``
            (definition A, the default: LIPIVERSE's thickness includes a headgroup
            term, so this is the like-for-like match) or ``"step"`` (definition B,
            the breakthrough penetration step). Both are always reported under
            ``height_step_nm`` / ``height_contact_substrate_nm``.
        """
        import numpy as np

        if height_definition not in ("contact_substrate", "step"):
            raise ValueError(
                "height_definition must be 'contact_substrate' or 'step', "
                f"got {height_definition!r}"
            )

        def med(a):
            a = np.asarray(a, dtype=float)
            a = a[np.isfinite(a)]
            return (float(np.median(a)) if a.size else float("nan")), int(a.size)

        step_med, step_n = med(analysis.heights())
        cs_med, cs_n = med(analysis.heights_contact_substrate())
        fb_med, fb_n = med(analysis.breakthrough_forces())
        ka_med, ka_n = med(analysis.kappa_A())

        thickness_med, thickness_n = (
            (cs_med, cs_n) if height_definition == "contact_substrate" else (step_med, step_n)
        )

        def entry(predicted, measured, n, unit):
            ratio = (measured / predicted) if (predicted and np.isfinite(measured)) else float("nan")
            return {
                "predicted": predicted,
                "measured_median": measured,
                "n": n,
                "ratio": ratio,
                "unit": unit,
            }

        return {
            "species": self.shorthand,
            "provenance": {"predicted": self.provenance, "measured": "AFM force map"},
            "height_definition": height_definition,
            "thickness_nm": entry(self.thickness_nm, thickness_med, thickness_n, "nm"),
            "breakthrough_nn": entry(self.breakthrough_nn, fb_med, fb_n, "nN"),
            "kappa_A_mN_per_m": entry(self.kappa_a_mn_per_m, ka_med, ka_n, "mN/m"),
            "height_step_nm": {"measured_median": step_med, "n": step_n, "unit": "nm"},
            "height_contact_substrate_nm": {"measured_median": cs_med, "n": cs_n, "unit": "nm"},
        }

    def summary(self) -> str:
        return (
            f"{self.shorthand} ({self.formula}): h={self.thickness_nm} nm, "
            f"Tm={self.tm_c} C, F_break={self.breakthrough_nn} nN, "
            f"kappa_A={self.kappa_a_mn_per_m} mN/m, P={self.packing} "
            f"[{self.provenance}]"
        )

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_record(cls, record: Dict[str, Any], *, source_url: Optional[str] = None
                    ) -> "MembranePrediction":
        """Build from a LIPIVERSE ``GET /species/:id`` JSON record.

        Raises
        ------
        NotAMembraneLipid
            If ``membrane`` is null (a neutral storage lipid).
        LipiverseError
            If the record is missing fields this adapter needs.
        """
        if not isinstance(record, dict):
            raise LipiverseError(f"expected a species record object, got {type(record).__name__}")

        shorthand = record.get("shorthand") or record.get("id") or "?"
        mem = record.get("membrane")
        if mem is None:
            raise NotAMembraneLipid(
                f"{shorthand} forms no bilayer (LIPIVERSE membrane=null) — "
                "neutral storage lipids have no thickness to simulate"
            )
        if not isinstance(mem, dict):
            raise LipiverseError(f"membrane block must be an object, got {type(mem).__name__}")

        def num(block: Dict[str, Any], key: str) -> float:
            v = block.get(key)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise LipiverseError(
                    f"membrane.{key} missing or non-numeric for {shorthand} (got {v!r}); "
                    "is the LIPIVERSE service a compatible version?"
                )
            return float(v)

        props = record.get("properties") or {}
        note = props.get("note") if isinstance(props, dict) else None

        return cls(
            species_id=str(record.get("id") or shorthand),
            shorthand=str(shorthand),
            formula=str(record.get("formula") or ""),
            exact_mass=float(record.get("exactMass") or 0.0),
            thickness_nm=num(mem, "thicknessNm"),
            area_per_lipid_a2=num(mem, "areaPerLipid"),
            packing=num(mem, "packing"),
            curvature=str(mem.get("curvature") or ""),
            bilayer=bool(mem.get("bilayer")),
            tm_c=num(mem, "tmC"),
            breakthrough_nn=num(mem, "breakthroughNn"),
            kappa_a_mn_per_m=num(mem, "compressibilityKa"),
            provenance="predicted",
            note=str(note or "LIPIVERSE membrane properties are heuristic predictions."),
            source_url=source_url,
            raw=record,
        )


# ==========================================================================
# transport
# ==========================================================================
Transport = Callable[[str], Dict[str, Any]]
"""A callable mapping an absolute URL to a decoded JSON object."""


def _urllib_transport(timeout_s: float) -> Transport:
    def fetch(url: str) -> Dict[str, Any]:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                payload = resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise UnknownSpecies(f"LIPIVERSE has no record at {url}") from exc
            raise LipiverseUnavailable(
                f"LIPIVERSE returned HTTP {exc.code} for {url}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise LipiverseUnavailable(
                f"cannot reach LIPIVERSE at {url} ({exc}). Start it with "
                "`npm run api` in the lipids repo, or pass a transport / use "
                "MockLipiverseTransport for offline work."
            ) from exc
        try:
            return json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise LipiverseError(f"LIPIVERSE returned non-JSON from {url}") from exc

    return fetch


class MockLipiverseTransport:
    """Offline stand-in for the LIPIVERSE REST API.

    Serves a small built-in library of common bilayer-forming species so examples
    and tests run with no service and no network. The records below are **verbatim
    output of LIPIVERSE's own service layer** (``getSpecies`` in
    ``src/lib/service.ts``, LIPIVERSE 0.1.0), not re-derived here — so they are
    predictions exactly as the live API's are, with the same caveats.

    Unknown species raise :class:`UnknownSpecies`, matching the live 404 path.

    >>> client = LipiverseClient(transport=MockLipiverseTransport())
    >>> client.get_species("PC 34:1").thickness_nm
    3.72
    """

    #: species id -> record (as the API emits it; captured from LIPIVERSE 0.1.0)
    LIBRARY: Dict[str, Dict[str, Any]] = {
        # POPC-like fluid PC — the workhorse SLB
        "PC-34-1": dict(
            shorthand="PC 34:1", formula="C42H82NO8P", exactMass=759.5778,
            membrane=dict(bilayer=True, thicknessNm=3.72, areaPerLipid=72,
                          packing=0.62, curvature="lamellar bilayer",
                          phase37="fluid (liquid-crystalline)", tmC=22,
                          breakthroughNn=2, compressibilityKa=277),
        ),
        # DOPC-like
        "PC-36-2": dict(
            shorthand="PC 36:2", formula="C44H84NO8P", exactMass=785.5935,
            membrane=dict(bilayer=True, thicknessNm=3.71, areaPerLipid=75,
                          packing=0.62, curvature="lamellar bilayer",
                          phase37="fluid (liquid-crystalline)", tmC=3,
                          breakthroughNn=1.4, compressibilityKa=261),
        ),
        # DPPC-like — gel at room temperature
        "PC-32-0": dict(
            shorthand="PC 32:0", formula="C40H80NO8P", exactMass=733.5622,
            membrane=dict(bilayer=True, thicknessNm=3.73, areaPerLipid=68,
                          packing=0.62, curvature="lamellar bilayer",
                          phase37="gel (ordered)", tmC=41,
                          breakthroughNn=2.7, compressibilityKa=292),
        ),
        # SM — sphingolipid; note LIPIVERSE's nanomechanics are headgroup-blind,
        # so this matches PC 34:1 exactly apart from area/packing (see VERIFY).
        "SM-34-1": dict(
            shorthand="SM 34:1", formula="C39H79N2O6P", exactMass=702.5676,
            membrane=dict(bilayer=True, thicknessNm=3.72, areaPerLipid=64,
                          packing=0.7, curvature="lamellar bilayer",
                          phase37="fluid (liquid-crystalline)", tmC=22,
                          breakthroughNn=2, compressibilityKa=277),
        ),
        # PE — smaller headgroup, higher packing parameter
        "PE-34-1": dict(
            shorthand="PE 34:1", formula="C39H76NO8P", exactMass=717.5309,
            membrane=dict(bilayer=True, thicknessNm=3.72, areaPerLipid=56,
                          packing=0.81, curvature="lamellar bilayer",
                          phase37="fluid (liquid-crystalline)", tmC=22,
                          breakthroughNn=2, compressibilityKa=277),
        ),
        # TG — a neutral storage lipid: exists, but forms no bilayer
        "TG-52-3": dict(
            shorthand="TG 52:3", formula="C55H100O6", exactMass=856.752,
            membrane=None,
        ),
    }

    def __init__(self, library: Optional[Dict[str, Dict[str, Any]]] = None):
        self.library = dict(self.LIBRARY if library is None else library)
        self.requests: list = []

    def __call__(self, url: str) -> Dict[str, Any]:
        self.requests.append(url)
        path = urllib.parse.urlparse(url).path
        prefix = "/species/"
        if not path.startswith(prefix):
            raise LipiverseError(f"MockLipiverseTransport only serves {prefix}:id, got {path!r}")
        species_id = urllib.parse.unquote(path[len(prefix):])
        record = self.library.get(species_id)
        if record is None:
            raise UnknownSpecies(f"MockLipiverseTransport has no species {species_id!r}")
        out = dict(record)
        out.setdefault("id", species_id)
        out.setdefault("properties", {
            "note": "CCS/RT/Tm are heuristic placeholders (see issue #3); "
                    "mass/formula are exact.",
        })
        return out


# ==========================================================================
# client
# ==========================================================================
class LipiverseClient:
    """Read-only client for the LIPIVERSE REST API.

    Parameters
    ----------
    base_url:
        Service root. Defaults to ``$LIPIVERSE_URL`` then
        :data:`LIPIVERSE_DEFAULT_URL`.
    timeout_s:
        Per-request timeout for the default urllib transport.
    transport:
        Optional ``Callable[[str], dict]`` replacing HTTP entirely — pass
        :class:`MockLipiverseTransport` to work offline, or any fake in tests.
    cache:
        Memoize species lookups per client (default True). LIPIVERSE records are
        deterministic for a given service version.

    >>> client = LipiverseClient(transport=MockLipiverseTransport())
    >>> client.get_species("PC 34:1").shorthand
    'PC 34:1'
    """

    def __init__(self, base_url: Optional[str] = None, *, timeout_s: float = _DEFAULT_TIMEOUT_S,
                 transport: Optional[Transport] = None, cache: bool = True):
        url = base_url or os.environ.get("LIPIVERSE_URL") or LIPIVERSE_DEFAULT_URL
        self.base_url = str(url).rstrip("/")
        self.timeout_s = float(timeout_s)
        self._transport: Transport = transport or _urllib_transport(self.timeout_s)
        self._cache: Optional[Dict[str, MembranePrediction]] = {} if cache else None

    def get_species(self, species: str) -> MembranePrediction:
        """Fetch one species' membrane prediction.

        Raises :class:`UnknownSpecies`, :class:`NotAMembraneLipid`, or
        :class:`LipiverseUnavailable`.
        """
        species_id = normalize_species_id(species)
        if self._cache is not None and species_id in self._cache:
            return self._cache[species_id]

        url = f"{self.base_url}/species/{urllib.parse.quote(species_id)}"
        record = self._transport(url)
        if record is None:
            raise UnknownSpecies(f"LIPIVERSE returned no record for {species_id}")
        prediction = MembranePrediction.from_record(record, source_url=url)
        if self._cache is not None:
            self._cache[species_id] = prediction
        return prediction

    def available(self) -> bool:
        """Whether the service answers a probe request (never raises)."""
        try:
            self.get_species("PC 34:1")
            return True
        except LipiverseUnavailable:
            return False
        except LipiverseError:
            return True   # it answered, just not with that species


# ==========================================================================
# convenience
# ==========================================================================
def fetch_membrane(species: str, *, base_url: Optional[str] = None,
                   transport: Optional[Transport] = None,
                   timeout_s: float = _DEFAULT_TIMEOUT_S,
                   default: Optional[MembranePrediction] = None,
                   required: bool = True) -> Optional[MembranePrediction]:
    """Fetch one species' membrane prediction in a single call.

    Parameters
    ----------
    required:
        When True (default), an unreachable service raises
        :class:`LipiverseUnavailable`. Set False to degrade gracefully to
        ``default`` (``None``) instead — useful when the lipid prior is an
        optional enrichment and the analysis must still run without it.

    >>> fetch_membrane("PC 34:1", transport=MockLipiverseTransport()).thickness_nm
    3.72
    """
    client = LipiverseClient(base_url, timeout_s=timeout_s, transport=transport)
    try:
        return client.get_species(species)
    except LipiverseUnavailable:
        if required:
            raise
        warnings.warn(
            f"LIPIVERSE unavailable at {client.base_url}; continuing without a "
            f"membrane prior for {species!r}.",
            stacklevel=2,
        )
        return default
