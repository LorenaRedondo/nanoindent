"""Cross-repo contract tests for the LIPIVERSE bridge (:mod:`nanoindent.lipids`).

LIPIVERSE ships from a **separate repository**, so nothing in this repo's CI sees
it change. If its ``GET /species/:id`` response drops a ``membrane`` field, renames
one, or silently changes a coefficient, the adapter would only find out at runtime
— inside somebody's analysis. These tests pin the contract instead.

The golden record below is the pivot, and drift is caught on both sides of it::

    live LIPIVERSE  <--(live tier)-->  GOLDEN  <--(hermetic tier)-->  MockLipiverseTransport

Hermetic tier (always runs, no service, no network)
    The frozen golden must satisfy the parser's contract, the parser must actually
    require every field in :data:`~nanoindent.lipids.REQUIRED_MEMBRANE_FIELDS`, and
    the offline mock library must agree with the golden. Catches drift *in this
    repo*: a parser change, or a hand-edited mock quietly diverging from reality.

Live tier (opt-in)
    Fetches from a running LIPIVERSE and asserts it still matches the golden.
    Catches drift *in that repo*. Skipped unless ``LIPIVERSE_URL`` is set::

        cd ../lipids && npm run api          # http://localhost:8787
        LIPIVERSE_URL=http://localhost:8787 pytest -m contract -q

    Run it whenever you touch either side of the bridge. A failure here is
    informative, not necessarily a bug: LIPIVERSE is entitled to improve its
    heuristics. It means "go look, then update the golden and the mock in the same
    commit, and say why" — the same rule as ``test_golden.py``.

Provenance of the golden
    Captured 2026-07-17 from LIPIVERSE 0.1.0 (commit 65dc786) by calling its own
    service layer (``getSpecies`` in ``src/lib/service.ts``) and confirmed against
    the live REST API. Values are LIPIVERSE's heuristic **predictions** — frozen
    here to detect change, not because they are physically authoritative.
"""

import os

import pytest

from nanoindent import lipids as lp
from nanoindent.lipids import (
    REQUIRED_MEMBRANE_FIELDS,
    LipiverseClient,
    LipiverseError,
    MembranePrediction,
    MockLipiverseTransport,
    NotAMembraneLipid,
    UnknownSpecies,
)

pytestmark = pytest.mark.contract

LIPIVERSE_URL = os.environ.get("LIPIVERSE_URL")
requires_service = pytest.mark.skipif(
    not LIPIVERSE_URL,
    reason="set LIPIVERSE_URL (e.g. http://localhost:8787) to run live contract tests",
)


# ==========================================================================
# the golden record
# ==========================================================================
#: species id -> the fields this adapter consumes, as LIPIVERSE 0.1.0 emits them.
GOLDEN = {
    "PC-34-1": {
        "shorthand": "PC 34:1", "formula": "C42H82NO8P", "exactMass": 759.5778,
        "membrane": {
            "bilayer": True, "thicknessNm": 3.72, "areaPerLipid": 72,
            "packing": 0.62, "curvature": "lamellar bilayer",
            "phase37": "fluid (liquid-crystalline)", "tmC": 22,
            "breakthroughNn": 2, "compressibilityKa": 277,
        },
    },
    "PC-36-2": {
        "shorthand": "PC 36:2", "formula": "C44H84NO8P", "exactMass": 785.5935,
        "membrane": {
            "bilayer": True, "thicknessNm": 3.71, "areaPerLipid": 75,
            "packing": 0.62, "curvature": "lamellar bilayer",
            "phase37": "fluid (liquid-crystalline)", "tmC": 3,
            "breakthroughNn": 1.4, "compressibilityKa": 261,
        },
    },
    "PC-32-0": {
        "shorthand": "PC 32:0", "formula": "C40H80NO8P", "exactMass": 733.5622,
        "membrane": {
            "bilayer": True, "thicknessNm": 3.73, "areaPerLipid": 68,
            "packing": 0.62, "curvature": "lamellar bilayer",
            "phase37": "gel (ordered)", "tmC": 41,
            "breakthroughNn": 2.7, "compressibilityKa": 292,
        },
    },
    "SM-34-1": {
        "shorthand": "SM 34:1", "formula": "C39H79N2O6P", "exactMass": 702.5676,
        "membrane": {
            "bilayer": True, "thicknessNm": 3.72, "areaPerLipid": 64,
            "packing": 0.7, "curvature": "lamellar bilayer",
            "phase37": "fluid (liquid-crystalline)", "tmC": 22,
            "breakthroughNn": 2, "compressibilityKa": 277,
        },
    },
    "PE-34-1": {
        "shorthand": "PE 34:1", "formula": "C39H76NO8P", "exactMass": 717.5309,
        "membrane": {
            "bilayer": True, "thicknessNm": 3.72, "areaPerLipid": 56,
            "packing": 0.81, "curvature": "lamellar bilayer",
            "phase37": "fluid (liquid-crystalline)", "tmC": 22,
            "breakthroughNn": 2, "compressibilityKa": 277,
        },
    },
    # a neutral storage lipid: exists, but membrane is null by contract
    "TG-52-3": {
        "shorthand": "TG 52:3", "formula": "C55H100O6", "exactMass": 856.752,
        "membrane": None,
    },
}

BILAYER_IDS = sorted(k for k, v in GOLDEN.items() if v["membrane"] is not None)


def _golden_record(species_id: str) -> dict:
    """The golden as a full API-shaped record (deep-copied: tests mutate it)."""
    import copy

    rec = copy.deepcopy(GOLDEN[species_id])
    rec["id"] = species_id
    return rec


# ==========================================================================
# hermetic tier — the parser and the mock, against the golden
# ==========================================================================
@pytest.mark.parametrize("species_id", BILAYER_IDS)
def test_golden_parses(species_id):
    """Every golden bilayer record must satisfy the parser."""
    pred = MembranePrediction.from_record(_golden_record(species_id))
    g = GOLDEN[species_id]["membrane"]
    assert pred.thickness_nm == g["thicknessNm"]
    assert pred.breakthrough_nn == g["breakthroughNn"]
    assert pred.kappa_a_mn_per_m == g["compressibilityKa"]
    assert pred.tm_c == g["tmC"]
    assert pred.packing == g["packing"]
    assert pred.area_per_lipid_a2 == g["areaPerLipid"]
    assert pred.bilayer is g["bilayer"]


@pytest.mark.parametrize("species_id", BILAYER_IDS)
def test_golden_carries_every_contract_field(species_id):
    mem = GOLDEN[species_id]["membrane"]
    for f in REQUIRED_MEMBRANE_FIELDS:
        assert f in mem, f"golden {species_id} missing contract field {f}"
        assert isinstance(mem[f], (int, float)) and not isinstance(mem[f], bool)


@pytest.mark.parametrize("field", REQUIRED_MEMBRANE_FIELDS)
def test_parser_requires_every_contract_field(field):
    """REQUIRED_MEMBRANE_FIELDS must be exactly what the parser enforces.

    Ties the documented constant to real behaviour: drop any one field and
    from_record must refuse the record rather than invent a value.
    """
    rec = _golden_record("PC-34-1")
    del rec["membrane"][field]
    with pytest.raises(LipiverseError, match=field):
        MembranePrediction.from_record(rec)


def test_storage_lipid_contract_is_membrane_null():
    """TG's membrane is null by contract — the adapter must refuse it clearly."""
    assert GOLDEN["TG-52-3"]["membrane"] is None
    with pytest.raises(NotAMembraneLipid):
        MembranePrediction.from_record(_golden_record("TG-52-3"))


@pytest.mark.parametrize("species_id", sorted(GOLDEN))
def test_mock_library_matches_golden(species_id):
    """The offline library claims to be captured verbatim from LIPIVERSE.

    This is what makes that claim true rather than aspirational: hand-editing
    MockLipiverseTransport.LIBRARY away from the golden fails here.
    """
    mock = MockLipiverseTransport.LIBRARY[species_id]
    golden = GOLDEN[species_id]
    assert mock["membrane"] == golden["membrane"]
    assert mock["shorthand"] == golden["shorthand"]
    assert mock["formula"] == golden["formula"]
    assert mock["exactMass"] == golden["exactMass"]


def test_mock_library_covers_the_golden():
    """No golden species may be missing from the offline library."""
    assert set(MockLipiverseTransport.LIBRARY) == set(GOLDEN)


# ==========================================================================
# live tier — the real service, against the golden
# ==========================================================================
@pytest.fixture(scope="module")
def live_client():
    client = LipiverseClient(LIPIVERSE_URL)
    if not client.available():
        pytest.skip(f"LIPIVERSE not answering at {LIPIVERSE_URL}")
    return client


@requires_service
def test_live_service_reachable(live_client):
    assert live_client.available() is True


@requires_service
@pytest.mark.parametrize("species_id", BILAYER_IDS)
def test_live_membrane_shape_unchanged(live_client, species_id):
    """The live membrane block must still carry every contract field, numerically.

    This is the shape half of the contract: a rename or a type change here breaks
    the adapter, and this is the test that says so at CI time rather than at
    analysis time.
    """
    pred = live_client.get_species(species_id)
    mem = pred.raw.get("membrane")
    assert isinstance(mem, dict), f"live {species_id} has no membrane object"
    for f in REQUIRED_MEMBRANE_FIELDS:
        assert f in mem, f"live {species_id} dropped contract field {f!r}"
        assert isinstance(mem[f], (int, float)) and not isinstance(mem[f], bool), (
            f"live {species_id}.{f} is {type(mem[f]).__name__}, expected a number"
        )


@requires_service
@pytest.mark.parametrize("species_id", BILAYER_IDS)
def test_live_values_match_golden(live_client, species_id):
    """The values half: LIPIVERSE's predictions must still be what we froze.

    A failure is not automatically a bug — LIPIVERSE may have improved a
    heuristic. Investigate, then update GOLDEN *and*
    MockLipiverseTransport.LIBRARY in one commit, and note why. If a coefficient
    moved, the VERIFY notes in nanoindent.lipids may need revisiting too.
    """
    live = live_client.get_species(species_id).raw["membrane"]
    golden = GOLDEN[species_id]["membrane"]
    for f in REQUIRED_MEMBRANE_FIELDS:
        assert live[f] == pytest.approx(golden[f], rel=1e-9), (
            f"{species_id}.{f}: live {live[f]!r} != golden {golden[f]!r} — "
            "LIPIVERSE drifted; see this test's docstring"
        )
    assert live["bilayer"] == golden["bilayer"]
    assert live["curvature"] == golden["curvature"]


@requires_service
@pytest.mark.parametrize("species_id", sorted(GOLDEN))
def test_live_top_level_identity_fields(live_client, species_id):
    """id / shorthand / formula / exactMass are part of the consumed surface too."""
    url = f"{live_client.base_url}/species/{species_id}"
    rec = live_client._transport(url)
    golden = GOLDEN[species_id]
    assert rec["id"] == species_id
    assert rec["shorthand"] == golden["shorthand"]
    assert rec["formula"] == golden["formula"]
    assert rec["exactMass"] == pytest.approx(golden["exactMass"], rel=1e-9)


@requires_service
def test_live_storage_lipid_still_has_null_membrane(live_client):
    """TG must still return membrane=null — the NotAMembraneLipid path depends on it."""
    with pytest.raises(NotAMembraneLipid):
        live_client.get_species("TG 52:3")


@requires_service
def test_live_unknown_species_still_404s(live_client):
    """The 404 -> UnknownSpecies mapping depends on the service's error behaviour."""
    with pytest.raises(UnknownSpecies):
        live_client.get_species("PC 99:9")


@requires_service
def test_live_mock_library_is_faithful(live_client):
    """End-to-end: the offline library must equal what the live service returns.

    Subsumes the two hermetic mock tests when a service is available, and is the
    single assertion that keeps `--offline` runs honest.
    """
    for species_id in sorted(GOLDEN):
        url = f"{live_client.base_url}/species/{species_id}"
        live = live_client._transport(url)
        mock = MockLipiverseTransport.LIBRARY[species_id]
        assert mock["membrane"] == live["membrane"], (
            f"offline library for {species_id} disagrees with the live service"
        )
