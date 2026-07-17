"""LIPIVERSE adapter tests.

Fully hermetic: every test injects a transport (``MockLipiverseTransport`` or a
local fake), so no LIPIVERSE service and no network are required. The urllib
error mapping is exercised by faking the transport's exceptions rather than by
opening sockets.
"""

import json
import urllib.error

import numpy as np
import pytest

from nanoindent import lipids as lp
from nanoindent.lipids import (
    LipiverseClient,
    LipiverseError,
    LipiverseUnavailable,
    MembranePrediction,
    MockLipiverseTransport,
    NotAMembraneLipid,
    UnknownSpecies,
    fetch_membrane,
    normalize_species_id,
)
from nanoindent import slb


@pytest.fixture
def client():
    return LipiverseClient(transport=MockLipiverseTransport())


# ==========================================================================
# species id parsing
# ==========================================================================
@pytest.mark.parametrize("raw,expected", [
    ("PC 34:1", "PC-34-1"),
    ("PC-34-1", "PC-34-1"),
    ("PC34:1", "PC-34-1"),
    ("  pc 34:1  ", "PC-34-1"),
    ("SM 34:1", "SM-34-1"),
    ("PC 36:2", "PC-36-2"),
    ("PC 32:0", "PC-32-0"),
])
def test_normalize_species_id(raw, expected):
    assert normalize_species_id(raw) == expected


@pytest.mark.parametrize("bad", ["", "   ", "PC", "34:1", "PC 34", "PC 34:1:2", "not a lipid"])
def test_normalize_species_id_rejects_junk(bad):
    with pytest.raises(UnknownSpecies):
        normalize_species_id(bad)


# ==========================================================================
# fetching
# ==========================================================================
def test_get_species_returns_prediction(client):
    pred = client.get_species("PC 34:1")
    assert isinstance(pred, MembranePrediction)
    assert pred.species_id == "PC-34-1"
    assert pred.shorthand == "PC 34:1"
    assert pred.thickness_nm == 3.72
    assert pred.breakthrough_nn == 2
    assert pred.kappa_a_mn_per_m == 277
    assert pred.tm_c == 22.0
    assert pred.bilayer is True


def test_prediction_is_always_tagged_predicted(client):
    """Provenance must survive the boundary — the whole point of the adapter."""
    pred = client.get_species("PC 34:1")
    assert pred.provenance == "predicted"
    assert pred.note
    assert "predicted" in pred.summary()


def test_source_url_recorded(client):
    pred = client.get_species("PC 34:1")
    assert pred.source_url.endswith("/species/PC-34-1")


def test_client_builds_url_from_base(client):
    transport = MockLipiverseTransport()
    LipiverseClient("http://example.test:9999/", transport=transport).get_species("PC 34:1")
    assert transport.requests == ["http://example.test:9999/species/PC-34-1"]


def test_base_url_from_env(monkeypatch):
    monkeypatch.setenv("LIPIVERSE_URL", "http://from-env:1234")
    assert LipiverseClient(transport=MockLipiverseTransport()).base_url == "http://from-env:1234"


def test_explicit_base_url_beats_env(monkeypatch):
    monkeypatch.setenv("LIPIVERSE_URL", "http://from-env:1234")
    c = LipiverseClient("http://explicit:1", transport=MockLipiverseTransport())
    assert c.base_url == "http://explicit:1"


def test_default_base_url(monkeypatch):
    monkeypatch.delenv("LIPIVERSE_URL", raising=False)
    c = LipiverseClient(transport=MockLipiverseTransport())
    assert c.base_url == lp.LIPIVERSE_DEFAULT_URL


def test_cache_avoids_refetch():
    transport = MockLipiverseTransport()
    c = LipiverseClient(transport=transport, cache=True)
    a, b = c.get_species("PC 34:1"), c.get_species("PC-34-1")
    assert a is b
    assert len(transport.requests) == 1


def test_cache_disabled_refetches():
    transport = MockLipiverseTransport()
    c = LipiverseClient(transport=transport, cache=False)
    c.get_species("PC 34:1")
    c.get_species("PC 34:1")
    assert len(transport.requests) == 2


def test_unknown_species_raises(client):
    with pytest.raises(UnknownSpecies):
        client.get_species("PC 99:9")


def test_storage_lipid_raises_not_a_membrane(client):
    """TG has membrane=null in LIPIVERSE — there is no thickness to simulate."""
    with pytest.raises(NotAMembraneLipid, match="no bilayer"):
        client.get_species("TG 52:3")


# ==========================================================================
# transport error mapping
# ==========================================================================
def _raising(exc):
    def transport(url):
        raise exc
    return transport


def test_unreachable_service_raises_unavailable():
    """Real urllib path against a closed local port (refused instantly, no
    external network), so the error mapping is exercised end-to-end."""
    c = LipiverseClient("http://localhost:9", timeout_s=0.25)
    with pytest.raises(LipiverseUnavailable, match="cannot reach LIPIVERSE"):
        c.get_species("PC 34:1")


def test_http_404_maps_to_unknown_species(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
    monkeypatch.setattr(lp.urllib.request, "urlopen", boom)
    c = LipiverseClient(transport=lp._urllib_transport(1.0))
    with pytest.raises(UnknownSpecies):
        c.get_species("PC 34:1")


def test_http_500_maps_to_unavailable(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 500, "Server Error", {}, None)
    monkeypatch.setattr(lp.urllib.request, "urlopen", boom)
    c = LipiverseClient(transport=lp._urllib_transport(1.0))
    with pytest.raises(LipiverseUnavailable, match="HTTP 500"):
        c.get_species("PC 34:1")


def test_non_json_response_raises(monkeypatch):
    class FakeResp:
        def read(self): return b"<html>not json</html>"
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(lp.urllib.request, "urlopen", lambda req, timeout=None: FakeResp())
    c = LipiverseClient(transport=lp._urllib_transport(1.0))
    with pytest.raises(LipiverseError, match="non-JSON"):
        c.get_species("PC 34:1")


def test_available_false_when_unreachable():
    c = LipiverseClient(transport=_raising(LipiverseUnavailable("down")))
    assert c.available() is False


def test_available_true_when_service_answers(client):
    assert client.available() is True


def test_fetch_membrane_optional_degrades_gracefully():
    with pytest.warns(UserWarning, match="LIPIVERSE unavailable"):
        got = fetch_membrane("PC 34:1", transport=_raising(LipiverseUnavailable("down")),
                             required=False)
    assert got is None


def test_fetch_membrane_required_raises():
    with pytest.raises(LipiverseUnavailable):
        fetch_membrane("PC 34:1", transport=_raising(LipiverseUnavailable("down")),
                       required=True)


def test_fetch_membrane_convenience():
    assert fetch_membrane("PC 34:1", transport=MockLipiverseTransport()).thickness_nm == 3.72


# ==========================================================================
# record parsing
# ==========================================================================
def _record(**membrane_overrides):
    mem = dict(bilayer=True, thicknessNm=3.72, areaPerLipid=72, packing=0.7,
               curvature="lamellar bilayer", phase37="fluid (liquid-crystalline)",
               tmC=22.0, breakthroughNn=1.6, compressibilityKa=222)
    mem.update(membrane_overrides)
    return {"id": "PC-34-1", "shorthand": "PC 34:1", "formula": "C42H82NO8P",
            "exactMass": 759.5778, "membrane": mem}


def test_from_record_rejects_missing_field():
    rec = _record()
    del rec["membrane"]["thicknessNm"]
    with pytest.raises(LipiverseError, match="thicknessNm"):
        MembranePrediction.from_record(rec)


def test_from_record_rejects_non_numeric_field():
    with pytest.raises(LipiverseError, match="non-numeric"):
        MembranePrediction.from_record(_record(thicknessNm="thick"))


def test_from_record_rejects_bool_masquerading_as_number():
    with pytest.raises(LipiverseError):
        MembranePrediction.from_record(_record(tmC=True))


def test_from_record_rejects_non_object():
    with pytest.raises(LipiverseError):
        MembranePrediction.from_record(["not", "a", "record"])


def test_from_record_rejects_non_object_membrane():
    with pytest.raises(LipiverseError, match="membrane block"):
        MembranePrediction.from_record({"shorthand": "PC 34:1", "membrane": "yes"})


def test_from_record_carries_api_note():
    pred = MembranePrediction.from_record(
        dict(_record(), properties={"note": "heuristic placeholders"}))
    assert pred.note == "heuristic placeholders"


def test_mock_transport_serves_json_roundtrippable(client):
    """The mock's records must be shaped like real API JSON."""
    t = MockLipiverseTransport()
    rec = t("http://x/species/PC-34-1")
    assert json.loads(json.dumps(rec)) == rec
    assert MembranePrediction.from_record(rec).shorthand == "PC 34:1"


def test_nanomechanics_are_headgroup_blind(client):
    """Documents a real limit of LIPIVERSE 0.1.0 (module docstring, VERIFY).

    breakthroughNn / compressibilityKa are functions of chain composition only,
    so PC / SM / PE 34:1 come back identical even though real bilayers differ.
    If a future LIPIVERSE makes these headgroup-aware, this test should fail and
    the VERIFY note should be revisited.
    """
    pc = client.get_species("PC 34:1")
    sm = client.get_species("SM 34:1")
    pe = client.get_species("PE 34:1")

    assert pc.breakthrough_nn == sm.breakthrough_nn == pe.breakthrough_nn
    assert pc.kappa_a_mn_per_m == sm.kappa_a_mn_per_m == pe.kappa_a_mn_per_m
    assert pc.thickness_nm == sm.thickness_nm == pe.thickness_nm
    # the headgroup does move area/packing, just not the nanomechanics
    assert pc.area_per_lipid_a2 != pe.area_per_lipid_a2
    assert pc.packing != pe.packing


def test_predicted_thickness_is_nearly_flat(client):
    """Also a documented limit: chain length/saturation barely move thickness."""
    thicknesses = [client.get_species(s).thickness_nm
                   for s in ("PC 36:2", "PC 34:1", "PC 32:0")]
    assert max(thicknesses) - min(thicknesses) < 0.05   # 3.71 / 3.72 / 3.73 nm


def test_mock_transport_rejects_foreign_paths():
    with pytest.raises(LipiverseError, match="only serves"):
        MockLipiverseTransport()("http://x/search?q=PC")


# ==========================================================================
# phase
# ==========================================================================
def test_phase_recomputed_at_experiment_temperature(client):
    """DPPC-like PC 32:0 (Tm 41 C) is gel at room temperature, fluid at 50 C."""
    pred = client.get_species("PC 32:0")
    assert pred.is_fluid_at(50.0) is True
    assert pred.is_fluid_at(25.0) is False
    assert "gel" in pred.phase_at(25.0)
    assert "fluid" in pred.phase_at(50.0)


def test_phase37_field_is_not_trusted(client):
    """PC 34:1 (Tm 22 C) is fluid at 25 C even though the API's phase37 is a
    hardcoded 37 C verdict — we recompute from tmC."""
    pred = client.get_species("PC 34:1")
    assert pred.raw["membrane"]["phase37"] == "fluid (liquid-crystalline)"
    assert pred.is_fluid_at(25.0) is True
    assert pred.is_fluid_at(10.0) is False   # below Tm; phase37 would still say fluid


# ==========================================================================
# simulation config
# ==========================================================================
def test_to_simulation_config_uses_predicted_thickness(client):
    pred = client.get_species("PC 34:1")
    cfg = pred.to_simulation_config(E=15.0, R=20.0)
    assert cfg.h == pred.thickness_nm
    assert cfg.E == 15.0
    assert cfg.R == 20.0


def test_config_label_records_provenance(client):
    """A predicted h must not silently become geometry."""
    cfg = client.get_species("PC 34:1").to_simulation_config(E=15.0)
    assert "predicted" in cfg.label
    assert "LIPIVERSE" in cfg.label
    assert "PC 34:1" in cfg.label


def test_config_label_reports_phase_at_temperature(client):
    cfg = client.get_species("PC 32:0").to_simulation_config(E=15.0, temperature_c=25.0)
    assert "gel" in cfg.label
    cfg_hot = client.get_species("PC 32:0").to_simulation_config(E=15.0, temperature_c=50.0)
    assert "fluid" in cfg_hot.label


def test_config_defaults_to_incompressible_mixed(client):
    cfg = client.get_species("PC 34:1").to_simulation_config(E=15.0)
    assert cfg.nu == 0.5
    assert cfg.formulation == "mixed"


def test_config_compressible_uses_primal(client):
    cfg = client.get_species("PC 34:1").to_simulation_config(E=15.0, nu=0.3)
    assert cfg.formulation == "primal"


def test_config_default_delta_stays_below_thickness(client):
    pred = client.get_species("PC 34:1")
    cfg = pred.to_simulation_config(E=15.0)
    assert 0 < cfg.delta_max < pred.thickness_nm


def test_config_overrides_pass_through(client):
    cfg = client.get_species("PC 34:1").to_simulation_config(
        E=15.0, n_steps=3, substrate="unbonded", penalty_alpha=123.0)
    assert cfg.n_steps == 3
    assert cfg.substrate == "unbonded"
    assert cfg.penalty_alpha == 123.0


def test_config_is_valid_for_the_solver(client):
    """SimulationConfig validates in __post_init__; a bad mapping would raise."""
    cfg = client.get_species("PC 36:2").to_simulation_config(E=20.0, R=20.0, delta_max=2.0)
    assert cfg.expected_contact_radius > 0
    assert cfg.E_star > cfg.E
    assert len(cfg.delta_schedule()) == cfg.n_steps


def test_non_lamellar_species_refuses_config():
    """A packing parameter outside the lamellar window means no film to indent."""
    pred = MembranePrediction.from_record(
        _record(bilayer=False, packing=1.4, curvature="inverted hexagonal"))
    with pytest.raises(NotAMembraneLipid, match="not lamellar"):
        pred.to_simulation_config(E=15.0)


def test_E_is_required(client):
    """LIPIVERSE predicts no modulus; the adapter must not invent one."""
    pred = client.get_species("PC 34:1")
    with pytest.raises(TypeError):
        pred.to_simulation_config()


# ==========================================================================
# comparison against measurement
# ==========================================================================
def _analysis(n=8, height=3.6, height_cs=3.9, force_pn=1500.0, kappa=240.0):
    results = [
        slb.CurveResult(pixel=i, breakthrough_force=force_pn, height=height,
                        contact_z=0.0, moduli={"dimitriadis": 15.0},
                        height_contact_substrate=height_cs, kappa_A=kappa)
        for i in range(n)
    ]
    return slb.SLBAnalysis(results=results, R=20.0, nu=0.5, tip_shape="paraboloid")


def test_compare_to_slb_pairs_predicted_and_measured(client):
    pred = client.get_species("PC 34:1")
    cmp = pred.compare_to_slb(_analysis())

    assert cmp["species"] == "PC 34:1"
    assert cmp["thickness_nm"]["predicted"] == 3.72
    assert cmp["thickness_nm"]["measured_median"] == pytest.approx(3.9)
    assert cmp["thickness_nm"]["n"] == 8
    assert cmp["thickness_nm"]["ratio"] == pytest.approx(3.9 / 3.72)


def test_compare_units_line_up(client):
    """Breakthrough: LIPIVERSE nN vs SLBAnalysis nN (pN/1e3) — no unit slip."""
    pred = client.get_species("PC 34:1")
    cmp = pred.compare_to_slb(_analysis(force_pn=2000.0))
    assert cmp["breakthrough_nn"]["predicted"] == 2
    assert cmp["breakthrough_nn"]["measured_median"] == pytest.approx(2.0)
    assert cmp["breakthrough_nn"]["ratio"] == pytest.approx(1.0)
    assert cmp["breakthrough_nn"]["unit"] == "nN"


def test_compare_kappa(client):
    cmp = client.get_species("PC 34:1").compare_to_slb(_analysis(kappa=240.0))
    assert cmp["kappa_A_mN_per_m"]["predicted"] == 277
    assert cmp["kappa_A_mN_per_m"]["measured_median"] == pytest.approx(240.0)


def test_compare_defaults_to_contact_substrate_height(client):
    """LIPIVERSE thickness includes a headgroup term -> definition A."""
    cmp = client.get_species("PC 34:1").compare_to_slb(_analysis(height=3.6, height_cs=3.9))
    assert cmp["height_definition"] == "contact_substrate"
    assert cmp["thickness_nm"]["measured_median"] == pytest.approx(3.9)


def test_compare_step_height_selectable(client):
    cmp = client.get_species("PC 34:1").compare_to_slb(
        _analysis(height=3.6, height_cs=3.9), height_definition="step")
    assert cmp["thickness_nm"]["measured_median"] == pytest.approx(3.6)


def test_compare_reports_both_height_definitions(client):
    cmp = client.get_species("PC 34:1").compare_to_slb(_analysis(height=3.6, height_cs=3.9))
    assert cmp["height_step_nm"]["measured_median"] == pytest.approx(3.6)
    assert cmp["height_contact_substrate_nm"]["measured_median"] == pytest.approx(3.9)


def test_compare_rejects_bad_height_definition(client):
    with pytest.raises(ValueError, match="height_definition"):
        client.get_species("PC 34:1").compare_to_slb(_analysis(), height_definition="bogus")


def test_compare_handles_empty_analysis(client):
    """No usable curves must yield nan, not a crash or a fake zero."""
    empty = slb.SLBAnalysis(results=[slb.CurveResult(0, None, None, None)],
                            R=20.0, nu=0.5, tip_shape="paraboloid")
    cmp = client.get_species("PC 34:1").compare_to_slb(empty)
    assert cmp["thickness_nm"]["n"] == 0
    assert np.isnan(cmp["thickness_nm"]["measured_median"])
    assert np.isnan(cmp["thickness_nm"]["ratio"])


def test_compare_marks_provenance_of_both_sides(client):
    cmp = client.get_species("PC 34:1").compare_to_slb(_analysis())
    assert cmp["provenance"]["predicted"] == "predicted"
    assert cmp["provenance"]["measured"] == "AFM force map"


# ==========================================================================
# package surface
# ==========================================================================
def test_exported_from_package_root():
    import nanoindent

    for name in ("lipids", "LipiverseClient", "MembranePrediction",
                 "MockLipiverseTransport", "fetch_membrane", "LipiverseError",
                 "LipiverseUnavailable", "UnknownSpecies", "NotAMembraneLipid"):
        assert name in nanoindent.__all__
        assert hasattr(nanoindent, name)


def test_transport_is_stdlib_only():
    """Importing the adapter must not pull in a third-party HTTP library.

    Checked in a subprocess: reloading the module in-process would rebuild its
    exception classes and break `pytest.raises` in every module that imported
    them (notably tests/test_lipiverse_contract.py).
    """
    import subprocess
    import sys

    code = (
        "import sys; import nanoindent.lipids; "
        "print(sorted(m for m in ('requests', 'httpx', 'aiohttp', 'urllib3') "
        "if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", f"adapter pulled in {out.stdout.strip()}"
