import os
import zipfile

import numpy as np
import pytest

from nanoindent import (
    SimulationConfig, synthetic_raw_curve, fit_modulus, analyze_experiment,
    ForceCurveData, read_csv, read_jpk, read_bruker, load_afm, analytics,
)
import nanoindent.experiment as exp


# -- calibration / round-trip ------------------------------------------------
def test_force_from_spring_constant():
    # F = k * deflection: 0.2 N/m * 1 nm = 0.2 nN = 200 pN.
    z = np.array([0.0, 1.0, 2.0, 3.0])
    d = np.array([0.0, 0.0, 1.0, 2.0])
    fc = ForceCurveData(z=z, deflection=d, spring_constant=0.2)
    delta, force = fc.to_force_indentation(baseline=0.0)
    # peak deflection 2 nm -> 0.2*2*1e3 = 400 pN
    assert np.isclose(force.max(), 400.0, rtol=1e-6)


def test_roundtrip_recovers_modulus_noiseless():
    cfg = SimulationConfig(R=17.0, E=130.0, nu=0.3, h=7.0, delta_max=1.0)
    raw = synthetic_raw_curve(cfg, spring_constant=0.2, true_E=130.0,
                              source_model="dimitriadis", baseline_nm=3.0,
                              z_contact=50.0, noise_nm=0.0, seed=0)
    delta, force = raw.to_force_indentation()
    fit = fit_modulus(delta, force, "dimitriadis", cfg.R, cfg.nu, cfg.h,
                      fit_contact_point=True)
    assert fit.E_fit == pytest.approx(130.0, rel=0.05)


def test_hertz_fit_overestimates_on_raw_thin_film():
    cfg = SimulationConfig(R=17.0, E=130.0, nu=0.3, h=7.0, delta_max=1.0)
    raw = synthetic_raw_curve(cfg, spring_constant=0.2, source_model="dimitriadis",
                              noise_nm=0.0)
    delta, force = raw.to_force_indentation()
    hertz = fit_modulus(delta, force, "hertz", cfg.R, cfg.nu, cfg.h,
                        fit_contact_point=True)
    assert hertz.E_fit > 1.3 * cfg.E     # bottom-effect inflation


def test_analyze_experiment_end_to_end():
    cfg = SimulationConfig(R=17.0, E=130.0, nu=0.3, h=7.0, delta_max=1.0)
    raw = synthetic_raw_curve(cfg, spring_constant=0.2, noise_nm=0.0)
    fit, (delta, force) = analyze_experiment(
        raw, R=cfg.R, nu=cfg.nu, h=cfg.h, model="dimitriadis_bonded")
    assert fit.E_fit == pytest.approx(130.0, rel=0.06)
    assert force.max() > 0


def test_volts_need_inv_ols():
    fc = ForceCurveData(z=np.arange(5.0), deflection=np.arange(5.0),
                        deflection_unit="V", spring_constant=0.2)
    with pytest.raises(ValueError):
        fc.to_force_indentation()          # no inv_ols
    d, f = fc.to_force_indentation(inv_ols=10.0)   # V -> nm
    assert f.size == d.size


# -- CSV ---------------------------------------------------------------------
def test_read_csv_roundtrip(tmp_path):
    cfg = SimulationConfig(R=17.0, E=130.0, nu=0.3, h=7.0, delta_max=1.0)
    raw = synthetic_raw_curve(cfg, spring_constant=0.2, noise_nm=0.0)
    p = tmp_path / "curve.csv"
    np.savetxt(p, np.column_stack([raw.z, raw.deflection]), delimiter=",",
               header="z_nm,defl_nm", comments="")
    c = read_csv(str(p), z_col=0, deflection_col=1, skip_header=1,
                 spring_constant=0.2)
    assert len(c.z) == len(raw.z)
    assert np.allclose(c.z, raw.z)
    assert c.fmt == "csv"


def test_read_csv_tab_delimiter(tmp_path):
    p = tmp_path / "curve.tsv"
    np.savetxt(p, np.column_stack([np.arange(5.0), np.arange(5.0)]),
               delimiter="\t")
    c = read_csv(str(p), spring_constant=0.1)
    assert len(c.z) == 5


# -- JPK ---------------------------------------------------------------------
def _make_jpk(path, z_m, defl_m):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("header.properties", "jpk-data-file=spm.forcefile\n")
        seg = "segments/0/"
        zf.writestr(seg + "segment-header.properties",
                    "channel.measuredHeight.data.encoder.type=float\n"
                    "channel.measuredHeight.data.encoder.scaling.multiplier=1.0\n"
                    "channel.measuredHeight.data.encoder.scaling.offset=0.0\n"
                    "channel.vDeflection.data.encoder.type=float\n"
                    "channel.vDeflection.data.encoder.scaling.multiplier=1.0\n"
                    "channel.vDeflection.data.encoder.scaling.offset=0.0\n"
                    "channel.vDeflection.spring-constant=0.2\n")
        zf.writestr(seg + "channels/measuredHeight.dat",
                    np.asarray(z_m, ">f4").tobytes())
        zf.writestr(seg + "channels/vDeflection.dat",
                    np.asarray(defl_m, ">f4").tobytes())


def test_read_jpk(tmp_path):
    z = np.linspace(0, 8, 40)
    d = np.linspace(2, 9, 40)
    p = tmp_path / "c.jpk-force"
    _make_jpk(str(p), z * 1e-9, d * 1e-9)     # stored in metres
    c = read_jpk(str(p))
    assert c.fmt == "jpk"
    assert c.spring_constant == pytest.approx(0.2)
    assert np.allclose(c.z, z, atol=1e-5)
    assert np.allclose(c.deflection, d, atol=1e-5)


# -- Bruker ------------------------------------------------------------------
def _make_bruker(path, z, defl):
    data = np.concatenate([z.astype("<i2"), defl.astype("<i2")]).tobytes()
    header = (
        "\\*Force file list\n"
        "\\Spring Constant: 0.2\n"
        "\\@Sens. Zsens: V 1.0 nm/V\n"
        "\\Data offset: 512\n"
        f"\\Data length: {len(data)}\n"
        "\\Bytes/pixel: 2\n"
        "\\*File list end\n"
    ).encode("latin-1")
    header = header + b" " * (512 - len(header))
    with open(path, "wb") as fh:
        fh.write(header + data)


def test_read_bruker(tmp_path):
    z = np.arange(40, dtype=int)
    d = (np.arange(40) * 10).astype(int)
    p = tmp_path / "c.spm"
    _make_bruker(str(p), z, d)
    c = read_bruker(str(p))
    assert c.fmt == "bruker"
    assert c.spring_constant == pytest.approx(0.2)
    assert len(c.z) == 40 and len(c.deflection) == 40


# -- IBW (stubbed igor2 loader) ----------------------------------------------
def test_read_ibw_stub(monkeypatch):
    def fake_load(path):
        z = np.linspace(0, 8e-9, 30)
        defl = np.linspace(2e-9, 9e-9, 30)
        return {"wave": {"wData": np.column_stack([defl, z]),
                         "labels": [[b""], [b"Defl", b"ZSnsr"]],
                         "note": b"SpringConstant: 0.25\rInvOLS: 45.0\r"}}
    monkeypatch.setattr(exp, "_load_ibw_raw", fake_load)
    c = exp.read_ibw("dummy.ibw")
    assert c.fmt == "ibw"
    assert c.spring_constant == pytest.approx(0.25)
    assert c.inv_ols == pytest.approx(45.0)
    assert c.z.max() == pytest.approx(8.0)          # m -> nm
    assert c.deflection.max() == pytest.approx(9.0)


# -- dispatch ----------------------------------------------------------------
def test_load_afm_dispatch(tmp_path):
    p = tmp_path / "c.csv"
    np.savetxt(p, np.column_stack([np.arange(5.0), np.arange(5.0)]),
               delimiter=",")
    assert load_afm(str(p), spring_constant=0.1).fmt == "csv"
    with pytest.raises(ValueError):
        load_afm("mystery.xyz")
