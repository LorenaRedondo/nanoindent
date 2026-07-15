"""JPK force-map reader + supported-lipid-bilayer analysis tests.

Uses a synthetic ``.jpk-force-map`` fixture built to the documented JPK layout
(shared-data lcd-info calibration chain, per-pixel segment channels) so the
reader's calibration decoding and the SLB analysis are exercised without a real
(proprietary) instrument file.
"""

import zipfile
import numpy as np
import pytest

from nanoindent import read_jpk_map, slb, JPKForceMap


# calibration used by the fixture (chosen so decoded z/defl equal the raw ints)
_SHARED = """\
lcd-info.0.encoder.scaling.multiplier=1.0
lcd-info.0.encoder.scaling.offset=0.0
lcd-info.0.conversion-set.conversions.base=volts
lcd-info.0.conversion-set.conversion.nominal.base-calibration-slot=volts
lcd-info.0.conversion-set.conversion.nominal.scaling.multiplier=1e-9
lcd-info.0.conversion-set.conversion.nominal.scaling.offset=0.0
lcd-info.1.encoder.scaling.multiplier=1.0
lcd-info.1.encoder.scaling.offset=0.0
lcd-info.1.conversion-set.conversions.base=volts
lcd-info.1.conversion-set.conversions.default=force
lcd-info.1.conversion-set.conversion.distance.base-calibration-slot=volts
lcd-info.1.conversion-set.conversion.distance.scaling.multiplier=1e-9
lcd-info.1.conversion-set.conversion.distance.scaling.offset=0.0
lcd-info.1.conversion-set.conversion.force.base-calibration-slot=distance
lcd-info.1.conversion-set.conversion.force.scaling.multiplier=0.6
lcd-info.1.conversion-set.conversion.force.scaling.offset=0.0
lcd-info.2.encoder.scaling.multiplier=1.0
lcd-info.2.encoder.scaling.offset=0.0
lcd-info.2.conversion-set.conversions.base=volts
lcd-info.2.conversion-set.conversion.nominal.base-calibration-slot=volts
lcd-info.2.conversion-set.conversion.nominal.scaling.multiplier=1e-9
lcd-info.2.conversion-set.conversion.nominal.scaling.offset=0.0
"""

_SEGHDR = ("channels.list=height vDeflection measuredHeight\n"
           "channel.height.lcd-info.*=0\n"
           "channel.vDeflection.lcd-info.*=1\n"
           "channel.measuredHeight.lcd-info.*=2\n")


def _synthetic_curve(z_contact=150.0, elastic_slope=0.5, break_travel=30.0,
                     drop=5.0, n=800):
    """z (nm, descending) and deflection (nm) with a breakthrough step."""
    z = np.linspace(300.0, 0.0, n)
    defl = np.zeros_like(z)
    travel = z_contact - z
    pre = (travel > 0) & (travel <= break_travel)
    defl[pre] = elastic_slope * travel[pre]
    post = travel > break_travel
    defl[post] = elastic_slope * break_travel - drop + 0.9 * (travel[post] - break_travel)
    return z, defl


def _make_jpk_map(path, n_pixels=4):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("header.properties", "force-scan-map.type=map\n")
        zf.writestr("shared-data/header.properties", _SHARED)
        for i in range(n_pixels):
            seg = f"index/{i}/segments/0/"
            zf.writestr(seg + "segment-header.properties", _SEGHDR)
            z, defl = _synthetic_curve(break_travel=28 + i, drop=4 + 0.2 * i)
            zf.writestr(seg + "channels/height.dat",
                        np.round(z).astype(">i4").tobytes())
            zf.writestr(seg + "channels/vDeflection.dat",
                        np.round(defl).astype(">i4").tobytes())
            zf.writestr(seg + "channels/measuredHeight.dat",
                        np.round(z).astype(">i4").tobytes())


def test_read_jpk_map_calibration(tmp_path):
    p = tmp_path / "m.jpk-force-map"
    _make_jpk_map(str(p), n_pixels=4)
    fmap = read_jpk_map(str(p))
    assert isinstance(fmap, JPKForceMap)
    assert len(fmap) == 4 and fmap.n_rows * fmap.n_cols == 4
    assert fmap.spring_constant == pytest.approx(0.6)
    assert fmap.inv_ols == pytest.approx(1.0)     # distance mult 1e-9 -> 1 nm/V
    c = fmap[0]
    assert c.z.max() == pytest.approx(300.0, abs=1.0)
    assert c.deflection.max() > 0


def test_slb_process_curve_finds_breakthrough(tmp_path):
    p = tmp_path / "m.jpk-force-map"
    _make_jpk_map(str(p), n_pixels=4)
    fmap = read_jpk_map(str(p))
    r = slb.process_curve(fmap[0], R=20.0, nu=0.5, film_thickness=5.0,
                          fit_max_delta=3.0)
    assert r.breakthrough_force is not None and r.breakthrough_force > 0
    assert r.height is not None                  # breakthrough step (B)
    assert r.height_contact_substrate is not None and r.height_contact_substrate > 0  # (A)
    # moduli fitted for every model
    for m in slb.MODULUS_MODELS:
        assert m in r.moduli and r.moduli[m] > 0


def test_slb_analyze_map_distributions(tmp_path):
    p = tmp_path / "m.jpk-force-map"
    _make_jpk_map(str(p), n_pixels=4)
    an = slb.analyze_map(read_jpk_map(str(p)), R=20.0, nu=0.5,
                         film_thickness=5.0, fit_max_delta=3.0)
    s = an.summary()
    assert s["breakthrough_force_nN"]["n"] >= 3
    assert s["height_step_nm"]["median"] > 0
    assert s["height_contact_substrate_nm"]["n"] >= 3
    for m in slb.MODULUS_MODELS:
        assert s[f"E_{m}_MPa"]["n"] >= 3
    # apparent Hertz modulus exceeds the bottom-effect-corrected ones
    assert an.moduli("hertz").mean() > an.moduli("garcia_bec").mean()


def test_slb_plot_analysis_runs(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    p = tmp_path / "m.jpk-force-map"
    _make_jpk_map(str(p), n_pixels=4)
    an = slb.analyze_map(read_jpk_map(str(p)), R=20.0, nu=0.5,
                         film_thickness=5.0, fit_max_delta=3.0)
    fig = slb.plot_analysis(an, film_thickness=5.0, fit_max_delta=3.0,
                            run_fem=False)
    assert len(fig.axes) >= 4
