"""nanoindent — interactive web dashboard (Streamlit).

Two pages:
  * FEM simulation — set tip/sample/substrate/mesh and explore the
    force-indentation curve, stress field, analytic-model comparison, and a
    modulus fit, all from the real solver.
  * SLB force-map analysis — upload a JPK force map (e.g. a supported lipid
    bilayer) and get distributions of breakthrough force, height, and Young's
    modulus (Hertz / Dimitriadis / Garcia & Garcia / Chadwick), with a FEM
    cross-check.

Launch
------
    pip install -e ".[web]"
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import streamlit as st

from nanoindent import (
    SimulationConfig, IndentationSolver, analytics, fit_modulus,
    read_jpk_map, slb,
)
from nanoindent.contact import make_tip

st.set_page_config(page_title="nanoindent — AFM FEM dashboard",
                   page_icon="🔬", layout="wide")

FIELD_LABELS = {
    "von_mises": "von Mises stress (MPa)", "sigma_zz": "σ_zz (MPa)",
    "pressure": "pressure (MPa)", "displacement": "|u| (nm)",
    "uz": "u_z (nm)", "ur": "u_r (nm)",
}


@st.cache_data(show_spinner=False)
def run_simulation(params: dict, field: str):
    cfg = SimulationConfig(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = IndentationSolver(cfg)
        curve = solver.run()
        vals = solver.nodal_field(field)
    mesh = solver.domain.mesh
    return {
        "delta": curve.delta, "force": curve.force,
        "equil": float(np.nanmax(curve.equilibrium_residual)),
        "n_dofs": int(curve.n_dofs),
        "mesh_p": mesh.p, "mesh_t": mesh.t, "field": vals,
    }


@st.cache_data(show_spinner=False)
def analyse_map(raw: bytes, R: float, nu: float, h: float, cap: float):
    with tempfile.NamedTemporaryFile(suffix=".jpk-force-map", delete=False) as fh:
        fh.write(raw)
        path = fh.name
    try:
        fmap = read_jpk_map(path)
        an = slb.analyze_map(fmap, R=R, nu=nu, film_thickness=h, fit_max_delta=cap)
    finally:
        os.unlink(path)
    return an, fmap.spring_constant, fmap.inv_ols, fmap.n_rows, fmap.n_cols


# ==========================================================================
# FEM page
# ==========================================================================
def fem_page():
    st.sidebar.header("Indenter")
    tip_shape = st.sidebar.selectbox(
        "Tip shape", ["paraboloid", "sphere", "cone", "blunted_cone"], index=0)
    R = st.sidebar.slider("Tip radius R (nm)", 1.0, 100.0, 20.0, 0.5)
    half_angle = None
    if tip_shape in ("cone", "blunted_cone"):
        half_angle = st.sidebar.slider("Half-angle from axis (deg)", 10.0, 89.0,
                                       70.3, 0.1)

    st.sidebar.header("Sample")
    E = st.sidebar.number_input("Young's modulus E (MPa)", 0.1, 1e5, 50.0)
    nu = st.sidebar.slider("Poisson's ratio ν", -0.5, 0.5, 0.5, 0.01)
    h = st.sidebar.slider("Thickness h (nm)", 1.0, 500.0, 6.2, 0.1)
    L = st.sidebar.slider("Lateral size L (nm)", 10.0, 500.0, 200.0, 5.0)
    substrate = st.sidebar.radio("Substrate", ["bonded", "unbonded"], index=0)

    st.sidebar.header("Loading & mesh")
    delta_max = st.sidebar.slider("Max indentation δ (nm)", 0.1,
                                  min(10.0, 0.95 * R), min(2.0, 0.9 * R), 0.1)
    n_steps = st.sidebar.slider("Load steps", 2, 40, 20)
    mesh_contact = st.sidebar.select_slider(
        "Near-contact element size (nm)",
        options=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2], value=0.02)
    penalty_alpha = st.sidebar.slider("Penalty α", 50.0, 2000.0, 600.0, 50.0)
    formulation = "mixed" if nu >= 0.45 else "primal"
    st.sidebar.caption(f"Formulation: **{formulation}**")

    params = dict(tip_shape=tip_shape, R=R, E=E, nu=nu, h=h, L=L,
                  substrate=substrate, delta_max=delta_max, n_steps=n_steps,
                  mesh_size_contact=mesh_contact, mesh_size_edge=4.0,
                  penalty_alpha=penalty_alpha, formulation=formulation)
    if half_angle is not None:
        params["tip_half_angle_deg"] = half_angle

    st.title("🔬 nanoindent — FEM simulation")
    field = st.selectbox("Field to display", list(FIELD_LABELS),
                         format_func=lambda k: FIELD_LABELS[k])
    compare = st.multiselect(
        "Analytic models to overlay",
        ["hertz", "dimitriadis_bonded", "garcia_bec", "chadwick_2002"],
        default=["hertz", "dimitriadis_bonded"])
    run = st.button("▶ Run simulation", type="primary")

    if not (run or st.session_state.get("fem_run")):
        st.info("Set parameters and press **Run simulation**. Incompressible "
                "samples (ν ≥ 0.45) use the locking-free mixed formulation.")
        return
    st.session_state["fem_run"] = True

    try:
        with st.spinner("Solving the contact problem…"):
            res = run_simulation(params, field)
    except Exception as exc:
        st.error(f"Simulation failed: {exc}"); return

    cfg = SimulationConfig(**params)
    c = st.columns(4)
    c[0].metric("F(δ_max)", f"{res['force'][-1]:.0f} pN")
    c[1].metric("Peak " + field.replace("_", " "), f"{np.nanmax(res['field']):.2f}")
    c[2].metric("Equilibrium", f"{res['equil']*100:.2f} %")
    c[3].metric("DOF", f"{res['n_dofs']:,}")

    a, b = st.columns(2)
    with a:
        st.subheader("Force–indentation")
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.plot(res["delta"], res["force"], "o-", color="#0072B2", ms=4, label="FEM")
        dd = np.linspace(0, cfg.delta_max, 200); reg = analytics.model_registry()
        for key in compare:
            if key in reg:
                ax.plot(dd, reg[key].func(dd, cfg), "--", lw=1.2, label=reg[key].label)
        ax.set_xlabel("indentation δ (nm)"); ax.set_ylabel("force F (pN)")
        ax.set_xlim(0); ax.set_ylim(bottom=0); ax.grid(alpha=0.25); ax.legend(fontsize=8)
        st.pyplot(fig)
        csv = "delta_nm,force_pN\n" + "\n".join(
            f"{d:.6g},{f:.6g}" for d, f in zip(res["delta"], res["force"]))
        st.download_button("⬇ F(δ) CSV", csv, "force_indentation.csv", "text/csv")
    with b:
        st.subheader(FIELD_LABELS[field])
        fig2, ax2 = plt.subplots(figsize=(5, 3.6))
        r0, z, tris = res["mesh_p"][0], res["mesh_p"][1], res["mesh_t"].T
        vmax = float(np.nanmax(res["field"])) or 1.0
        lv = np.linspace(min(0, float(np.nanmin(res["field"]))), vmax, 40)
        for sgn in (1, -1):
            tcf = ax2.tricontourf(mtri.Triangulation(sgn * r0, z, tris),
                                  res["field"], levels=lv, cmap="jet")
        rr = np.linspace(-6 * np.sqrt(R * delta_max), 6 * np.sqrt(R * delta_max), 400)
        ztip = make_tip(cfg).surface_z(np.abs(rr), delta_max)
        fin = np.isfinite(ztip)
        ax2.plot(rr[fin], ztip[fin], color="crimson", lw=1.2)
        ax2.set_xlim(rr.min(), rr.max()); ax2.set_ylim(0, h * 1.3)
        ax2.set_aspect("equal"); ax2.set_xlabel("r (nm)"); ax2.set_ylabel("z (nm)")
        fig2.colorbar(tcf, ax=ax2, fraction=0.046, pad=0.03)
        st.pyplot(fig2)


# ==========================================================================
# SLB force-map page
# ==========================================================================
def slb_page():
    st.title("🧫 SLB force-map analysis")
    st.caption("Upload a JPK force map (`.jpk-force-map`). Extracts breakthrough "
               "force, bilayer height, and Young's modulus (Hertz / Dimitriadis "
               "/ Garcia & Garcia / Chadwick) per curve, with a FEM cross-check.")

    up = st.file_uploader("JPK force map", type=["jpk-force-map", "jpk", "zip"])
    st.sidebar.header("Analysis parameters")
    R = st.sidebar.slider("Tip radius R (nm)", 1.0, 100.0, 20.0, 0.5)
    nu = st.sidebar.slider("Poisson's ratio ν", 0.0, 0.5, 0.5, 0.01)
    h = st.sidebar.slider("Film thickness h (nm)", 1.0, 30.0, 5.0, 0.5)
    cap = st.sidebar.slider("Fit window δ ≤ (nm)", 0.5, 15.0, 3.0, 0.5)

    if up is None:
        st.info("Upload a `.jpk-force-map` file to begin.")
        return
    try:
        with st.spinner("Decoding calibration and analysing curves…"):
            an, k, invols, nr, nc = analyse_map(up.getvalue(), R, nu, h, cap)
    except Exception as exc:
        st.error(f"Could not analyse file: {exc}"); return

    s = an.summary()
    st.success(f"{len(an.results)} curves ({nr}×{nc}), "
               f"k = {k:.3f} N/m, InvOLS = {invols:.2f} nm/V"
               if invols else f"{len(an.results)} curves")

    m = st.columns(4)
    m[0].metric("Breakthrough force",
                f"{s['breakthrough_force_nN']['median']:.1f} nN")
    m[1].metric("Height contact→substrate",
                f"{s['height_contact_substrate_nm']['median']:.2f} nm",
                help="Bilayer thickness from the measured-height channel; the "
                     "breakthrough step is in the summary table below.")
    m[2].metric("E (Hertz)", f"{s['E_hertz_MPa']['median']:.1f} MPa")
    m[3].metric("κ_A (Connell 2010)",
                f"{s['kappa_A_mN_per_m']['median']:.0f} mN/m",
                help="Area compressibility modulus for a fluid bilayer; fluid "
                     "SLBs are ~200–300 mN/m. Much lower values indicate a "
                     "non-fluid / soft-solid response.")

    with st.spinner("Rendering distributions and FEM cross-check…"):
        fig = slb.plot_analysis(an, film_thickness=h, fit_max_delta=cap)
    st.pyplot(fig)

    # summary table + CSV
    rows = ["quantity,n,median,mean,std"]
    for key, st_ in s.items():
        rows.append(f"{key},{st_['n']},{st_['median']:.4g},{st_['mean']:.4g},"
                    f"{st_['std']:.4g}")
    st.download_button("⬇ Summary CSV", "\n".join(rows), "slb_summary.csv",
                       "text/csv")
    st.dataframe({"quantity": list(s), **{
        "median": [f"{v['median']:.3g}" for v in s.values()],
        "mean": [f"{v['mean']:.3g}" for v in s.values()],
        "n": [v["n"] for v in s.values()]}})


# ==========================================================================
PAGE = st.sidebar.radio("Page", ["FEM simulation", "SLB force-map analysis"])
st.sidebar.markdown("---")
if PAGE == "FEM simulation":
    fem_page()
else:
    slb_page()
