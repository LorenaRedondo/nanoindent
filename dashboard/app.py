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
    read_jpk_map, slb, video as hsvideo, streaming, ExperimentIndex, load_run,
    forcemap, viscoelastic as ve,
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
# HS-AFM video page
# ==========================================================================
@st.cache_data(show_spinner=False)
def load_movie(raw: bytes, name: str, px: float, dt: float):
    suffix = os.path.splitext(name)[1] or ".npy"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
        fh.write(raw)
        path = fh.name
    try:
        if suffix.lower() in (".npy", ".npz"):
            vid = hsvideo.read_numpy(path, pixel_size_nm=px, frame_interval_s=dt)
        elif suffix.lower() in (".tif", ".tiff"):
            vid = hsvideo.read_tiff(path, pixel_size_nm=px, frame_interval_s=dt)
        elif suffix.lower() == ".asd":
            vid = hsvideo.read_asd(path)
        else:
            vid = hsvideo.load_video(path)
    finally:
        os.unlink(path)
    return vid


def hsafm_page():
    st.title("🎬 HS-AFM movie analysis")
    st.caption("Upload a high-speed AFM movie (`.asd`, multi-page TIFF, or "
               "NumPy `.npy`/`.npz`) — or use a synthetic diffusing-particle "
               "movie — for drift correction, per-frame statistics, particle "
               "tracking, diffusion coefficients, and event detection.")

    src = st.sidebar.radio("Source", ["Synthetic demo", "Upload file"])
    st.sidebar.header("Calibration")
    px = st.sidebar.number_input("Pixel size (nm/px)", 0.1, 100.0, 2.0, 0.1)
    dt = st.sidebar.number_input("Frame interval (s)", 0.001, 10.0, 0.1,
                                 0.001, format="%.3f")

    vid = None
    if src == "Synthetic demo":
        st.sidebar.header("Synthetic movie")
        nfr = st.sidebar.slider("Frames", 20, 200, 100, 10)
        npart = st.sidebar.slider("Particles", 1, 12, 5)
        D0 = st.sidebar.slider("True D (nm²/s)", 20.0, 600.0, 200.0, 10.0)
        drift = st.sidebar.slider("Drift (nm/frame)", 0.0, 2.0, 0.0, 0.1)
        vid = hsvideo.synthetic_video(n_frames=nfr, size=80, n_particles=npart,
                                      pixel_size_nm=px, frame_interval_s=dt,
                                      D_nm2_per_s=D0, drift_nm_per_frame=drift,
                                      seed=0)
    else:
        up = st.file_uploader("HS-AFM movie",
                              type=["asd", "tif", "tiff", "npy", "npz"])
        if up is None:
            st.info("Upload an `.asd`, TIFF, or NumPy movie to begin.")
            return
        try:
            vid = load_movie(up.getvalue(), up.name, px, dt)
        except Exception as exc:
            st.error(f"Could not read movie: {exc}"); return

    st.sidebar.header("Analysis")
    do_drift = st.sidebar.checkbox("Drift-correct (phase correlation)", False)
    max_step = st.sidebar.slider("Max linking step (px)", 2.0, 20.0, 6.0, 0.5)
    min_len = st.sidebar.slider("Min track length", 3, 30, 8)

    if do_drift:
        vid = hsvideo.drift_correct(vid, reference="previous")

    c = st.columns(4)
    c[0].metric("Frames", vid.n_frames)
    c[1].metric("Frame size", f"{vid.shape[1]}×{vid.shape[2]}")
    c[2].metric("Duration", f"{vid.duration():.2f} s")
    c[3].metric("Frame rate", f"{vid.fps:.1f} fps")

    with st.spinner("Tracking particles and fitting diffusion…"):
        tracks = hsvideo.track(vid, max_step_px=max_step)
        D = hsvideo.ensemble_diffusion(tracks, vid.pixel_size_nm,
                                       vid.frame_interval_s, min_length=min_len)

    m = st.columns(3)
    m[0].metric("Tracks", len(tracks))
    m[1].metric("Fitted (≥ min length)", int(D.size))
    m[2].metric("Median D",
                f"{np.median(D):.0f} nm²/s" if D.size else "—",
                help="From the initial MSD slope, MSD = 4·D·t (2-D).")
    if vid.metadata.get("true_D_nm2_per_s"):
        st.caption(f"Ground-truth D for this synthetic movie: "
                   f"**{vid.metadata['true_D_nm2_per_s']:.0f} nm²/s** — centroid "
                   f"tracking of overlapping blobs typically underestimates it.")

    with st.spinner("Rendering movie summary…"):
        fig = hsvideo.plot_summary(vid, tracks=tracks)
    st.pyplot(fig)

    # per-frame stats + event detection on the mean-height series
    stats = hsvideo.per_frame_stats(vid)
    t, series = hsvideo.height_timeseries(vid)
    events = hsvideo.detect_events(series, t)
    a, b = st.columns(2)
    with a:
        st.subheader("Height / roughness vs time")
        fig2, ax = plt.subplots(figsize=(5, 3.4))
        ax.plot(stats["time_s"], stats["mean_height_nm"], label="mean height")
        ax.plot(stats["time_s"], stats["rms_roughness_nm"], label="RMS roughness")
        for _, te, _ in events:
            ax.axvline(te, color="crimson", ls=":", lw=0.8)
        ax.set_xlabel("time (s)"); ax.set_ylabel("nm")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        st.pyplot(fig2)
    with b:
        st.subheader("Detected events")
        if events:
            st.dataframe({"frame": [e[0] for e in events],
                          "time_s": [f"{e[1]:.3f}" for e in events],
                          "Δheight_nm": [f"{e[2]:.2f}" for e in events]})
        else:
            st.write("No step-like events above the MAD threshold.")
        csv = "time_s,mean_height_nm,rms_roughness_nm,coverage\n" + "\n".join(
            f"{ti:.6g},{h:.6g},{r:.6g},{cv:.6g}" for ti, h, r, cv in zip(
                stats["time_s"], stats["mean_height_nm"],
                stats["rms_roughness_nm"], stats["coverage"]))
        st.download_button("⬇ Per-frame stats CSV", csv, "hsafm_stats.csv",
                           "text/csv")


# ==========================================================================
# Live control (simulated) page
# ==========================================================================
@st.cache_data(show_spinner=False)
def run_control(scan_px, px, dt, npart, D0, nframes, apply, seed):
    kw = dict(scan_px=scan_px, pixel_size_nm=px, frame_interval_s=dt,
              n_particles=npart, D_nm2_per_s=D0, max_frames=nframes, seed=seed)
    inst = streaming.SimulatedInstrument(**kw)
    an = streaming.StreamingAnalyzer(px, dt, max_step_px=6.0)
    rep = streaming.run_control_loop(
        inst, an,
        streaming.combine_policies(streaming.recenter_on_brightest(),
                                   streaming.ramp_on_event()),
        apply=apply)
    off = float(np.sqrt(np.mean(np.sum(inst.target_offset_px() ** 2, axis=1))))
    return {
        "proc_ms": rep.proc_ms, "commands": rep.commands, "events": rep.events,
        "kept_up": rep.kept_up(), "rtf": rep.real_time_factor(),
        "n_ramps": len(inst.ramp_events), "rms_offset": off,
        "offsets": inst.target_offset_px(), "scan_px": scan_px, "dt": dt,
    }


def control_page():
    st.title("🎯 Live control (simulated)")
    st.caption("Pseudo-realtime closed loop against a **reactive simulated "
               "instrument** — no hardware. Frames stream in, particles are "
               "tracked online, and a policy re-centres the scan window on the "
               "brightest feature. Swap the simulator for a vendor adapter with "
               "the same `__iter__` / `send` contract to drive a real "
               "microscope. This loop is *supervisory* only — fast tip "
               "protection stays on the instrument's controller.")

    st.sidebar.header("Simulated instrument")
    scan_px = st.sidebar.select_slider("Scan size (px)", [32, 48, 64, 96, 128],
                                       value=64)
    px = st.sidebar.number_input("Pixel size (nm/px)", 0.5, 20.0, 2.0, 0.5)
    fps = st.sidebar.slider("Frame rate (fps)", 5, 100, 20, 5)
    dt = 1.0 / fps
    npart = st.sidebar.slider("Particles", 1, 10, 4)
    D0 = st.sidebar.slider("Diffusion D (nm²/s)", 20.0, 600.0, 200.0, 10.0)
    nframes = st.sidebar.slider("Frames", 30, 300, 150, 10)

    st.sidebar.header("Control")
    apply = st.sidebar.checkbox("Close the loop (re-center + ramps)", True)

    if not st.button("▶ Run control loop", type="primary"):
        st.info("Set parameters and press **Run control loop**. Compare the "
                "closed loop (feature stays centred) against the open loop "
                "(feature drifts out of view).")
        return

    with st.spinner("Streaming frames and closing the loop…"):
        res = run_control(scan_px, px, dt, npart, D0, nframes, apply, seed=1)
        # open-loop baseline (same seed) for comparison
        base = run_control(scan_px, px, dt, npart, D0, nframes, False, seed=1)

    c = st.columns(4)
    c[0].metric("Analysis / frame",
                f"{np.median(res['proc_ms'][1:]):.2f} ms",
                help=f"Frame period is {dt*1e3:.0f} ms.")
    c[1].metric("Real-time factor", f"{res['rtf']:.2f}×",
                help="Median analysis time ÷ frame period (<1 = spare headroom).")
    c[2].metric("Keeps up", "✅ yes" if res["kept_up"] else "⚠️ no")
    c[3].metric("Commands", f"{len(res['commands'])}")

    d = st.columns(2)
    d[0].metric("Target offset (closed loop)", f"{res['rms_offset']:.1f} px",
                help="RMS distance of the followed feature from the window "
                     "centre — small means the loop holds it in view.")
    d[1].metric("Target offset (open loop)", f"{base['rms_offset']:.1f} px",
                delta=f"{base['rms_offset'] - res['rms_offset']:.1f} px tighter",
                delta_color="inverse")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    fr = np.arange(len(res["proc_ms"]))
    axs[0].plot(fr, res["proc_ms"], ".-", ms=3, color="#0072B2")
    axs[0].axhline(dt * 1e3, color="crimson", ls="--",
                   label=f"frame period {dt*1e3:.0f} ms")
    axs[0].set_xlabel("frame"); axs[0].set_ylabel("analysis time (ms)")
    axs[0].set_ylim(bottom=0); axs[0].legend(fontsize=8); axs[0].grid(alpha=0.25)
    axs[0].set_title("real-time headroom", fontsize=10)

    half = scan_px / 2
    axs[1].plot(res["offsets"][:, 1], res["offsets"][:, 0], ".-", ms=3,
                color="#009E73", label="closed loop")
    axs[1].plot(base["offsets"][:, 1], base["offsets"][:, 0], ".-", ms=2,
                color="gray", alpha=0.6, label="open loop")
    axs[1].scatter([0], [0], marker="+", s=140, color="crimson")
    axs[1].set_xlim(-half, half); axs[1].set_ylim(-half, half)
    axs[1].set_aspect("equal"); axs[1].legend(fontsize=8)
    axs[1].set_xlabel("x offset (px)"); axs[1].set_ylabel("y offset (px)")
    axs[1].set_title("feature vs window centre", fontsize=10)
    fig.tight_layout()
    st.pyplot(fig)


# ==========================================================================
# Run browser page — compare & replay recorded control runs
# ==========================================================================
def run_browser_page():
    st.title("📁 Run browser")
    st.caption("Point at a folder of recorded runs (a `RunRecorder` output tree). "
               "Compare them in a table, then drill into one to replay its "
               "per-frame stats, commands, events, and frames.")

    root = st.text_input("Runs directory", value="runs")
    if not os.path.isdir(root):
        st.info(f"Directory `{root}` not found. Record runs with "
                "`nanoindent-control … --outdir runs/<name>` (or the Python API), "
                "then set the folder here.")
        return

    idx = ExperimentIndex(root)
    if len(idx) == 0:
        st.warning(f"No run manifests found under `{root}`.")
        return

    st.subheader(f"{len(idx)} run(s)")
    rows = idx.rows()
    metric_cols = [c for c in idx.columns() if c.startswith("metrics.")][:6]
    show = ["run_id", "started_at"] + [c for c in idx.columns()
                                       if c.startswith("tags.")][:3] + metric_cols
    st.dataframe({c.split(".")[-1]: [f"{r.get(c, '')}" for r in rows] for c in show})
    st.download_button("⬇ Comparison CSV",
                       "\n".join([",".join(idx.columns())] +
                                 [",".join(f"{r.get(c, '')}" for c in idx.columns())
                                  for r in rows]),
                       "comparison.csv", "text/csv")

    st.markdown("---")
    run_ids = [r["run_id"] for r in rows]
    chosen = st.selectbox("Replay a run", run_ids)
    man = next(m for m in idx.manifests() if m.get("run_id") == chosen)
    try:
        run = load_run(man["outdir"])
    except Exception as exc:
        st.error(f"Could not load run: {exc}"); return

    c = st.columns(4)
    c[0].metric("Frames", run.n_frames)
    c[1].metric("Commands", len(run.commands))
    c[2].metric("Events", len(run.events))
    rtf = run.metrics.get("real_time_factor")
    c[3].metric("Real-time factor", f"{rtf:.2f}×" if rtf is not None else "—")

    with st.expander("Params / metrics / tags"):
        st.json({"params": run.params, "metrics": run.metrics, "tags": run.tags})

    st.subheader("Replay summary")
    with st.spinner("Rendering recorded run…"):
        st.pyplot(run.plot_summary())

    if run.frames is not None:
        st.subheader("Recorded frames")
        t = st.slider("Frame", 0, run.n_frames - 1, 0)
        figf, axf = plt.subplots(figsize=(4.5, 4.2))
        axf.imshow(run.frames[t], cmap="afmhot", origin="lower")
        axf.set_title(f"frame {t}", fontsize=9)
        axf.set_xlabel("x (px)"); axf.set_ylabel("y (px)")
        st.pyplot(figf)
    else:
        st.caption("This run was recorded without frames (`save_frames=False`).")


# ==========================================================================
# Viscoelastic fit page
# ==========================================================================
@st.cache_data(show_spinner=False)
def geometric_kernel(R: float, nu: float, h: float, delta_max: float):
    """Geometric kernel phi(delta)=F/E from one FEM solve (cached)."""
    cfg = SimulationConfig(R=R, nu=nu, h=h, L=max(100.0, 8 * h),
                           delta_max=delta_max, formulation="mixed",
                           mesh_size_contact=0.35, mesh_size_edge=12.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return forcemap.geometric_kernel(cfg)


def _parse_csv(raw: bytes, n_cols: int):
    import io
    txt = raw.decode("utf-8", "replace")
    rows = []
    for line in txt.splitlines():
        line = line.strip()
        if not line or line[0] in "#":
            continue
        parts = [p for p in line.replace(",", " ").split() if p]
        try:
            vals = [float(p) for p in parts[:n_cols]]
        except ValueError:
            continue          # header row
        if len(vals) == n_cols:
            rows.append(vals)
    return np.array(rows, dtype=float)


def viscoelastic_fit_page():
    st.title("🧪 Viscoelastic fit")
    st.caption("Recover a sample's Standard-Linear-Solid parameters — glassy E0, "
               "equilibrium E∞, relaxation time τ — from a stress-relaxation "
               "(ramp-and-hold) curve or a frequency sweep. The FEM geometric "
               "kernel separates tip/geometry from material response.")

    protocol = st.sidebar.radio("Protocol", ["Stress relaxation", "Frequency sweep"])
    source = st.sidebar.radio("Data", ["Synthetic demo", "Upload CSV"])

    st.sidebar.header("Tip & geometry")
    R = st.sidebar.slider("Tip radius R (nm)", 1.0, 100.0, 20.0, 0.5)
    nu = st.sidebar.slider("Poisson's ratio ν", 0.0, 0.5, 0.5, 0.01)
    h = st.sidebar.slider("Film thickness h (nm)", 1.0, 40.0, 6.0, 0.5)
    delta_hold = st.sidebar.slider("Hold / operating indentation δ (nm)",
                                   0.2, 2.5, 1.2, 0.1)
    kern = geometric_kernel(R, nu, h, max(delta_hold, 1.5))
    phi = float(kern.phi_of(np.array([delta_hold]))[0])
    st.sidebar.caption(f"geometric kernel φ(δ) = {phi:.2f} pN/MPa")

    truth = None
    if source == "Synthetic demo":
        st.sidebar.header("True material (hidden from the fit)")
        E0 = st.sidebar.slider("E0 glassy (MPa)", 2.0, 200.0, 45.0, 1.0)
        Einf = st.sidebar.slider("E∞ equilibrium (MPa)", 1.0, float(E0) - 1.0,
                                 min(12.0, float(E0) - 1.0), 1.0)
        tau = st.sidebar.select_slider(
            "τ relaxation (s)",
            options=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0], value=0.03)
        truth = (E0, Einf, tau)
        sls = ve.standard_linear_solid(E0=E0, E_inf=Einf, tau=tau)

    if protocol == "Stress relaxation":
        noise = st.sidebar.slider("Force noise (pN)", 0.0, 60.0, 15.0, 1.0)
        if source == "Synthetic demo":
            rng = np.random.default_rng(0)
            t, d, F = ve.ramp_hold_curve(sls, kern.phi_of, delta_max=delta_hold,
                                         approach_velocity_nm_s=2e4,
                                         hold_s=8 * tau, n_ramp=60, n_hold=300)
            hold = t >= t[d.argmax()]
            th, Fh = t[hold] - t[hold][0], F[hold] + rng.normal(0, noise, hold.sum())
        else:
            up = st.file_uploader("CSV: two columns — time (s), force (pN)",
                                  type=["csv", "txt", "dat"])
            if up is None:
                st.info("Upload a hold curve, or switch to Synthetic demo.")
                return
            data = _parse_csv(up.getvalue(), 2)
            if data.size == 0:
                st.error("No numeric (time, force) rows found."); return
            th, Fh = data[:, 0], data[:, 1]

        try:
            fit = ve.fit_relaxation(th, Fh, phi)
        except Exception as exc:
            st.error(f"Fit failed: {exc}"); return

        c = st.columns(4)
        c[0].metric("E0 glassy", f"{fit.E0:.1f} MPa", f"± {fit.E0_std:.2g}")
        c[1].metric("E∞ equilibrium", f"{fit.E_inf:.1f} MPa", f"± {fit.E_inf_std:.2g}")
        c[2].metric("τ relaxation", f"{fit.tau*1e3:.0f} ms", f"± {fit.tau_std*1e3:.2g}")
        c[3].metric("β = 1−E∞/E0", f"{fit.beta:.2f}")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(th * 1e3, Fh, "o", ms=3, color="0.6", label="measured")
        tg = np.linspace(th.min(), th.max(), 300)
        ax.plot(tg * 1e3, phi * fit.model.E_relax(tg - tg.min()), "-",
                color="#D55E00", lw=2.2, label="SLS fit")
        ax.set_xlabel("time in hold (ms)"); ax.set_ylabel("force (pN)")
        ax.legend(frameon=False); st.pyplot(fig)

    else:  # Frequency sweep
        noise = st.sidebar.slider("E′/E″ noise (MPa)", 0.0, 3.0, 0.6, 0.1)
        if source == "Synthetic demo":
            rng = np.random.default_rng(0)
            freqs = np.logspace(-0.5, 3.0, 18)
            om = 2 * np.pi * freqs
            spec = ve.frequency_sweep(
                sls, kern.phi_of, lambda d0: ve.kernel_slope(kern.phi_of, d0),
                om, delta0=delta_hold, delta_amp=0.05)
            Ep = spec.storage + rng.normal(0, noise, spec.storage.size)
            Epp = spec.loss + rng.normal(0, noise, spec.loss.size)
        else:
            up = st.file_uploader("CSV: three columns — freq (Hz), E′ (MPa), E″ (MPa)",
                                  type=["csv", "txt", "dat"])
            if up is None:
                st.info("Upload a spectrum, or switch to Synthetic demo.")
                return
            data = _parse_csv(up.getvalue(), 3)
            if data.size == 0:
                st.error("No numeric (freq, E', E'') rows found."); return
            freqs, Ep, Epp = data[:, 0], data[:, 1], data[:, 2]
            om = 2 * np.pi * freqs

        try:
            fit = ve.fit_dynamic_spectrum(om, Ep, Epp)
        except Exception as exc:
            st.error(f"Fit failed: {exc}"); return

        c = st.columns(4)
        c[0].metric("E0 glassy", f"{fit.E0:.1f} MPa", f"± {fit.E0_std:.2g}")
        c[1].metric("E∞ equilibrium", f"{fit.E_inf:.1f} MPa", f"± {fit.E_inf_std:.2g}")
        c[2].metric("τ relaxation", f"{fit.tau*1e3:.0f} ms", f"± {fit.tau_std*1e3:.2g}")
        c[3].metric("f corner", f"{1/(2*np.pi*fit.tau):.1f} Hz")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.loglog(freqs, Ep, "o", ms=5, mfc="none", color="#0072B2", label="E′")
        ax.loglog(freqs, Epp, "s", ms=5, mfc="none", color="#D55E00", label="E″")
        fg = np.logspace(np.log10(freqs.min()), np.log10(freqs.max()), 200)
        wg = 2 * np.pi * fg
        ax.loglog(fg, fit.model.storage(wg), "-", color="#0072B2", lw=2)
        ax.loglog(fg, fit.model.loss(wg), "-", color="#D55E00", lw=2)
        ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("modulus (MPa)")
        ax.legend(frameon=False); st.pyplot(fig)

    if truth is not None:
        st.caption(f"ground truth (synthetic): E0 = {truth[0]:g} MPa, "
                   f"E∞ = {truth[1]:g} MPa, τ = {truth[2]*1e3:g} ms — "
                   "the fit above never sees these.")
    st.markdown(
        "**Requirements on real data:** a good tip/geometry calibration (the "
        "kernel φ), a detected contact point, and an experiment whose timescale "
        "brackets τ. Relaxation and frequency protocols live on the loading/hold "
        "branch where the theory is exact; hysteresis-loop fitting does not.")


# ==========================================================================
PAGE = st.sidebar.radio(
    "Page",
    ["FEM simulation", "SLB force-map analysis", "Viscoelastic fit",
     "HS-AFM movie analysis", "Live control (simulated)", "Run browser"])
st.sidebar.markdown("---")
if PAGE == "FEM simulation":
    fem_page()
elif PAGE == "SLB force-map analysis":
    slb_page()
elif PAGE == "Viscoelastic fit":
    viscoelastic_fit_page()
elif PAGE == "HS-AFM movie analysis":
    hsafm_page()
elif PAGE == "Live control (simulated)":
    control_page()
else:
    run_browser_page()
