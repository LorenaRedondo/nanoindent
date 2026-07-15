"""HS-AFM (high-speed AFM) movie analysis tests.

A synthetic diffusing-particle movie (:func:`nanoindent.synthetic_video`) with a
known ground-truth diffusion coefficient exercises the universal NumPy/TIFF path
plus every analysis (drift correction, per-frame stats, tracking, MSD/diffusion,
event detection). The ``.asd`` reader is checked with a round-trip against a
file written to the documented version-1 layout.
"""

import struct

import numpy as np
import pytest

from nanoindent import video as V
from nanoindent import AFMVideo, synthetic_video


# --------------------------------------------------------------------------
# container + synthetic generator
# --------------------------------------------------------------------------
def test_afmvideo_container():
    frames = np.zeros((5, 8, 8))
    vid = AFMVideo(frames, pixel_size_nm=2.0, frame_interval_s=0.25)
    assert vid.n_frames == 5 and vid.shape == (5, 8, 8)
    assert vid.fps == pytest.approx(4.0)
    assert vid.duration() == pytest.approx(1.25)
    assert np.allclose(vid.times(), [0, 0.25, 0.5, 0.75, 1.0])


def test_afmvideo_rejects_non_3d():
    with pytest.raises(ValueError):
        AFMVideo(np.zeros((8, 8)))


def test_synthetic_video_shape_and_truth():
    vid = synthetic_video(n_frames=20, size=32, n_particles=4, seed=0)
    assert vid.shape == (20, 32, 32)
    assert vid.metadata["true_D_nm2_per_s"] == 200.0
    assert vid.metadata["n_particles"] == 4
    # particles rise above the noise floor
    assert vid.frames.max() > 2.0


# --------------------------------------------------------------------------
# drift correction
# --------------------------------------------------------------------------
def test_drift_correct_recovers_translation():
    # a static scene translated by a known integer shift each frame
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (48, 48))
    base[18:24, 30:36] += 8.0                      # a feature to lock onto
    shift = 2
    frames = [np.roll(np.roll(base, shift * t, axis=0), shift * t, axis=1)
              for t in range(6)]
    vid = AFMVideo(np.stack(frames), pixel_size_nm=1.0, frame_interval_s=0.1)
    dc = V.drift_correct(vid, reference="first")
    # after correction every frame matches frame 0
    for t in range(1, vid.n_frames):
        assert np.corrcoef(dc.frames[t].ravel(), dc.frames[0].ravel())[0, 1] > 0.99
    assert "drift_shifts_px" in dc.metadata


# --------------------------------------------------------------------------
# per-frame statistics
# --------------------------------------------------------------------------
def test_per_frame_stats_keys_and_shapes():
    vid = synthetic_video(n_frames=15, size=32, seed=1)
    s = V.per_frame_stats(vid)
    for k in ("time_s", "mean_height_nm", "rms_roughness_nm", "coverage",
              "max_height_nm"):
        assert k in s and len(s[k]) == vid.n_frames
    assert np.all(s["coverage"] >= 0) and np.all(s["coverage"] <= 1)


def test_height_timeseries_region():
    vid = synthetic_video(n_frames=10, size=32, seed=1)
    t, whole = V.height_timeseries(vid)
    assert len(t) == 10
    _, region = V.height_timeseries(vid, region=(0, 16, 0, 16))
    assert region.shape == (10,)


# --------------------------------------------------------------------------
# detection + tracking + diffusion
# --------------------------------------------------------------------------
def test_detect_particles_counts():
    vid = synthetic_video(n_frames=1, size=48, n_particles=3,
                          particle_sigma_nm=4.0, noise_nm=0.05, seed=2)
    cents = V.detect_particles(vid.frames[0])
    assert cents.shape[1] == 2
    assert 1 <= len(cents) <= 5            # ~3, tolerating merges/splits


def test_track_and_diffusion_recovery():
    D0 = 200.0
    vid = synthetic_video(n_frames=120, size=96, n_particles=4,
                          pixel_size_nm=2.0, frame_interval_s=0.05,
                          D_nm2_per_s=D0, particle_sigma_nm=5.0, seed=3)
    tracks = V.track(vid, max_step_px=6.0)
    assert len(tracks) >= 1
    assert max(len(t) for t in tracks) >= 10
    for tr in tracks:
        assert tr.shape[1] == 3            # (frame, y, x)

    D = V.ensemble_diffusion(tracks, vid.pixel_size_nm, vid.frame_interval_s,
                             min_length=10)
    assert D.size >= 1
    # centroid tracking of overlapping blobs underestimates D; require the
    # recovered median within a factor of ~2.5 of the ground truth.
    med = float(np.median(D))
    assert 0.4 * D0 <= med <= 1.6 * D0


def test_msd_scales_with_lag():
    vid = synthetic_video(n_frames=60, size=64, n_particles=2,
                          frame_interval_s=0.05, D_nm2_per_s=150.0, seed=4)
    tracks = V.track(vid)
    tr = max(tracks, key=len)
    lags, m = V.msd(tr, vid.pixel_size_nm)
    good = np.isfinite(m)
    # MSD is (statistically) increasing: last valid lag exceeds the first
    assert m[good][-1] > m[good][0]


def test_diffusion_coefficient_none_for_short_track():
    tr = np.array([[0, 1.0, 1.0]])
    assert V.diffusion_coefficient(tr, 1.0, 0.1) is None


# --------------------------------------------------------------------------
# event detection
# --------------------------------------------------------------------------
def test_detect_events_flags_step():
    t = np.arange(50) * 0.1
    series = np.zeros(50)
    series[25:] += 6.0                      # a single step of +6
    series += np.random.default_rng(0).normal(0, 0.05, 50)
    events = V.detect_events(series, t)
    assert len(events) == 1
    idx, tt, delta = events[0]
    assert idx == 25 and delta > 4.0


def test_detect_events_none_for_flat():
    series = np.random.default_rng(0).normal(0, 0.1, 40)
    events = V.detect_events(series, min_step=5.0)
    assert events == []


# --------------------------------------------------------------------------
# readers: NumPy + ASD round-trip
# --------------------------------------------------------------------------
def test_read_numpy_roundtrip(tmp_path):
    vid = synthetic_video(n_frames=8, size=24, seed=5)
    p = tmp_path / "movie.npz"
    np.savez(p, frames=vid.frames, pixel_size_nm=vid.pixel_size_nm,
             frame_interval_s=vid.frame_interval_s)
    back = V.read_numpy(str(p))
    assert back.shape == vid.shape
    assert back.pixel_size_nm == pytest.approx(vid.pixel_size_nm)
    assert np.allclose(back.frames, vid.frames)


def _write_asd_v1(path, frames, z_ext=100.0, z_gain=1.0, ad_res=12,
                  scan_range=128, dt_ms=100.0):
    """Write a minimal ASD version-1 file matching ``video._ASD_V1_HEADER``."""
    frames = np.asarray(frames)
    n, yp, xp = frames.shape
    scale = z_ext * z_gain / float(2 ** ad_res)
    hdr = dict(
        file_version=1, file_header_size=0, frame_header_size=0,
        text_encoding=0, operator_name_size=0, comment_size=0,
        data_kind_1ch=0, data_kind_2ch=0, init_frame=0, num_frames=n,
        scanning_direction=0, file_id=0, x_pixel=xp, y_pixel=yp,
        x_scanning_range=scan_range, y_scanning_range=scan_range,
        is_averaged=0, average_window=0, year=2026, month=7, day=15,
        hour=0, minute=0, second=0, x_rounding_degree=0, y_rounding_degree=0,
        frame_acquision_time=dt_ms, sensor_sensitivity=1.0,
        phase_sensitivity=1.0, offset=0, machine_id=0, ad_resolution=ad_res,
        x_max_scanning_range=1.0, y_max_scanning_range=1.0,
        x_piezo_extension=1.0, y_piezo_extension=1.0, z_piezo_extension=z_ext,
        z_piezo_gain=z_gain)

    def pack(h):
        return b"".join(struct.pack("<" + fmt, h[name])
                        for name, fmt in V._ASD_V1_HEADER)

    hdr["file_header_size"] = len(pack(hdr))
    body = pack(hdr)
    levels = np.round(frames / scale).astype("<i2")
    with open(path, "wb") as fh:
        fh.write(body)
        fh.write(levels.tobytes())
    return scale


def test_read_asd_roundtrip(tmp_path):
    truth = synthetic_video(n_frames=6, size=32, n_particles=3, seed=6)
    p = tmp_path / "movie.asd"
    scale = _write_asd_v1(str(p), truth.frames)
    back = V.read_asd(str(p))
    assert back.shape == truth.shape
    assert back.pixel_size_nm == pytest.approx(128.0 / 32.0)
    assert back.frame_interval_s == pytest.approx(0.1)   # 100 ms
    # heights match up to the int16 quantisation of one AD level
    assert np.allclose(back.frames, truth.frames, atol=2 * scale)


def test_read_asd_rejects_wrong_version(tmp_path):
    p = tmp_path / "bad.asd"
    p.write_bytes(struct.pack("<i", 2) + b"\x00" * 200)
    with pytest.raises(ValueError):
        V.read_asd(str(p))


def test_load_video_dispatch(tmp_path):
    vid = synthetic_video(n_frames=4, size=16, seed=7)
    p = tmp_path / "m.npy"
    np.save(p, vid.frames)
    back = V.load_video(str(p))
    assert back.shape == vid.shape
    with pytest.raises(ValueError):
        V.load_video("movie.unknownext")


# --------------------------------------------------------------------------
# visualisation
# --------------------------------------------------------------------------
def test_plot_summary_runs():
    import matplotlib
    matplotlib.use("Agg")
    vid = synthetic_video(n_frames=20, size=48, n_particles=4, seed=8)
    fig = V.plot_summary(vid)
    assert len(fig.axes) >= 4


def test_animate_video_gif(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    vid = synthetic_video(n_frames=6, size=24, n_particles=2, seed=9)
    out = V.animate_video(vid, path=str(tmp_path / "m.gif"), fps=5)
    assert out.endswith(".gif")
    import os
    assert os.path.getsize(out) > 0
