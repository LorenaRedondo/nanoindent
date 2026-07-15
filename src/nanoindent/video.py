"""High-speed AFM (HS-AFM) movie analysis — imaging dynamics.

This is a distinct domain from the force-indentation FEM core: HS-AFM records a
time series of topography images (frames) at video rate, capturing molecular
dynamics. The module provides

* an :class:`AFMVideo` container (frames T x H x W in nm, pixel size, frame rate);
* readers for ``.asd`` (Ando/RIBM), multi-page TIFF, JPK QI/movie, and NumPy;
* drift correction (phase-correlation registration);
* per-frame statistics (mean height, roughness, coverage);
* particle detection, frame-to-frame tracking, MSD and diffusion coefficients;
* per-region height-vs-time series and simple event (step) detection;
* animation of the movie.

The universal (NumPy/TIFF) path and every analysis are validated by a synthetic
diffusing-particle movie (:func:`synthetic_video`). Instrument readers target
the documented layouts — verify against your own files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple
import struct
import warnings

import numpy as np


# ==========================================================================
# container
# ==========================================================================
@dataclass
class AFMVideo:
    """A HS-AFM movie: height frames plus spatial/temporal calibration.

    Attributes
    ----------
    frames:
        ``(T, H, W)`` array of heights (nm).
    pixel_size_nm:
        Pixel size (nm/px); scalar or ``(dy, dx)``.
    frame_interval_s:
        Time between frames (s).
    """

    frames: np.ndarray
    pixel_size_nm: float = 1.0
    frame_interval_s: float = 0.1
    metadata: dict = field(default_factory=dict)
    source: Optional[str] = None

    def __post_init__(self):
        self.frames = np.asarray(self.frames, dtype=float)
        if self.frames.ndim != 3:
            raise ValueError("frames must be a (T, H, W) array")

    @property
    def n_frames(self) -> int:
        return self.frames.shape[0]

    @property
    def shape(self) -> tuple:
        return self.frames.shape

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_interval_s if self.frame_interval_s else 0.0

    def times(self) -> np.ndarray:
        return np.arange(self.n_frames) * self.frame_interval_s

    def duration(self) -> float:
        return self.n_frames * self.frame_interval_s


# ==========================================================================
# synthetic generator (for tests / demos)
# ==========================================================================
def synthetic_video(n_frames: int = 60, size: int = 64, n_particles: int = 6,
                    pixel_size_nm: float = 2.0, frame_interval_s: float = 0.1,
                    D_nm2_per_s: float = 200.0, particle_height: float = 3.0,
                    particle_sigma_nm: float = 6.0, noise_nm: float = 0.15,
                    drift_nm_per_frame: float = 0.0, seed: int = 0) -> AFMVideo:
    """Generate a HS-AFM movie of Gaussian particles diffusing on a surface.

    Particles undergo Brownian motion with diffusion coefficient ``D`` so the
    tracking/MSD analysis can be validated against a known ground truth. An
    optional constant drift exercises the drift-correction step.
    """
    rng = np.random.default_rng(seed)
    L = size * pixel_size_nm
    pos = rng.uniform(0.2 * L, 0.8 * L, size=(n_particles, 2))
    step = np.sqrt(2 * D_nm2_per_s * frame_interval_s)   # per-axis rms step (nm)
    yy, xx = np.mgrid[0:size, 0:size] * pixel_size_nm

    frames = np.zeros((n_frames, size, size))
    for t in range(n_frames):
        img = np.zeros((size, size))
        for p in range(n_particles):
            cy, cx = pos[p]
            img += particle_height * np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * particle_sigma_nm ** 2))
        img += rng.normal(0, noise_nm, img.shape)
        img += t * drift_nm_per_frame            # global height ramp (drift)
        frames[t] = img
        pos += rng.normal(0, step, pos.shape)
        pos = np.clip(pos, 0, L)
    return AFMVideo(frames, pixel_size_nm=pixel_size_nm,
                    frame_interval_s=frame_interval_s,
                    metadata={"true_D_nm2_per_s": D_nm2_per_s,
                              "n_particles": n_particles}, source="synthetic")


# ==========================================================================
# readers
# ==========================================================================
def read_numpy(path, pixel_size_nm: float = 1.0,
               frame_interval_s: float = 0.1) -> AFMVideo:
    """Read frames from a ``.npy`` (T,H,W array) or ``.npz`` (keys frames /
    pixel_size_nm / frame_interval_s)."""
    if str(path).endswith(".npz"):
        d = np.load(path)
        return AFMVideo(d["frames"],
                        pixel_size_nm=float(d["pixel_size_nm"]) if "pixel_size_nm" in d else pixel_size_nm,
                        frame_interval_s=float(d["frame_interval_s"]) if "frame_interval_s" in d else frame_interval_s,
                        source=str(path))
    arr = np.load(path)
    return AFMVideo(arr, pixel_size_nm=pixel_size_nm,
                   frame_interval_s=frame_interval_s, source=str(path))


def read_tiff(path, pixel_size_nm: float = 1.0,
              frame_interval_s: float = 0.1) -> AFMVideo:
    """Read a multi-page TIFF stack of height images (via Pillow)."""
    from PIL import Image, ImageSequence
    frames = []
    with Image.open(path) as im:
        for page in ImageSequence.Iterator(im):
            frames.append(np.asarray(page, dtype=float))
    return AFMVideo(np.stack(frames), pixel_size_nm=pixel_size_nm,
                   frame_interval_s=frame_interval_s, source=str(path))


# ASD version-1 header: (name, struct-format) in file byte order (little-endian).
_ASD_V1_HEADER = [
    ("file_version", "i"), ("file_header_size", "i"), ("frame_header_size", "i"),
    ("text_encoding", "i"), ("operator_name_size", "i"), ("comment_size", "i"),
    ("data_kind_1ch", "i"), ("data_kind_2ch", "i"), ("init_frame", "i"),
    ("num_frames", "i"), ("scanning_direction", "i"), ("file_id", "i"),
    ("x_pixel", "i"), ("y_pixel", "i"), ("x_scanning_range", "i"),
    ("y_scanning_range", "i"), ("is_averaged", "i"), ("average_window", "i"),
    ("year", "i"), ("month", "i"), ("day", "i"), ("hour", "i"),
    ("minute", "i"), ("second", "i"), ("x_rounding_degree", "i"),
    ("y_rounding_degree", "i"), ("frame_acquision_time", "f"),
    ("sensor_sensitivity", "f"), ("phase_sensitivity", "f"), ("offset", "i"),
    ("machine_id", "i"), ("ad_resolution", "i"), ("x_max_scanning_range", "f"),
    ("y_max_scanning_range", "f"), ("x_piezo_extension", "f"),
    ("y_piezo_extension", "f"), ("z_piezo_extension", "f"), ("z_piezo_gain", "f"),
]


def read_asd(path, channel: int = 0) -> AFMVideo:
    """Read an Ando/RIBM HS-AFM ``.asd`` movie (file version 1).

    Parses the version-1 header (see the ``libasd`` reference layout), then each
    frame's ``x_pixel*y_pixel`` int16 samples, converting AD levels to height in
    nm via ``z_piezo_extension * z_piezo_gain``. Height calibration conventions
    vary between setups — verify the scale against your instrument. Only file
    version 1 is supported here.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    version = struct.unpack_from("<i", raw, 0)[0]
    if version != 1:
        raise ValueError(f"read_asd supports ASD version 1, got {version}")

    hdr, off = {}, 0
    for name, fmt in _ASD_V1_HEADER:
        hdr[name] = struct.unpack_from("<" + fmt, raw, off)[0]
        off += struct.calcsize(fmt)

    xp, yp = hdr["x_pixel"], hdr["y_pixel"]
    n = hdr["num_frames"]
    fh_size = hdr["file_header_size"]
    frame_hdr = hdr["frame_header_size"]
    ad_res = hdr["ad_resolution"] or 12
    scale = hdr["z_piezo_extension"] * hdr["z_piezo_gain"] / float(2 ** ad_res)

    frames = []
    pos = fh_size
    npix = xp * yp
    for _ in range(n):
        pos += frame_hdr                       # skip the per-frame header
        block = np.frombuffer(raw, dtype="<i2", count=npix, offset=pos).astype(float)
        pos += npix * 2
        if hdr["data_kind_2ch"] not in (0, -1) and channel == 1:
            block = np.frombuffer(raw, dtype="<i2", count=npix, offset=pos).astype(float)
            pos += npix * 2
        elif hdr["data_kind_2ch"] not in (0, -1):
            pos += npix * 2                     # skip the second channel
        frames.append(block.reshape(yp, xp) * scale)

    px = (hdr["x_scanning_range"] / xp) if xp else 1.0
    dt = hdr["frame_acquision_time"] * 1e-3 if hdr["frame_acquision_time"] else 0.1
    return AFMVideo(np.stack(frames), pixel_size_nm=px, frame_interval_s=dt,
                   metadata={"asd_header": hdr}, source=str(path))


def read_jpk_qi(path, channel="height", frame_interval_s: float = 0.1) -> AFMVideo:
    """Read a JPK QI / force-map image as a single-frame movie (topography).

    Reuses :func:`nanoindent.read_jpk_map`: each pixel's contact height gives one
    image pixel. A true QI *movie* is a stack of such maps; this returns the
    topography of one map (extend to a stack by concatenating).
    """
    from .experiment import read_jpk_map
    fmap = read_jpk_map(path)
    heights = np.array([c.z.max() - c.z.min() for c in fmap.curves])
    img = heights.reshape(fmap.n_rows, fmap.n_cols)
    return AFMVideo(img[None, :, :], pixel_size_nm=1.0,
                   frame_interval_s=frame_interval_s,
                   metadata={"note": "single QI topography frame"}, source=str(path))


def load_video(path, **kwargs) -> AFMVideo:
    """Dispatch to the right reader by extension."""
    p = str(path).lower()
    if p.endswith(".asd"):
        return read_asd(path, **kwargs)
    if p.endswith((".tif", ".tiff")):
        return read_tiff(path, **kwargs)
    if p.endswith((".npy", ".npz")):
        return read_numpy(path, **kwargs)
    if p.endswith((".jpk-qi-data", ".jpk-qi-image", ".jpk-force-map")):
        return read_jpk_qi(path, **kwargs)
    raise ValueError(f"unrecognised HS-AFM movie format: {path!r}")


# ==========================================================================
# drift correction
# ==========================================================================
def _phase_shift(ref: np.ndarray, img: np.ndarray) -> Tuple[float, float]:
    """Integer-pixel shift (dy, dx) aligning ``img`` to ``ref`` by phase corr."""
    f0 = np.fft.fft2(ref - ref.mean())
    f1 = np.fft.fft2(img - img.mean())
    r = f0 * np.conj(f1)
    r /= np.abs(r) + 1e-12
    cc = np.fft.ifft2(r).real
    peak = np.unravel_index(np.argmax(cc), cc.shape)
    dy, dx = peak
    if dy > ref.shape[0] // 2:
        dy -= ref.shape[0]
    if dx > ref.shape[1] // 2:
        dx -= ref.shape[1]
    return float(dy), float(dx)


def drift_correct(video: AFMVideo, reference: str = "first") -> AFMVideo:
    """Register every frame to a reference by phase correlation (integer px).

    ``reference="first"`` aligns to frame 0; ``"previous"`` aligns cumulatively.
    Returns a new :class:`AFMVideo` with rolled frames.
    """
    frames = video.frames
    out = np.empty_like(frames)
    out[0] = frames[0]
    shifts = [(0.0, 0.0)]
    cum = np.array([0.0, 0.0])
    for t in range(1, len(frames)):
        ref = frames[0] if reference == "first" else out[t - 1]
        dy, dx = _phase_shift(ref, frames[t])
        if reference == "previous":
            cum += (dy, dx)
            sy, sx = cum
        else:
            sy, sx = dy, dx
        out[t] = np.roll(np.roll(frames[t], int(round(sy)), axis=0),
                         int(round(sx)), axis=1)
        shifts.append((sy, sx))
    v = AFMVideo(out, video.pixel_size_nm, video.frame_interval_s,
                 dict(video.metadata, drift_shifts_px=shifts), video.source)
    return v


# ==========================================================================
# per-frame statistics
# ==========================================================================
def per_frame_stats(video: AFMVideo, coverage_threshold: Optional[float] = None) -> dict:
    """Per-frame mean height, RMS roughness, and coverage fraction.

    ``coverage_threshold`` (nm) defaults to mean + 1 std of the whole movie.
    """
    f = video.frames
    if coverage_threshold is None:
        coverage_threshold = float(f.mean() + f.std())
    return {
        "time_s": video.times(),
        "mean_height_nm": f.mean(axis=(1, 2)),
        "rms_roughness_nm": f.std(axis=(1, 2)),
        "coverage": (f > coverage_threshold).mean(axis=(1, 2)),
        "max_height_nm": f.max(axis=(1, 2)),
    }


# ==========================================================================
# particle detection + tracking + diffusion
# ==========================================================================
def detect_particles(frame: np.ndarray, threshold: Optional[float] = None,
                     min_size: int = 2) -> np.ndarray:
    """Return an ``(n, 2)`` array of particle centroids ``(y, x)`` in pixels."""
    from scipy import ndimage
    if threshold is None:
        threshold = float(frame.mean() + frame.std())
    mask = frame > threshold
    labels, n = ndimage.label(mask)
    if n == 0:
        return np.empty((0, 2))
    sizes = ndimage.sum(np.ones_like(frame), labels, range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_size]
    if not keep:
        return np.empty((0, 2))
    cents = ndimage.center_of_mass(frame, labels, keep)
    return np.array(cents, dtype=float)


def track(video: AFMVideo, threshold: Optional[float] = None,
          max_step_px: float = 8.0, min_size: int = 2) -> list:
    """Detect and link particles frame-to-frame into trajectories.

    Nearest-neighbour linking with a maximum per-frame displacement. Returns a
    list of trajectories, each an ``(m, 3)`` array of ``(frame, y, x)`` in px.
    """
    detections = [detect_particles(video.frames[t], threshold, min_size)
                  for t in range(video.n_frames)]
    tracks, active = [], []   # active: list of (traj_list, last_yx)
    for t, cents in enumerate(detections):
        used = set()
        for traj, last in active:
            if len(cents) == 0:
                continue
            d = np.hypot(cents[:, 0] - last[0], cents[:, 1] - last[1])
            j = int(np.argmin(d))
            if d[j] <= max_step_px and j not in used:
                traj.append((t, cents[j, 0], cents[j, 1]))
                used.add(j)
        # update last positions; drop tracks that missed this frame
        new_active = []
        for traj, last in active:
            if traj[-1][0] == t:
                new_active.append((traj, (traj[-1][1], traj[-1][2])))
        for j, c in enumerate(cents):
            if j not in used:
                traj = [(t, c[0], c[1])]
                tracks.append(traj)
                new_active.append((traj, (c[0], c[1])))
        active = new_active
    return [np.array(tr, dtype=float) for tr in tracks]


def msd(trajectory: np.ndarray, pixel_size_nm: float,
        max_lag: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Mean-squared displacement (nm^2) vs lag (frames) for one trajectory."""
    yx = trajectory[:, 1:3] * pixel_size_nm
    n = len(yx)
    max_lag = max_lag or n // 2
    lags = np.arange(1, max(max_lag, 1) + 1)
    out = []
    for lag in lags:
        if lag >= n:
            out.append(np.nan)
            continue
        d = yx[lag:] - yx[:-lag]
        out.append(float(np.mean(np.sum(d ** 2, axis=1))))
    return lags, np.array(out)


def diffusion_coefficient(trajectory: np.ndarray, pixel_size_nm: float,
                          frame_interval_s: float, n_points: int = 4) -> Optional[float]:
    """2-D diffusion coefficient (nm^2/s) from the initial MSD slope
    (``MSD = 4 D t``)."""
    lags, m = msd(trajectory, pixel_size_nm, max_lag=n_points)
    good = np.isfinite(m)
    if good.sum() < 2:
        return None
    t = lags[good] * frame_interval_s
    slope = float(np.polyfit(t, m[good], 1)[0])
    return slope / 4.0


def ensemble_diffusion(tracks: Sequence[np.ndarray], pixel_size_nm: float,
                       frame_interval_s: float, min_length: int = 5) -> np.ndarray:
    """Diffusion coefficient (nm^2/s) for every track long enough to fit."""
    out = []
    for tr in tracks:
        if len(tr) >= min_length:
            D = diffusion_coefficient(tr, pixel_size_nm, frame_interval_s)
            if D is not None and D >= 0:
                out.append(D)
    return np.array(out)


# ==========================================================================
# height dynamics + event detection
# ==========================================================================
def height_timeseries(video: AFMVideo, region=None) -> Tuple[np.ndarray, np.ndarray]:
    """Mean height (nm) vs time over a region ``(y0, y1, x0, x1)`` (whole frame
    if None)."""
    if region is None:
        series = video.frames.mean(axis=(1, 2))
    else:
        y0, y1, x0, x1 = region
        series = video.frames[:, y0:y1, x0:x1].mean(axis=(1, 2))
    return video.times(), series


def detect_events(series: np.ndarray, times: Optional[np.ndarray] = None,
                  min_step: Optional[float] = None) -> list:
    """Detect step-like events in a 1-D time series (assembly/binding, etc.).

    Flags frames where the point-to-point change exceeds ``min_step`` (default
    4x the median absolute step), returning ``(index, time, delta)`` per event.
    """
    series = np.asarray(series, float)
    d = np.diff(series)
    if min_step is None:
        mad = np.median(np.abs(d - np.median(d))) or (np.std(d) or 1e-9)
        min_step = 4.0 * 1.4826 * mad
    events = []
    for i, delta in enumerate(d):
        if abs(delta) >= min_step:
            t = float(times[i + 1]) if times is not None else i + 1
            events.append((i + 1, t, float(delta)))
    return events


# ==========================================================================
# visualisation
# ==========================================================================
def animate_video(video: AFMVideo, path: str = "hsafm.mp4", fps: Optional[int] = None,
                  cmap: str = "afmhot", tracks: Optional[Sequence[np.ndarray]] = None,
                  dpi: int = 130):
    """Animate the movie to MP4 (ffmpeg) or GIF, optionally overlaying tracks."""
    import matplotlib.pyplot as plt
    from matplotlib import animation

    f = video.frames
    vmin, vmax = float(np.percentile(f, 1)), float(np.percentile(f, 99))
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    im = ax.imshow(f[0], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="height (nm)")
    scat = ax.scatter([], [], s=25, facecolors="none", edgecolors="lime", lw=1.0)

    def upd(t):
        im.set_data(f[t])
        ax.set_title(f"t = {t * video.frame_interval_s:.2f} s", fontsize=9)
        if tracks:
            pts = [(tr[tr[:, 0] == t][0, 2], tr[tr[:, 0] == t][0, 1])
                   for tr in tracks if np.any(tr[:, 0] == t)]
            scat.set_offsets(np.array(pts) if pts else np.empty((0, 2)))
        return [im, scat]

    anim = animation.FuncAnimation(fig, upd, frames=video.n_frames, blit=False)
    fps = fps or max(1, int(round(video.fps)) or 10)
    want_gif = str(path).lower().endswith(".gif")
    writer = None
    if not want_gif:
        try:
            import imageio_ffmpeg
            plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
        if animation.FFMpegWriter.isAvailable():
            writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
        else:
            path = str(path).rsplit(".", 1)[0] + ".gif"; want_gif = True
    if want_gif:
        writer = animation.PillowWriter(fps=fps)
    anim.save(path, writer=writer, dpi=dpi)
    plt.close(fig)
    return path


def plot_summary(video: AFMVideo, tracks: Optional[Sequence[np.ndarray]] = None):
    """Four-panel movie summary: first frame, per-frame stats, tracks, MSD."""
    import matplotlib.pyplot as plt

    stats = per_frame_stats(video)
    fig, axs = plt.subplots(2, 2, figsize=(9, 7))
    axs[0, 0].imshow(video.frames[0], cmap="afmhot", origin="lower")
    axs[0, 0].set_title("frame 0", fontsize=9)
    axs[0, 1].plot(stats["time_s"], stats["mean_height_nm"], label="mean height")
    axs[0, 1].plot(stats["time_s"], stats["rms_roughness_nm"], label="roughness")
    axs[0, 1].set_xlabel("time (s)"); axs[0, 1].set_ylabel("nm")
    axs[0, 1].legend(fontsize=7); axs[0, 1].set_title("per-frame stats", fontsize=9)

    if tracks is None:
        tracks = track(video)
    axs[1, 0].imshow(video.frames.mean(0), cmap="gray", origin="lower")
    for tr in tracks:
        if len(tr) >= 3:
            axs[1, 0].plot(tr[:, 2], tr[:, 1], lw=0.8)
    axs[1, 0].set_title(f"tracks (n={len(tracks)})", fontsize=9)

    D = ensemble_diffusion(tracks, video.pixel_size_nm, video.frame_interval_s)
    if D.size:
        axs[1, 1].hist(D, bins=15, color="#0072B2", edgecolor="k", lw=0.3)
        axs[1, 1].axvline(np.median(D), color="crimson", ls="--")
        axs[1, 1].set_title(f"diffusion (median {np.median(D):.0f} nm²/s)", fontsize=9)
    axs[1, 1].set_xlabel("D (nm²/s)"); axs[1, 1].set_ylabel("count")
    for a in axs.flat:
        a.grid(alpha=0.2)
    fig.tight_layout()
    return fig
