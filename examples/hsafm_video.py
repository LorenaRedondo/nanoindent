"""High-speed AFM (HS-AFM) movie analysis — imaging dynamics.

Generates a synthetic diffusing-particle movie with a known ground-truth
diffusion coefficient, then runs the full HS-AFM pipeline: drift correction,
per-frame statistics, particle tracking, MSD / diffusion-coefficient recovery,
event detection, and a summary figure + animation.

The same functions read real movies — swap ``synthetic_video(...)`` for
``load_video("scan.asd")`` (Ando/RIBM ``.asd``), ``read_tiff("stack.tif")``,
or ``read_numpy("movie.npz")``.

    python examples/hsafm_video.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")

from nanoindent import video as V


def main():
    # A movie of 6 particles diffusing at D = 200 nm^2/s, 10 fps, 2 nm/px.
    vid = V.synthetic_video(n_frames=120, size=96, n_particles=6,
                            pixel_size_nm=2.0, frame_interval_s=0.05,
                            D_nm2_per_s=200.0, particle_sigma_nm=5.0, seed=1)
    print(f"movie: {vid.n_frames} frames, {vid.shape[1]}x{vid.shape[2]} px, "
          f"{vid.pixel_size_nm} nm/px, {vid.fps:.0f} fps, "
          f"{vid.duration():.1f} s")

    # ---- per-frame statistics -------------------------------------------
    stats = V.per_frame_stats(vid)
    print(f"mean height {stats['mean_height_nm'].mean():.2f} nm, "
          f"roughness {stats['rms_roughness_nm'].mean():.2f} nm, "
          f"coverage {stats['coverage'].mean()*100:.1f} %")

    # ---- particle tracking + diffusion ----------------------------------
    tracks = V.track(vid, max_step_px=6.0)
    D = V.ensemble_diffusion(tracks, vid.pixel_size_nm, vid.frame_interval_s,
                             min_length=10)
    print(f"{len(tracks)} tracks, {D.size} long enough to fit")
    if D.size:
        print(f"recovered D = {np.median(D):.0f} nm^2/s "
              f"(ground truth {vid.metadata['true_D_nm2_per_s']:.0f}); "
              f"centroid tracking of overlapping blobs underestimates D")

    # ---- event detection on the mean-height trace -----------------------
    t, series = V.height_timeseries(vid)
    events = V.detect_events(series, t)
    print(f"{len(events)} step-like events detected")

    # ---- figure + animation ---------------------------------------------
    fig = V.plot_summary(vid, tracks=tracks)
    fig.savefig("hsafm_summary.png", dpi=150, bbox_inches="tight")
    out = V.animate_video(vid, path="hsafm_movie.gif", tracks=tracks, fps=15)
    print(f"wrote hsafm_summary.png and {out}")


if __name__ == "__main__":
    main()
