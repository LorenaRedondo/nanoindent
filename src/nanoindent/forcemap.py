"""Synthetic AFM force-volume (force-map) generation.

An AFM force map rasters the tip over an ``n_rows x n_cols`` grid at a spatial
pitch ``X`` (nm) and records a force-distance ramp at each pixel, at a ramp
frequency ``Y`` (Hz). This module generates such a map from the FEM, for a
sample whose Young's modulus varies pixel to pixel.

The physics that makes this cheap and exact: for linear elasticity the
indentation force is *strictly proportional* to the modulus at fixed geometry,

    F(delta; E) = E * phi(delta),

where ``phi`` is a purely geometric kernel (it depends on tip radius, film
thickness, Poisson's ratio and the mesh, but not on E). This is verified to
0.0000% across a 400x range of E -- see ``docs/design.md``. So one FEM solve
gives ``phi``, and every pixel of the map -- any modulus field, heterogeneous
or continuous -- is an exact analytic scaling of it. A 64x64 map costs one
solve, not 4096.

The generated map is a :class:`ForceVolume` whose ``curves`` are raw
``(z, deflection)`` :class:`~nanoindent.experiment.ForceCurveData`, i.e. exactly
what an instrument writes and what :func:`nanoindent.slb.analyze_map` reads. So
the map round-trips: generate at a known modulus field, analyse it back, and
compare the recovered modulus to the ground truth.

Frequency ``Y`` sets the *time base* only -- ramp velocity, per-pixel duration,
total scan time, sample timestamps. For a purely elastic sample it does not
change the forces (there is no rate dependence in the constitutive law); that is
the regime this module models. Rate/frequency dependence of the modulus itself
is viscoelasticity -- see :mod:`nanoindent.viscoelastic`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Union

import numpy as np

from .config import SimulationConfig
from .solver import IndentationSolver
from .experiment import ForceCurveData

# A modulus field is either a 2D array (n_rows x n_cols) of moduli in MPa, or a
# callable (row, col) -> E that returns the modulus at a pixel.
ModulusField = Union[np.ndarray, Callable[[int, int], float]]


# --------------------------------------------------------------------------
# geometric kernel: one FEM solve, reused for every pixel and every modulus
# --------------------------------------------------------------------------
@dataclass
class GeometricKernel:
    """The modulus-independent force response ``phi(delta) = F(delta) / E``.

    Built from a single FEM solve. ``force(delta, E)`` gives the elastic force
    at any modulus by exact linear scaling; ``delta_max`` is the depth the FEM
    curve covers (the kernel must not be evaluated beyond it).
    """

    delta: np.ndarray            # nm, monotone increasing from 0
    phi: np.ndarray              # pN / MPa  (force per unit modulus)
    nu: float
    R: float
    h: float
    substrate: str

    @property
    def delta_max(self) -> float:
        return float(self.delta[-1])

    def phi_of(self, delta: np.ndarray) -> np.ndarray:
        """Interpolate the geometric kernel at arbitrary indentation (nm)."""
        d = np.clip(np.asarray(delta, dtype=float), 0.0, self.delta_max)
        return np.interp(d, self.delta, self.phi)

    def force(self, delta: np.ndarray, E: float) -> np.ndarray:
        """Elastic force (pN) at modulus ``E`` (MPa): ``E * phi(delta)``."""
        return float(E) * self.phi_of(delta)


def geometric_kernel(cfg: SimulationConfig, n_steps: int = 12) -> GeometricKernel:
    """Run the FEM once and return the geometric kernel ``phi(delta)``.

    The modulus used for the solve is irrelevant to the kernel (``phi = F / E``
    removes it); ``cfg.E`` is used as the reference and divided back out.
    """
    solve_cfg = replace(cfg, n_steps=n_steps)
    curve = IndentationSolver(solve_cfg).run()
    # Prepend the origin so interpolation is well defined from delta = 0.
    d = np.concatenate([[0.0], curve.delta])
    f = np.concatenate([[0.0], curve.force])
    d, idx = np.unique(d, return_index=True)
    phi = (f[idx]) / float(solve_cfg.E)
    return GeometricKernel(delta=d, phi=phi, nu=cfg.nu, R=cfg.R, h=cfg.h,
                           substrate=str(cfg.substrate))


# --------------------------------------------------------------------------
# acquisition geometry and timing
# --------------------------------------------------------------------------
@dataclass
class ForceVolumeSpec:
    """Grid geometry and acquisition timing of a force map.

    Parameters
    ----------
    n_rows, n_cols:
        Grid dimensions (pixels).
    pitch_nm:
        Spatial spacing ``X`` between pixels (nm).
    ramp_freq_hz:
        Ramp frequency ``Y`` (Hz): one approach-retract force curve per
        ``1 / Y`` seconds at each pixel.
    z_amplitude_nm:
        Peak indentation budget of the z-ramp (nm); the approach drives the tip
        from just above contact down to this depth.
    points_per_ramp:
        Samples recorded on the approach segment.
    dwell_s:
        Optional hold time at maximum force (s), added to the per-pixel time.
    """

    n_rows: int
    n_cols: int
    pitch_nm: float
    ramp_freq_hz: float
    z_amplitude_nm: float = 1.0
    points_per_ramp: int = 128
    dwell_s: float = 0.0

    @property
    def n_pixels(self) -> int:
        return self.n_rows * self.n_cols

    @property
    def ramp_period_s(self) -> float:
        """Duration of one full approach-retract cycle (s)."""
        return 1.0 / self.ramp_freq_hz + self.dwell_s

    @property
    def approach_velocity_nm_s(self) -> float:
        """Tip velocity on the approach segment (nm/s).

        A triangular z-drive covers ``z_amplitude`` up then down over ``1/Y``
        seconds, so the approach half takes ``1/(2Y)`` and the velocity is
        ``2 * z_amplitude * Y``.
        """
        return 2.0 * self.z_amplitude_nm * self.ramp_freq_hz

    @property
    def per_pixel_time_s(self) -> float:
        return self.ramp_period_s

    @property
    def total_scan_time_s(self) -> float:
        return self.n_pixels * self.per_pixel_time_s

    @property
    def field_of_view_nm(self) -> tuple:
        """(width, height) of the scanned area (nm)."""
        return ((self.n_cols - 1) * self.pitch_nm,
                (self.n_rows - 1) * self.pitch_nm)

    def summary(self) -> str:
        w, h = self.field_of_view_nm
        return (
            f"ForceVolumeSpec({self.n_rows}x{self.n_cols} px, pitch={self.pitch_nm} nm, "
            f"FOV={w:.0f}x{h:.0f} nm, ramp={self.ramp_freq_hz} Hz, "
            f"v_approach={self.approach_velocity_nm_s:.0f} nm/s, "
            f"scan_time={self.total_scan_time_s:.1f} s)"
        )


# --------------------------------------------------------------------------
# modulus field helpers
# --------------------------------------------------------------------------
def uniform_field(n_rows: int, n_cols: int, E: float) -> np.ndarray:
    """A homogeneous modulus map."""
    return np.full((n_rows, n_cols), float(E))


def two_phase_field(n_rows: int, n_cols: int, E_soft: float, E_stiff: float,
                    stiff_fraction: float = 0.4, seed: int = 0,
                    blob_scale: float = 0.25) -> np.ndarray:
    """A two-phase sample: stiff domains embedded in a soft matrix.

    Mimics coexisting lipid phases (e.g. liquid-ordered domains in a
    liquid-disordered matrix). Smooth low-pass noise is thresholded so the
    stiff phase forms connected blobs rather than salt-and-pepper pixels.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n_rows, n_cols))
    # Cheap isotropic smoothing by repeated 4-neighbour averaging.
    passes = max(1, int(blob_scale * max(n_rows, n_cols)))
    f = noise.copy()
    for _ in range(passes):
        f = (f
             + np.roll(f, 1, 0) + np.roll(f, -1, 0)
             + np.roll(f, 1, 1) + np.roll(f, -1, 1)) / 5.0
    thresh = np.quantile(f, 1.0 - stiff_fraction)
    E = np.where(f >= thresh, E_stiff, E_soft)
    return E.astype(float)


def gradient_field(n_rows: int, n_cols: int, E_left: float, E_right: float,
                   axis: int = 1) -> np.ndarray:
    """A linear modulus gradient across the sample (stiffness ramp)."""
    n = n_cols if axis == 1 else n_rows
    ramp = np.linspace(E_left, E_right, n)
    E = np.tile(ramp, (n_rows, 1)) if axis == 1 else np.tile(ramp[:, None], (1, n_cols))
    return E.astype(float)


def _resolve_field(e_field: ModulusField, n_rows: int, n_cols: int) -> np.ndarray:
    if callable(e_field):
        return np.array([[float(e_field(i, j)) for j in range(n_cols)]
                         for i in range(n_rows)], dtype=float)
    E = np.asarray(e_field, dtype=float)
    if E.shape != (n_rows, n_cols):
        raise ValueError(f"modulus field shape {E.shape} != grid ({n_rows}, {n_cols})")
    return E


# --------------------------------------------------------------------------
# the force volume
# --------------------------------------------------------------------------
@dataclass
class ForceVolume:
    """A synthetic force map: raw curves plus the ground truth they encode.

    ``curves`` are raw ``ForceCurveData`` (row-major), so this object is a drop-in
    for :func:`nanoindent.slb.analyze_map` (which reads ``.curves``). ``E_true``
    is the ground-truth modulus map that generated them.
    """

    curves: list
    n_rows: int
    n_cols: int
    spring_constant: float
    E_true: np.ndarray                       # (n_rows, n_cols), MPa
    spec: ForceVolumeSpec
    kernel: GeometricKernel
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    inv_ols: Optional[float] = None
    source: str = "synthetic-fem"

    def __len__(self):
        return len(self.curves)

    def __getitem__(self, i):
        return self.curves[i]

    def recover_modulus_map(self, model: str = "dimitriadis",
                            fit_max_delta: Optional[float] = None,
                            fit_contact_point: bool = True) -> np.ndarray:
        """Recover the per-pixel modulus (MPa) by the ordinary analysis path.

        Each raw curve is calibrated to ``(delta, force)`` with
        :meth:`ForceCurveData.to_force_indentation` and fit with
        :func:`nanoindent.fit_modulus` -- the same inverse analysis a user runs
        on real elastic force curves. Reshapes the result onto the grid; a
        failed fit comes back as ``nan``. This closes the round-trip: generate
        at a known modulus field, recover it, compare.

        ``model="kernel"`` fits against the FEM geometric kernel itself. Because
        the curves *are* ``E * phi(delta)``, this recovers E exactly (to
        calibration and interpolation error) and so validates the machinery --
        timing, calibration inversion, scaling -- independently of any analytic
        model's approximation error. The analytic models (``"hertz"``,
        ``"dimitriadis"``, ``"garcia_bec"``, ``"chadwick"``) instead surface that
        approximation error: Hertz over-estimates on a thin bonded film (the
        bottom effect), the finite-thickness corrections less so.

        (This is the elastic path. The breakthrough-based SLB pipeline
        :func:`nanoindent.slb.analyze_map` is for bilayer snap-through curves,
        not the monotonic elastic curves generated here.)
        """
        from .dataset import fit_modulus

        cap = fit_max_delta if fit_max_delta is not None else self.kernel.delta_max
        shape = self.kernel.phi_of if model == "kernel" else None
        out = np.full(self.n_rows * self.n_cols, np.nan)
        for pixel, curve in enumerate(self.curves):
            try:
                d, f = curve.to_force_indentation(clip_negative=False)
                m = d <= cap
                fit = fit_modulus(d[m], f[m], model=model, R=self.kernel.R,
                                  nu=self.kernel.nu, h=self.kernel.h,
                                  fit_contact_point=fit_contact_point, shape=shape)
                if fit is not None and np.isfinite(fit.E_fit) and fit.E_fit > 0:
                    out[pixel] = fit.E_fit
            except Exception:
                pass
        return out.reshape(self.n_rows, self.n_cols)


def _raw_curve_from_force(delta_contact: np.ndarray, force_pN: np.ndarray,
                          spring_constant: float, z_contact: float,
                          n_precontact: int, baseline_nm: float,
                          noise_nm: float, rng, pixel: int, E_true: float,
                          timestamp: float) -> ForceCurveData:
    """Invert the calibration: (delta, force) -> raw (z, deflection) curve.

    Deflection ``d = F / k``; piezo ``z = z_contact + delta + d`` on contact,
    flat below it. Feeding this back through ``to_force_indentation`` recovers
    the input, which is what makes the map a genuine round-trip.
    """
    defl_contact = force_pN / (spring_constant * 1e3)      # pN -> nN, /k -> nm
    z_c = z_contact + delta_contact + defl_contact

    z_pre = np.linspace(z_contact - 0.4 * float(delta_contact[-1]), z_contact,
                        n_precontact, endpoint=False)
    defl_pre = np.zeros_like(z_pre)

    z = np.concatenate([z_pre, z_c])
    defl = np.concatenate([defl_pre, defl_contact]) + baseline_nm
    if noise_nm:
        defl = defl + rng.normal(0.0, noise_nm, size=defl.shape)

    return ForceCurveData(
        z=z, deflection=defl, deflection_unit="nm",
        spring_constant=spring_constant, segment="approach",
        metadata={"pixel": pixel, "true_E": float(E_true),
                  "timestamp_s": float(timestamp)},
        fmt="synthetic-fem",
    )


def simulate_force_volume(
    spec: ForceVolumeSpec,
    e_field: ModulusField,
    cfg: SimulationConfig,
    spring_constant: float = 0.1,
    kernel: Optional[GeometricKernel] = None,
    noise_nm: float = 0.0,
    baseline_nm: float = 2.0,
    z_contact: float = 50.0,
    n_precontact: int = 24,
    seed: int = 0,
) -> ForceVolume:
    """Generate a synthetic force map over a modulus field.

    One FEM solve builds the geometric kernel (unless ``kernel`` is supplied);
    every pixel is then an exact analytic scaling of it by that pixel's modulus,
    converted to a raw AFM curve on the ramp waveform.

    Parameters
    ----------
    spec:
        Grid geometry and timing.
    e_field:
        Ground-truth modulus map (MPa): a 2D array or a callable ``(i, j) -> E``.
    cfg:
        Geometry/material context for the FEM kernel (R, nu, h, substrate,
        mesh). ``cfg.delta_max`` must cover ``spec.z_amplitude_nm``.
    spring_constant:
        Cantilever ``k`` (N/m) used to synthesise the raw deflection.
    kernel:
        Pre-computed kernel to reuse (skips the FEM solve). Must match ``cfg``'s
        geometry.
    noise_nm:
        Deflection noise std (nm), applied per curve.

    Returns
    -------
    ForceVolume
    """
    if kernel is None:
        kernel = geometric_kernel(cfg)
    if spec.z_amplitude_nm > kernel.delta_max + 1e-9:
        raise ValueError(
            f"ramp z_amplitude ({spec.z_amplitude_nm} nm) exceeds the FEM kernel "
            f"depth ({kernel.delta_max} nm); raise cfg.delta_max")

    E_true = _resolve_field(e_field, spec.n_rows, spec.n_cols)
    rng = np.random.default_rng(seed)

    delta_contact = np.linspace(0.0, spec.z_amplitude_nm, spec.points_per_ramp)
    phi = kernel.phi_of(delta_contact)          # geometric kernel on the ramp

    curves = []
    timestamps = np.zeros(spec.n_pixels)
    for pixel in range(spec.n_pixels):
        i, j = divmod(pixel, spec.n_cols)
        E = float(E_true[i, j])
        force_pN = E * phi                       # exact linear scaling
        t = pixel * spec.per_pixel_time_s
        timestamps[pixel] = t
        curves.append(_raw_curve_from_force(
            delta_contact, force_pN, spring_constant, z_contact,
            n_precontact, baseline_nm, noise_nm, rng, pixel, E, t))

    return ForceVolume(
        curves=curves, n_rows=spec.n_rows, n_cols=spec.n_cols,
        spring_constant=spring_constant, E_true=E_true, spec=spec,
        kernel=kernel, timestamps=timestamps,
    )
