"""Ingest raw AFM force curves and turn them into force-indentation data.

An AFM does not record force vs. indentation directly; it records cantilever
**deflection** vs. **piezo Z-position**.  This module

1. **reads** raw curves from the common instrument formats
   (:func:`read_csv`, :func:`read_jpk`, :func:`read_bruker`, :func:`read_ibw`),
2. **calibrates** them to force-indentation ``(delta, force)`` using the
   spring constant ``k`` and deflection sensitivity (InvOLS), subtracting the
   baseline and locating the contact point
   (:meth:`ForceCurveData.to_force_indentation`), and
3. **analyses** them with the library's contact models
   (:func:`analyze_experiment`, which calls :func:`nanoindent.fit_modulus`).

Calibration recap
-----------------
* force:        ``F = k * deflection`` (deflection in nm, ``k`` in N/m -> nN;
  reported in pN to match the library units).
* deflection:   photodiode signal ``V`` times InvOLS (nm/V) if not already nm.
* indentation:  ``delta = (Z - Z0) - (d - d0)`` — piezo travel past the contact
  point ``Z0`` minus the cantilever deflection (baseline ``d0``).

The pure-CSV path and the calibration/contact pipeline are backend-free and
fully validated by a synthetic round-trip (generate a curve at known E -> raw
signals -> invert -> recover E).  Instrument-format readers target the
documented layouts; verify against your own exports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple
import os

import numpy as np


# --------------------------------------------------------------------------
# container
# --------------------------------------------------------------------------
@dataclass
class ForceCurveData:
    """A raw AFM force curve: piezo Z and cantilever deflection, plus metadata.

    Attributes
    ----------
    z:
        Piezo / height-sensor position (nm).
    deflection:
        Cantilever deflection, in nm (``deflection_unit="nm"``) or photodiode
        volts (``"V"``, needs ``inv_ols`` to convert).
    spring_constant:
        Cantilever spring constant ``k`` (N/m), if known from the file.
    inv_ols:
        Deflection sensitivity / inverse optical lever sensitivity (nm/V).
    segment:
        ``"approach"`` or ``"retract"``.
    """

    z: np.ndarray
    deflection: np.ndarray
    deflection_unit: str = "nm"
    spring_constant: Optional[float] = None
    inv_ols: Optional[float] = None
    segment: str = "approach"
    metadata: dict = field(default_factory=dict)
    source: Optional[str] = None
    fmt: Optional[str] = None

    def __post_init__(self):
        self.z = np.asarray(self.z, dtype=float)
        self.deflection = np.asarray(self.deflection, dtype=float)
        if self.z.shape != self.deflection.shape:
            raise ValueError("z and deflection must have the same length")

    # -- calibration -> (delta, force) ------------------------------------
    def deflection_nm(self, inv_ols: Optional[float] = None) -> np.ndarray:
        """Deflection converted to nm (applies InvOLS if the data are in V)."""
        d = self.deflection
        if self.deflection_unit == "nm":
            return d
        s = inv_ols if inv_ols is not None else self.inv_ols
        if s is None:
            raise ValueError("deflection is in volts; provide inv_ols (nm/V)")
        return d * s

    def to_force_indentation(
        self,
        spring_constant: Optional[float] = None,
        inv_ols: Optional[float] = None,
        contact_point: Optional[float] = None,
        baseline: Optional[float] = None,
        baseline_fraction: float = 0.1,
        contact_sigma: float = 5.0,
        clip_negative: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert the raw curve to ``(delta_nm, force_pN)``.

        Parameters
        ----------
        spring_constant:
            ``k`` in N/m (overrides the file value; required if the file has none).
        inv_ols:
            Deflection sensitivity (nm/V), needed only if deflection is in volts.
        contact_point, baseline:
            Fix the contact point (nm, in oriented-Z) and/or deflection baseline
            (nm); if ``None`` they are estimated (see
            :meth:`detect_contact_point`).
        baseline_fraction:
            Fraction of the pre-contact tail used to estimate the baseline.
        contact_sigma:
            Contact is where the deflection first rises ``contact_sigma`` noise
            standard deviations above baseline.
        clip_negative:
            Return only the in-contact branch (``delta > 0``).
        """
        k = spring_constant if spring_constant is not None else self.spring_constant
        if k is None:
            raise ValueError("spring_constant (N/m) is required")

        d_nm = self.deflection_nm(inv_ols)
        z = self.z.astype(float)

        # Orient so the pushing direction has z increasing and, in contact,
        # deflection rising with z. Sort by oriented z (pre-contact first). A
        # perfectly flat deflection (no contact) needs no reorientation.
        s = 1.0
        if np.std(d_nm) > 0 and np.std(z) > 0 and np.corrcoef(z, d_nm)[0, 1] < 0:
            s = -1.0
        zc = s * z
        order = np.argsort(zc)
        zc, d_nm = zc[order], d_nm[order]

        # Baseline / noise from a small, safely pre-contact leading fraction.
        n = len(zc)
        ntail = max(int(baseline_fraction * n), 3)
        if baseline is None:
            baseline = float(np.median(d_nm[:ntail]))
        noise = float(np.std(d_nm[:ntail]))
        d_rel = d_nm - baseline

        # Provisional contact point. Its exact value only sets a constant
        # offset in delta (delta = delta_true + const), which a downstream
        # model fit with a contact-point scan absorbs; so a robust rough
        # estimate is enough. Threshold has an absolute floor (1% of the
        # deflection range) so it works even for noiseless data.
        if contact_point is None:
            thresh = max(contact_sigma * noise, 0.01 * float(np.ptp(d_rel)))
            contact_point = self.detect_contact_point(zc, d_rel, thresh)

        # Force (pN) and indentation (nm). The "- d_rel" term is the cantilever
        # compliance correction (piezo travel is shared between indenting the
        # sample and deflecting the cantilever).
        force = k * d_rel * 1e3                       # N/m * nm -> nN; *1e3 -> pN
        delta = (zc - contact_point) - d_rel

        if clip_negative:
            m = delta > 0.0
            delta, force = delta[m], force[m]
        order_d = np.argsort(delta)
        return delta[order_d], force[order_d]

    @staticmethod
    def detect_contact_point(zc, d_rel, thresh) -> float:
        """Return the contact point in oriented-Z: the first sustained crossing
        of ``d_rel`` above ``thresh``."""
        above = d_rel > thresh
        for i in range(len(above) - 1):
            if above[i] and above[i + 1]:
                return float(zc[i])
        return float(zc[int(np.argmax(d_rel))])


# --------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------
def read_csv(
    path: str,
    z_col=0,
    deflection_col=1,
    z_unit: str = "nm",
    deflection_unit: str = "nm",
    spring_constant: Optional[float] = None,
    inv_ols: Optional[float] = None,
    delimiter: Optional[str] = None,
    skip_header: int = 0,
    z_scale: float = 1.0,
    deflection_scale: float = 1.0,
) -> ForceCurveData:
    """Read a generic two-column CSV/TSV of ``z`` and ``deflection``.

    Columns may be given by 0-based index or by header name (set
    ``skip_header``/``delimiter`` as needed).  ``z_scale`` /
    ``deflection_scale`` multiply the raw columns (e.g. 1e9 to convert metres
    to nm).
    """
    if delimiter is None:
        delimiter = _sniff_delimiter(path, skip_header)
    names = None
    if isinstance(z_col, str) or isinstance(deflection_col, str):
        names = True
    data = np.genfromtxt(path, delimiter=delimiter, names=names,
                         skip_header=skip_header, dtype=float)
    if names:
        z = data[z_col]
        d = data[deflection_col]
    else:
        arr = np.atleast_2d(data)
        z = arr[:, z_col]
        d = arr[:, deflection_col]

    z = np.asarray(z, float) * z_scale
    d = np.asarray(d, float) * deflection_scale
    # Drop non-numeric rows (e.g. a header line read as NaN, or blank lines).
    good = np.isfinite(z) & np.isfinite(d)
    z, d = z[good], d[good]
    if z_unit == "m":
        z = z * 1e9
    return ForceCurveData(z=z, deflection=d, deflection_unit=deflection_unit,
                          spring_constant=spring_constant, inv_ols=inv_ols,
                          source=path, fmt="csv")


def _load_ibw_raw(path: str):
    """Thin wrapper over igor2 (isolated so tests can stub the parser)."""
    try:
        from igor2 import binarywave
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "reading Asylum .ibw files needs the 'igor2' package "
            "(pip install igor2)"
        ) from exc
    return binarywave.load(path)


def read_ibw(
    path: str,
    z_label: str = "ZSnsr",
    deflection_label: str = "Defl",
    z_scale: float = 1e9,
    deflection_scale: float = 1e9,
) -> ForceCurveData:
    """Read an Asylum Research Igor Binary Wave (``.ibw``) force curve.

    Asylum stores the channels as columns of a single 2-D wave with the column
    names in the wave's dimension labels, in SI units (metres); ``z_scale`` /
    ``deflection_scale`` default to 1e9 (m -> nm).  Spring constant / InvOLS are
    parsed from the wave note when present.
    """
    d = _load_ibw_raw(path)
    wave = d["wave"]
    data = np.asarray(wave["wData"], dtype=float)
    if data.ndim == 1:
        data = data[:, None]

    # Column labels live in wave['labels'] (list per dimension) for Asylum.
    labels = []
    for lab in wave.get("labels", []):
        for item in lab:
            if item:
                labels.append(item.decode() if isinstance(item, bytes) else str(item))
    note = wave.get("note", b"")
    note = note.decode("latin-1", "ignore") if isinstance(note, bytes) else str(note)
    meta = _parse_keyvalue_note(note)

    def column(label, default_index):
        if label in labels:
            return data[:, labels.index(label)]
        return data[:, min(default_index, data.shape[1] - 1)]

    z = column(z_label, data.shape[1] - 1) * z_scale
    defl = column(deflection_label, 1 if data.shape[1] > 1 else 0) * deflection_scale

    k = meta.get("springconstant") or meta.get("spring_constant")
    invols = meta.get("invols") or meta.get("sens")
    return ForceCurveData(
        z=z, deflection=defl, deflection_unit="nm",
        spring_constant=float(k) if k else None,
        inv_ols=float(invols) if invols else None,
        metadata=meta, source=path, fmt="ibw",
    )


def read_jpk(
    path: str,
    segment: int = 0,
    height_channels=("measuredHeight", "height", "capacitiveSensorHeight"),
    deflection_channel: str = "vDeflection",
) -> ForceCurveData:
    """Read a JPK ``.jpk-force`` file (a ZIP of segment headers + channel data).

    Decodes the requested segment's height and deflection channels: raw
    big-endian samples scaled by the encoder's linear ``multiplier``/``offset``
    (and a nominal->calibrated conversion where declared).  Real JPK calibration
    chains vary; verify against your files.
    """
    import zipfile

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        seg_prefix = f"segments/{segment}/"
        hdr = _read_properties(zf, seg_prefix + "segment-header.properties")
        shared = None
        if "shared-data/header.properties" in names:
            shared = _read_properties(zf, "shared-data/header.properties")

        def channel_array(chan):
            dat = seg_prefix + f"channels/{chan}.dat"
            if dat not in names:
                return None
            raw = zf.read(dat)
            return _decode_jpk_channel(raw, hdr, shared, chan)

        z = None
        for chan in height_channels:
            z = channel_array(chan)
            if z is not None:
                break
        defl = channel_array(deflection_channel)
        if z is None or defl is None:
            raise ValueError(
                f"JPK file missing height/deflection channels in segment {segment}")

        top = _read_properties(zf, "header.properties") if \
            "header.properties" in names else {}

    k = _first_float(hdr, [k for k in hdr if k.endswith("spring-constant")]) \
        or _first_float(top, [k for k in top if k.endswith("spring-constant")])
    return ForceCurveData(
        z=z * 1e9, deflection=defl * 1e9, deflection_unit="nm",
        spring_constant=k, metadata={"jpk_header": hdr}, source=path, fmt="jpk",
    )


def read_bruker(
    path: str,
    z_sens_key: str = "@Sens. ZsensSens",
) -> ForceCurveData:
    """Read a Bruker/Veeco Nanoscope force ramp (text header + int16 data).

    Parses the ASCII header (``\\`` lines) for the spring constant, Z sensitivity
    and the ramp data offset/length, then reads the two int16 channels.  This
    targets the common Nanoscope force-curve layout; multi-channel or
    PeakForce files may need channel selection.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    # The ASCII header ends at the "\*File list end" marker.
    header_end = raw.find(b"\\*File list end")
    header = raw[: header_end if header_end > 0 else 4096].decode("latin-1", "ignore")
    meta = _parse_nanoscope_header(header)

    offset = int(meta.get("data offset", 0))
    length = int(meta.get("data length", 0))
    bpp = int(meta.get("bytes/pixel", 2))
    dtype = {2: "<i2", 4: "<i4"}.get(bpp, "<i2")
    npts = length // bpp
    data = np.frombuffer(raw, dtype=dtype, count=npts, offset=offset).astype(float)

    # Nanoscope ramps interleave/segment Z and deflection; split in half as the
    # common two-segment (approach/retract or Z/Defl) layout.
    half = npts // 2
    z = data[:half]
    defl = data[half:2 * half] if 2 * half <= npts else data[half:]
    n = min(len(z), len(defl))
    z, defl = z[:n], defl[:n]

    zsens = _first_float(meta, [k for k in meta if "zsens" in k.lower()
                                or "z sens" in k.lower()])
    k = _first_float(meta, [key for key in meta if "spring constant" in key.lower()])
    if zsens:
        z = z * zsens
    return ForceCurveData(
        z=z, deflection=defl, deflection_unit="V",
        spring_constant=k, metadata=meta, source=path, fmt="bruker",
    )


_READERS = {
    ".csv": read_csv, ".txt": read_csv, ".tsv": read_csv,
    ".ibw": read_ibw,
    ".jpk-force": read_jpk, ".jpk": read_jpk,
    ".spm": read_bruker, ".pfc": read_bruker,
}


def load_afm(path: str, **kwargs) -> ForceCurveData:
    """Dispatch to the right reader by file extension.

    ``.csv/.txt/.tsv`` -> CSV, ``.ibw`` -> Asylum, ``.jpk-force/.jpk`` -> JPK,
    ``.spm/.pfc`` -> Bruker.  Extra kwargs pass through to the reader.
    """
    lower = path.lower()
    for ext, reader in _READERS.items():
        if lower.endswith(ext):
            return reader(path, **kwargs)
    raise ValueError(f"unrecognised AFM file extension: {path!r}")


# --------------------------------------------------------------------------
# high-level analysis
# --------------------------------------------------------------------------
def analyze_experiment(
    path_or_curve,
    R: float,
    nu: float,
    h: float,
    model: str = "dimitriadis_bonded",
    tip_shape: str = "paraboloid",
    spring_constant: Optional[float] = None,
    inv_ols: Optional[float] = None,
    fit_contact_point: bool = True,
    **read_kwargs,
):
    """End-to-end: load a raw AFM curve, calibrate it, and fit the modulus.

    ``path_or_curve`` may be a file path (dispatched via :func:`load_afm`) or an
    existing :class:`ForceCurveData`.  Returns the
    :class:`nanoindent.dataset.ModulusFit`, with the calibrated ``(delta,
    force)`` attached as ``.delta`` / ``.force`` on the returned object's
    ``meta`` for inspection.
    """
    from .dataset import fit_modulus

    if isinstance(path_or_curve, ForceCurveData):
        curve = path_or_curve
    else:
        curve = load_afm(path_or_curve, **read_kwargs)

    delta, force = curve.to_force_indentation(
        spring_constant=spring_constant, inv_ols=inv_ols)
    # Model names: analytics uses "dimitriadis_bonded"; fit_modulus uses
    # "dimitriadis" for the general-nu bonded shape.
    model_key = "dimitriadis" if model.startswith("dimitriadis") else model
    fit = fit_modulus(delta, force, model=model_key, R=R, nu=nu, h=h,
                      fit_contact_point=fit_contact_point)
    return fit, (delta, force)


# --------------------------------------------------------------------------
# synthetic raw generator (for tests / demos)
# --------------------------------------------------------------------------
def synthetic_raw_curve(
    cfg,
    spring_constant: float = 0.2,
    true_E: Optional[float] = None,
    source_model: str = "dimitriadis",
    n_contact: int = 120,
    n_precontact: int = 40,
    z_contact: float = 50.0,
    baseline_nm: float = 2.0,
    noise_nm: float = 0.0,
    seed: int = 0,
) -> ForceCurveData:
    """Build a realistic raw (Z, deflection) curve from a known model.

    Inverts the calibration: given a ground-truth force law ``F(delta)`` and a
    spring constant, produce the deflection ``d = F/k`` and piezo
    ``Z = Z0 + delta + d`` that an AFM would record, with a flat pre-contact
    region, a baseline offset, and optional deflection noise.  Feeding the
    result back through :meth:`ForceCurveData.to_force_indentation` recovers the
    original curve.
    """
    from . import analytics

    E = cfg.E if true_E is None else true_E
    rng = np.random.default_rng(seed)

    d_contact = np.linspace(0.0, cfg.delta_max, n_contact)
    reg = analytics.model_registry()
    key = "dimitriadis_bonded" if source_model.startswith("dimitriadis") else source_model
    force_pN = reg[key].func(d_contact, type("C", (), dict(
        R=cfg.R, E=E, nu=cfg.nu, h=cfg.h))())        # pN

    defl_contact = force_pN / (spring_constant * 1e3)   # pN -> nN/k -> nm
    z_c = z_contact + d_contact + defl_contact

    # Pre-contact: approach with zero deflection, Z below the contact point.
    z_pre = np.linspace(z_contact - 0.4 * cfg.delta_max, z_contact,
                        n_precontact, endpoint=False)
    defl_pre = np.zeros_like(z_pre)

    z = np.concatenate([z_pre, z_c])
    defl = np.concatenate([defl_pre, defl_contact]) + baseline_nm
    if noise_nm:
        defl = defl + rng.normal(0.0, noise_nm, size=defl.shape)

    return ForceCurveData(
        z=z, deflection=defl, deflection_unit="nm",
        spring_constant=spring_constant, segment="approach",
        metadata={"true_E": E, "source_model": source_model}, fmt="synthetic",
    )


# --------------------------------------------------------------------------
# small parsing helpers
# --------------------------------------------------------------------------
def _parse_keyvalue_note(note: str) -> dict:
    """Parse an Asylum-style ``key: value`` note (CR/LF separated)."""
    out = {}
    for line in note.replace("\r", "\n").split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip().lower().replace(" ", "")] = v.strip()
    return out


def _sniff_delimiter(path: str, skip_header: int) -> Optional[str]:
    """Guess the column delimiter (comma, tab, semicolon, or whitespace)."""
    with open(path) as fh:
        for _ in range(skip_header):
            fh.readline()
        line = fh.readline()
    for cand in (",", "\t", ";"):
        if cand in line:
            return cand
    return None  # whitespace


def _read_properties(zf, name: str) -> dict:
    """Parse a Java ``.properties`` file (``key=value`` lines) from a zip."""
    out = {}
    try:
        text = zf.read(name).decode("latin-1", "ignore")
    except KeyError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _decode_jpk_channel(raw: bytes, hdr: dict, shared: Optional[dict], chan: str):
    """Decode a JPK channel .dat: big-endian samples scaled by the encoder."""
    base = f"channel.{chan}.data.encoder"
    dtype_name = hdr.get(f"{base}.type", "short")
    dt = {"short": ">i2", "unsignedshort": ">u2", "integer": ">i4",
          "integer-data": ">i4", "float-data": ">f4", "float": ">f4",
          "double": ">f8"}.get(dtype_name, ">i2")
    values = np.frombuffer(raw, dtype=dt).astype(float)

    mult = _first_float(hdr, [f"{base}.scaling.multiplier"]) or 1.0
    off = _first_float(hdr, [f"{base}.scaling.offset"]) or 0.0
    values = values * mult + off

    # Optional nominal->calibrated linear conversion (e.g. distance / force).
    conv_base = f"channel.{chan}.data.conversion-set"
    default = hdr.get(f"{conv_base}.conversions.default")
    if default:
        c = f"{conv_base}.conversion.{default}.scaling"
        m = _first_float(hdr, [f"{c}.multiplier"])
        o = _first_float(hdr, [f"{c}.offset"])
        if m is not None:
            values = values * m + (o or 0.0)
    return values


def _parse_nanoscope_header(header: str) -> dict:
    """Parse Bruker Nanoscope ``\\Key: Value`` header lines."""
    out = {}
    for line in header.splitlines():
        line = line.strip()
        if not line.startswith("\\"):
            continue
        line = line.lstrip("\\").lstrip("@")
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip().lower()] = v.strip()
    return out


def _first_float(d: dict, keys) -> Optional[float]:
    """Return the first parseable float among ``keys`` (last numeric token)."""
    for k in keys:
        if k in d:
            for tok in reversed(str(d[k]).replace(",", " ").split()):
                try:
                    return float(tok)
                except ValueError:
                    continue
    return None
