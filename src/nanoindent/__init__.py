"""nanoindent — FEM simulation of AFM nanoindentation of thin elastic films.

An open-source (scikit-fem / dolfinx) alternative to the commercial-FEM
workflow for a rigid spherical tip indenting a linear-elastic film bonded to a
rigid substrate, validated against analytic contact models rather than against
another solver (docs/design.md section 6.2).

Quick start
-----------
>>> from nanoindent import SimulationConfig, IndentationSolver
>>> cfg = SimulationConfig(h=7.0, nu=0.3)
>>> result = IndentationSolver(cfg).run()
>>> result.plot(compare=["hertz", "dimitriadis_bonded", "garcia_bec"])
"""

from .config import SimulationConfig, MPA_NM2_TO_PN
from .material import LinearElastic
from .contact import (
    RigidIndenter, RigidSphere, Paraboloid, Cone, BluntedCone,
    make_tip, equivalent_cone_half_angle,
)
from .geometry import build_mesh, Domain
from .solver import IndentationSolver
from .postproc import ForceCurve
from . import analytics
from .analytics import compare_models, BEC_MODELS
from . import convergence
from . import dataset
from .dataset import synthetic_force_curve, fit_modulus, SyntheticDataset, ModulusFit
from . import experiment
from .experiment import (
    ForceCurveData, load_afm, read_csv, read_ibw, read_jpk, read_jpk_map,
    read_bruker, analyze_experiment, synthetic_raw_curve, JPKForceMap,
)
from . import slb
from . import forcemap
from .forcemap import (
    ForceVolumeSpec, ForceVolume, GeometricKernel, geometric_kernel,
    simulate_force_volume, uniform_field, two_phase_field, gradient_field,
)
from . import viscoelastic
from .viscoelastic import (
    RelaxationModel, standard_linear_solid, generalized_maxwell,
    hereditary_force, ramp_hold_curve, frequency_sweep, DynamicSpectrum,
    kernel_slope,
)
from . import recovery
from .recovery import (
    SurfaceFootprint, surface_footprint, RasterRecovery, raster_surface,
    residual_fraction, recovery_limit_map,
)
from . import lipids
from .lipids import (
    LipiverseClient, MembranePrediction, MockLipiverseTransport, fetch_membrane,
    LipiverseError, LipiverseUnavailable, UnknownSpecies, NotAMembraneLipid,
)
from . import video
from .video import (
    AFMVideo, synthetic_video, load_video, read_asd, read_tiff, read_numpy,
    read_jpk_qi, drift_correct, per_frame_stats, detect_particles, track,
    msd, diffusion_coefficient, ensemble_diffusion, height_timeseries,
    detect_events, animate_video, plot_summary,
)
from . import streaming
from .streaming import (
    StreamingAnalyzer, AnalysisState, OnlineTrack, Command, FrameSource,
    VideoFrameSource, SimulatedInstrument, run_control_loop, RunReport,
    plot_control_run, recenter_on_brightest, recenter_on_longest_track,
    ramp_on_event, pause_on_event, combine_policies,
)
from . import adapters
from .adapters import (
    InstrumentAdapter, BackendSimulator, NanosurfAdapter, JPKAdapter,
    AsylumAdapter,
)
from . import sources
from .sources import DirectoryWatchSource, RetryFrameSource, read_frame_file
from . import recording
from .recording import RunRecorder, RecordingSource, make_run_id
from . import safety
from .safety import SafetyEnvelope, SafetyLimits
from . import tracking
from .tracking import ExperimentIndex, log_run_to_mlflow, load_manifest
from . import replay
from .replay import RecordedRun, load_run
from . import control
from . import plotting

__all__ = [
    "SimulationConfig",
    "MPA_NM2_TO_PN",
    "LinearElastic",
    "RigidIndenter",
    "RigidSphere",
    "Paraboloid",
    "Cone",
    "BluntedCone",
    "make_tip",
    "equivalent_cone_half_angle",
    "build_mesh",
    "Domain",
    "IndentationSolver",
    "ForceCurve",
    "analytics",
    "compare_models",
    "BEC_MODELS",
    "convergence",
    "dataset",
    "experiment",
    "ForceCurveData",
    "load_afm",
    "read_csv",
    "read_ibw",
    "read_jpk",
    "read_jpk_map",
    "read_bruker",
    "analyze_experiment",
    "synthetic_raw_curve",
    "JPKForceMap",
    "slb",
    # force-volume (force-map) generation
    "forcemap",
    "ForceVolumeSpec",
    "ForceVolume",
    "GeometricKernel",
    "geometric_kernel",
    "simulate_force_volume",
    "uniform_field",
    "two_phase_field",
    "gradient_field",
    # viscoelastic / dynamic mechanics
    "viscoelastic",
    "RelaxationModel",
    "standard_linear_solid",
    "generalized_maxwell",
    "hereditary_force",
    "ramp_hold_curve",
    "frequency_sweep",
    "DynamicSpectrum",
    "kernel_slope",
    # recovery between consecutive indentations
    "recovery",
    "SurfaceFootprint",
    "surface_footprint",
    "RasterRecovery",
    "raster_surface",
    "residual_fraction",
    "recovery_limit_map",
    "lipids",
    "LipiverseClient",
    "MembranePrediction",
    "MockLipiverseTransport",
    "fetch_membrane",
    "LipiverseError",
    "LipiverseUnavailable",
    "UnknownSpecies",
    "NotAMembraneLipid",
    "video",
    "AFMVideo",
    "synthetic_video",
    "load_video",
    "read_asd",
    "read_tiff",
    "read_numpy",
    "read_jpk_qi",
    "drift_correct",
    "per_frame_stats",
    "detect_particles",
    "track",
    "msd",
    "diffusion_coefficient",
    "ensemble_diffusion",
    "height_timeseries",
    "detect_events",
    "animate_video",
    "plot_summary",
    "streaming",
    "StreamingAnalyzer",
    "AnalysisState",
    "OnlineTrack",
    "Command",
    "FrameSource",
    "VideoFrameSource",
    "SimulatedInstrument",
    "run_control_loop",
    "RunReport",
    "plot_control_run",
    "recenter_on_brightest",
    "recenter_on_longest_track",
    "ramp_on_event",
    "pause_on_event",
    "combine_policies",
    "adapters",
    "InstrumentAdapter",
    "BackendSimulator",
    "NanosurfAdapter",
    "JPKAdapter",
    "AsylumAdapter",
    "sources",
    "DirectoryWatchSource",
    "RetryFrameSource",
    "read_frame_file",
    "recording",
    "RunRecorder",
    "RecordingSource",
    "make_run_id",
    "safety",
    "SafetyEnvelope",
    "SafetyLimits",
    "tracking",
    "ExperimentIndex",
    "log_run_to_mlflow",
    "load_manifest",
    "replay",
    "RecordedRun",
    "load_run",
    "control",
    "plotting",
    "synthetic_force_curve",
    "fit_modulus",
    "SyntheticDataset",
    "ModulusFit",
]

__version__ = "0.1.0"
