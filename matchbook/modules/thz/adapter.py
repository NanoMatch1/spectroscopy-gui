"""Framework adapter for the THz TDS module.

This is the **only** file in the THz module that imports from the matchbook
framework.  It bridges between the module's standalone types (models,
loaders, analysis) and the framework's DataService / Pipeline / descriptors.

Delete this file and the rest of the THz package works standalone.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from matchbook.core.data_service import DataEntry, DataKey, DataService
from matchbook.core.module_base import (
    DataGroupDescriptor,
    ParameterDescriptor,
    ParamType,
    PipelineStepDescriptor,
    TraceDescriptor,
)
from matchbook.core.pipeline import PipelineStep

import json

from matchbook.modules.thz import analysis
from matchbook.modules.thz.models import AnalysisDataset

from matchbook.io.file_ingestor import FileIngestor, IngestedFile
from matchbook.io.recognisers import (
    get_atomiser,
    recognise,
    register_atomiser,
    register_recogniser,
)
from matchbook.modules.thz.containers import THzData
from matchbook.services.grouping_step import (
    create_grouping_step,
    grouping_step_descriptor,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: push an (x, y) pair into the DataService
# ---------------------------------------------------------------------------

def _put_xy(
    data_service: DataService,
    series_id: str,
    group: str,
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    x_label: str = "",
    y_label: str = "",
    display_label: str = "",
) -> None:
    """Convenience wrapper to insert one trace into the DataService."""
    data_service.put(DataEntry(
        key=DataKey(series_id, group, name),
        x=x, y=y,
        metadata={
            "x_label": x_label,
            "y_label": y_label,
            "display_label": display_label,
        },
    ))


# ---------------------------------------------------------------------------
# Pipeline step functions
#
# Signature:  fn(data_service, series_id, **params) -> None
# Each reads its inputs from data_service and writes its outputs back.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Recogniser + Atomiser registrations
# ---------------------------------------------------------------------------

@register_recogniser("thz_tds")
def recognise_thz(ingested: IngestedFile) -> float:
    """Return 1.0 if the loaded data is a THzData object."""
    return 1.0 if isinstance(ingested.data, THzData) else 0.0


@register_atomiser("thz_tds")
def atomise_thz(
    ingested: IngestedFile,
    data_service: DataService,
    series_id: str,
) -> None:
    """Write a single THzData object into the DataService."""
    thz_obj: THzData = ingested.data
    filename = ingested.filename
    file_series = f"{series_id}/{filename}"

    # Store raw compiled array
    if thz_obj.raw_data is not None:
        data_service.put(DataEntry(
            key=DataKey(file_series, "raw", "compiled"),
            x=thz_obj.raw_data[:, 0],
            y=thz_obj.raw_data[:, 1] if thz_obj.raw_data.shape[1] > 1 else thz_obj.raw_data[:, 0],
            metadata={
                "n_scans": len(thz_obj.data_list),
                "filename": filename,
                "data_type": thz_obj.data_type,
                "full_raw_shape": list(thz_obj.raw_data.shape),
            },
        ))

    # Store averaged time-domain data
    if thz_obj.data is not None:
        _put_xy(
            data_service, file_series,
            "time_domain", "mean",
            thz_obj.time, thz_obj.y_mean,
            "Time (ps)", "Amplitude (a.u.)", filename,
        )
        if thz_obj.y_err is not None:
            _put_xy(
                data_service, file_series,
                "time_domain", "stderr",
                thz_obj.time, thz_obj.y_err,
                "Time (ps)", "Std Error", f"{filename} stderr",
            )


# ---------------------------------------------------------------------------
# Registry-based file loading
# ---------------------------------------------------------------------------

def step_load_from_files(
    data_service: DataService,
    series_id: str,
    *,
    file_paths: list[str] | str = "",
) -> None:
    """Load THz data files via the FileIngestor.

    Accepts a list of file paths (or a single newline-separated string).
    Uses the FileIngestor to parse each file, then atomises the results
    into the DataService.  Grouping is handled by a separate pipeline step.
    """
    if isinstance(file_paths, str):
        file_paths = [p.strip() for p in file_paths.splitlines() if p.strip()]
    if not file_paths:
        return

    # Ensure loader modules are imported (triggers @register_loader)
    import matchbook.io.loaders  # noqa: F401

    # --- Generic: load via FileIngestor ---
    ingestor = FileIngestor()
    report = ingestor.load_files(file_paths)

    # --- Route each file to the best-matching module's atomiser ---
    for ingested in report.succeeded:
        matches = recognise(ingested)
        if not matches:
            logger.warning("No recogniser matched %s — skipping", ingested.filename)
            continue
        best_module, confidence = matches[0]
        logger.debug(
            "Recognised %s as '%s' (confidence %.2f)",
            ingested.filename, best_module, confidence,
        )
        atomiser = get_atomiser(best_module)
        atomiser(ingested, data_service, series_id)


def step_align_on_peak(
    data_service: DataService,
    series_id: str,
    *,
    peak_ranges_json: str = "{}",
    half_window_ps: float = 5.0,
) -> None:
    """Trim each series' time-domain trace to a window around its pulse peak.

    For each series, a user-supplied time range (from the span selection
    session) is searched for the largest-|amplitude| sample (after
    Savitzky-Golay smoothing).  The trace is then extracted as
    ``[peak_t - half_window_ps, peak_t + half_window_ps]`` and stored as
    ``time_domain/aligned_mean``.

    Because each series is processed independently, this step is compatible
    with the per-series pipeline execution model.  The half-window ensures
    output traces of equal length when all series share the same dt.

    Parameters
    ----------
    peak_ranges_json:
        JSON-encoded ``{series_id: [t_start, t_end]}`` produced by the
        span selection session in the GUI.
    half_window_ps:
        Half-width of the extraction window around the detected peak (ps).
    """
    try:
        ranges: dict = json.loads(peak_ranges_json) if peak_ranges_json else {}
    except json.JSONDecodeError:
        logger.warning("Invalid peak_ranges_json for %s — skipping", series_id)
        return

    if series_id not in ranges:
        logger.debug("No peak range configured for %s — skipping", series_id)
        return

    entry = data_service.get(DataKey(series_id, "time_domain", "mean"))
    if entry is None:
        logger.warning("No time_domain/mean data found for %s", series_id)
        return

    t_arr = entry.x
    y_arr = entry.y

    # Convert time range to array indices
    t_start, t_end = float(ranges[series_id][0]), float(ranges[series_id][1])
    idx_s = int(np.searchsorted(t_arr, t_start))
    idx_e = int(np.searchsorted(t_arr, t_end))
    if idx_s >= idx_e:
        logger.warning("Empty search range for %s ([%.3f, %.3f] ps)", series_id, t_start, t_end)
        return

    # Smooth and find peak
    try:
        from thz_core.preprocess import find_extremum_in_index_range
        from thz_core._common import smooth_trace_savgol
        y_smooth = smooth_trace_savgol(y_arr, window_length=11, polyorder=3)
        result = find_extremum_in_index_range(t_arr, y_smooth, idx_s, idx_e, mode="abs")
        peak_t = float(result["last_pick"]["x"])
    except Exception as exc:
        logger.error("Peak detection failed for %s: %s", series_id, exc)
        return

    # Extract fixed-width window around the peak
    mask = (t_arr >= peak_t - half_window_ps) & (t_arr <= peak_t + half_window_ps)
    if not np.any(mask):
        logger.warning("Window [%.3f, %.3f] ps contains no samples for %s",
                       peak_t - half_window_ps, peak_t + half_window_ps, series_id)
        return

    _put_xy(
        data_service, series_id,
        "time_domain", "aligned_mean",
        t_arr[mask], y_arr[mask],
        "Time (ps)", "Amplitude (a.u.)", f"Aligned: {series_id}",
    )
    logger.info("Aligned %s: peak @ %.3f ps, window [%.3f, %.3f] ps",
                series_id, peak_t, float(t_arr[mask][0]), float(t_arr[mask][-1]))


def step_compute_transfer_function(
    data_service: DataService,
    series_id: str,
    **_params: Any,
) -> None:
    """Compute transfer function from FFT data in the DataService."""
    ref_amp_entry = data_service.get(DataKey(series_id, "fft", "ref_amp"))
    sam_amp_entry = data_service.get(DataKey(series_id, "fft", "sam_amp"))
    ref_phase_entry = data_service.get(DataKey(series_id, "fft", "ref_phase"))
    sam_phase_entry = data_service.get(DataKey(series_id, "fft", "sam_phase"))

    if ref_amp_entry is None or sam_amp_entry is None:
        return

    freq, amp_ratio = analysis.compute_transfer_function_amplitude(
        ref_amp_entry.x, ref_amp_entry.y,
        sam_amp_entry.x, sam_amp_entry.y,
    )
    _put_xy(data_service, series_id, "transfer_function", "amplitude",
            freq, amp_ratio,
            "Frequency (THz)", "|H(f)|", "|H(f)| amplitude")

    if ref_phase_entry is not None and sam_phase_entry is not None:
        freq_p, phase_diff = analysis.compute_transfer_function_phase(
            ref_phase_entry.x, ref_phase_entry.y,
            sam_phase_entry.x, sam_phase_entry.y,
        )
        _put_xy(data_service, series_id, "transfer_function", "phase",
                freq_p, phase_diff,
                "Frequency (THz)", "Δφ (rad)", "Phase difference")


# ---------------------------------------------------------------------------
# Module façade (satisfies AnalysisModule protocol)
# ---------------------------------------------------------------------------

class THzModule:
    """THz TDS analysis module.

    This class satisfies the ``AnalysisModule`` protocol structurally —
    it does **not** inherit from any base class.
    """

    @property
    def name(self) -> str:
        return "thz_tds"

    @property
    def display_name(self) -> str:
        return "THz Time-Domain Spectroscopy"

    def data_groups(self) -> list[DataGroupDescriptor]:
        """Declare all plottable data groups and traces."""
        return [
            DataGroupDescriptor(
                key="raw",
                label="Raw / Compiled",
                x_label="Time (ps)",
                order=0,
                traces=[
                    TraceDescriptor("raw_compiled", "Compiled scan",
                                    "raw", "compiled",
                                    y_label="Amplitude (a.u.)", default_visible=True),
                ],
            ),
            DataGroupDescriptor(
                key="time_domain",
                label="Time Domain",
                x_label="Time (ps)",
                order=1,
                traces=[
                    TraceDescriptor("td_mean", "Mean amplitude",
                                    "time_domain", "mean",
                                    y_label="Amplitude (a.u.)", default_visible=True),
                    TraceDescriptor("td_stderr", "Std error",
                                    "time_domain", "stderr",
                                    y_label="Std Error"),
                    TraceDescriptor("td_aligned_mean", "Aligned mean",
                                    "time_domain", "aligned_mean",
                                    y_label="Amplitude (a.u.)", default_visible=False),
                ],
            ),
            DataGroupDescriptor(
                key="fft",
                label="FFT",
                x_label="Frequency (THz)",
                order=2,
                traces=[
                    TraceDescriptor("fft_ref_amp", "Reference amplitude",
                                    "fft", "ref_amp",
                                    y_label="Amplitude", default_visible=True),
                    TraceDescriptor("fft_sam_amp", "Sample amplitude",
                                    "fft", "sam_amp",
                                    y_label="Amplitude", default_visible=True),
                    TraceDescriptor("fft_ref_phase", "Reference phase",
                                    "fft", "ref_phase",
                                    y_label="Phase (rad)"),
                    TraceDescriptor("fft_sam_phase", "Sample phase",
                                    "fft", "sam_phase",
                                    y_label="Phase (rad)"),
                ],
            ),
            DataGroupDescriptor(
                key="transfer_function",
                label="Transfer Function",
                x_label="Frequency (THz)",
                order=3,
                traces=[
                    TraceDescriptor("tf_amp", "|H(f)| amplitude",
                                    "transfer_function", "amplitude",
                                    y_label="|H(f)|", default_visible=True),
                    TraceDescriptor("tf_phase", "Phase difference",
                                    "transfer_function", "phase",
                                    y_label="Δφ (rad)", default_visible=True),
                ],
            ),
            DataGroupDescriptor(
                key="optical_constants",
                label="Optical Constants",
                x_label="Frequency (THz)",
                order=4,
                traces=[
                    TraceDescriptor("oc_n", "Refractive index n",
                                    "optical_constants", "n",
                                    y_label="n", default_visible=True),
                    TraceDescriptor("oc_k", "Extinction coeff k",
                                    "optical_constants", "k",
                                    y_label="k", default_visible=True),
                    TraceDescriptor("oc_eps1", "ε₁ (real dielectric)",
                                    "optical_constants", "eps1",
                                    y_label="ε₁"),
                    TraceDescriptor("oc_eps2", "ε₂ (imag dielectric)",
                                    "optical_constants", "eps2",
                                    y_label="ε₂"),
                    TraceDescriptor("oc_sre", "σ_Re (S/m)",
                                    "optical_constants", "sigma_re",
                                    y_label="σ_Re (S/m)"),
                    TraceDescriptor("oc_sim", "σ_Im (S/m)",
                                    "optical_constants", "sigma_im",
                                    y_label="σ_Im (S/m)"),
                ],
            ),
        ]

    def pipeline_step_descriptors(self) -> list[PipelineStepDescriptor]:
        """Declare pipeline steps and their tunable parameters."""
        return [
            PipelineStepDescriptor(
                id="load_from_files",
                name="Load Data",
                params=[
                    ParameterDescriptor("file_paths", "File paths (one per line)",
                                        ParamType.STRING, ""),
                ],
            ),
            grouping_step_descriptor(depends_on=["load_from_files"]),
            PipelineStepDescriptor(
                id="align_on_peak",
                name="Align on Peak",
                params=[
                    ParameterDescriptor(
                        name="peak_ranges_json",
                        label="Peak ranges",
                        type=ParamType.SPAN_SESSION,
                        default="{}",
                        data_group="time_domain",
                        tooltip="Per-series time ranges for pulse peak detection",
                    ),
                    ParameterDescriptor(
                        name="half_window_ps",
                        label="Half-window (ps)",
                        type=ParamType.FLOAT,
                        default=5.0,
                        min=0.5,
                        max=50.0,
                        tooltip="Half-width of extraction window around detected peak",
                    ),
                ],
                depends_on=["group_files"],
            ),
            PipelineStepDescriptor(
                id="transfer_function",
                name="Compute Transfer Function",
                depends_on=["align_on_peak"],
            ),
        ]

    def create_pipeline_steps(self) -> list[PipelineStep]:
        """Return bound PipelineStep objects for the engine."""
        descriptors = {s.id: s for s in self.pipeline_step_descriptors()}
        return [
            PipelineStep(
                id="load_from_files",
                name="Load Data",
                fn=step_load_from_files,
                params=_defaults_from(descriptors["load_from_files"]),
                param_descriptors=descriptors["load_from_files"].params,
            ),
            create_grouping_step(depends_on=["load_from_files"]),
            PipelineStep(
                id="align_on_peak",
                name="Align on Peak",
                fn=step_align_on_peak,
                params=_defaults_from(descriptors["align_on_peak"]),
                param_descriptors=descriptors["align_on_peak"].params,
                depends_on=["group_files"],
            ),
            PipelineStep(
                id="transfer_function",
                name="Compute Transfer Function",
                fn=step_compute_transfer_function,
                params=_defaults_from(descriptors["transfer_function"]),
                param_descriptors=descriptors["transfer_function"].params,
                depends_on=["align_on_peak"],
            ),
        ]


def _defaults_from(desc: PipelineStepDescriptor) -> dict[str, Any]:
    """Extract {param_name: default_value} from a step descriptor."""
    return {p.name: p.default for p in desc.params}
