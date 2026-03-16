"""Framework adapter for the THz TDS module.

This is the **only** file in the THz module that imports from the matchbook
framework.  It bridges between the module's standalone types (models,
loaders, analysis) and the framework's DataService / Pipeline / descriptors.

Delete this file and the rest of the THz package works standalone.
"""

from __future__ import annotations

import logging
import os
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

from matchbook.modules.thz import analysis, loaders
from matchbook.modules.thz.models import AnalysisDataset

# New infrastructure imports
from matchbook.io.loaders.registry import get_loader_for_extension, LoaderError
from matchbook.modules.thz.containers import THzData
from matchbook.services.grouping import GroupingService

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

def step_load_data(
    data_service: DataService,
    series_id: str,
    *,
    base_dir: str = "",
    person_name: str = "",
    sample_json: str = "",
) -> None:
    """Load raw + processed data files for one person into the DataService.

    This wraps the standalone ``loaders.load_person`` call and pushes
    every array into the DataService under the appropriate group/name keys.
    """
    if not base_dir:
        return

    data_dir = os.path.join(base_dir, "data")
    original_dir = os.path.join(data_dir, "original")
    person_dir = os.path.join(data_dir, person_name)

    if not os.path.isdir(person_dir):
        logger.warning(f"Person directory not found: {person_dir}")
        return

    sample_info = loaders.load_sample_info(sample_json)

    raw_ref = loaders.load_raw_measurement(
        loaders._find_file(original_dir, r'reference') or ""
    )
    raw_sam = loaders.load_raw_measurement(
        loaders._find_file(original_dir, r'sample') or ""
    )

    dataset = loaders.load_person(
        person_name, person_dir, raw_ref, raw_sam, sample_info
    )

    # Store the dataset object for later reference
    data_service.put(DataEntry(
        key=DataKey(series_id, "_meta", "dataset"),
        x=np.array([0.0]), y=np.array([0.0]),
        metadata={"dataset": dataset},
    ))

    _publish_dataset(data_service, series_id, dataset)


def _publish_dataset(
    data_service: DataService,
    series_id: str,
    dataset: AnalysisDataset,
) -> None:
    """Push all arrays from an AnalysisDataset into the DataService."""
    xl_time = "Time (ps)"
    xl_freq = "Frequency (THz)"
    yl_amp = "Amplitude (V)"

    # -- Raw measurements --
    _put_xy(data_service, series_id, "time_domain", "orig_ref_time",
            dataset.raw_reference.time_ps, dataset.raw_reference.time_ps,
            xl_time, xl_time, "Original ref time")
    _put_xy(data_service, series_id, "time_domain", "orig_ref_amp",
            dataset.raw_reference.time_ps, dataset.raw_reference.amplitude,
            xl_time, yl_amp, "Original reference")
    _put_xy(data_service, series_id, "time_domain", "orig_sam_amp",
            dataset.raw_sample.time_ps, dataset.raw_sample.amplitude,
            xl_time, yl_amp, "Original sample")

    # -- Time domain --
    td = dataset.time_domain
    if td is not None:
        _put_xy(data_service, series_id, "time_domain", "win_ref_amp",
                td.ref_time_ps, td.ref_amplitude,
                xl_time, yl_amp, "Windowed reference")
        _put_xy(data_service, series_id, "time_domain", "win_sam_amp",
                td.sample_time_ps, td.sample_amplitude,
                xl_time, yl_amp, "Windowed sample")
        if td.raw_ref_time_ps is not None:
            _put_xy(data_service, series_id, "time_domain", "raw_ref_amp",
                    td.raw_ref_time_ps, td.raw_ref_amplitude,
                    xl_time, yl_amp, "Pre-windowed reference")
        if td.raw_sample_time_ps is not None:
            _put_xy(data_service, series_id, "time_domain", "raw_sam_amp",
                    td.raw_sample_time_ps, td.raw_sample_amplitude,
                    xl_time, yl_amp, "Pre-windowed sample")

    # -- FFT --
    fft = dataset.fft
    if fft is not None:
        _put_xy(data_service, series_id, "fft", "ref_amp",
                fft.ref_frequency_thz, fft.ref_amplitude,
                xl_freq, "Amplitude", "Reference amplitude")
        _put_xy(data_service, series_id, "fft", "sam_amp",
                fft.sample_frequency_thz, fft.sample_amplitude,
                xl_freq, "Amplitude", "Sample amplitude")
        _put_xy(data_service, series_id, "fft", "ref_phase",
                fft.ref_frequency_thz, fft.ref_phase,
                xl_freq, "Phase (rad)", "Reference phase")
        _put_xy(data_service, series_id, "fft", "sam_phase",
                fft.sample_frequency_thz, fft.sample_phase,
                xl_freq, "Phase (rad)", "Sample phase")

    # -- Transfer function --
    if fft is not None:
        tf_freq, tf_amp = analysis.compute_transfer_function_amplitude(
            fft.ref_frequency_thz, fft.ref_amplitude,
            fft.sample_frequency_thz, fft.sample_amplitude,
        )
        _put_xy(data_service, series_id, "transfer_function", "amplitude",
                tf_freq, tf_amp,
                xl_freq, "|H(f)|", "|H(f)| amplitude")

        tf_freq_p, tf_phase = analysis.compute_transfer_function_phase(
            fft.ref_frequency_thz, fft.ref_phase,
            fft.sample_frequency_thz, fft.sample_phase,
        )
        _put_xy(data_service, series_id, "transfer_function", "phase",
                tf_freq_p, tf_phase,
                xl_freq, "Δφ (rad)", "Phase difference")

    # -- Optical constants --
    oc = dataset.optical_constants
    if oc is not None:
        for attr, name, ylabel in [
            ("n",        "n",        "n"),
            ("k",        "k",        "k"),
            ("eps1",     "eps1",     "ε₁"),
            ("eps2",     "eps2",     "ε₂"),
            ("sigma_re", "sigma_re", "σ_Re (S/m)"),
            ("sigma_im", "sigma_im", "σ_Im (S/m)"),
        ]:
            arr = getattr(oc, attr)
            if arr is not None and not np.all(np.isnan(arr)):
                _put_xy(data_service, series_id, "optical_constants", name,
                        oc.frequency_thz, arr,
                        xl_freq, ylabel, ylabel)


# ---------------------------------------------------------------------------
# Registry-based file loading
# ---------------------------------------------------------------------------

def step_load_from_files(
    data_service: DataService,
    series_id: str,
    *,
    file_paths: list[str] | str = "",
    grouping_keywords: list[str] | None = None,
    grouping_delimiter: str = "_",
) -> None:
    """Load THz data files via the loader registry and auto-group them.

    Accepts a list of file paths (or a single newline-separated string).
    Uses the loader registry to parse each file, then runs GroupingService
    to pair samples with references.  Associations are stored in the
    DataService.
    """
    if isinstance(file_paths, str):
        file_paths = [p.strip() for p in file_paths.splitlines() if p.strip()]
    if not file_paths:
        return

    # Ensure loader modules are imported (triggers @register_loader)
    import matchbook.io.loaders  # noqa: F401

    loaded: dict[str, THzData] = {}
    for fpath in file_paths:
        ext = os.path.splitext(fpath)[1]
        try:
            loader_cls = get_loader_for_extension(ext)
        except LoaderError:
            logger.warning(f"No loader for {ext}, skipping: {fpath}")
            continue
        loader = loader_cls(fpath)
        result = loader.load()
        if isinstance(result, THzData):
            loaded[os.path.basename(fpath)] = result

    if not loaded:
        return

    # Group files by filename conventions
    gs = GroupingService(
        keywords=grouping_keywords or ["type", "series", "temp"],
        delimiter=grouping_delimiter,
        filelist=list(loaded.keys()),
    )
    gs.simple_grouping(delimiter=grouping_delimiter, keywords=gs.keywords)

    # Atomise each loaded file into the DataService
    for filename, thz_obj in loaded.items():
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

        # Record file-level metadata
        info = gs(filename)
        if info is not None:
            data_service.put(DataEntry(
                key=DataKey(file_series, "_meta", "grouping"),
                x=np.array([0.0]), y=np.array([0.0]),
                metadata={
                    "data_type": info.data_type,
                    "series": info.series,
                    "temperature": info.temperature,
                },
            ))

    # Create associations from grouping results
    for filename, info in gs.file_items.items():
        file_series = f"{series_id}/{filename}"
        if info.substrate_reference:
            ref_series = f"{series_id}/{info.substrate_reference}"
            data_service.associate(file_series, "substrate_reference", ref_series)
        if info.air_reference:
            air_series = f"{series_id}/{info.air_reference}"
            data_service.associate(file_series, "air_reference", air_series)


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
                key="time_domain",
                label="Time Domain",
                x_label="Time (ps)",
                order=0,
                traces=[
                    TraceDescriptor("td_orig_ref", "Original reference",
                                    "time_domain", "orig_ref_amp",
                                    y_label="Amplitude (V)"),
                    TraceDescriptor("td_orig_sam", "Original sample",
                                    "time_domain", "orig_sam_amp",
                                    y_label="Amplitude (V)"),
                    TraceDescriptor("td_win_ref", "Windowed reference",
                                    "time_domain", "win_ref_amp",
                                    y_label="Amplitude (V)", default_visible=True),
                    TraceDescriptor("td_win_sam", "Windowed sample",
                                    "time_domain", "win_sam_amp",
                                    y_label="Amplitude (V)", default_visible=True),
                    TraceDescriptor("td_raw_ref", "Pre-windowed reference",
                                    "time_domain", "raw_ref_amp",
                                    y_label="Amplitude (V)"),
                    TraceDescriptor("td_raw_sam", "Pre-windowed sample",
                                    "time_domain", "raw_sam_amp",
                                    y_label="Amplitude (V)"),
                ],
            ),
            DataGroupDescriptor(
                key="fft",
                label="FFT",
                x_label="Frequency (THz)",
                order=1,
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
                order=2,
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
                order=3,
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
                id="load_data",
                name="Load Data (comparison)",
                params=[
                    ParameterDescriptor("base_dir", "Data directory",
                                        ParamType.FILE_PATH, ""),
                    ParameterDescriptor("person_name", "Person / series name",
                                        ParamType.STRING, ""),
                    ParameterDescriptor("sample_json", "Sample details JSON",
                                        ParamType.FILE_PATH, ""),
                ],
            ),
            PipelineStepDescriptor(
                id="load_from_files",
                name="Load Data (files)",
                params=[
                    ParameterDescriptor("file_paths", "File paths (one per line)",
                                        ParamType.STRING, ""),
                    ParameterDescriptor("grouping_keywords", "Grouping keywords",
                                        ParamType.STRING, "type,series,temp"),
                    ParameterDescriptor("grouping_delimiter", "Filename delimiter",
                                        ParamType.STRING, "_"),
                ],
            ),
            PipelineStepDescriptor(
                id="transfer_function",
                name="Compute Transfer Function",
                depends_on=["load_data"],
            ),
        ]

    def create_pipeline_steps(self) -> list[PipelineStep]:
        """Return bound PipelineStep objects for the engine."""
        descriptors = {s.id: s for s in self.pipeline_step_descriptors()}
        return [
            PipelineStep(
                id="load_data",
                name="Load Data (comparison)",
                fn=step_load_data,
                params=_defaults_from(descriptors["load_data"]),
                param_descriptors=descriptors["load_data"].params,
            ),
            PipelineStep(
                id="load_from_files",
                name="Load Data (files)",
                fn=step_load_from_files,
                params=_defaults_from(descriptors["load_from_files"]),
                param_descriptors=descriptors["load_from_files"].params,
            ),
            PipelineStep(
                id="transfer_function",
                name="Compute Transfer Function",
                fn=step_compute_transfer_function,
                params=_defaults_from(descriptors["transfer_function"]),
                param_descriptors=descriptors["transfer_function"].params,
                depends_on=["load_data"],
            ),
        ]


def _defaults_from(desc: PipelineStepDescriptor) -> dict[str, Any]:
    """Extract {param_name: default_value} from a step descriptor."""
    return {p.name: p.default for p in desc.params}
