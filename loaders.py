"""Load THz TDS data files and run sanity checks.

Handles format variations across different analysis pipelines:
  - Tab- and space-delimited files
  - 1- or 2-row headers
  - Frequency in Hz or THz (auto-detected and normalised to THz)
  - Masked / missing values ('--', empty cells, 'inf')  → NaN
  - 4-column (Chris) and 12-column (Marco/Hari) time-domain layouts
  - 9-column (Chris) and 10-column (Marco/Hari) FFT layouts
  - 7- or 8+-column optical-constants layouts
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import OrderedDict
from typing import Optional

import numpy as np

from models import (
    AnalysisDataset,
    FFTData,
    OpticalConstants,
    RawMeasurement,
    SampleInfo,
    TimeDomainData,
)

logger = logging.getLogger(__name__)

# ── Low-level parsing helpers ────────────────────────────────────────────────

_NAN_TOKENS = frozenset(('', '--', '- -', '\u2013 \u2013', 'inf', '-inf', '+inf', 'nan'))


def _parse_value(s: str) -> float:
    """Parse a single cell to float, returning NaN for masks/empties/inf."""
    s = s.strip()
    if s.lower() in _NAN_TOKENS:
        return float('nan')
    try:
        return float(s)
    except ValueError:
        return float('nan')


def _read_lines(filepath: str) -> list[str]:
    """Read non-blank lines from a text file."""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        return [line.rstrip('\n\r') for line in f if line.strip()]


def _detect_delimiter(lines: list[str]) -> Optional[str]:
    """Return '\\t' if the file is tab-delimited, else None (whitespace)."""
    for line in lines[:5]:
        if '\t' in line:
            return '\t'
    return None


def _split_line(line: str, delimiter: Optional[str]) -> list[str]:
    if delimiter:
        return line.split(delimiter)
    return line.split()


def _is_data_line(line: str, delimiter: Optional[str]) -> bool:
    """Heuristic: True if the line contains at least one parseable float."""
    for part in _split_line(line, delimiter):
        p = part.strip()
        if p.lower() in _NAN_TOKENS:
            continue
        try:
            float(p)
            return True
        except ValueError:
            continue
    return False


def _count_header_rows(lines: list[str], delimiter: Optional[str]) -> int:
    for i, line in enumerate(lines):
        if _is_data_line(line, delimiter):
            return i
    return len(lines)


def _parse_data_block(
    lines: list[str],
    delimiter: Optional[str],
    n_header: int,
) -> tuple[list[str], np.ndarray]:
    """Parse a tabular text file into header lines and a 2-D numpy array."""
    headers = lines[:n_header]
    data_lines = lines[n_header:]

    # Determine column count from data
    col_counts = [len(_split_line(l, delimiter)) for l in data_lines[:30]]
    n_cols = max(col_counts) if col_counts else 0

    rows: list[list[float]] = []
    for line in data_lines:
        parts = _split_line(line, delimiter)
        while len(parts) < n_cols:
            parts.append('')
        rows.append([_parse_value(p) for p in parts[:n_cols]])

    data = np.array(rows, dtype=np.float64) if rows else np.empty((0, n_cols))
    return headers, data


def _normalise_frequency(freq: np.ndarray) -> tuple[np.ndarray, bool]:
    """If values look like Hz (max > 1e6), divide by 1e12.  Returns (array, was_converted)."""
    valid = freq[np.isfinite(freq)]
    if len(valid) == 0:
        return freq.copy(), False
    if np.nanmax(np.abs(valid)) > 1e6:
        return freq / 1e12, True
    return freq.copy(), False


def _optional_array(arr: np.ndarray) -> Optional[np.ndarray]:
    """Return None if every element is NaN, else the array."""
    if arr is None or len(arr) == 0 or np.all(np.isnan(arr)):
        return None
    return arr


# ── File-type loaders ────────────────────────────────────────────────────────

def load_raw_measurement(filepath: str) -> RawMeasurement:
    """Load a 2-column space-delimited raw measurement file (no header)."""
    lines = _read_lines(filepath)
    delim = _detect_delimiter(lines)
    n_hdr = _count_header_rows(lines, delim)
    _, data = _parse_data_block(lines, delim, n_hdr)
    logger.info(f"  Raw measurement: {os.path.basename(filepath)} -> "
                f"{data.shape[0]} rows, {n_hdr} header row(s)")
    return RawMeasurement(time_ps=data[:, 0], amplitude=data[:, 1])


def load_time_domain(filepath: str) -> TimeDomainData:
    """Load a time-domain file (4-col Chris or 12-col Marco/Hari)."""
    lines = _read_lines(filepath)
    delim = _detect_delimiter(lines)
    n_hdr = _count_header_rows(lines, delim)
    _, data = _parse_data_block(lines, delim, n_hdr)
    n_cols = data.shape[1] if data.ndim == 2 else 0
    logger.info(f"  Time-domain: {os.path.basename(filepath)} -> "
                f"{data.shape[0]} rows x {n_cols} cols, {n_hdr} header row(s)")

    if n_cols <= 4:
        # Chris-style: 4 columns – already windowed
        return TimeDomainData(
            ref_time_ps=data[:, 0],
            ref_amplitude=data[:, 1],
            sample_time_ps=data[:, 2],
            sample_amplitude=data[:, 3],
        )

    if n_cols >= 12:
        # Marco/Hari-style: raw(0-2) | win_ref(3-5) | raw_sam(6-8) | win_sam(9-11)
        # Raw data only exists for rows where the raw time column is finite.
        raw_ref_mask = np.isfinite(data[:, 0])
        raw_sam_mask = np.isfinite(data[:, 6])

        return TimeDomainData(
            ref_time_ps=data[:, 3],
            ref_amplitude=data[:, 4],
            sample_time_ps=data[:, 9],
            sample_amplitude=data[:, 10],
            ref_std_error=_optional_array(data[:, 5]),
            sample_std_error=_optional_array(data[:, 11]),
            raw_ref_time_ps=data[raw_ref_mask, 0] if np.any(raw_ref_mask) else None,
            raw_ref_amplitude=data[raw_ref_mask, 1] if np.any(raw_ref_mask) else None,
            raw_ref_std_error=_optional_array(data[raw_ref_mask, 2]) if np.any(raw_ref_mask) else None,
            raw_sample_time_ps=data[raw_sam_mask, 6] if np.any(raw_sam_mask) else None,
            raw_sample_amplitude=data[raw_sam_mask, 7] if np.any(raw_sam_mask) else None,
            raw_sample_std_error=_optional_array(data[raw_sam_mask, 8]) if np.any(raw_sam_mask) else None,
        )

    raise ValueError(f"Unexpected column count {n_cols} in {filepath}")


def load_fft(filepath: str) -> FFTData:
    """Load an FFT-output file (9-col Chris or 10-col Marco/Hari)."""
    lines = _read_lines(filepath)
    delim = _detect_delimiter(lines)
    n_hdr = _count_header_rows(lines, delim)
    _, data = _parse_data_block(lines, delim, n_hdr)
    n_cols = data.shape[1]
    logger.info(f"  FFT: {os.path.basename(filepath)} -> "
                f"{data.shape[0]} rows x {n_cols} cols, {n_hdr} header row(s)")

    if n_cols == 10:
        # ref: freq(0) amp(1) Δamp(2) phase(3) Δphase(4)
        # sam: freq(5) amp(6) Δamp(7) phase(8) Δphase(9)
        ref_freq, converted = _normalise_frequency(data[:, 0])
        sam_freq, _ = _normalise_frequency(data[:, 5])
        if converted:
            logger.info("    Converted frequencies from Hz → THz")
        return FFTData(
            ref_frequency_thz=ref_freq,
            ref_amplitude=data[:, 1],
            ref_phase=data[:, 3],
            sample_frequency_thz=sam_freq,
            sample_amplitude=data[:, 6],
            sample_phase=data[:, 8],
            ref_delta_amplitude=_optional_array(data[:, 2]),
            ref_delta_phase=_optional_array(data[:, 4]),
            sample_delta_amplitude=_optional_array(data[:, 7]),
            sample_delta_phase=_optional_array(data[:, 9]),
        )

    if n_cols == 9:
        # Chris-style: ref(freq(0) amp(1) phase(2)), sam(freq(3) amp(4) phase(5)),
        #              raw_diff(6), phase_offset(7), phase_diff(8)
        ref_freq, converted = _normalise_frequency(data[:, 0])
        sam_freq, _ = _normalise_frequency(data[:, 3])
        if converted:
            logger.info("    Converted frequencies from Hz → THz")
        return FFTData(
            ref_frequency_thz=ref_freq,
            ref_amplitude=data[:, 1],
            ref_phase=data[:, 2],
            sample_frequency_thz=sam_freq,
            sample_amplitude=data[:, 4],
            sample_phase=data[:, 5],
            phase_offset=_optional_array(data[:, 7]),
            phase_diff=_optional_array(data[:, 8]),
        )

    raise ValueError(f"Unexpected column count {n_cols} in {filepath}")


def load_optical_constants(filepath: str) -> OpticalConstants:
    """Load an optical-constants file (7- or 8+-column variants)."""
    lines = _read_lines(filepath)
    delim = _detect_delimiter(lines)
    n_hdr = _count_header_rows(lines, delim)
    _, data = _parse_data_block(lines, delim, n_hdr)
    n_cols = data.shape[1]
    logger.info(f"  Optical constants: {os.path.basename(filepath)} -> "
                f"{data.shape[0]} rows x {n_cols} cols, {n_hdr} header row(s)")

    freq, converted = _normalise_frequency(data[:, 0])
    if converted:
        logger.info("    Converted frequencies from Hz → THz")

    eps_infty: Optional[float] = None
    if n_cols >= 8:
        col7 = data[:, 7]
        valid = col7[np.isfinite(col7)]
        if len(valid) > 0:
            eps_infty = float(valid[0])
            logger.info(f"    eps_infty = {eps_infty}")

    return OpticalConstants(
        frequency_thz=freq,
        n=data[:, 1],
        k=data[:, 2],
        eps1=data[:, 3],
        eps2=data[:, 4],
        sigma_re=data[:, 5],
        sigma_im=data[:, 6],
        eps_infty=eps_infty,
    )


# ── Sanity checks ───────────────────────────────────────────────────────────

def _check_monotonic(arr: np.ndarray, label: str, warnings: list[str]):
    valid = arr[np.isfinite(arr)]
    if len(valid) < 2:
        return
    n_violations = int(np.sum(np.diff(valid) < 0))
    if n_violations:
        warnings.append(f"{label}: not monotonically increasing ({n_violations} violations)")


def _check_range(arr: np.ndarray, label: str, lo: float, hi: float, warnings: list[str]):
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        warnings.append(f"{label}: all values are NaN")
        return
    vmin, vmax = float(np.nanmin(valid)), float(np.nanmax(valid))
    if vmin < lo or vmax > hi:
        warnings.append(f"{label}: range [{vmin:.4g}, {vmax:.4g}] outside expected [{lo}, {hi}]")


def _check_positive(arr: np.ndarray, label: str, warnings: list[str]):
    valid = arr[np.isfinite(arr)]
    if len(valid) > 0 and np.nanmin(valid) < 0:
        warnings.append(f"{label}: contains negative values (min={np.nanmin(valid):.4g})")


def sanity_check(dataset: AnalysisDataset) -> list[str]:
    """Run sanity checks on a loaded dataset.  Returns warning strings."""
    w: list[str] = []
    name = dataset.name

    # ── Time domain ──
    if dataset.time_domain is not None:
        td = dataset.time_domain
        _check_monotonic(td.ref_time_ps, f"[{name}] TD ref time", w)
        _check_monotonic(td.sample_time_ps, f"[{name}] TD sample time", w)
        _check_range(td.ref_time_ps, f"[{name}] TD ref time (ps)", 0, 500, w)
        _check_range(td.sample_time_ps, f"[{name}] TD sample time (ps)", 0, 500, w)

    # ── FFT ──
    if dataset.fft is not None:
        fft = dataset.fft
        _check_monotonic(fft.ref_frequency_thz, f"[{name}] FFT ref freq", w)
        _check_monotonic(fft.sample_frequency_thz, f"[{name}] FFT sample freq", w)
        _check_range(fft.ref_frequency_thz, f"[{name}] FFT ref freq (THz)", 0, 20, w)
        _check_range(fft.sample_frequency_thz, f"[{name}] FFT sample freq (THz)", 0, 20, w)
        _check_positive(fft.ref_amplitude, f"[{name}] FFT ref amplitude", w)
        _check_positive(fft.sample_amplitude, f"[{name}] FFT sample amplitude", w)

    # ── Optical constants ──
    if dataset.optical_constants is not None:
        oc = dataset.optical_constants
        _check_monotonic(oc.frequency_thz, f"[{name}] OC freq", w)
        _check_range(oc.frequency_thz, f"[{name}] OC freq (THz)", 0, 20, w)
        _check_positive(oc.n, f"[{name}] OC refractive index n", w)
        _check_range(oc.n, f"[{name}] OC refractive index n", 0.5, 50, w)
        valid_k = oc.k[np.isfinite(oc.k)]
        if len(valid_k) > 0 and float(np.nanmin(valid_k)) < -0.01:
            w.append(f"[{name}] OC extinction k has significant negative values "
                     f"(min={np.nanmin(valid_k):.4g})")

    return w


# ── Discovery and orchestration ─────────────────────────────────────────────

def discover_people(data_dir: str) -> list[str]:
    """Find person names from subdirectories (excluding 'original')."""
    return sorted(
        entry for entry in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, entry)) and entry != 'original'
    )


def load_sample_info(json_path: str) -> SampleInfo:
    with open(json_path, 'r') as f:
        d = json.load(f)
    return SampleInfo(thickness_m=d['thickness'], resistivity_ohm_m=d['resistivity'])


def _find_file(directory: str, pattern: str) -> Optional[str]:
    """Return path of first file in *directory* whose name matches *pattern*."""
    for fname in os.listdir(directory):
        if re.search(pattern, fname, re.IGNORECASE):
            return os.path.join(directory, fname)
    return None


def load_person(
    name: str,
    person_dir: str,
    raw_ref: RawMeasurement,
    raw_sam: RawMeasurement,
    sample_info: SampleInfo,
) -> AnalysisDataset:
    """Load all available data files for one person."""
    logger.info(f"Loading dataset for '{name}'")

    td = fft_data = oc = None

    td_file = _find_file(person_dir, r'time.?domain')
    if td_file:
        td = load_time_domain(td_file)
    else:
        logger.info(f"  No time-domain file found for {name}")

    fft_file = _find_file(person_dir, r'fft')
    if fft_file:
        fft_data = load_fft(fft_file)
    else:
        logger.info(f"  No FFT file found for {name}")

    oc_file = _find_file(person_dir, r'optical.?constant')
    if oc_file:
        oc = load_optical_constants(oc_file)
    else:
        logger.info(f"  No optical-constants file found for {name}")

    return AnalysisDataset(
        name=name,
        sample_info=sample_info,
        raw_reference=raw_ref,
        raw_sample=raw_sam,
        time_domain=td,
        fft=fft_data,
        optical_constants=oc,
    )


def load_all(base_dir: str, sample_json: str) -> list[AnalysisDataset]:
    """Discover and load every person's dataset."""
    sample_info = load_sample_info(sample_json)
    data_dir = os.path.join(base_dir, 'data')
    original_dir = os.path.join(data_dir, 'original')

    ref_file = _find_file(original_dir, r'reference')
    sam_file = _find_file(original_dir, r'sample')
    if not ref_file or not sam_file:
        raise FileNotFoundError(
            "Could not find reference/sample raw measurement files in data/original/"
        )

    raw_ref = load_raw_measurement(ref_file)
    raw_sam = load_raw_measurement(sam_file)
    logger.info(f"Raw reference: {len(raw_ref.time_ps)} points, "
                f"time [{raw_ref.time_ps[0]:.2f}, {raw_ref.time_ps[-1]:.2f}] ps")
    logger.info(f"Raw sample:    {len(raw_sam.time_ps)} points, "
                f"time [{raw_sam.time_ps[0]:.2f}, {raw_sam.time_ps[-1]:.2f}] ps")

    people = discover_people(data_dir)
    logger.info(f"Discovered people: {people}")

    datasets = []
    for name in people:
        person_dir = os.path.join(data_dir, name)
        ds = load_person(name, person_dir, raw_ref, raw_sam, sample_info)
        datasets.append(ds)

    return datasets
