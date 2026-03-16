"""Generic THz TDS data-file loaders and sanity checks.

This module has **no framework dependencies** — it uses only stdlib and numpy.
It provides format-agnostic loading for common THz data file layouts:
  - Tab- and space-delimited files
  - 1- or 2-row headers
  - Frequency in Hz or THz (auto-detected and normalised to THz)
  - Masked / missing values ('--', empty cells, 'inf')  -> NaN
  - 4-column and 12-column time-domain layouts
  - 9-column and 10-column FFT layouts
  - 7- or 8+-column optical-constants layouts

Discovery functions scan directories for series sub-folders, load whatever
recognised files they contain, and return ``AnalysisDataset`` bundles.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

import numpy as np

from matchbook.modules.thz.models import (
    AnalysisDataset,
    FFTData,
    OpticalConstants,
    RawMeasurement,
    TimeDomainData,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Low-level parsing helpers
# ---------------------------------------------------------------------------

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
    r"""Return '\t' if the file is tab-delimited, else None (whitespace)."""
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
    """If values look like Hz (max > 1e6), divide by 1e12."""
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


# ---------------------------------------------------------------------------
# File-type loaders
# ---------------------------------------------------------------------------

def load_two_column(filepath: str) -> RawMeasurement:
    """Load a 2-column numeric file (any delimiter, auto-detected header).

    Returns a ``RawMeasurement`` with the first column as *time_ps* and
    the second as *amplitude*.  Suitable for any simple (x, y) data.
    """
    lines = _read_lines(filepath)
    delim = _detect_delimiter(lines)
    n_hdr = _count_header_rows(lines, delim)
    _, data = _parse_data_block(lines, delim, n_hdr)
    logger.info(f"  Two-column load: {os.path.basename(filepath)} -> "
                f"{data.shape[0]} rows, {n_hdr} header row(s)")
    return RawMeasurement(time_ps=data[:, 0], amplitude=data[:, 1])


# Backward-compatible alias
load_raw_measurement = load_two_column


def load_time_domain(filepath: str) -> TimeDomainData:
    """Load a time-domain file (4-col or 12-col layouts)."""
    lines = _read_lines(filepath)
    delim = _detect_delimiter(lines)
    n_hdr = _count_header_rows(lines, delim)
    _, data = _parse_data_block(lines, delim, n_hdr)
    n_cols = data.shape[1] if data.ndim == 2 else 0
    logger.info(f"  Time-domain: {os.path.basename(filepath)} -> "
                f"{data.shape[0]} rows x {n_cols} cols, {n_hdr} header row(s)")

    if n_cols <= 4:
        return TimeDomainData(
            ref_time_ps=data[:, 0],
            ref_amplitude=data[:, 1],
            sample_time_ps=data[:, 2],
            sample_amplitude=data[:, 3],
        )

    if n_cols >= 12:
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
    """Load an FFT-output file (9-col or 10-col layouts)."""
    lines = _read_lines(filepath)
    delim = _detect_delimiter(lines)
    n_hdr = _count_header_rows(lines, delim)
    _, data = _parse_data_block(lines, delim, n_hdr)
    n_cols = data.shape[1]
    logger.info(f"  FFT: {os.path.basename(filepath)} -> "
                f"{data.shape[0]} rows x {n_cols} cols, {n_hdr} header row(s)")

    if n_cols == 10:
        ref_freq, converted = _normalise_frequency(data[:, 0])
        sam_freq, _ = _normalise_frequency(data[:, 5])
        if converted:
            logger.info("    Converted frequencies from Hz -> THz")
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
        ref_freq, converted = _normalise_frequency(data[:, 0])
        sam_freq, _ = _normalise_frequency(data[:, 3])
        if converted:
            logger.info("    Converted frequencies from Hz -> THz")
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
    """Load an optical-constants file with flexible column mapping."""
    lines = _read_lines(filepath)
    delim = _detect_delimiter(lines)
    n_hdr = _count_header_rows(lines, delim)
    header_lines, data = _parse_data_block(lines, delim, n_hdr)
    n_cols = data.shape[1] if data.ndim == 2 else 0
    logger.info(f"  Optical constants: {os.path.basename(filepath)} -> "
                f"{data.shape[0]} rows x {n_cols} cols, {n_hdr} header row(s)")

    if header_lines:
        raw_labels = _split_line(header_lines[-1], delim)
    else:
        raw_labels = [f"col{i}" for i in range(n_cols)]

    col_labels = [_normalise_label(l) for l in raw_labels]
    col_map = _map_oc_columns(col_labels)
    logger.info(f"    Column mapping: {col_map}")

    n_rows = data.shape[0]

    def _get_col(name: str) -> np.ndarray:
        idx = col_map.get(name)
        if idx is not None and idx < n_cols:
            return data[:, idx]
        return np.full(n_rows, np.nan)

    freq, converted = _normalise_frequency(_get_col("frequency"))
    if converted:
        logger.info("    Converted frequencies from Hz -> THz")

    eps_infty: Optional[float] = None
    eps_inf_col = _get_col("eps_infty")
    valid_ei = eps_inf_col[np.isfinite(eps_inf_col)]
    if len(valid_ei) > 0:
        eps_infty = float(valid_ei[0])
        logger.info(f"    eps_infty = {eps_infty}")

    return OpticalConstants(
        frequency_thz=freq,
        n=_get_col("n"),
        k=_get_col("k"),
        eps1=_get_col("eps1"),
        eps2=_get_col("eps2"),
        sigma_re=_get_col("sigma_re"),
        sigma_im=_get_col("sigma_im"),
        eps_infty=eps_infty,
    )


# ---------------------------------------------------------------------------
# Optical-constants header sniffing
# ---------------------------------------------------------------------------

_OC_LABEL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("frequency", re.compile(r'freq', re.I)),
    ("n",         re.compile(r'^n$|refrac', re.I)),
    ("k",         re.compile(r'^k$|extinct', re.I)),
    ("eps1",      re.compile(r'eps.*1|ε.*1|e1|real.*dielec', re.I)),
    ("eps2",      re.compile(r'eps.*2|ε.*2|e2|imag.*dielec', re.I)),
    ("sigma_re",  re.compile(r'σ\s*re|sigma.*re|σ_?re|sre|real.*cond', re.I)),
    ("sigma_im",  re.compile(r'σ\s*im|sigma.*im|σ_?im|sim|imag.*cond', re.I)),
    ("eps_infty", re.compile(r'eps.*inf|ε.*inf|infty|static', re.I)),
]


def _normalise_label(raw: str) -> str:
    """Strip whitespace, units in parentheses, and common decoration."""
    s = raw.strip()
    s = re.sub(r'\(.*?\)', '', s).strip()
    return s


def _map_oc_columns(labels: list[str]) -> dict[str, int]:
    """Map canonical field names to column indices using header sniffing."""
    mapping: dict[str, int] = {}
    used_indices: set[int] = set()
    for canonical, pattern in _OC_LABEL_PATTERNS:
        for idx, lbl in enumerate(labels):
            if idx in used_indices:
                continue
            if pattern.search(lbl):
                mapping[canonical] = idx
                used_indices.add(idx)
                break
    return mapping


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

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

    if dataset.time_domain is not None:
        td = dataset.time_domain
        _check_monotonic(td.ref_time_ps, f"[{name}] TD ref time", w)
        _check_monotonic(td.sample_time_ps, f"[{name}] TD sample time", w)
        _check_range(td.ref_time_ps, f"[{name}] TD ref time (ps)", 0, 500, w)
        _check_range(td.sample_time_ps, f"[{name}] TD sample time (ps)", 0, 500, w)

    if dataset.fft is not None:
        fft = dataset.fft
        _check_monotonic(fft.ref_frequency_thz, f"[{name}] FFT ref freq", w)
        _check_monotonic(fft.sample_frequency_thz, f"[{name}] FFT sample freq", w)
        _check_range(fft.ref_frequency_thz, f"[{name}] FFT ref freq (THz)", 0, 20, w)
        _check_range(fft.sample_frequency_thz, f"[{name}] FFT sample freq (THz)", 0, 20, w)
        _check_positive(fft.ref_amplitude, f"[{name}] FFT ref amplitude", w)
        _check_positive(fft.sample_amplitude, f"[{name}] FFT sample amplitude", w)

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


# ---------------------------------------------------------------------------
# Discovery and orchestration
# ---------------------------------------------------------------------------

def discover_series(
    data_dir: str,
    exclude: set[str] | None = None,
) -> list[str]:
    """Find series names from sub-directories of *data_dir*.

    Parameters
    ----------
    data_dir:
        Parent directory containing one sub-folder per series.
    exclude:
        Optional set of directory names to skip (e.g. ``{"original"}``).  
        Defaults to ``{"original"}`` when *None*.
    """
    if exclude is None:
        exclude = {"original"}
    return sorted(
        entry for entry in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, entry)) and entry not in exclude
    )


# Backward-compatible alias
def discover_people(data_dir: str) -> list[str]:
    return discover_series(data_dir)


def _find_file(directory: str, pattern: str) -> Optional[str]:
    """Return path of first file in *directory* whose name matches *pattern*."""
    for fname in os.listdir(directory):
        if re.search(pattern, fname, re.IGNORECASE):
            return os.path.join(directory, fname)
    return None


def load_series_folder(
    name: str,
    folder_path: str,
    raw_ref: RawMeasurement | None = None,
    raw_sam: RawMeasurement | None = None,
) -> AnalysisDataset:
    """Load all recognised data files from a single series folder.

    Scans *folder_path* for time-domain, FFT, and optical-constants files
    matching common naming conventions.  All parameters besides *name* and
    *folder_path* are optional — the result will simply have ``None`` for
    any data category that was not found.
    """
    logger.info(f"Loading dataset for '{name}'")

    td = fft_data = oc = None

    td_file = _find_file(folder_path, r'time.?domain')
    if td_file:
        td = load_time_domain(td_file)
    else:
        logger.info(f"  No time-domain file found for {name}")

    fft_file = _find_file(folder_path, r'fft')
    if fft_file:
        fft_data = load_fft(fft_file)
    else:
        logger.info(f"  No FFT file found for {name}")

    oc_file = _find_file(folder_path, r'optical.?constant')
    if oc_file:
        oc = load_optical_constants(oc_file)
    else:
        logger.info(f"  No optical-constants file found for {name}")

    return AnalysisDataset(
        name=name,
        raw_reference=raw_ref,
        raw_sample=raw_sam,
        time_domain=td,
        fft=fft_data,
        optical_constants=oc,
    )


def load_directory(
    data_dir: str,
    exclude: set[str] | None = None,
    raw_dir: str | None = None,
) -> list[AnalysisDataset]:
    """Discover series sub-folders in *data_dir* and load each one.

    Parameters
    ----------
    data_dir:
        Top-level directory containing one sub-folder per series.
    exclude:
        Directory names to skip (passed to :func:`discover_series`).
    raw_dir:
        Optional path to a folder containing shared reference / sample raw
        files (e.g. a pre-existing ``original/`` folder).  If provided,
        files matching ``reference`` and ``sample`` are loaded and attached
        to every series.
    """
    raw_ref = raw_sam = None
    if raw_dir and os.path.isdir(raw_dir):
        ref_file = _find_file(raw_dir, r'reference')
        sam_file = _find_file(raw_dir, r'sample')
        if ref_file:
            raw_ref = load_two_column(ref_file)
            logger.info(f"Raw reference: {len(raw_ref.time_ps)} pts")
        if sam_file:
            raw_sam = load_two_column(sam_file)
            logger.info(f"Raw sample:    {len(raw_sam.time_ps)} pts")

    series_names = discover_series(data_dir, exclude=exclude)
    logger.info(f"Discovered series: {series_names}")

    datasets = []
    for name in series_names:
        folder = os.path.join(data_dir, name)
        ds = load_series_folder(name, folder, raw_ref=raw_ref, raw_sam=raw_sam)
        datasets.append(ds)

    return datasets
