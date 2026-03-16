"""Export AnalysisDataset objects to CSV and verify roundtrip fidelity."""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Optional

import numpy as np

from models import (
    AnalysisDataset,
    FFTData,
    OpticalConstants,
    RawMeasurement,
    TimeDomainData,
)

logger = logging.getLogger(__name__)

# ── CSV helpers ──────────────────────────────────────────────────────────────


def _format_val(v: float) -> str:
    """Format a float for CSV output.  NaN / inf → empty string."""
    if not np.isfinite(v):
        return ''
    return repr(float(v))


def _write_csv(filepath: str, headers: list[str], data: np.ndarray):
    with open(filepath, 'w', newline='') as f:
        f.write(','.join(headers) + '\n')
        for row in data:
            f.write(','.join(_format_val(v) for v in row) + '\n')
    logger.info(f"  Wrote {os.path.basename(filepath)}  "
                f"({data.shape[0]} rows x {data.shape[1]} cols)")


def _read_csv(filepath: str) -> tuple[list[str], np.ndarray]:
    """Read back a CSV produced by _write_csv."""
    with open(filepath, 'r') as f:
        lines = [l.rstrip('\n\r') for l in f if l.strip()]
    headers = lines[0].split(',')
    rows: list[list[float]] = []
    for line in lines[1:]:
        parts = line.split(',')
        rows.append([float(p) if p != '' else float('nan') for p in parts])
    return headers, np.array(rows, dtype=np.float64)


def _pad_to_same_length(*arrays: Optional[np.ndarray]) -> list[np.ndarray]:
    """Pad shorter arrays with NaN so that all share the longest length."""
    lengths = [len(a) for a in arrays if a is not None]
    if not lengths:
        return [np.empty(0) for _ in arrays]
    max_len = max(lengths)
    result: list[np.ndarray] = []
    for a in arrays:
        if a is None:
            result.append(np.full(max_len, np.nan))
        elif len(a) < max_len:
            padded = np.full(max_len, np.nan)
            padded[:len(a)] = a
            result.append(padded)
        else:
            result.append(a)
    return result


def _arrays_match(a: np.ndarray, b: np.ndarray, tol: float = 1e-10) -> tuple[bool, str]:
    """Element-wise comparison treating NaN == NaN.  Returns (ok, detail)."""
    if a.shape != b.shape:
        return False, f"shape mismatch {a.shape} vs {b.shape}"
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if not np.array_equal(nan_a, nan_b):
        n = int(np.sum(nan_a != nan_b))
        return False, f"NaN pattern differs at {n} positions"
    valid = ~nan_a
    if np.any(valid):
        max_diff = float(np.max(np.abs(a[valid] - b[valid])))
        if max_diff > tol:
            return False, f"max |diff| = {max_diff:.3e}"
    return True, "OK"


# ── Column builders (shared by export and verify) ───────────────────────────

def _raw_columns(meas: RawMeasurement) -> OrderedDict:
    return OrderedDict([('time_ps', meas.time_ps), ('amplitude', meas.amplitude)])


def _td_columns(td: TimeDomainData) -> OrderedDict:
    cols = OrderedDict()
    cols['ref_time_ps'] = td.ref_time_ps
    cols['ref_amplitude'] = td.ref_amplitude
    cols['sample_time_ps'] = td.sample_time_ps
    cols['sample_amplitude'] = td.sample_amplitude
    for name, arr in [
        ('ref_std_error', td.ref_std_error),
        ('sample_std_error', td.sample_std_error),
        ('raw_ref_time_ps', td.raw_ref_time_ps),
        ('raw_ref_amplitude', td.raw_ref_amplitude),
        ('raw_ref_std_error', td.raw_ref_std_error),
        ('raw_sample_time_ps', td.raw_sample_time_ps),
        ('raw_sample_amplitude', td.raw_sample_amplitude),
        ('raw_sample_std_error', td.raw_sample_std_error),
    ]:
        if arr is not None:
            cols[name] = arr
    return cols


def _fft_columns(fft: FFTData) -> OrderedDict:
    cols = OrderedDict()
    cols['ref_frequency_thz'] = fft.ref_frequency_thz
    cols['ref_amplitude'] = fft.ref_amplitude
    cols['ref_phase'] = fft.ref_phase
    cols['sample_frequency_thz'] = fft.sample_frequency_thz
    cols['sample_amplitude'] = fft.sample_amplitude
    cols['sample_phase'] = fft.sample_phase
    for name, arr in [
        ('ref_delta_amplitude', fft.ref_delta_amplitude),
        ('ref_delta_phase', fft.ref_delta_phase),
        ('sample_delta_amplitude', fft.sample_delta_amplitude),
        ('sample_delta_phase', fft.sample_delta_phase),
        ('phase_offset', fft.phase_offset),
        ('phase_diff', fft.phase_diff),
    ]:
        if arr is not None:
            cols[name] = arr
    return cols


def _oc_columns(oc: OpticalConstants) -> OrderedDict:
    return OrderedDict([
        ('frequency_thz', oc.frequency_thz),
        ('n', oc.n),
        ('k', oc.k),
        ('eps1', oc.eps1),
        ('eps2', oc.eps2),
        ('sigma_re', oc.sigma_re),
        ('sigma_im', oc.sigma_im),
    ])


def _build_csv_data(cols: OrderedDict) -> tuple[list[str], np.ndarray]:
    """Convert an ordered dict of column arrays → (headers, 2d array)."""
    headers = list(cols.keys())
    padded = _pad_to_same_length(*cols.values())
    return headers, np.column_stack(padded) if padded else (headers, np.empty((0, 0)))


# ── Public export / verify API ───────────────────────────────────────────────

def export_dataset(dataset: AnalysisDataset, export_dir: str):
    """Export one person's full dataset to CSV files under *export_dir*."""
    os.makedirs(export_dir, exist_ok=True)
    name = dataset.name

    _write_csv(
        os.path.join(export_dir, f'{name}_raw_reference.csv'),
        *_build_csv_data(_raw_columns(dataset.raw_reference)),
    )
    _write_csv(
        os.path.join(export_dir, f'{name}_raw_sample.csv'),
        *_build_csv_data(_raw_columns(dataset.raw_sample)),
    )

    if dataset.time_domain is not None:
        _write_csv(
            os.path.join(export_dir, f'{name}_time-domain.csv'),
            *_build_csv_data(_td_columns(dataset.time_domain)),
        )

    if dataset.fft is not None:
        _write_csv(
            os.path.join(export_dir, f'{name}_fft.csv'),
            *_build_csv_data(_fft_columns(dataset.fft)),
        )

    if dataset.optical_constants is not None:
        _write_csv(
            os.path.join(export_dir, f'{name}_optical-constants.csv'),
            *_build_csv_data(_oc_columns(dataset.optical_constants)),
        )
        if dataset.optical_constants.eps_infty is not None:
            meta_path = os.path.join(export_dir, f'{name}_optical-constants_meta.csv')
            with open(meta_path, 'w') as f:
                f.write('parameter,value\n')
                f.write(f'eps_infty,{dataset.optical_constants.eps_infty}\n')
            logger.info(f"  Wrote {os.path.basename(meta_path)}")


def verify_roundtrip(dataset: AnalysisDataset, export_dir: str) -> list[str]:
    """Re-read exported CSVs and compare to in-memory data.  Returns issues."""
    issues: list[str] = []
    name = dataset.name

    def _check(filepath: str, cols_builder, label: str):
        if not os.path.exists(filepath):
            issues.append(f"[{name}] {label}: file missing")
            return
        expected_headers, expected_data = _build_csv_data(cols_builder())
        actual_headers, actual_data = _read_csv(filepath)
        if expected_headers != actual_headers:
            issues.append(f"[{name}] {label}: header mismatch")
            return
        ok, detail = _arrays_match(expected_data, actual_data)
        if not ok:
            issues.append(f"[{name}] {label}: {detail}")
        else:
            logger.info(f"  [{name}] {label}: roundtrip verified")

    _check(
        os.path.join(export_dir, f'{name}_raw_reference.csv'),
        lambda: _raw_columns(dataset.raw_reference),
        'raw_reference',
    )
    _check(
        os.path.join(export_dir, f'{name}_raw_sample.csv'),
        lambda: _raw_columns(dataset.raw_sample),
        'raw_sample',
    )
    if dataset.time_domain is not None:
        _check(
            os.path.join(export_dir, f'{name}_time-domain.csv'),
            lambda: _td_columns(dataset.time_domain),
            'time-domain',
        )
    if dataset.fft is not None:
        _check(
            os.path.join(export_dir, f'{name}_fft.csv'),
            lambda: _fft_columns(dataset.fft),
            'fft',
        )
    if dataset.optical_constants is not None:
        _check(
            os.path.join(export_dir, f'{name}_optical-constants.csv'),
            lambda: _oc_columns(dataset.optical_constants),
            'optical-constants',
        )

    return issues
