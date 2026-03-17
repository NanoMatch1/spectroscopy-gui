"""Standalone THz acquisition editor.

A self-contained tool for interactively reviewing and cleaning THz
time-domain data.  Works with ``.acc`` files without requiring the full
``thz`` library.

Quick-start
-----------
>>> import acquisition_editor
>>> acquisition_editor.process_directory(r"C:\\data\\my_measurements")

This will:

1. Scan the directory for ``.acc`` files.
2. Open an interactive matplotlib window for each file so you can
   exclude bad acquisitions and patch individual points.
3. Save the cleaned data into an ``export/`` sub-folder as both
   ``.acc`` (kept scans only) and ``.dat`` (averaged) files.

Public API
----------
process_directory(directory, *, export_folder, page_size)
    Main batch workflow — load, edit, export.
load_file(filepath)
    Load a single ``.acc`` file → dict.
edit_data(data_dict, *, page_size)
    Open the interactive editor on an already-loaded dict.
save_acc(data_dict, dest_path)
    Write the kept scans back as an ``.acc`` file.
save_dat(data_dict, dest_path)
    Write the averaged trace as a ``.dat`` file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np

from acquisition_editor._loaders import (
    SUPPORTED_EXTENSIONS,
    load_file,
)
from acquisition_editor._editor import edit_acquisitions

__all__ = [
    "process_directory",
    "load_file",
    "edit_data",
    "save_acc",
    "save_dat",
    "SUPPORTED_EXTENSIONS",
]

# Number format matching the instrument software output.
_NUM_FMT = "%.18e"
_NEWLINE = "\r\n"


# ---------------------------------------------------------------------------
# Editing wrapper that operates on the dict structure
# ---------------------------------------------------------------------------

def edit_data(
    data_dict: dict,
    *,
    page_size: int = 25,
) -> dict:
    """Open the interactive editor for a single loaded file dict.

    Parameters
    ----------
    data_dict
        As returned by :func:`load_file`.
    page_size
        Number of items per list page in the editor UI.

    Returns
    -------
    dict
        Same structure with ``data`` replaced by the edited array.
        ``scan_headers`` is trimmed to match the kept scans.
        If the user closes the window without saving, the original data
        is returned unchanged.
    """
    edited_array = edit_acquisitions(
        data_dict["data"],
        filename=data_dict.get("filename", "array"),
        page_size=page_size,
    )

    # Determine which scans survived (edit_acquisitions drops excluded cols).
    n_kept = edited_array.shape[1] - 1 if edited_array.ndim == 2 else 0
    original_headers = data_dict.get("scan_headers", [])

    # If the editor returned fewer scans, we can't know exactly which were
    # kept (indices aren't tracked), so we keep the first n_kept headers.
    # For most workflows the headers only differ in scan number + timestamp,
    # and we renumber on export anyway.
    kept_headers = original_headers[:n_kept] if original_headers else []

    return {
        "filename": data_dict["filename"],
        "header": data_dict["header"],
        "scan_headers": kept_headers,
        "data": edited_array,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_data_row(row: np.ndarray) -> str:
    """Format one row of numeric data exactly like the instrument output."""
    return " ".join(_NUM_FMT % v for v in row)


def _build_scan_block(
    header_lines: list[str],
    time: np.ndarray,
    signal: np.ndarray,
) -> str:
    """Build one scan block (header + data rows) as a string."""
    lines: list[str] = []
    for h in header_lines:
        lines.append(f"%{h}")
    for i in range(len(time)):
        lines.append(_format_data_row(np.array([time[i], signal[i]])))
    return _NEWLINE.join(lines)


# ---------------------------------------------------------------------------
# Exporting / saving
# ---------------------------------------------------------------------------

def save_acc(
    data_dict: dict,
    dest_path: Union[str, Path],
) -> Path:
    """Write the kept scans as an ``.acc`` file matching instrument format.

    Scans are renumbered 1…N and separated by ``%%``.
    Uses ``\\r\\n`` line endings and ``%.18e`` number format.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    data = data_dict["data"]
    scan_headers = data_dict.get("scan_headers", [])
    n_scans = data.shape[1] - 1 if data.ndim == 2 and data.shape[1] > 1 else 0
    time = data[:, 0]

    blocks: list[str] = []
    for scan_idx in range(n_scans):
        signal = data[:, scan_idx + 1]

        # Use existing per-scan header if available, otherwise fall back to
        # the first-scan header.
        if scan_idx < len(scan_headers):
            hdr = list(scan_headers[scan_idx])
        elif scan_headers:
            hdr = list(scan_headers[0])
        else:
            hdr = list(data_dict.get("header", []))

        # Renumber the title line: "title <name> acc <N>"
        for i, line in enumerate(hdr):
            if line.lower().startswith("title "):
                parts = line.rsplit(" acc ", 1)
                hdr[i] = f"{parts[0]} acc {scan_idx + 1}"
                break

        blocks.append(_build_scan_block(hdr, time, signal))

    content = (_NEWLINE + "%%" + _NEWLINE).join(blocks) + _NEWLINE
    dest_path.write_bytes(content.encode("utf-8"))
    return dest_path


def save_dat(
    data_dict: dict,
    dest_path: Union[str, Path],
) -> Path:
    """Write the averaged trace as a ``.dat`` file matching instrument format.

    The header is based on the first scan's header with the ``acc N`` suffix
    stripped from the title and an ``accumulations`` param added.
    Uses ``\\r\\n`` line endings and ``%.18e`` number format.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    data = data_dict["data"]
    time = data[:, 0]
    n_scans = data.shape[1] - 1 if data.ndim == 2 and data.shape[1] > 1 else 0

    if n_scans > 0:
        mean_signal = np.mean(data[:, 1:], axis=1)
    else:
        mean_signal = np.zeros_like(time)

    # Build header from the first scan's header
    hdr = list(data_dict.get("header", []))

    # Strip " acc N" from the title line
    for i, line in enumerate(hdr):
        if line.lower().startswith("title "):
            parts = line.rsplit(" acc ", 1)
            hdr[i] = parts[0]
            break

    # Add accumulations param if not already present
    has_accum = any("accumulations" in h.lower() for h in hdr)
    if not has_accum:
        hdr.append(f"param accumulations,{n_scans}")

    lines: list[str] = []
    for h in hdr:
        lines.append(f"%{h}")
    for i in range(len(time)):
        lines.append(_format_data_row(np.array([time[i], mean_signal[i]])))

    content = _NEWLINE.join(lines) + _NEWLINE
    dest_path.write_bytes(content.encode("utf-8"))
    return dest_path


# ---------------------------------------------------------------------------
# Batch directory workflow
# ---------------------------------------------------------------------------

def process_directory(
    directory: Union[str, Path],
    *,
    export_folder: str = "export",
    page_size: int = 25,
) -> list[dict]:
    """Load every ``.acc`` file, open the editor, then save to *export_folder*.

    For each file two outputs are written:

    - ``<name>.acc`` — the kept (edited) scans in full ``.acc`` format.
    - ``<name>.dat`` — the average of the kept scans.

    Parameters
    ----------
    directory
        Path to the folder containing raw ``.acc`` data files.
    export_folder
        Name of the sub-folder (inside *directory*) where edited files are
        written.  Created automatically if it doesn't exist.
    page_size
        Number of items per list page in the editor UI.

    Returns
    -------
    list[dict]
        List of the edited data dicts (one per file).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() == ".acc"
    )

    if not files:
        print(f"No .acc files found in {directory}")
        return []

    export_dir = directory / export_folder
    export_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for filepath in files:
        print(f"\n{'='*60}")
        print(f"Loading: {filepath.name}")
        print(f"{'='*60}")

        data_dict = load_file(filepath)
        n_acq = max(0, data_dict["data"].shape[1] - 1) if data_dict["data"].ndim == 2 else 0
        print(f"  {data_dict['data'].shape[0]} points, {n_acq} acquisition(s)")

        edited = edit_data(data_dict, page_size=page_size)

        n_kept = max(0, edited["data"].shape[1] - 1) if edited["data"].ndim == 2 else 0
        print(f"  Kept {n_kept} of {n_acq} acquisition(s)")

        # Export .acc (kept scans)
        acc_path = export_dir / filepath.name
        save_acc(edited, acc_path)
        print(f"  Saved .acc → {acc_path}")

        # Export .dat (average)
        dat_path = export_dir / (filepath.stem + ".dat")
        save_dat(edited, dat_path)
        print(f"  Saved .dat → {dat_path}")

        results.append(edited)

    print(f"\nDone. {len(results)} file(s) processed → {export_dir}")
    return results
