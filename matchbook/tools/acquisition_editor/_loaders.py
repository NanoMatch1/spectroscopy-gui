"""Minimal standalone loader for .acc THz files.

Returns a dict::

    {
        "filename":     str,
        "header":       list[str],
        "scan_headers": list[list[str]],
        "data":         np.ndarray,   # (N_points, 1 + N_scans)
    }

Used by the acquisition editor to load raw scan data without requiring
the full loader registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


SUPPORTED_EXTENSIONS = (".acc",)


def _parse_rows(rows: list[str]) -> np.ndarray:
    """Parse space-separated numeric rows into a 2-D float array."""
    numeric = []
    for row in rows:
        parts = row.split()
        if not parts:
            continue
        try:
            numeric.append([float(v) for v in parts])
        except ValueError:
            continue
    if not numeric:
        return np.empty((0, 0))
    return np.array(numeric, dtype=np.float64)


def _split_header_and_data(text: str) -> tuple[list[str], list[str]]:
    header: list[str] = []
    data: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("%"):
            header.append(line.lstrip("%").strip())
        else:
            data.append(line)
    return header, data


def load_acc(filepath: Union[str, Path]) -> dict:
    """Load an ``.acc`` file and return a standardised dict.

    Multiple scans (delimited by ``%%``) are combined so that ``data``
    has shape ``(N_points, 1 + N_scans)``.
    """
    filepath = Path(filepath)
    raw_text = filepath.read_text(encoding="utf-8")

    blocks = raw_text.split("%%")

    scan_headers: list[list[str]] = []
    signal_columns: list[np.ndarray] = []
    time_axis: np.ndarray | None = None

    for block in blocks:
        header, rows = _split_header_and_data(block)
        if not rows:
            continue
        arr = _parse_rows(rows)
        if arr.size == 0:
            continue

        scan_headers.append(header)

        if time_axis is None:
            time_axis = arr[:, 0]

        if arr.shape[1] >= 2:
            for col_idx in range(1, arr.shape[1]):
                signal_columns.append(arr[:, col_idx])

    if time_axis is None:
        return {
            "filename": filepath.name,
            "header": [],
            "scan_headers": [],
            "data": np.empty((0, 0)),
        }

    data = np.column_stack([time_axis] + signal_columns)

    return {
        "filename": filepath.name,
        "header": scan_headers[0] if scan_headers else [],
        "scan_headers": scan_headers,
        "data": data,
    }


def load_file(filepath: Union[str, Path]) -> dict:
    """Load an ``.acc`` file.

    Raises ``ValueError`` if the extension is not ``.acc``.
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()
    if ext != ".acc":
        raise ValueError(
            f"Unsupported extension '{ext}'. Only .acc files are supported."
        )
    return load_acc(filepath)
