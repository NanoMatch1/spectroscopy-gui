"""Loaders for CSV style numeric files.

These return a `Spectrum` instance for simple 1D/XY data files.

Behavior:
- Uses Python's CSV sniffer to detect a delimiter for CSV files.
- Lines that cannot be parsed entirely as floats are retained in
  `headers` and returned on the `Spectrum` object.
"""

from __future__ import annotations

import csv
from os import path
from typing import List, Optional, Tuple

import numpy as np

from dataset_core.data_structures.spectrum import Spectrum
from dataset_core.io.loaders.registry import BaseLoader, register_loader


def _sniff_delimiter(sample: str) -> str:
    # Try csv.Sniffer first, fall back to common delimiters
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        return dialect.delimiter
    except Exception:
        # Heuristic fallback: choose the delimiter with the largest count
        candidates = [",", "\t", ";", " "]
        counts = {d: sample.count(d) for d in candidates}
        # Prefer comma or tab when present
        best = max(counts, key=counts.get)
        return best


def _parse_text_to_numeric(lines: List[str], delimiter: Optional[str]) -> Tuple[np.ndarray, List[str]]:
    numeric_rows: List[List[float]] = []
    headers: List[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # Choose splitting strategy
        if delimiter == " ":
            tokens = line.split()
        else:
            reader = csv.reader([line], delimiter=delimiter)
            tokens = next(reader)

        # Try converting all tokens to float
        try:
            row = [float(tok) for tok in tokens if tok != ""]
            if len(row) == 0:
                # empty after trimming
                continue
            numeric_rows.append(row)
        except Exception:
            headers.append(raw)

    if len(numeric_rows) == 0:
        numeric = np.empty((0, 2), dtype=float)
    else:
        # Pad short rows with NaN to produce a rectangular array
        maxcols = max(len(r) for r in numeric_rows)
        numeric = np.full((len(numeric_rows), maxcols), np.nan, dtype=float)
        for i, r in enumerate(numeric_rows):
            numeric[i, : len(r)] = r

    return numeric, headers


@register_loader
class CSVLoader(BaseLoader):
    extension = ".csv"

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.filename = path.basename(filepath)

    def _simple_load(self) -> str:
        with open(self.filepath, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()

    def load(self) -> Spectrum:
        raw = self._simple_load()
        # Build a sample for sniffing
        sample = "\n".join(raw.splitlines()[:20])
        delim = _sniff_delimiter(sample)
        lines = raw.splitlines()
        data, headers = _parse_text_to_numeric(lines, delim)
        return Spectrum(data=data, headers=headers, filename=self.filename, data_type="csv")
