"""Loader for .csv files — delimiter-sniffing numeric loader.

Uses Python's CSV sniffer to detect the delimiter, then parses all
numeric rows into a :class:`~matchbook.data_structures.spectrum.Spectrum`.
"""

from __future__ import annotations

import csv
from os import path
from typing import List, Optional, Tuple

import numpy as np

from matchbook.data_structures.spectrum import Spectrum
from matchbook.io.loaders.registry import BaseLoader, register_loader


def _sniff_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except Exception:
        candidates = [",", "\t", ";", " "]
        counts = {d: sample.count(d) for d in candidates}
        return max(counts, key=counts.get)


def _parse_text_to_numeric(
    lines: List[str], delimiter: Optional[str],
) -> Tuple[np.ndarray, List[str]]:
    numeric_rows: List[List[float]] = []
    headers: List[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if delimiter == " ":
            tokens = line.split()
        else:
            reader = csv.reader([line], delimiter=delimiter)
            tokens = next(reader)

        try:
            row = [float(tok) for tok in tokens if tok != ""]
            if row:
                numeric_rows.append(row)
        except Exception:
            headers.append(raw)

    if not numeric_rows:
        return np.empty((0, 2), dtype=float), headers

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
        sample = "\n".join(raw.splitlines()[:20])
        delim = _sniff_delimiter(sample)
        data, headers = _parse_text_to_numeric(raw.splitlines(), delim)
        return Spectrum(
            data=data,
            headers=headers,
            filename=self.filename,
            data_type="csv",
        )
