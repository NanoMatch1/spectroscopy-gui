"""Loader for .txt files — simple text-based numeric data.

Parses space-separated numeric rows; non-numeric lines are kept as headers.
Returns a THzData container (single scan).
"""

from os import path

import numpy as np

from matchbook.io.loaders.registry import BaseLoader, register_loader
from matchbook.modules.thz.containers import BaseTHzData, THzData


@register_loader
class TXTLoader(BaseLoader):

    extension = ".txt"

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.filename = path.basename(filepath)
        self.errors: list[str] = []

    def _simple_load(self) -> str:
        with open(self.filepath, "r") as fh:
            return fh.read()

    def _parse_data(self, filestring: str) -> tuple[np.ndarray, list[str]]:
        numeric_data: list[list[float]] = []
        headers: list[str] = []

        for row in filestring.split("\n"):
            try:
                numeric_row = [float(v) for v in row.split()]
                if numeric_row:
                    numeric_data.append(numeric_row)
            except ValueError:
                self.errors.append(f"Could not parse row: {row}")
                headers.append(row)

        try:
            return np.array(numeric_data), headers
        except Exception as exc:
            self.errors.append(f"Could not convert data to numpy array: {exc}")
            return np.empty((0, 0)), headers

    def load(self) -> THzData:
        raw_data = self._simple_load()
        data, headers = self._parse_data(raw_data)
        scan = BaseTHzData(data=data, headers=headers)
        return THzData(
            data=[scan],
            header=None,
            filename=self.filename,
            data_type="txt",
        )
