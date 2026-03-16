"""Loader for .acc files — multi-scan THz acquisition data.

.acc files are text documents containing multiple scans (unaveraged).
Headers are indicated by leading ``%`` symbols; scans are separated by ``%%``.
"""

from os import path

import numpy as np

from matchbook.io.loaders.registry import BaseLoader, register_loader
from matchbook.modules.thz.containers import BaseTHzData, THzData


@register_loader
class ACCLoader(BaseLoader):

    extension = ".acc"

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.filename = path.basename(filepath)
        self.errors: list[str] = []

    def _simple_load(self) -> str:
        with open(self.filepath, "r") as fh:
            return fh.read()

    def _parse_data(self, spectrum: list[str]) -> np.ndarray:
        numeric_data: list[list[float]] = []
        for row in spectrum:
            try:
                numeric_row = [float(v) for v in row.split()]
                numeric_data.append(numeric_row)
            except ValueError:
                self.errors.append(f"Could not parse row: {row}")
        try:
            return np.array(numeric_data)
        except Exception as exc:
            self.errors.append(f"Could not convert data to numpy array: {exc}")
            return np.empty((0, 0))

    def _simple_split(self, raw_data: str) -> dict:
        scans = raw_data.split("%%")
        scan_dict: dict = {}

        for index, item in enumerate(scans):
            header: list[str] = []
            spectrum: list[str] = []
            for row in item.split("\n"):
                row = row.strip()
                if row.startswith("%"):
                    header.append(row.strip("%").strip())
                elif row:
                    spectrum.append(row)
            scan_dict[f"scan_{index}"] = {
                "header": header,
                "spectrum": spectrum,
            }
        return scan_dict

    def load(self) -> THzData:
        new_data: list[BaseTHzData] = []
        raw_data = self._simple_load()
        parsed = self._simple_split(raw_data)
        for value in parsed.values():
            data = self._parse_data(value["spectrum"])
            new_data.append(BaseTHzData(data=data, headers=value["header"]))
        return THzData(
            data=new_data,
            header=None,
            filename=self.filename,
            data_type="acc",
        )
