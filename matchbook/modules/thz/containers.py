"""Data container classes for THz time-domain spectroscopy datasets.

``BaseTHzData`` holds a single scan; ``THzData`` holds multiple scans and
provides averaged data access and statistical helpers.

Processing methods (FFT, baseline, windowing, padding, etc.) are
intentionally excluded — they belong in separate analysis packages.

No pandas dependency — all data is stored as numpy arrays.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Statistical helper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeDomainStats:
    time: np.ndarray                              # (N_time,)
    mean: np.ndarray                              # (N_time,)
    std: np.ndarray                               # (N_time,)
    stderr: np.ndarray                            # (N_time,)
    n_repeats: int
    baseline_sigma: Optional[float]               # single-number sigma
    snr_from_baseline: Optional[np.ndarray]       # (N_time,)


def _as_slice(
    idx: Union[slice, np.ndarray, list, tuple, None], n: int,
) -> Union[slice, np.ndarray]:
    if idx is None:
        return slice(None)

    if isinstance(idx, slice):
        start = 0 if idx.start is None else idx.start
        stop = n if idx.stop is None else idx.stop
        if start < 0 or stop < 0 or start > n or stop > n or start >= stop:
            raise ValueError(f"Invalid baseline_slice={idx} for length {n}.")
        return idx

    arr = np.asarray(idx)
    if arr.dtype == bool and arr.shape != (n,):
        raise ValueError(
            f"Boolean baseline mask must have shape {(n,)}, got {arr.shape}."
        )
    return arr


def calculate_time_domain_stats(
    raw_data: np.ndarray,
    *,
    limit: Optional[int] = None,
    ddof: int = 1,
    baseline_slice: Union[slice, np.ndarray, list, tuple, None] = None,
    compute_baseline_snr: bool = False,
) -> TimeDomainStats:
    """Compute per-timepoint mean/std/stderr across repeated scans.

    Parameters
    ----------
    raw_data : ndarray, shape (N_time, 1 + N_repeats)
        Column 0 is the time axis; columns 1… are repeated traces.
    """
    if raw_data is None or raw_data.size == 0:
        empty = np.array([])
        return TimeDomainStats(
            time=empty, mean=empty, std=empty, stderr=empty,
            n_repeats=0, baseline_sigma=None, snr_from_baseline=None,
        )

    if raw_data.ndim != 2 or raw_data.shape[1] < 2:
        raise ValueError(
            f"raw_data must be 2D with at least 2 columns. Got {raw_data.shape}."
        )

    time = raw_data[:, 0]
    traces = raw_data[:, 1:]

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be a positive integer.")
        traces = traces[:, :limit]

    n_time, n_repeats = traces.shape
    if n_repeats < 1:
        raise ValueError("No repeat traces available after slicing.")

    mean = np.mean(traces, axis=1)
    ddof_eff = ddof if n_repeats > 1 else 0
    std = np.std(traces, axis=1, ddof=ddof_eff)
    stderr = std / np.sqrt(n_repeats)

    baseline_sigma = None
    snr = None

    if compute_baseline_snr:
        bidx = _as_slice(baseline_slice, n_time)
        baseline_vals = mean[bidx]
        if baseline_vals.size < 2:
            raise ValueError(
                "Baseline region must include at least 2 points."
            )
        baseline_sigma = float(np.std(baseline_vals, ddof=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.where(
                baseline_sigma > 0, np.abs(mean) / baseline_sigma, 0.0
            )

    return TimeDomainStats(
        time=time,
        mean=mean,
        std=std,
        stderr=stderr,
        n_repeats=n_repeats,
        baseline_sigma=baseline_sigma,
        snr_from_baseline=snr,
    )


# ---------------------------------------------------------------------------
# Single-scan container
# ---------------------------------------------------------------------------

class BaseTHzData:
    """Holds one scan and metadata information."""

    def __init__(self, data: np.ndarray, headers: list) -> None:
        self.raw_data = data
        self.headers = headers
        self.filename, self.scan_index = self._resolve_filename()
        self.timestamp = self._resolve_timestamp()

    def __repr__(self) -> str:
        return (
            f"<BaseTHzData:{self.filename}, scan_{self.scan_index}, "
            f"timestamp:{self.timestamp}>"
        )

    def _compress_data(self) -> None:
        """Drop the redundant X-axis column to save memory."""
        self.raw_data = self.raw_data[:, 1]
        self.headers = None

    def _resolve_filename(self) -> tuple:
        for item in self.headers:
            if "title" in item.lower():
                parts = item.split(" ")
                title = parts[1] if len(parts) > 1 else "unknown_file"
                scan_index = int(parts[4]) if len(parts) > 4 else None
                return title, scan_index
        return "unknown_file", None

    def _resolve_timestamp(self) -> datetime.datetime | str:
        for item in self.headers:
            if "date" in item.lower() and "time" in item.lower():
                parts = item.split(",")
                if len(parts) > 1:
                    timestr = parts[1].split(".")[0].strip()
                    try:
                        return datetime.datetime.strptime(
                            timestr, "%Y-%m-%d %H:%M:%S"
                        )
                    except ValueError:
                        pass
        return "unknown_timestamp"


# ---------------------------------------------------------------------------
# Multi-scan container
# ---------------------------------------------------------------------------

class THzData:
    """Data container for THz time-domain spectroscopy measurements.

    Holds multiple :class:`BaseTHzData` scan objects and provides averaged
    data access and basic statistical helpers.
    """

    def __init__(self, data: list, header: list | None, **kwargs) -> None:
        self.data_list = data
        self.raw_data = self._compile_data_array()
        self.headers = (
            header if header is not None else self._grabonedata().headers
        )
        self.data_type = kwargs.get("data_type", None)
        self.filename = kwargs.get("filename", "unknown_file")

        self._meta_data: dict = {}
        self.processing_dict: dict = {}

        self._time_data = self._average_data()
        self._data = self._time_data

    def __repr__(self) -> str:
        return (
            f"\nTHzData:{self.filename}\n"
            f"   -> Scans: {len(self.data_list)}\n"
            f"   -> Data type: {self.data_type}\n"
        )

    # -- Properties --

    @property
    def type(self) -> tuple:
        return self.__class__, self.data_type

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def time(self) -> np.ndarray | None:
        if self._data is not None:
            return self._data[:, 0]
        return None

    @property
    def y_mean(self) -> np.ndarray | None:
        if self._data is not None:
            return self._data[:, 1]
        return None

    @property
    def y_err(self) -> np.ndarray | None:
        if self._data is not None:
            return self._data[:, 2]
        return None

    @property
    def time_const(self) -> float | None:
        tc = self._meta_data.get("time_constant", None)
        if tc is None:
            tc = self._identify_time_constant()
        return tc

    # -- Data manipulation --

    def update_data(self, new_data: np.ndarray) -> None:
        """Re-average after an external modification (e.g. acquisition editor)."""
        self.raw_data = new_data
        self._time_data = self._average_data()
        self._data = self._time_data

    def average_data(self, limit: int | None = None) -> np.ndarray:
        """Average the data across a chosen number of scans."""
        data_matrix = np.array(
            [obj.raw_data[:, 1] for obj in self.data_list[:limit]]
        )
        std_error = np.std(data_matrix, axis=0) / np.sqrt(data_matrix.shape[0])
        mean_data = np.mean(data_matrix, axis=0)
        time_axis = self.data_list[0].raw_data[:, 0]
        return np.column_stack((time_axis, mean_data, std_error))

    # -- Internal helpers --

    def _compile_data_array(self, limit: int | None = None) -> np.ndarray:
        compiled = None
        for idx, obj in enumerate(self.data_list):
            if limit is not None and idx >= limit:
                break
            if compiled is None:
                compiled = obj.raw_data
            else:
                compiled = np.column_stack((compiled, obj.raw_data[:, 1]))
        return compiled

    def _grabonedata(self, index: int = 0) -> BaseTHzData:
        return self.data_list[index]

    def _compress_dataset(self) -> None:
        for obj in self.data_list:
            obj._compress_data()

    def _calculate_std_error(self) -> np.ndarray:
        data_matrix = np.array(
            [obj.raw_data[:, 1] for obj in self.data_list]
        )
        return np.std(data_matrix, axis=0) / np.sqrt(len(self.data_list))

    def _average_data(self) -> np.ndarray:
        if self.raw_data.shape[1] < 3:
            time_axis = self.raw_data[:, 0]
            mean_data = self.raw_data[:, 1]
            std_error = np.zeros_like(mean_data)
            averaged = np.column_stack((time_axis, mean_data, std_error))
        else:
            data_matrix = np.array(
                [obj.raw_data[:, 1] for obj in self.data_list]
            )
            std_error = np.std(data_matrix, axis=0) / np.sqrt(
                len(self.data_list)
            )
            mean_data = np.mean(data_matrix, axis=0)
            time_axis = self.data_list[0].raw_data[:, 0]
            averaged = np.column_stack((time_axis, mean_data, std_error))

        self.processing_dict["time_domain"] = averaged.copy()
        return averaged

    def _identify_time_constant(self, time_unit: str = "ms") -> float | None:
        if "time" in self.filename.lower() and "const" in self.filename.lower():
            stritem = self.filename.split(time_unit)[0].strip()
            stritem = stritem[::-1]
            tc_str = ""
            for char in stritem:
                if char.isdigit() or char == ".":
                    tc_str = char + tc_str
                else:
                    break
            try:
                tc = float(tc_str)
                self._meta_data["time_constant"] = tc
            except ValueError:
                self._meta_data["time_constant"] = None

        return self._meta_data.get("time_constant", None)
