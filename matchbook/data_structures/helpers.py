"""Utility functions for data manipulation — pure numpy, no pandas.

Provides: extremum finding, monotonicity checks, Savitzky-Golay smoothing,
simple interpolation, and multi-dataset interpolation to a common axis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Extremum finding
# ---------------------------------------------------------------------------

def find_maxima(
    data: np.ndarray,
    window: tuple,
    mode: str = "abs",
    values: str = "index",
    return_idx: bool = False,
) -> tuple | None:
    """Find an extremum (abs/max/min) within a window.

    Parameters
    ----------
    data : (N, 2) array_like
        Column 0 is x, column 1 is y.
    window : tuple
        (xmin, xmax) — meaning depends on *values*.
    mode : ``'abs'`` | ``'max'`` | ``'min'``
    values : ``'index'`` | ``'value'``
        Whether *window* refers to integer indices or x-axis values.
    return_idx : bool
        If True, return ``(idx, x0, y0)`` instead of ``(x0, y0)``.
    """
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("data must be an (N, 2) array.")

    a, b = window
    x, y = data[:, 0], data[:, 1]

    if values == "value":
        xmin, xmax = (a, b) if a <= b else (b, a)
        idxs = np.flatnonzero((x >= xmin) & (x <= xmax))
    elif values == "index":
        i0, i1 = int(a), int(b)
        if i1 < i0:
            i0, i1 = i1, i0
        i0 = max(i0, 0)
        i1 = min(i1, len(x))
        idxs = np.arange(i0, i1, dtype=int)
    else:
        raise ValueError("values must be 'index' or 'value'")

    if idxs.size == 0:
        return None

    y_sel = y[idxs]

    if mode == "abs":
        rel = int(np.argmax(np.abs(y_sel)))
    elif mode == "max":
        rel = int(np.argmax(y_sel))
    elif mode == "min":
        rel = int(np.argmin(y_sel))
    else:
        raise ValueError("mode must be 'abs', 'max', or 'min'")

    idx = int(idxs[rel])
    x0, y0 = float(x[idx]), float(y[idx])
    return (idx, x0, y0) if return_idx else (x0, y0)


# ---------------------------------------------------------------------------
# Monotonicity check
# ---------------------------------------------------------------------------

def is_monotonic(x: np.ndarray) -> bool:
    d = np.diff(x)
    return bool(np.all(d >= 0) or np.all(d <= 0))


# ---------------------------------------------------------------------------
# Savitzky-Golay smoothing
# ---------------------------------------------------------------------------

def smooth_trace_savgol(
    y: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
    mode: str = "interp",
) -> np.ndarray:
    """Smooth a 1D trace using a Savitzky-Golay filter.

    Parameters
    ----------
    y : 1D array
    window_length : int (must be odd; auto-corrected if even)
    polyorder : int (must be < window_length)
    mode : str — boundary handling mode for ``savgol_filter``
    """
    y = np.asarray(y)
    if window_length % 2 == 0:
        window_length += 1
    if window_length >= y.size:
        raise ValueError("window_length must be smaller than y.size")
    return savgol_filter(y, window_length=window_length, polyorder=polyorder, mode=mode)


# ---------------------------------------------------------------------------
# Simple interpolation
# ---------------------------------------------------------------------------

def interpolate_data(
    data: np.ndarray,
    resolution: float,
    new_limits: tuple[float, float] | None = None,
) -> np.ndarray:
    """Interpolate a 2D array ``[x, y, ...]`` onto a regular grid.

    Parameters
    ----------
    data : (N, M) array — column 0 is x; remaining columns are interpolated.
    resolution : desired x-spacing in the output.
    new_limits : optional (xmin, xmax); defaults to min/max of input x.

    Returns
    -------
    (N_new, M) array on the new regular grid.
    """
    if new_limits is None:
        new_limits = (float(data[:, 0].min()), float(data[:, 0].max()))

    new_x = np.arange(new_limits[0], new_limits[1], resolution)
    cols = [new_x]
    for col_idx in range(1, data.shape[1]):
        cols.append(np.interp(new_x, data[:, 0], data[:, col_idx]))
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Multi-dataset interpolation helpers
# ---------------------------------------------------------------------------

def _validate_data_dict(obj: dict) -> tuple[np.ndarray, list[str]]:
    """Extract and validate ``data`` and ``headers`` from a dict.

    Expected format: ``{"data": ndarray (N, M), "headers": list[str]}``.
    """
    if not isinstance(obj, dict) or "data" not in obj or "headers" not in obj:
        raise TypeError("Expected dict with 'data' and 'headers' keys.")
    data = obj["data"]
    headers = obj["headers"]
    if not isinstance(data, np.ndarray):
        raise TypeError("dict['data'] must be a numpy array")
    if not isinstance(headers, (list, tuple)):
        raise TypeError("dict['headers'] must be a list of column names")
    if data.ndim != 2:
        raise ValueError("dict['data'] must be a 2D array")
    return data, list(headers)


def interpolate_to_common_axis(
    *args: dict[str, Any],
    axis_col_name: str,
    clip_to_overlap: bool = True,
    phase_col_names: tuple[str, ...] = ("Phase", "Δ(Phase)", "delta Phase"),
    wrap_phase_output: bool = False,
) -> list[dict[str, Any]]:
    """Interpolate multiple datasets onto the highest-resolution common axis.

    Each argument must be ``{"data": ndarray, "headers": list[str]}``.

    Parameters
    ----------
    axis_col_name : name of the x-axis column in *headers*.
    clip_to_overlap : if True, output is limited to the overlapping x-range.
    phase_col_names : columns that need unwrap → interp → (optional re-wrap).
    wrap_phase_output : re-wrap phase columns after interpolation.
    """
    normalized: list[tuple[np.ndarray, list[str]]] = []
    axes: list[np.ndarray] = []
    spacings: list[float] = []

    for obj in args:
        data, headers = _validate_data_dict(obj)
        if axis_col_name not in headers:
            raise KeyError(
                f"Axis column '{axis_col_name}' not found in headers {headers}"
            )
        axis_idx = headers.index(axis_col_name)
        x = data[:, axis_idx]

        if len(x) > 1 and x[1] < x[0]:
            x = x[::-1]
            data = data[::-1, :]

        normalized.append((data, headers))
        axes.append(x)
        spacings.append(float(np.mean(np.abs(np.diff(x)))) if len(x) > 1 else np.inf)

    idx_best = int(np.argmin(spacings))
    x_target = axes[idx_best]

    if clip_to_overlap:
        start = max(ax[0] for ax in axes)
        stop = min(ax[-1] for ax in axes)
        mask = (x_target >= start) & (x_target <= stop)
        x_target = x_target[mask]

    results: list[dict[str, Any]] = []

    for (data, headers), x in zip(normalized, axes):
        axis_idx = headers.index(axis_col_name)
        cols: list[np.ndarray] = []

        for j, col_name in enumerate(headers):
            if j == axis_idx:
                cols.append(x_target)
                continue

            y = data[:, j]

            if col_name in phase_col_names:
                if not is_monotonic(y):
                    y = np.unwrap(y)
                y_interp = np.interp(x_target, x, y)
                if wrap_phase_output:
                    y_interp = np.angle(np.exp(1j * y_interp))
            else:
                y_interp = np.interp(x_target, x, y)

            cols.append(y_interp)

        results.append({"data": np.column_stack(cols), "headers": headers})

    return results


def interpolate_to_longest_axis(
    *args: dict[str, Any],
    axis_col_name: str,
    phase_col_names: tuple[str, ...] = ("Phase", "Δ(Phase)", "delta Phase"),
) -> list[dict[str, Any]]:
    """Interpolate all datasets onto the axis from the longest array.

    Each argument must be ``{"data": ndarray, "headers": list[str]}``.
    """
    normalized: list[tuple[np.ndarray, list[str]]] = []
    axes: list[np.ndarray] = []

    for obj in args:
        data, headers = _validate_data_dict(obj)
        if axis_col_name not in headers:
            raise KeyError(
                f"Axis column '{axis_col_name}' not found in headers {headers}"
            )
        axis_idx = headers.index(axis_col_name)
        x = data[:, axis_idx]

        if len(x) > 1 and x[1] < x[0]:
            x = x[::-1]
            data = data[::-1, :]

        normalized.append((data, headers))
        axes.append(x)

    idx_best = int(np.argmax([len(x) for x in axes]))
    x_target = axes[idx_best]

    results: list[dict[str, Any]] = []

    for (data, headers), x in zip(normalized, axes):
        axis_idx = headers.index(axis_col_name)
        cols: list[np.ndarray] = []

        for j, col_name in enumerate(headers):
            if j == axis_idx:
                cols.append(x_target)
                continue

            y = data[:, j]
            if col_name in phase_col_names and not is_monotonic(y):
                y = np.unwrap(y)

            cols.append(np.interp(x_target, x, y))

        results.append({"data": np.column_stack(cols), "headers": headers})

    return results
