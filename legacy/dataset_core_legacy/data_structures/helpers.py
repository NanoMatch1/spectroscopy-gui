import pandas as pd
import numpy as np
from typing import Any
from scipy.signal import savgol_filter

import numpy as np

def find_maxima(data, window, mode="abs", values="index", return_idx=False):
    """
    Find an extremum (abs/max/min) within a window.

    Parameters
    ----------
    data : (N, 2) array_like
        Column 0 is x, column 1 is y.
    window : tuple
        (xmin, xmax) where meaning depends on `values`:
        - values='index' -> integer indices [xmin, xmax)
        - values='value' -> x-axis values within [xmin, xmax]
    mode : {'abs', 'max', 'min'}
        Which extremum to pick inside the window.
    values : {'index', 'value'}
        Whether `window` is in index space or x-value space.
    return_idx : bool
        If True, also return the integer index into `data`.

    Returns
    -------
    (x0, y0) or (idx, x0, y0)
        The picked extremum. Returns None if the window selects no points.
    """
    # --- validate inputs ---
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("data must be an (N, 2) array (x in col 0, y in col 1).")

    try:
        a, b = window
    except Exception as e:
        raise ValueError("window must be a tuple/list like (min, max).") from e

    x = data[:, 0]
    y = data[:, 1]

    # --- select indices for the window ---
    if values == "value":
        xmin, xmax = (a, b) if a <= b else (b, a)
        idxs = np.flatnonzero((x >= xmin) & (x <= xmax))
    elif values == "index":
        # allow floats but interpret as indices
        i0, i1 = int(a), int(b)
        if i1 < i0:
            i0, i1 = i1, i0
        # treat as python slice [i0, i1)
        i0 = max(i0, 0)
        i1 = min(i1, len(x))
        idxs = np.arange(i0, i1, dtype=int)
    else:
        raise ValueError("values must be one of: 'index', 'value'")

    if idxs.size == 0:
        return None

    y_sel = y[idxs]

    # --- pick extremum within selected region ---
    if mode == "abs":
        rel = int(np.argmax(np.abs(y_sel)))
    elif mode == "max":
        rel = int(np.argmax(y_sel))
    elif mode == "min":
        rel = int(np.argmin(y_sel))
    else:
        raise ValueError("mode must be one of: 'abs', 'max', 'min'")

    idx = int(idxs[rel])
    x0 = float(x[idx])
    y0 = float(y[idx])

    return (idx, x0, y0) if return_idx else (x0, y0)

def df_to_array(df: pd.DataFrame) -> np.ndarray:
    """Convert a pandas DataFrame to a numpy ndarray - columns not preserved."""
    return df.to_numpy()

def df_to_dict(df: pd.DataFrame) -> dict[str, Any]:
    """Convert a pandas DataFrame to a dictionary with dictonary keys as the column names and values as numpy ndarrays."""
    return {col: df[col].to_numpy() for col in df.columns}

def is_monotonic(x):
    d = np.diff(x)
    return np.all(d >= 0) or np.all(d <= 0)


def dict_to_df(data_dict: dict) -> pd.DataFrame:
    """Convert a dictionary of numpy ndarrays to a pandas DataFrame."""
    data = data_dict['data']
    headers = data_dict['headers']
    return pd.DataFrame(data, columns=headers)

def array_to_df(array: np.ndarray, columns: list[str] | None = None) -> pd.DataFrame:
    """Convert a numpy ndarray to a pandas DataFrame."""
    return pd.DataFrame(array, columns=columns)



def _extract_data_and_headers(obj):
    """
    Convert supported objects into a canonical (data, headers, fmt) tuple.

    Supported:
      - pandas.DataFrame
      - dict with keys {'data': ndarray, 'headers': list}

    Unsupported:
      - bare numpy.ndarray (raises TypeError)
      - anything else
    """
    if isinstance(obj, pd.DataFrame):
        data = obj.to_numpy()
        headers = obj.columns.tolist()
        fmt = "dataframe"

    elif isinstance(obj, dict) and "data" in obj and "headers" in obj:
        data = obj["data"]
        headers = obj["headers"]
        fmt = "dict"

        if not isinstance(data, np.ndarray):
            raise TypeError("dict['data'] must be a numpy array")
        if not isinstance(headers, (list, tuple)):
            raise TypeError("dict['headers'] must be a list of column names")
        if data.ndim != 2:
            raise ValueError("dict['data'] must be a 2D array")

    else:
        raise TypeError(
            f"Unsupported type {type(obj)} — only DataFrame or "
            "dict{'data': ndarray, 'headers': list} are allowed."
        )

    return data, headers, fmt


def interpolate_to_max_resolution(
    *args,
    axis_col_name: str = None,      # must be provided for header safety
    clip_to_overlap: bool = True,
    phase_col_names: tuple[str, ...] = ('Phase', 'Δ(Phase)', 'delta Phase'),
    wrap_phase_output: bool = False,
):
    """
    Interpolate all datasets onto a common axis defined by the dataset
    with the smallest spacing.

    Only DataFrames or dict{'data','headers'} are accepted.

    Phase columns (matched by name) are unwrapped → interpolated → rewrapped.
    """
    if axis_col_name is None:
        raise ValueError(
            "axis_col_name must be provided (e.g., 'Time (ps)' or "
            "'Frequency (THz)') to ensure header-safe alignment."
        )

    objs = list(args)
    normalized = []
    axes = []
    spacings = []

    # Normalize and collect metadata
    for obj in objs:
        data, headers, fmt = _extract_data_and_headers(obj)

        if axis_col_name not in headers:
            raise KeyError(f"Axis column '{axis_col_name}' not found in headers {headers}")

        axis_idx = headers.index(axis_col_name)
        x = data[:, axis_idx]

        # Ensure x is increasing
        if len(x) > 1 and x[1] < x[0]:
            x = x[::-1]
            data = data[::-1, :]

        normalized.append((data, headers, fmt))
        axes.append(x)

        # Compute resolution
        if len(x) > 1:
            dx = np.diff(x)
            spacings.append(np.mean(np.abs(dx)))
        else:
            spacings.append(np.inf)

    # Pick the target axis = smallest spacing
    idx_best = int(np.argmin(spacings))
    x_target_full = axes[idx_best]

    # Clip to overlap region
    if clip_to_overlap:
        start = max(ax[0] for ax in axes)
        stop = min(ax[-1] for ax in axes)
        mask = (x_target_full >= start) & (x_target_full <= stop)
        x_target = x_target_full[mask]
    else:
        x_target = x_target_full

    resampled = []

    # Interpolate each dataset
    for (data, headers, fmt), x in zip(normalized, axes):

        axis_idx = headers.index(axis_col_name)
        cols = []

        for j, col_name in enumerate(headers):
            if j == axis_idx:
                cols.append(x_target)
                continue

            y = data[:, j]

            if col_name in phase_col_names:
                if is_monotonic(y):
                    # no need to unwrap monotonic data, it is already unwrapped
                    y_unwrapped = y
                else:
                    y_unwrapped = np.unwrap(y)
                y_interp = np.interp(x_target, x, y_unwrapped)
                if wrap_phase_output:
                    y_interp = np.angle(np.exp(1j * y_interp))
            else:
                y_interp = np.interp(x_target, x, y)

            cols.append(y_interp)

        new_data = np.column_stack(cols)
        resampled.append((new_data, headers, fmt))

    # Convert back to original format
    final_results = []
    for (data, headers, fmt) in resampled:
        if fmt == "dataframe":
            final_results.append(pd.DataFrame(data, columns=headers))
        else:  # fmt == 'dict'
            final_results.append({"data": data, "headers": headers})

    return final_results

def interpolate_to_max_resolution_simple(
    *args,
    axis_col_name: str = None,      # must be provided for header safety
    phase_col_names: tuple[str, ...] = ('Phase', 'Δ(Phase)', 'delta Phase'),
    **kwargs
):
    """
    Interpolate all datasets onto a common axis defined by the dataset
    with the longest array (i.e., highest resolution).

    Only DataFrames or dict{'data','headers'} are accepted.

    Phase columns (matched by name) are checked if need unwrapping (unwrapped) → interpolated.
    """
    if axis_col_name is None:
        raise ValueError(
            "axis_col_name must be provided (e.g., 'Time (ps)' or "
            "'Frequency (THz)') to ensure header-safe alignment."
        )

    objs = list(args)
    normalized = []
    axes = []

    # Normalize and collect metadata
    for obj in objs:
        data, headers, fmt = _extract_data_and_headers(obj)

        if axis_col_name not in headers:
            raise KeyError(f"Axis column '{axis_col_name}' not found in headers {headers}")

        axis_idx = headers.index(axis_col_name)
        x = data[:, axis_idx]

        # Ensure x is increasing
        if len(x) > 1 and x[1] < x[0]:
            x = x[::-1]
            data = data[::-1, :]

        normalized.append((data, headers, fmt))
        axes.append(x)

    # Pick the target axis = longest array
    idx_best = int(np.argmax([len(x) for x in axes]))
    x_target = axes[idx_best]

    resampled = []

    # Interpolate each dataset
    for (data, headers, fmt), x in zip(normalized, axes):

        axis_idx = headers.index(axis_col_name)
        cols = []

        for j, col_name in enumerate(headers):
            if j == axis_idx:
                cols.append(x_target)
                continue

            y = data[:, j]

            if col_name in phase_col_names:
                if not is_monotonic(y):
                    # non-monotonic phase needs unwrapping
                    y = np.unwrap(y)

            y_interp = np.interp(x_target, x, y)

            cols.append(y_interp)

        new_data = np.column_stack(cols)
        resampled.append((new_data, headers, fmt))

    # Convert back to original format
    final_results = []
    for (data, headers, fmt) in resampled:
        if fmt == "dataframe":
            final_results.append(pd.DataFrame(data, columns=headers))
        else:  # fmt == 'dict'
            final_results.append({"data": data, "headers": headers})

    return final_results


def smooth_trace_savgol(
    y: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
    mode: str = "interp"
) -> np.ndarray:
    """
    Smooth a 1D THz time trace using a Savitzky-Golay filter.

    Parameters
    ----------
    y : np.ndarray
        1D array of field values.
    window_length : int
        Length of the filter window (must be odd).
        Typical THz values: 7–21 samples.
    polyorder : int
        Polynomial order (must be < window_length).
        2–3 is typical.
    mode : str
        Boundary handling mode passed to savgol_filter.

    Returns
    -------
    y_smooth : np.ndarray
        Smoothed y-axis, same shape as input.
    """
    y = np.asarray(y)

    if window_length % 2 == 0:
        window_length += 1  # enforce odd window

    if window_length >= y.size:
        raise ValueError("window_length must be smaller than y.size")

    return savgol_filter(
        y,
        window_length=window_length,
        polyorder=polyorder,
        mode=mode
    )

def interpolate_data(data, resolution, new_limits=None) -> np.ndarray:
    '''Interpolates data using np.interp.
    
    Parameters:
    - data: 2D array with columns [x, y]
    - resolution: desired spacing between x values in the output
    - new_limits: tuple (min, max) for x values in the output. If None, uses min and max of input data.
    
    Returns:
    - new_data: 2D array with columns [x_interp, y_interp]'''

    if new_limits is None:
        new_limits = (data[:, 0].min(), data[:, 0].max())
    dataX = data[:, 0]
    dataY = data[:, 1]
    
    new_dataX = np.arange(new_limits[0], new_limits[1], resolution)
    dataY_interp = np.interp(new_dataX, dataX, dataY)
    std_error_interp = np.interp(new_dataX, dataX, data[:, 2])

    data = np.column_stack((new_dataX, dataY_interp, std_error_interp))
    return data
