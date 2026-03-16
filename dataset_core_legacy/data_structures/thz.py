'''
Data container classes for THz time-domain spectroscopy datasets.

BaseTHzData holds a single scan; THzData holds multiple scans and
provides averaged data access, statistical helpers, and plotting.

Processing methods (FFT, baseline, windowing, padding, etc.) are
intentionally excluded — they belong in separate analysis packages.
'''

from __future__ import annotations
import numpy as np
import datetime
import pandas as pd

from dataclasses import dataclass
from typing import Optional, Union


@dataclass(frozen=True)
class TimeDomainStats:
    time: np.ndarray                 # (N_time,)
    mean: np.ndarray                 # (N_time,)
    std: np.ndarray                  # (N_time,)  scatter across repeats at each timepoint
    stderr: np.ndarray               # (N_time,)  std / sqrt(N_repeats)
    n_repeats: int                   # number of repeats used
    baseline_sigma: Optional[float]  # single-number sigma from baseline window on mean (if requested)
    snr_from_baseline: Optional[np.ndarray]  # (N_time,) |mean| / baseline_sigma (if requested)


def _as_slice(idx: Union[slice, np.ndarray, list, tuple, None], n: int) -> Union[slice, np.ndarray]:
    """
    Accept slice or index array/list. Validates bounds for slice.
    """
    if idx is None:
        return slice(None)

    if isinstance(idx, slice):
        # Basic bounds safety for slice
        start = 0 if idx.start is None else idx.start
        stop = n if idx.stop is None else idx.stop
        if start < 0 or stop < 0 or start > n or stop > n or start >= stop:
            raise ValueError(f"Invalid baseline_slice={idx} for length {n}.")
        return idx

    # Otherwise treat as array-like of indices / boolean mask
    arr = np.asarray(idx)
    if arr.dtype == bool and arr.shape != (n,):
        raise ValueError(f"Boolean baseline mask must have shape {(n,)}, got {arr.shape}.")
    return arr


def calculate_time_domain_stats(
    raw_data: np.ndarray,
    *,
    limit: Optional[int] = None,
    ddof: int = 1,
    baseline_slice: Union[slice, np.ndarray, list, tuple, None] = None,
    compute_baseline_snr: bool = False,
) -> TimeDomainStats:
    """
    raw_data expected shape: (N_time, 1 + N_repeats)
      - column 0 = time axis
      - columns 1: = repeated traces (same time grid)

    Returns per-timepoint mean/std/stderr across repeats.
    Optionally computes a single baseline sigma from a baseline time window on the MEAN trace,
    and SNR(t) = |mean(t)| / baseline_sigma.
    """

    if raw_data is None or raw_data.size == 0:
        # Return empty object with consistent fields
        empty = np.array([])
        return TimeDomainStats(
            time=empty, mean=empty, std=empty, stderr=empty,
            n_repeats=0, baseline_sigma=None, snr_from_baseline=None
        )

    if raw_data.ndim != 2 or raw_data.shape[1] < 2:
        raise ValueError(f"raw_data must be 2D with at least 2 columns (time + >=1 repeat). Got {raw_data.shape}.")

    time = raw_data[:, 0]
    traces = raw_data[:, 1:]  # (N_time, N_repeats_total)

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be a positive integer.")
        traces = traces[:, :limit]

    n_time, n_repeats = traces.shape
    if n_repeats < 1:
        raise ValueError("No repeat traces available after slicing.")

    mean = np.mean(traces, axis=1)

    # If only one repeat, ddof=1 would give NaNs; fall back safely
    ddof_eff = ddof if n_repeats > 1 else 0
    std = np.std(traces, axis=1, ddof=ddof_eff)
    stderr = std / np.sqrt(n_repeats)

    baseline_sigma = None
    snr = None

    if compute_baseline_snr:
        # baseline window is applied to the MEAN trace (classic approach)
        bidx = _as_slice(baseline_slice, n_time)

        # Again: if baseline region is too small, ddof=1 could be problematic
        baseline_vals = mean[bidx]
        if baseline_vals.size < 2:
            raise ValueError("Baseline region must include at least 2 points to estimate sigma.")

        baseline_sigma = float(np.std(baseline_vals, ddof=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.where(baseline_sigma > 0, np.abs(mean) / baseline_sigma, 0.0)

    return TimeDomainStats(
        time=time,
        mean=mean,
        std=std,
        stderr=stderr,
        n_repeats=n_repeats,
        baseline_sigma=baseline_sigma,
        snr_from_baseline=snr,
    )



class BaseTHzData:
    '''Base class for THz data structures. Holds one scan and metadata information.
    '''
    
    def __init__(self, data: np.array, headers: dict) -> None:
        self.raw_data = data  # Numpy array of [time, amplitude] pairs
        self.headers = headers  # list of header strings
        self.filename, self.scan_index = self._resolve_filename()
        self.timestamp = self._resolve_timestamp()

    def __repr__(self):
        return f"<BaseTHzData:{self.filename}, scan_{self.scan_index}, timestamp:{self.timestamp}>"
    
    def _compress_data(self):
        '''Dump the redundant X-axis if needed to save memory, deletes headers.'''
        self.raw_data = self.raw_data[:, 1]
        self.headers = None

    def _resolve_filename(self) -> tuple:
        # TODO: check if else logic is correct at this indentation level.
        '''Extracts filename and scan index from headers if available.'''
        for item in self.headers:
            if 'title' in item.lower():
                stringlist = item.split(' ') # Assumes format 'title filename ...'
                title = stringlist[1]
                scan_index = int(stringlist[4]) if len(stringlist) > 4 else None
                return title, scan_index
            else:
                return 'unknown_file', None
        return 'unknown_file', None # if header is empty

    def _resolve_timestamp(self) -> str:
        '''Extracts timestamp from headers if available.'''
        for item in self.headers:
            if 'date' in item.lower() and 'time' in item.lower():
                stringlist = item.split(',') # Assumes format 'Data and time,YYYY-MM-DD HH:MM:SS'
                timeobj = stringlist[1].split('.')[0].strip()
                # convert to datetime object
                timeobj = datetime.datetime.strptime(timeobj, '%Y-%m-%d %H:%M:%S')
                return timeobj
        return 'unknown_timestamp'


class THzData:
    '''Data container for THz time-domain spectroscopy measurements.

    Holds multiple BaseTHzData scan objects and provides averaged data access,
    basic statistical helpers, and a plotting method. Processing methods
    (FFT, baseline, windowing, padding, etc.) live in separate packages.

    Old dataframe compatibility: allows access via thzdata['Mean'], thzdata['Time (ps)'], etc.
    '''

    # df compatibility mapping
    _column_map = {
        "Time (ps)": 0,
        "Mean": 1,
        "std error": 2,
    }

    def __init__(self, data: list, header: list, **kwargs) -> None:
        self.data_list = data  # list of BaseTHz objects for each scan
        self.raw_data = self._compile_data_array()  # np.array of compiled data from all scans
        self.headers = header if header is not None else self._grabonedata().headers  # retain headers from first scan
        self.data_type = kwargs.get('data_type', None) # e.g. 'acc', 'dat', etc.
        self.filename = kwargs.get('filename', 'unknown_file')
        self.reference_filename = None

        self._meta_data = {} # stores statistical data like noise estimates, phase offset, etc.
        self.processing_dict = {}  # stores processed data at various steps

        self._time_data = self._average_data()  # Averaged dataset
        self._time_data_headers = ["Time (ps)", "Mean", "std error"]

        self.reference_data = None

        self._data = self._time_data  # Current working data (time or frequency domain)

    def __getitem__(self, key):
        """
        Backwards-compatible dictionary-style access.
        Allows: thz['Mean'], thz['Time (ps)'], etc.
        """
        if key not in self._column_map:
            raise KeyError(f"{key} not found in THzData columns {list(self._column_map)}")

        col_idx = self._column_map[key]
        return self._data[:, col_idx]

    # df compatibility assignment
    def __setitem__(self, key, value):
        """
        Optional: allow assignment like thz['Mean'] = new_array.
        """
        if key not in self._column_map:
            raise KeyError(f"{key} not found in THzData columns {list(self._column_map)}")
        col_idx = self._column_map[key]
        self._data[:, col_idx] = value

    @property
    def type(self) -> str:
        '''Returns the data type (e.g. 'acc', 'dat', etc.) if available.'''
        return self.__class__, self.data_type

    @property
    def time_const(self) -> float | None:
        '''Returns the time constant metadata if available. Tries to parse if not, else None.'''
        time_const = self._meta_data.get('time_constant', None)
        if time_const is None:
            time_const = self._identify_time_constant()

        return time_const

    @property
    def data(self) -> np.array:
        '''Returns the current working data array (time or frequency domain).'''
        return self._data

    # df compatibility properties
    @property
    def columns(self):
        """Backwards-compatible .columns attribute, like a DataFrame."""
        return list(self._column_map.keys())

    # df compatibility contains
    def __contains__(self, key):
        return key in self._column_map

    def __repr__(self):
        return f"\nTHzData:{self.filename}\n   -> Scans: {len(self.data_list)}\n   -> Data type: {self.data_type}\n" 
    
    def update_data(self, new_data: np.ndarray) -> None:
        '''Takes a modified raw_data np.array and updates the internal state of the object, including re-averaging and recalculating stats. Used for instance after modifying the acquisitions.'''
        self.raw_data = new_data
        self._time_data = self._average_data()
        self._time_data_headers = ["Time (ps)", "Mean", "std error"]
        self._data = self._time_data
    
    def _identify_time_constant(self, time_unit='ms') -> None:
        '''Work around function to pull time constant from filename, if available.'''
        if 'time' in self.filename.lower() and 'const' in self.filename.lower():
            stritem = self.filename.split(time_unit)[0].strip()
            # check string before time for float until non-float character
            stritem = stritem[::-1] # reverse string
            timeconst_str = ''
            for char in stritem:
                if char.isdigit() or char == '.':
                    timeconst_str = char + timeconst_str # adds to front so we dont need to reverse again
                else:
                    break

            try:
                timeconst = float(timeconst_str)
                self._meta_data['time_constant'] = timeconst
            except ValueError:
                print(f"Could not parse time constant from filename: {self.filename}. Setting to None.")
                self._meta_data['time_constant'] = None

        return self._meta_data.get('time_constant', None)

    def _calculate_std_error(self) -> np.array:
        '''Calculates the standard error across all scans for each time point.'''
        data_matrix = np.array([obj.raw_data[:, 1] for obj in self.data_list])
        std_error = np.std(data_matrix, axis=0) / np.sqrt(len(self.data_list))
        return std_error

    def _compile_data_array(self, limit=None) -> np.array:
        '''Takes the data from all scans and compiles it into a single numpy array.'''
        compiled_data = None
        for idx, obj in enumerate(self.data_list):
            if limit is not None and idx >= limit: # condition to limit number of scans compiled
                break
            if compiled_data is None:
                compiled_data = obj.raw_data
            else:
                compiled_data = np.column_stack((compiled_data, obj.raw_data[:, 1]))
        return compiled_data
    
    def _grabonedata(self, index=0) -> BaseTHzData:
        '''Returns a single BaseTHzData object from the data_list by index.'''
        return self.data_list[index]
    
    def _compress_dataset(self) -> None:
        '''Compresses the dataset by removing redundant X-axis data from each scan.'''
        for obj in self.data_list:
            obj._compress_data()

    def average_data(self, limit=None) -> np.array:
        '''Public method to average the data across number of chosen scans.'''
        data_matrix = np.array([obj.raw_data[:, 1] for obj in self.data_list[:limit]])
        std_error = np.std(data_matrix, axis=0) / np.sqrt(len(self.data_list[:limit]))
        mean_data = np.mean(data_matrix, axis=0)
        time_axis = self.data_list[0].raw_data[:, 0]
        averaged_data = np.column_stack((time_axis, mean_data, std_error))
        return averaged_data

        
    def _average_data(self) -> np.array:
        '''returns array of:
         0: time (x-axis),
         1: averaged data across all scans (y-axis),
         2: Standard error as third column.
         
         If there is only one scan in raw_data, returns the mean with zero error.'''
        
        if self.raw_data.shape[1] < 3:
            time_axis = self.raw_data[:, 0]
            mean_data = self.raw_data[:, 1]
            std_error = np.zeros_like(mean_data)
            averaged_data = np.column_stack((time_axis, mean_data, std_error))
        else:
            data_matrix = np.array([obj.raw_data[:, 1] for obj in self.data_list])
            std_error = np.std(data_matrix, axis=0) / np.sqrt(len(self.data_list))
            mean_data = np.mean(data_matrix, axis=0)
            time_axis = self.data_list[0].raw_data[:, 0]
            averaged_data = np.column_stack((time_axis, mean_data, std_error))

        self.processing_dict['time_domain'] = averaged_data.copy()
        return averaged_data

    @property
    def time(self) -> np.array:
        '''Returns the time axis of the working dataset in data_current.'''
        if self._data is not None:
            return self._data[:, 0]
        return None
    
    @property
    def y_mean(self) -> np.array:
        '''Returns the mean amplitude of the working dataset in data_current.'''
        if self._data is not None:
            return self._data[:, 1]
        return None
    
    @property
    def y_err(self) -> np.array:
        '''Returns the standard error of the working dataset in data_current.'''
        if self._data is not None:
            return self._data[:, 2]
        return None

    def to_legacy_dataframe(self) -> pd.DataFrame:
        """
        Return a pandas DataFrame mimicking the original Marco-style timedata:

        Columns:
        'Time (ps)':   time axis in ps
        'Mean':        averaged signal
        'std error':   standard error across scans
        """
        if self._data is None:
            raise ValueError("THzData.data is None; nothing to convert.")

        df = pd.DataFrame(
            self._data,
            columns=["Time (ps)", "Mean", "std error"],
        )
        return df

    def plot_current(self, *, figure_obj=None, **kwargs) -> None:
        """Plot the current averaged data.

        Accepts an optional `figure_obj` (the dataset's FigureObject). If it is
        provided the method will draw onto `figure_obj.ax` and will not call
        `plt.show()`; otherwise a new figure is created and shown.
        """
        import matplotlib.pyplot as plt

        error_bars = kwargs.get('error_bars', True)
        normalise = kwargs.get('normalise', False)
        index_axis = kwargs.get('index_axis', False)

        if self._data is None:
            print("No averaged data to plot.")
            return

        # do not mutate self._data in-place; operate on a local view
        if isinstance(self._data, pd.DataFrame):
            data_view = self._data.to_numpy()
        else:
            data_view = self._data

        if index_axis:
            time = np.arange(data_view.shape[0])
        else:
            time = data_view[:, 0]

        mean_amplitude = data_view[:, 1]
        std_error = data_view[:, 2]

        if normalise:
            max_amp = np.max(np.abs(mean_amplitude))
            if max_amp != 0:
                mean_amplitude = mean_amplitude / max_amp
                std_error = std_error / max_amp

        # Acquire axis: prefer provided FigureObject, otherwise create a temporary
        show_plot = False
        if figure_obj is None:
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
            show_plot = True
        else:
            ax = getattr(figure_obj, 'ax', None)
            if ax is None:
                # fallback to creating a new figure if the object is malformed
                fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
                show_plot = True

        line_alpha = kwargs.get('line_alpha', 1.0)
        ax.plot(time, mean_amplitude, '-', label=self.filename, alpha=line_alpha)
        if error_bars:
            ax.fill_between(
                time,
                mean_amplitude - std_error,
                mean_amplitude + std_error,
                alpha=kwargs.get('alpha', 0.3),
                color=kwargs.get('error_color', 'tab:red'),
            )

        ax.set_title(kwargs.get('title', 'Averaged THz Data'))
        ax.set_xlabel(kwargs.get('xlabel', 'Index' if index_axis else 'Time (ps)'))
        ax.set_ylabel(kwargs.get('ylabel', 'Amplitude (a.u.)'))
        ax.legend()
        ax.grid(kwargs.get('show_grid', True))

        if show_plot:
            plt.show()
