import functools
import inspect
from typing import Iterable, Any

import numpy as np
import pandas as pd
from dataset_core.data_structures.helpers import interpolate_to_max_resolution_simple, _extract_data_and_headers

def align_to_max_resolution(
    axis_col_name: str,
    clip_to_overlap: bool = True,
    phase_col_names: tuple[str, ...] = ('Phase', 'Δ(Phase)', 'delta Phase'),
    wrap_phase_output: bool = False,
    **kwargs
):
    """
    Decorator that aligns multiple data-like arguments before passing them
    to the wrapped function.

    Only accepts:
      - pandas.DataFrame
      - dict{'data': ndarray, 'headers': list}

    Rejects bare ndarrays to avoid silent column misalignment.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args = list(args)

            # Which args need alignment?
            idx_data = []
            data_objs = []
            for i, arg in enumerate(args):
                if isinstance(arg, pd.DataFrame) or (
                    isinstance(arg, dict) and "data" in arg and "headers" in arg
                ):
                    idx_data.append(i)
                    data_objs.append(arg)

            # Only align if more than one dataset is present
            if len(data_objs) >= 2:
                aligned = interpolate_to_max_resolution_simple(
                    *data_objs,
                    axis_col_name=axis_col_name,
                    clip_to_overlap=clip_to_overlap,
                    phase_col_names=phase_col_names,
                    wrap_phase_output=wrap_phase_output,
                )
                for i, new_obj in zip(idx_data, aligned):
                    args[i] = new_obj

            return func(*args, **kwargs)

        return wrapper
    return decorator


#TODO: Remove with_dataframes alltogether by constructing the dataframes before function call
def with_dataframe(
    columns: Iterable[str] | None = None,
    ):
    """
    Decorator to adapt a legacy function that expects a pandas.DataFrame
    so it can also accept a numpy.ndarray or dict of 'data' and 'headers'.

    - If the input arguments are numpy arrays, they are converted to DataFrames
      with the given `columns` (or those stored on the function).
    Returns:
    - If any input was an array and output is a DataFrame, converts back to dict with 'data' and 'headers'.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            # track which args were arrays (by name)
            format_type: dict[str, bool] = {}

            # Determine columns: decorator > function attribute
            cols = columns
            if cols is None:
                cols = getattr(func, "_expected_columns", None)
            if cols is None:
                raise ValueError(
                    f"No column labels provided for {func.__name__}. "
                    f"Pass 'columns=...' to the decorator or set "
                    f"func._expected_columns = [...]"
                )

            for arg_name, value in bound.arguments.items():
                if isinstance(value, pd.DataFrame):
                    format_type[arg_name] = "dataframe"
                    continue
                is_arr = isinstance(value, np.ndarray)
                is_dict = isinstance(value, dict) and 'data' in value and 'headers' in value
                if not (is_arr or is_dict):
                    format_type[arg_name] = "other"
                    continue
                format_type[arg_name] = "array" if is_arr else "dict"



                # Only convert *this* argument if it is an ndarray
                if is_arr:
                    df = pd.DataFrame(value, columns=list(cols))
                    bound.arguments[arg_name] = df
                elif is_dict:
                    df = pd.DataFrame(value['data'], columns=value['headers'])
                    bound.arguments[arg_name] = df

            # Call original function
            result = func(*bound.args, **bound.kwargs)

            # If any input was an array and output is a DataFrame, convert back
            if any(ft in ("array", "dict") for ft in format_type.values()) and isinstance(result, pd.DataFrame):
                result = {
                    'data': result.to_numpy(),
                    'headers': result.columns.tolist()
                }

            return result

        return wrapper

    return decorator
