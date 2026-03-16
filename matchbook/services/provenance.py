"""Provenance tracking — records what happened to data and why.

Provides a decorator ``@record_provenance`` that can wrap any function
taking a data-like object with a ``.history`` attribute.  Also provides
``record_step_provenance`` for integration with the Pipeline engine.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from datetime import datetime, timezone
from typing import Any, Callable, Optional


def _jsonable(x: Any) -> Any:
    """Best-effort conversion to something JSON-serializable."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if hasattr(x, "item") and callable(getattr(x, "item")):
        try:
            return _jsonable(x.item())
        except Exception:
            pass
    if hasattr(x, "__array__"):
        try:
            arr = x.__array__()
            return {
                "__ndarray__": True,
                "shape": list(getattr(arr, "shape", [])),
                "dtype": str(getattr(arr, "dtype", "")),
            }
        except Exception:
            pass
    return {"__repr__": repr(x)[:200]}


def _return_summary(ret: Any, max_chars: int = 500) -> Any:
    j = _jsonable(ret)
    s = json.dumps(j, default=str)
    if len(s) > max_chars:
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()
        return {"__summary__": True, "sha256": h, "truncated": True}
    return j


def record_provenance(
    *, store_return: bool = False, note: Optional[str] = None,
) -> Callable:
    """Decorator that appends a provenance record to ``data.history``.

    The decorated function's first positional argument (or keyword ``data``)
    must expose a ``history`` list attribute.
    """
    def deco(func: Callable) -> Callable:
        sig = inspect.signature(func)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            data = bound.arguments.get("data") or (args[0] if args else None)
            if data is None or not hasattr(data, "history"):
                raise TypeError(
                    "Expected first arg (or 'data') to have a .history attribute."
                )

            rec: dict[str, Any] = {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "func": f"{func.__module__}.{func.__qualname__}",
                "args": {
                    k: _jsonable(v)
                    for k, v in bound.arguments.items()
                    if k != "data"
                },
            }
            if note:
                rec["note"] = note

            ret = func(*args, **kwargs)

            if store_return:
                rec["return"] = _return_summary(ret)

            data.history.append(rec)
            return ret

        wrapper.__name__ = func.__name__
        wrapper.__qualname__ = func.__qualname__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return deco


def record_step_provenance(
    db: Any,
    series_id: str,
    step_id: str,
    params: dict[str, Any],
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    note: str = "",
) -> None:
    """Record a pipeline step execution to the database provenance table.

    Parameters
    ----------
    db:
        A Database instance (or anything with a ``record_provenance`` method).
    series_id, step_id:
        Identifies what was processed and by which step.
    params:
        Snapshot of parameters used for this execution.
    inputs, outputs:
        Lists of DataKey string representations consumed / produced.
    note:
        Optional human-readable annotation.
    """
    if hasattr(db, "record_provenance"):
        db.record_provenance(
            series_id=series_id,
            step_id=step_id,
            inputs=inputs,
            params=params,
            outputs=outputs,
            note=note,
        )
