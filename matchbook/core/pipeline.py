"""Rewindable pipeline engine.

The Pipeline manages an ordered sequence of processing steps, each with
named parameters.  It supports:

- **Partial re-run**: execute from an arbitrary step onward.
- **Parameter history**: every ``run_from`` snapshots all parameters so
  any previous state can be restored.
- **Observer events**: listeners are notified on step start / step done /
  rewind so the GUI can update progress indicators.

Step functions have the signature::

    fn(data_service: DataService, series_id: str, **params) -> None

They read inputs from ``data_service.get()`` and write results via
``data_service.put()``.  This makes them identical whether run from the
GUI, a script, or a test.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from matchbook.core.module_base import ParameterDescriptor


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------

@dataclass
class PipelineStep:
    """One processing step in a pipeline.

    Parameters
    ----------
    id:
        Unique identifier (e.g. ``"window"``).
    name:
        Human-readable label (e.g. ``"Apply Window"``).
    fn:
        The processing function called as ``fn(data_service, series_id, **params)``.
    params:
        Current parameter values keyed by name.
    param_descriptors:
        Metadata describing each parameter (for GUI rendering).
    depends_on:
        List of step ids that must run before this one.
    """
    id: str
    name: str
    fn: Callable[..., Any]
    params: dict[str, Any] = field(default_factory=dict)
    param_descriptors: list[ParameterDescriptor] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cached result
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Record of a step execution (for staleness checking)."""
    step_id: str
    params_snapshot: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Pipeline engine
# ---------------------------------------------------------------------------

# Listener signature:  (event: str, step_id: str) -> None
#   events: "step_start", "step_done", "rewind", "param_changed"
PipelineListener = Callable[[str, str], None]


class Pipeline:
    """Rewindable, parametric pipeline engine.

    Typical usage::

        pipe = Pipeline()
        pipe.add_step(PipelineStep(id="load", ...))
        pipe.add_step(PipelineStep(id="window", ...))
        pipe.add_step(PipelineStep(id="fft", ...))

        pipe.run_all(data_service, series_id)

        # User tweaks a parameter:
        pipe.set_param("window", "start_ps", 10.0)
        pipe.run_from("window", data_service, series_id)
    """

    def __init__(self) -> None:
        self._steps: dict[str, PipelineStep] = {}
        self._order: list[str] = []
        self._cache: dict[str, StepResult] = {}
        self._history: list[dict[str, dict[str, Any]]] = []
        self._listeners: list[PipelineListener] = []

    # -- step management -------------------------------------------------------

    def add_step(self, step: PipelineStep) -> None:
        """Append a step.  Duplicate ids overwrite the previous step."""
        self._steps[step.id] = step
        if step.id not in self._order:
            self._order.append(step.id)

    def get_step(self, step_id: str) -> PipelineStep:
        """Return the step with *step_id*, raising ``KeyError`` if missing."""
        return self._steps[step_id]

    @property
    def steps(self) -> list[PipelineStep]:
        """All steps in execution order."""
        return [self._steps[sid] for sid in self._order]

    @property
    def step_ids(self) -> list[str]:
        """Step ids in execution order."""
        return list(self._order)

    # -- parameter access ------------------------------------------------------

    def set_param(self, step_id: str, param_name: str, value: Any) -> None:
        """Update a parameter value.  Does **not** auto-run."""
        self._steps[step_id].params[param_name] = value
        self._notify("param_changed", step_id)

    def get_params(self, step_id: str) -> dict[str, Any]:
        """Return current parameter dict for *step_id*."""
        return dict(self._steps[step_id].params)

    # -- execution -------------------------------------------------------------

    def run_from(self, step_id: str, data_service: Any, series_id: str) -> None:
        """Re-execute from *step_id* through all downstream steps."""
        self._snapshot_history()
        idx = self._order.index(step_id)
        for sid in self._order[idx:]:
            step = self._steps[sid]
            self._notify("step_start", sid)
            step.fn(data_service, series_id, **step.params)
            self._cache[sid] = StepResult(
                step_id=sid,
                params_snapshot=copy.deepcopy(step.params),
            )
            self._notify("step_done", sid)

    def run_all(self, data_service: Any, series_id: str) -> None:
        """Execute the full pipeline from the first step."""
        if self._order:
            self.run_from(self._order[0], data_service, series_id)

    # -- staleness / cache -----------------------------------------------------

    def downstream_of(self, step_id: str) -> list[str]:
        """Return *step_id* and every step after it in execution order."""
        idx = self._order.index(step_id)
        return self._order[idx:]

    def is_stale(self, step_id: str) -> bool:
        """True if step has never run or its params changed since last run."""
        cached = self._cache.get(step_id)
        if cached is None:
            return True
        return cached.params_snapshot != self._steps[step_id].params

    # -- history / rewind ------------------------------------------------------

    @property
    def history_length(self) -> int:
        return len(self._history)

    def rewind(self, history_index: int) -> None:
        """Restore all step parameters to a previous snapshot."""
        if history_index < 0 or history_index >= len(self._history):
            raise IndexError(f"History index {history_index} out of range "
                             f"[0, {len(self._history)})")
        snapshot = self._history[history_index]
        for step_id, params in snapshot.items():
            if step_id in self._steps:
                self._steps[step_id].params = copy.deepcopy(params)
        self._notify("rewind", "")

    def _snapshot_history(self) -> None:
        snap = {
            sid: copy.deepcopy(step.params)
            for sid, step in self._steps.items()
        }
        self._history.append(snap)

    # -- observers -------------------------------------------------------------

    def subscribe(self, listener: PipelineListener) -> None:
        self._listeners.append(listener)

    def unsubscribe(self, listener: PipelineListener) -> None:
        self._listeners = [fn for fn in self._listeners if fn is not listener]

    def _notify(self, event: str, step_id: str) -> None:
        for fn in self._listeners:
            fn(event, step_id)
