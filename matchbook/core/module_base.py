"""Descriptor types and protocol for analysis modules.

Modules produce **descriptors** — plain dataclasses that tell the framework
what data groups, traces, pipeline steps, and parameters they expose.
The GUI interprets these generically; no module ever imports tkinter.

The ``AnalysisModule`` protocol defines the structural contract.  Modules
do **not** need to import or inherit it — they just need to expose the
right attributes and methods.  The protocol lives here so the framework
can verify compatibility at registration time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Parameter descriptors
# ---------------------------------------------------------------------------

class ParamType(Enum):
    """Supported parameter widget types."""
    FLOAT = "float"
    INT = "int"
    STRING = "string"
    BOOL = "bool"
    CHOICE = "choice"
    FILE_PATH = "file_path"


@dataclass
class ParameterDescriptor:
    """Describes one tunable parameter for a pipeline step.

    Parameters
    ----------
    name:
        Internal key used in ``step.params``.
    label:
        Human-readable label shown in the GUI.
    type:
        Widget type — determines what control the GUI renders.
    default:
        Initial value.
    min, max, step:
        For numeric types: slider range and increment.
    choices:
        For ``CHOICE`` type: list of option strings.
    tooltip:
        Optional help text.
    """
    name: str
    label: str
    type: ParamType
    default: Any
    min: float | int | None = None
    max: float | int | None = None
    step: float | int | None = None
    choices: list[str] | None = None
    tooltip: str = ""


# ---------------------------------------------------------------------------
# Data-group and trace descriptors
# ---------------------------------------------------------------------------

@dataclass
class TraceDescriptor:
    """Describes one plottable trace within a data group.

    The ``group`` and ``name`` fields correspond to the DataKey used to
    look up the entry in the DataService.
    """
    key: str
    label: str
    group: str
    name: str
    y_label: str = ""
    default_visible: bool = False
    line_style: str | None = None


@dataclass
class DataGroupDescriptor:
    """Describes a collapsible group of traces in the sidebar.

    One descriptor becomes one collapsible section in the GUI, containing
    toggle-checkboxes for each trace.
    """
    key: str
    label: str
    x_label: str
    traces: list[TraceDescriptor] = field(default_factory=list)
    order: int = 0


# ---------------------------------------------------------------------------
# Pipeline-step descriptor
# ---------------------------------------------------------------------------

@dataclass
class PipelineStepDescriptor:
    """Enough information for the GUI to build controls for one step."""
    id: str
    name: str
    params: list[ParameterDescriptor] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Module protocol (structural typing — modules need not import this)
# ---------------------------------------------------------------------------

@runtime_checkable
class AnalysisModule(Protocol):
    """Structural contract that the framework expects from a module.

    A module satisfies this protocol purely by having the right attributes
    and methods — it does **not** need to import or subclass anything.
    """

    @property
    def name(self) -> str: ...

    @property
    def display_name(self) -> str: ...

    def data_groups(self) -> list[DataGroupDescriptor]: ...

    def pipeline_step_descriptors(self) -> list[PipelineStepDescriptor]: ...

    def create_pipeline_steps(self) -> list[Any]: ...
    # Returns list[PipelineStep] — typed as Any here to avoid circular import.
    # The registry validates the actual type at registration time.
