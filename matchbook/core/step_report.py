"""Step reporting framework — collects per-step diagnostics.

Each pipeline step can produce a StepReport summarising what happened:
data counts, warnings, quality metrics, etc.  The Pipeline collects
these and makes them available to the GUI and to provenance logging.

This is a skeleton — specific reporting content for each processing step
will be defined later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DiagnosticLevel(Enum):
    """Severity level for a single diagnostic item."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Diagnostic:
    """One diagnostic message produced during a pipeline step."""
    level: DiagnosticLevel
    message: str
    detail: str = ""


@dataclass
class StepReport:
    """Summary of a single pipeline step execution.

    Pipeline steps optionally return a StepReport.  If they return None,
    a minimal "step completed" report is synthesised automatically.
    """
    step_id: str
    series_id: str
    diagnostics: list[Diagnostic] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    entries_written: int = 0
    entries_read: int = 0

    # -- convenience builders --

    def info(self, message: str, detail: str = "") -> None:
        self.diagnostics.append(
            Diagnostic(DiagnosticLevel.INFO, message, detail))

    def warn(self, message: str, detail: str = "") -> None:
        self.diagnostics.append(
            Diagnostic(DiagnosticLevel.WARNING, message, detail))

    def error(self, message: str, detail: str = "") -> None:
        self.diagnostics.append(
            Diagnostic(DiagnosticLevel.ERROR, message, detail))

    # -- queries --

    @property
    def has_errors(self) -> bool:
        return any(d.level == DiagnosticLevel.ERROR for d in self.diagnostics)

    @property
    def has_warnings(self) -> bool:
        return any(d.level == DiagnosticLevel.WARNING for d in self.diagnostics)

    @property
    def ok(self) -> bool:
        return not self.has_errors

    def summary(self) -> str:
        """One-line human-readable summary."""
        n_err = sum(1 for d in self.diagnostics if d.level == DiagnosticLevel.ERROR)
        n_warn = sum(1 for d in self.diagnostics if d.level == DiagnosticLevel.WARNING)
        parts = [f"step={self.step_id}"]
        if self.entries_written:
            parts.append(f"wrote={self.entries_written}")
        if n_err:
            parts.append(f"errors={n_err}")
        if n_warn:
            parts.append(f"warnings={n_warn}")
        if self.metrics:
            parts.append(f"metrics={list(self.metrics.keys())}")
        return " | ".join(parts)
