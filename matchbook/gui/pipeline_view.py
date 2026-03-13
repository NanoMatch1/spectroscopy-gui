"""Pipeline view — visual step list with status, run, and rewind controls."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from matchbook.core.pipeline import Pipeline
from matchbook.gui import theme


class PipelineView:
    """Displays the pipeline as a vertical step list with status indicators.

    Status indicators:
      - checkmark  (done / cached)
      - bullet     (selected / editing)
      - circle     (not yet run)
      - warning    (stale — upstream params changed since last run)
    """

    STATUS_ICONS = {
        "done":     "\u2713",   # ✓
        "editing":  "\u25CF",   # ●
        "pending":  "\u25CB",   # ○
        "stale":    "\u26A0",   # ⚠
    }

    def __init__(
        self,
        parent: tk.Widget,
        pipeline: Pipeline,
        on_step_selected: Callable[[str], None],
        on_run_from: Callable[[str], None],
        on_rewind: Callable[[int], None],
    ) -> None:
        self._pipeline = pipeline
        self._on_step_selected = on_step_selected
        self._on_run_from = on_run_from
        self._on_rewind = on_rewind
        self._step_labels: dict[str, ttk.Label] = {}
        self._selected_step: str | None = None

        self.frame = ttk.LabelFrame(parent, text="Pipeline")
        self.frame.pack(fill=tk.X, padx=6, pady=(6, 0))

        self._build_step_list()
        self._build_controls()

        # Subscribe to pipeline events
        pipeline.subscribe(self._on_pipeline_event)

    def _build_step_list(self) -> None:
        for step in self._pipeline.steps:
            row = ttk.Frame(self.frame)
            row.pack(fill=tk.X, padx=4, pady=1)

            status = self._get_status(step.id)
            icon = self.STATUS_ICONS.get(status, "?")
            lbl = ttk.Label(
                row, text=f" {icon}  {step.name}",
                cursor="hand2",
                font=("TkDefaultFont", 10),
            )
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            lbl.bind("<Button-1>",
                     lambda _e, sid=step.id: self._select_step(sid))

            self._step_labels[step.id] = lbl

    def _build_controls(self) -> None:
        ctrl = ttk.Frame(self.frame)
        ctrl.pack(fill=tk.X, padx=4, pady=4)

        ttk.Button(ctrl, text="\u25B6 Run from here", width=14,
                   command=self._run_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="\u25C0 Undo", width=6,
                   command=self._undo).pack(side=tk.LEFT, padx=2)

        self._history_label = ttk.Label(ctrl, text="History: 0")
        self._history_label.pack(side=tk.RIGHT, padx=4)

    def _select_step(self, step_id: str) -> None:
        self._selected_step = step_id
        self._on_step_selected(step_id)
        self.refresh()

    def _run_selected(self) -> None:
        if self._selected_step:
            self._on_run_from(self._selected_step)

    def _undo(self) -> None:
        hl = self._pipeline.history_length
        if hl > 0:
            self._on_rewind(hl - 1)

    def _get_status(self, step_id: str) -> str:
        if step_id == self._selected_step:
            return "editing"
        if self._pipeline.is_stale(step_id):
            cache = self._pipeline.get_cache(step_id)
            if cache is not None:
                return "stale"
            return "pending"
        return "done"

    def refresh(self) -> None:
        """Update all status icons and the history counter."""
        for step_id, lbl in self._step_labels.items():
            step = self._pipeline.get_step(step_id)
            status = self._get_status(step_id)
            icon = self.STATUS_ICONS.get(status, "?")
            lbl.configure(text=f" {icon}  {step.name}")
        self._history_label.configure(
            text=f"History: {self._pipeline.history_length}")

    def _on_pipeline_event(self, event: str, step_id: str) -> None:
        self.refresh()
