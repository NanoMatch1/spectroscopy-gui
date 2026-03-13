"""Dynamic parameter panel — renders widgets from ParameterDescriptors.

Builds controls (sliders, dropdowns, entries, checkboxes, file pickers)
for each pipeline step's parameters.  Changes can optionally auto-trigger
pipeline re-runs.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog
from typing import Any, Callable

from matchbook.core.module_base import (
    ParameterDescriptor,
    ParamType,
    PipelineStepDescriptor,
)
from matchbook.gui import theme


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ParameterPanel:
    """Right-hand panel showing pipeline steps and their parameter widgets."""

    def __init__(
        self,
        parent: tk.Widget,
        step_descriptors: list[PipelineStepDescriptor],
        on_param_changed: Callable[[str, str, Any], None],
        on_run_from: Callable[[str], None],
    ) -> None:
        """
        Parameters
        ----------
        parent:
            Parent widget.
        step_descriptors:
            Pipeline step descriptors from the registered module.
        on_param_changed:
            Called as ``on_param_changed(step_id, param_name, new_value)``
            whenever a parameter widget changes.
        on_run_from:
            Called as ``on_run_from(step_id)`` when the user clicks
            'Run from here'.
        """
        self._on_param_changed = on_param_changed
        self._on_run_from = on_run_from
        self._param_widgets: dict[str, dict[str, tk.Variable]] = {}

        self.frame = ttk.Frame(parent, width=250)
        self.frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 6), pady=6)
        self.frame.pack_propagate(False)

        # Scrollable interior
        canvas = tk.Canvas(self.frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL,
                                   command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>",
                   lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Auto-rerun checkbox
        self.auto_rerun = tk.BooleanVar(value=False)
        ttk.Checkbutton(inner, text="Auto-rerun on change",
                        variable=self.auto_rerun).pack(anchor="w", padx=6, pady=(6, 2))

        ttk.Separator(inner, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=4, pady=4)

        # Build per-step sections
        for step_desc in step_descriptors:
            self._build_step_section(inner, step_desc)

    def _build_step_section(
        self,
        parent: ttk.Frame,
        step_desc: PipelineStepDescriptor,
    ) -> None:
        """Build one collapsible section for a pipeline step."""
        header = ttk.Frame(parent)
        header.pack(fill=tk.X, padx=4, pady=(6, 0))

        ttk.Label(header, text=step_desc.name,
                  font=theme.SECTION_HEADER_FONT).pack(side=tk.LEFT)
        ttk.Button(
            header, text="\u25B6 Run", width=6,
            command=lambda sid=step_desc.id: self._on_run_from(sid),
        ).pack(side=tk.RIGHT)

        body = ttk.Frame(parent)
        body.pack(fill=tk.X, padx=8)

        self._param_widgets[step_desc.id] = {}

        for param in step_desc.params:
            self._build_param_widget(body, step_desc.id, param)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(
            fill=tk.X, padx=4, pady=4)

    def _build_param_widget(
        self,
        parent: ttk.Frame,
        step_id: str,
        param: ParameterDescriptor,
    ) -> None:
        """Build a single parameter widget based on its type."""
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)

        ttk.Label(row, text=param.label, width=18, anchor="w").pack(
            side=tk.LEFT, padx=(0, 4))

        var: tk.Variable

        if param.type == ParamType.BOOL:
            var = tk.BooleanVar(value=bool(param.default))
            ttk.Checkbutton(row, variable=var).pack(side=tk.LEFT)

        elif param.type == ParamType.CHOICE and param.choices:
            var = tk.StringVar(value=str(param.default))
            cb = ttk.Combobox(row, textvariable=var, values=param.choices,
                              state="readonly", width=12)
            cb.pack(side=tk.LEFT)

        elif param.type == ParamType.FILE_PATH:
            var = tk.StringVar(value=str(param.default))
            entry = ttk.Entry(row, textvariable=var, width=14)
            entry.pack(side=tk.LEFT, padx=(0, 2))
            ttk.Button(
                row, text="...", width=3,
                command=lambda v=var: self._browse_file(v),
            ).pack(side=tk.LEFT)

        elif param.type in (ParamType.FLOAT, ParamType.INT):
            var = tk.StringVar(value=str(param.default))
            if param.min is not None and param.max is not None:
                # Slider + entry
                scale_var = tk.DoubleVar(value=float(param.default))
                scale = ttk.Scale(
                    row, from_=param.min, to=param.max,
                    variable=scale_var, orient=tk.HORIZONTAL, length=100,
                )
                scale.pack(side=tk.LEFT)
                entry = ttk.Entry(row, textvariable=var, width=8)
                entry.pack(side=tk.LEFT, padx=2)

                # Sync slider -> entry
                def _on_scale(val, v=var, sv=scale_var, p=param):
                    if p.type == ParamType.INT:
                        v.set(str(int(float(val))))
                    else:
                        v.set(f"{float(val):.6g}")
                scale.configure(command=_on_scale)

                # Sync entry -> slider
                def _on_entry(*_args, v=var, sv=scale_var, p=param):
                    try:
                        sv.set(float(v.get()))
                    except ValueError:
                        pass
                var.trace_add("write", _on_entry)
            else:
                entry = ttk.Entry(row, textvariable=var, width=14)
                entry.pack(side=tk.LEFT)

        else:
            # STRING or fallback
            var = tk.StringVar(value=str(param.default))
            entry = ttk.Entry(row, textvariable=var, width=14)
            entry.pack(side=tk.LEFT)

        # Store and bind change notification
        self._param_widgets[step_id][param.name] = var
        var.trace_add("write", lambda *_a, sid=step_id, pn=param.name, v=var:
                      self._handle_change(sid, pn, v))

    def _handle_change(self, step_id: str, param_name: str,
                       var: tk.Variable) -> None:
        """Called when any parameter widget value changes."""
        value = var.get()
        self._on_param_changed(step_id, param_name, value)
        if self.auto_rerun.get():
            self._on_run_from(step_id)

    @staticmethod
    def _browse_file(var: tk.StringVar) -> None:
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def get_param_values(self, step_id: str) -> dict[str, Any]:
        """Return current widget values for a pipeline step."""
        result: dict[str, Any] = {}
        for name, var in self._param_widgets.get(step_id, {}).items():
            result[name] = var.get()
        return result
