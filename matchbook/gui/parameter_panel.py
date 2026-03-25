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

# Signature for the span-session activation callback provided by the app:
#   on_activate_span_session(step_id, param_desc, controls_frame, on_done, on_cancel)
# where:
#   on_done(results_json: str, n_configured: int) -> None
#   on_cancel() -> None
SpanSessionActivator = Callable[
    [str, ParameterDescriptor, Any, Callable[[str, int], None], Callable[[], None]],
    None,
]
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
        on_activate_span_session: SpanSessionActivator | None = None,
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
        on_activate_span_session:
            Optional callback invoked when a SPAN_SESSION parameter's
            'Configure...' button is clicked.  If None, SPAN_SESSION
            parameters render as disabled.
        """
        self._on_param_changed = on_param_changed
        self._on_run_from = on_run_from
        self._on_activate_span_session = on_activate_span_session
        self._param_widgets: dict[str, dict[str, tk.Variable]] = {}

        self.frame = ttk.Frame(parent)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=(0, 4), pady=6)

        # Fixed header area — pipeline view and other top-pinned widgets go here
        self.header_frame = ttk.Frame(self.frame)
        self.header_frame.pack(fill=tk.X)

        # Scrollable interior — fills remaining space below the header
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
        """Build a single parameter widget using a two-row stacked layout.

        The parameter label sits on its own row above the control widgets.
        This prevents overflow in the fixed-width panel.
        """
        wrapper = ttk.Frame(parent)
        wrapper.pack(fill=tk.X, pady=(2, 0))

        ttk.Label(wrapper, text=param.label, anchor="w").pack(
            anchor="w", padx=2)

        control_row = ttk.Frame(wrapper)
        control_row.pack(fill=tk.X, padx=4, pady=(0, 2))

        var: tk.Variable

        if param.type == ParamType.BOOL:
            var = tk.BooleanVar(value=bool(param.default))
            ttk.Checkbutton(control_row, variable=var).pack(side=tk.LEFT)

        elif param.type == ParamType.CHOICE and param.choices:
            var = tk.StringVar(value=str(param.default))
            cb = ttk.Combobox(control_row, textvariable=var,
                               values=param.choices, state="readonly")
            cb.pack(side=tk.LEFT, fill=tk.X, expand=True)

        elif param.type == ParamType.FILE_PATH:
            var = tk.StringVar(value=str(param.default))
            entry = ttk.Entry(control_row, textvariable=var)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
            ttk.Button(
                control_row, text="...", width=3,
                command=lambda v=var: self._browse_file(v),
            ).pack(side=tk.LEFT)

        elif param.type in (ParamType.FLOAT, ParamType.INT):
            var = tk.StringVar(value=str(param.default))
            if param.min is not None and param.max is not None:
                # Slider fills available width; fixed-width entry beside it
                scale_var = tk.DoubleVar(value=float(param.default))
                scale = ttk.Scale(
                    control_row, from_=param.min, to=param.max,
                    variable=scale_var, orient=tk.HORIZONTAL,
                )
                scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
                entry = ttk.Entry(control_row, textvariable=var, width=7)
                entry.pack(side=tk.LEFT, padx=2)

                # Sync slider -> entry
                def _on_scale(val, v=var, sv=scale_var, p=param):
                    if p.type == ParamType.INT:
                        v.set(str(int(float(val))))
                    else:
                        v.set(f"{float(val):.6g}")
                scale.configure(command=_on_scale)

                # Sync entry -> slider
                def _on_entry(*_args, v=var, sv=scale_var):
                    try:
                        sv.set(float(v.get()))
                    except ValueError:
                        pass
                var.trace_add("write", _on_entry)
            else:
                entry = ttk.Entry(control_row, textvariable=var)
                entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        elif param.type == ParamType.SPAN_SESSION:
            # A JSON-string var holds the per-series result dict.
            var = tk.StringVar(value=str(param.default))
            summary_var = tk.StringVar(value="Not configured")
            ttk.Label(control_row, textvariable=summary_var,
                      font=("TkDefaultFont", theme.SIDEBAR_FONT_SIZE,
                            "italic")).pack(side=tk.LEFT)

            # controls_frame: hidden when idle, populated by the session
            controls_frame = ttk.Frame(parent)
            controls_frame.pack(fill=tk.X)

            configure_btn = ttk.Button(control_row, text="Configure…")
            configure_btn.pack(side=tk.LEFT, padx=(4, 0))

            def _on_done(results_json: str, n: int,
                         _sv=summary_var, _v=var, _btn=configure_btn,
                         _sid=step_id, _pn=param.name) -> None:
                _sv.set(f"{n}/{n} configured")
                _v.set(results_json)
                _btn.pack(side=tk.LEFT, padx=(4, 0))
                self._handle_change(_sid, _pn, _v)

            def _on_cancel_session(_btn=configure_btn) -> None:
                _btn.pack(side=tk.LEFT, padx=(4, 0))

            def _activate(_btn=configure_btn, _p=param,
                          _sid=step_id, _cf=controls_frame) -> None:
                _btn.pack_forget()
                if self._on_activate_span_session is not None:
                    self._on_activate_span_session(
                        _sid, _p, _cf, _on_done, _on_cancel_session)

            configure_btn.configure(
                command=_activate,
                state=tk.NORMAL if self._on_activate_span_session else tk.DISABLED,
            )

        else:
            # STRING or fallback
            var = tk.StringVar(value=str(param.default))
            entry = ttk.Entry(control_row, textvariable=var)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

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
