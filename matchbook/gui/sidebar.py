"""Dynamic sidebar builder — renders series toggles and trace groups.

Everything is built from descriptors provided by registered modules.
No domain-specific knowledge lives here.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from matchbook.core.module_base import DataGroupDescriptor
from matchbook.gui import theme


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_sidebar(
    parent: tk.Widget,
    series_names: list[str],
    series_vars: dict[str, tk.BooleanVar],
    data_groups: list[DataGroupDescriptor],
    trace_vars: dict[str, tk.BooleanVar],
    autoscale_vars: dict[str, tk.BooleanVar],
    manual_x_enabled: dict[str, tk.BooleanVar],
    manual_y_enabled: dict[str, tk.BooleanVar],
    manual_x_min: dict[str, tk.StringVar],
    manual_x_max: dict[str, tk.StringVar],
    manual_y_min: dict[str, tk.StringVar],
    manual_y_max: dict[str, tk.StringVar],
    schedule_redraw: Callable[[], None],
) -> tuple[ttk.Frame, ttk.Frame]:
    """Build the full sidebar widget.

    Returns
    -------
    (sidebar_frame, series_body_frame)
        The outer sidebar frame and the inner frame holding series
        checkboxes — the latter can be passed to
        :func:`refresh_series_section` when new series arrive.
    """

    sidebar = ttk.Frame(parent, width=theme.SIDEBAR_WIDTH)
    sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0), pady=6)
    sidebar.pack_propagate(False)

    canvas = tk.Canvas(sidebar, highlightthickness=0)
    scrollbar = ttk.Scrollbar(sidebar, orient=tk.VERTICAL, command=canvas.yview)
    inner = ttk.Frame(canvas)

    inner.bind("<Configure>",
               lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    _bind_mousewheel(canvas)

    # -- Series section --
    series_body = _build_series_section(inner, series_names, series_vars)

    ttk.Separator(inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6, padx=4)

    # -- Data-group trace sections --
    for group_desc in sorted(data_groups, key=lambda g: g.order):
        _build_group_section(
            inner, group_desc, trace_vars,
            autoscale_vars, manual_x_enabled, manual_y_enabled,
            manual_x_min, manual_x_max, manual_y_min, manual_y_max,
            schedule_redraw,
        )
        ttk.Separator(inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4, padx=4)

    return sidebar, series_body


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------

def refresh_series_section(
    series_body: ttk.Frame,
    series_names: list[str],
    series_vars: dict[str, tk.BooleanVar],
) -> None:
    """Rebuild the series checkboxes inside an existing series body frame.

    Call this whenever the series list changes after the sidebar was built.
    *series_vars* must already contain a ``BooleanVar`` for each name in
    *series_names* before this is called.
    """
    for widget in series_body.winfo_children():
        widget.destroy()
    _populate_series_body(series_body, series_names, series_vars)


def _build_series_section(
    parent: ttk.Frame,
    series_names: list[str],
    series_vars: dict[str, tk.BooleanVar],
) -> ttk.Frame:
    """Build the 'Series' collapsible section.  Returns the body frame."""
    body = _build_collapsible(parent, "Series")
    _populate_series_body(body, series_names, series_vars)
    return body


def _populate_series_body(
    body: ttk.Frame,
    series_names: list[str],
    series_vars: dict[str, tk.BooleanVar],
) -> None:
    """Create one colour-coded checkbox per series name inside *body*."""
    for idx, name in enumerate(series_names):
        colour = theme.series_colour(idx)
        frame = ttk.Frame(body)
        frame.pack(anchor="w", padx=4)
        cb = tk.Checkbutton(
            frame,
            text=f"  {name}",
            variable=series_vars[name],
            fg=colour,
            selectcolor="#ffffff",
            font=("TkDefaultFont", theme.SIDEBAR_FONT_SIZE, "bold"),
            anchor="w",
        )
        cb.pack(anchor="w")


def _build_group_section(
    parent: ttk.Frame,
    group_desc: DataGroupDescriptor,
    trace_vars: dict[str, tk.BooleanVar],
    autoscale_vars: dict[str, tk.BooleanVar],
    manual_x_enabled: dict[str, tk.BooleanVar],
    manual_y_enabled: dict[str, tk.BooleanVar],
    manual_x_min: dict[str, tk.StringVar],
    manual_x_max: dict[str, tk.StringVar],
    manual_y_min: dict[str, tk.StringVar],
    manual_y_max: dict[str, tk.StringVar],
    schedule_redraw: Callable[[], None],
) -> None:
    """Build one collapsible data-group section with trace toggles + axis controls."""
    body = _build_collapsible(parent, group_desc.label)
    gkey = group_desc.key

    # All / None buttons
    btn_frame = ttk.Frame(body)
    btn_frame.pack(anchor="w", padx=4)
    ttk.Button(
        btn_frame, text="All", width=4,
        command=lambda: _set_traces(group_desc, trace_vars, True),
    ).pack(side=tk.LEFT)
    ttk.Button(
        btn_frame, text="None", width=4,
        command=lambda: _set_traces(group_desc, trace_vars, False),
    ).pack(side=tk.LEFT, padx=2)

    # Trace checkboxes
    for trace_desc in group_desc.traces:
        cb = ttk.Checkbutton(body, text=trace_desc.label,
                             variable=trace_vars[trace_desc.key])
        cb.pack(anchor="w", padx=12)

    # Axis controls
    axis_lbl = ttk.Label(body, text="Axis controls", font=theme.AXIS_CONTROL_FONT)
    axis_lbl.pack(anchor="w", padx=8, pady=(4, 0))

    auto_cb = ttk.Checkbutton(body, text="Autoscale",
                               variable=autoscale_vars[gkey])
    auto_cb.pack(anchor="w", padx=12)

    # X limits
    xf = ttk.Frame(body)
    xf.pack(anchor="w", padx=12, pady=1)
    ttk.Checkbutton(xf, text="X:", variable=manual_x_enabled[gkey],
                    width=3).pack(side=tk.LEFT)
    ex_min = ttk.Entry(xf, textvariable=manual_x_min[gkey], width=8)
    ex_min.pack(side=tk.LEFT, padx=1)
    ex_min.bind("<Return>", lambda _e: schedule_redraw())
    ttk.Label(xf, text="\u2013").pack(side=tk.LEFT)
    ex_max = ttk.Entry(xf, textvariable=manual_x_max[gkey], width=8)
    ex_max.pack(side=tk.LEFT, padx=1)
    ex_max.bind("<Return>", lambda _e: schedule_redraw())

    # Y limits
    yf = ttk.Frame(body)
    yf.pack(anchor="w", padx=12, pady=1)
    ttk.Checkbutton(yf, text="Y:", variable=manual_y_enabled[gkey],
                    width=3).pack(side=tk.LEFT)
    ey_min = ttk.Entry(yf, textvariable=manual_y_min[gkey], width=8)
    ey_min.pack(side=tk.LEFT, padx=1)
    ey_min.bind("<Return>", lambda _e: schedule_redraw())
    ttk.Label(yf, text="\u2013").pack(side=tk.LEFT)
    ey_max = ttk.Entry(yf, textvariable=manual_y_max[gkey], width=8)
    ey_max.pack(side=tk.LEFT, padx=1)
    ey_max.bind("<Return>", lambda _e: schedule_redraw())


def _set_traces(
    group_desc: DataGroupDescriptor,
    trace_vars: dict[str, tk.BooleanVar],
    state: bool,
) -> None:
    for trace in group_desc.traces:
        trace_vars[trace.key].set(state)


# ---------------------------------------------------------------------------
# Collapsible section helper
# ---------------------------------------------------------------------------

def _build_collapsible(parent: ttk.Frame, title: str) -> ttk.Frame:
    """Create a collapsible section.  Returns the body frame."""
    visible = tk.BooleanVar(value=True)

    header = ttk.Frame(parent)
    header.pack(fill=tk.X, padx=4, pady=(4, 0))

    body = ttk.Frame(parent)
    body.pack(fill=tk.X, padx=4)

    def _toggle():
        if visible.get():
            body.pack_forget()
            visible.set(False)
            btn.configure(text="\u25B6")
        else:
            body.pack(fill=tk.X, padx=4, after=header)
            visible.set(True)
            btn.configure(text="\u25BC")

    btn = ttk.Button(header, text="\u25BC", width=2, command=_toggle)
    btn.pack(side=tk.LEFT)
    lbl = ttk.Label(header, text=title, font=theme.SECTION_HEADER_FONT,
                     cursor="hand2")
    lbl.pack(side=tk.LEFT, padx=4)
    lbl.bind("<Button-1>", lambda _e: _toggle())

    return body


# ---------------------------------------------------------------------------
# Mousewheel scrolling
# ---------------------------------------------------------------------------

def _bind_mousewheel(canvas: tk.Canvas) -> None:
    """Bind mousewheel to scroll the sidebar canvas on hover."""
    def _on_mousewheel(event: tk.Event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
    canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))
