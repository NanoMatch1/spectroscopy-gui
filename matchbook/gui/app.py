"""Main GUI application — orchestrates sidebar, plot area, and parameter panel.

``MatchbookApp`` is the top-level window.  It is entirely generic:
all domain content comes from registered modules via their descriptors.
"""

from __future__ import annotations

import logging
import os
import tkinter as tk
from tkinter import ttk
from typing import Any

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from matchbook.core.data_service import DataKey, DataService
from matchbook.core.module_base import DataGroupDescriptor
from matchbook.core.pipeline import Pipeline
from matchbook.core.registry import Registry
from matchbook.gui import plot_area, sidebar, theme
from matchbook.gui.file_panel import FilePanel
from matchbook.gui.parameter_panel import ParameterPanel
from matchbook.gui.pipeline_view import PipelineView
from matchbook.gui.debug_console import DebugConsole
from matchbook.gui.span_select_session import SpanSelectSession
from matchbook.gui.status_bar import StatusBar

logger = logging.getLogger(__name__)


class MatchbookApp:
    """Top-level GUI window.

    Usage::

        ds = DataService()
        registry = Registry(ds)
        registry.register(some_module)

        app = MatchbookApp(ds, registry)
        app.run()
    """

    def __init__(
        self,
        data_service: DataService,
        registry: Registry,
        title: str = "Matchbook Analysis",
        database: Any | None = None,
    ) -> None:
        self._data_service = data_service
        self._registry = registry
        self._database = database
        self._echo_calls = False   # for debugging: set to True to log all calls to the console
        self.script_dir = os.path.abspath(__file__)
        # self.dev_dir = os.path.join(self.script_dir., "dev_script.py")

        # -- Root window -----------------------------------------------
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("1500x850")
        self.root.minsize(1000, 500)

        # -- Collect descriptors from all registered modules -----------
        self._data_groups: list[DataGroupDescriptor] = registry.all_data_groups
        self._series_names: list[str] = sorted(data_service.list_series())

        # -- State variables -------------------------------------------

        # Series toggles
        self._series_vars: dict[str, tk.BooleanVar] = {}
        for name in self._series_names:
            v = tk.BooleanVar(value=True)
            v.trace_add("write", lambda *_a: self._schedule_redraw())
            self._series_vars[name] = v

        # Trace toggles
        self._trace_vars: dict[str, tk.BooleanVar] = {}
        for group in self._data_groups:
            for trace in group.traces:
                v = tk.BooleanVar(value=trace.default_visible)
                v.trace_add("write", lambda *_a: self._schedule_redraw())
                self._trace_vars[trace.key] = v

        # Per-group axis controls
        self._autoscale_vars: dict[str, tk.BooleanVar] = {}
        self._manual_x_enabled: dict[str, tk.BooleanVar] = {}
        self._manual_y_enabled: dict[str, tk.BooleanVar] = {}
        self._manual_x_min: dict[str, tk.StringVar] = {}
        self._manual_x_max: dict[str, tk.StringVar] = {}
        self._manual_y_min: dict[str, tk.StringVar] = {}
        self._manual_y_max: dict[str, tk.StringVar] = {}
        self._saved_xlim: dict[str, tuple[float, float]] = {}
        self._saved_ylim: dict[str, tuple[float, float]] = {}

        for group in self._data_groups:
            gk = group.key
            v = tk.BooleanVar(value=True)
            v.trace_add("write", lambda *_a: self._schedule_redraw())
            self._autoscale_vars[gk] = v

            mx = tk.BooleanVar(value=False)
            mx.trace_add("write", lambda *_a: self._schedule_redraw())
            self._manual_x_enabled[gk] = mx
            my = tk.BooleanVar(value=False)
            my.trace_add("write", lambda *_a: self._schedule_redraw())
            self._manual_y_enabled[gk] = my

            self._manual_x_min[gk] = tk.StringVar(value="")
            self._manual_x_max[gk] = tk.StringVar(value="")
            self._manual_y_min[gk] = tk.StringVar(value="")
            self._manual_y_max[gk] = tk.StringVar(value="")

        self._redraw_pending = False
        self._span_session: SpanSelectSession | None = None
        self._debug_console: DebugConsole | None = None

        # -- Build layout ----------------------------------------------
        self._build_ui()

        # -- Subscribe to DataService changes --------------------------
        self._data_service.subscribe(self._on_data_changed)

        # -- Subscribe to pipeline events for status bar ---------------
        if self._active_pipeline is not None:
            self._active_pipeline.subscribe(self._on_pipeline_event)

        # -- Initial draw ----------------------------------------------
        self._do_redraw()

    # -----------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------

    def _build_ui(self) -> None:
        # Status bar must be packed first (side=BOTTOM) so it reserves its
        # strip before the other widgets claim all remaining space.
        self._status_bar = StatusBar(self.root)

        # Far left: file panel — keeps its own collapse logic outside the paned area
        self._file_panel = FilePanel(
            self.root,
            on_load_requested=self._on_files_loaded,
            on_db_search_requested=self._on_db_search,
            on_db_load_requested=self._on_db_load,
        )

        # Resizable paned area: sidebar | plot | parameter panel
        # tk.PanedWindow is used (not ttk) for per-pane minsize and width support.
        self._paned = tk.PanedWindow(
            self.root, orient=tk.HORIZONTAL,
            sashwidth=5, sashrelief=tk.GROOVE,
        )
        self._paned.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        _sidebar_pane = ttk.Frame(self._paned)
        _plot_pane = ttk.Frame(self._paned)
        _param_pane = ttk.Frame(self._paned)

        self._paned.add(_sidebar_pane, minsize=120, width=260, stretch="never")
        self._paned.add(_plot_pane,    minsize=300,             stretch="always")
        self._paned.add(_param_pane,   minsize=120, width=300,  stretch="never")

        # Sidebar pane
        _sidebar_frame, self._series_body = sidebar.build_sidebar(
            _sidebar_pane,
            self._series_names,
            self._series_vars,
            self._data_groups,
            self._trace_vars,
            self._autoscale_vars,
            self._manual_x_enabled,
            self._manual_y_enabled,
            self._manual_x_min,
            self._manual_x_max,
            self._manual_y_min,
            self._manual_y_max,
            self._schedule_redraw,
            on_remove_series=self._on_remove_series,
        )

        # Parameter panel pane
        all_step_descs = []
        self._active_pipeline: Pipeline | None = None
        for record in self._registry.registered_modules.values():
            all_step_descs.extend(record.pipeline_step_descriptors)
            if self._active_pipeline is None:
                self._active_pipeline = record.pipeline

        if all_step_descs:
            self._param_panel = ParameterPanel(
                _param_pane,
                all_step_descs,
                on_param_changed=self._on_param_changed,
                on_run_step=self._on_run_step,
                on_activate_span_session=self._start_span_session,
            )

            # Pipeline view sits in the fixed header above the scrollable params
            if self._active_pipeline is not None:
                self._pipeline_view = PipelineView(
                    self._param_panel.header_frame,
                    self._active_pipeline,
                    on_step_selected=self._on_step_selected,
                    on_run_from=self._on_run_from,
                    on_rewind=self._on_rewind,
                )
        else:
            self._param_panel = None
            self._pipeline_view = None

        # Plot pane — canvas fills the pane directly
        self._plot_frame = ttk.Frame(_plot_pane)
        self._plot_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._fig = Figure(figsize=(10, 7), dpi=100)
        self._canvas = FigureCanvasTkAgg(self._fig, master=self._plot_frame)
        self._toolbar = NavigationToolbar2Tk(self._canvas, self._plot_frame)
        self._toolbar.update()
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Debug console toggle — small button on the toolbar + Ctrl+` shortcut
        debug_btn = tk.Button(
            self._toolbar, text="⚙ Debug",
            font=("TkDefaultFont", 8),
            relief="flat", padx=4, pady=1,
            command=self._toggle_debug_console,
        )
        debug_btn.pack(side=tk.RIGHT, padx=4)
        self.root.bind("<Control-grave>", lambda _e: self._toggle_debug_console())

    # -----------------------------------------------------------------
    # Redraw
    # -----------------------------------------------------------------

    def _schedule_redraw(self) -> None:
        """Coalesce rapid toggles into a single redraw (50 ms debounce)."""
        if not self._redraw_pending:
            self._redraw_pending = True
            self.root.after(50, self._do_redraw)

    def _do_redraw(self) -> None:
        self._redraw_pending = False

        # While a span selection session is active, delegate entirely to it
        if self._span_session is not None:
            self._span_session.draw_current_trace()
            self._canvas.draw_idle()
            return

        active_series = [
            name for name in self._series_names
            if self._series_vars[name].get()
        ]
        trace_vis = {k: v.get() for k, v in self._trace_vars.items()}
        autoscale = {k: v.get() for k, v in self._autoscale_vars.items()}
        mx_en = {k: v.get() for k, v in self._manual_x_enabled.items()}
        my_en = {k: v.get() for k, v in self._manual_y_enabled.items()}
        mx_min = {k: v.get() for k, v in self._manual_x_min.items()}
        mx_max = {k: v.get() for k, v in self._manual_x_max.items()}
        my_min = {k: v.get() for k, v in self._manual_y_min.items()}
        my_max = {k: v.get() for k, v in self._manual_y_max.items()}

        plot_area.redraw(
            self._fig,
            self._data_service,
            active_series,
            self._series_names,
            self._data_groups,
            trace_vis,
            autoscale,
            mx_en, my_en,
            mx_min, mx_max,
            my_min, my_max,
            self._saved_xlim,
            self._saved_ylim,
        )
        self._canvas.draw_idle()

    # -----------------------------------------------------------------
    # DataService observer
    # -----------------------------------------------------------------

    def _on_data_changed(self, event: str, key: Any) -> None:
        """Called when data is added/removed — refresh the series list and redraw."""
        new_series = sorted(self._data_service.list_series())
        if new_series != self._series_names:
            # Add toggle vars for newly arrived series
            for name in new_series:
                if name not in self._series_vars:
                    v = tk.BooleanVar(value=True)
                    v.trace_add("write", lambda *_a: self._schedule_redraw())
                    self._series_vars[name] = v
            # Remove vars for series that no longer exist
            for name in list(self._series_vars):
                if name not in new_series:
                    del self._series_vars[name]
            self._series_names = new_series
            # Rebuild the series checkboxes in the sidebar
            sidebar.refresh_series_section(
                self._series_body, self._series_names, self._series_vars,
                on_remove_series=self._on_remove_series)
        self._schedule_redraw()

    def _on_remove_series(self, series_id: str) -> None:
        """Remove all data for a series from the DataService.

        The DataService subscriber (_on_data_changed) handles the subsequent
        sidebar refresh and redraw automatically.
        """
        self._data_service.clear_series(series_id)

    # -----------------------------------------------------------------
    # Span selection session
    # -----------------------------------------------------------------

    def _start_span_session(
        self,
        step_id: str,
        param_desc: Any,
        controls_frame: Any,
        on_done: Any,
        on_cancel: Any,
    ) -> None:
        """Activate a per-trace span selection session.

        Fetches time-domain traces from the DataService (references first),
        creates a SpanSelectSession, and hands control of the plot over to
        it until the user finishes or cancels.
        """
        import json

        data_group = getattr(param_desc, "data_group", "time_domain")

        # Collect traces, annotating with data_type for sorting
        raw: list[tuple[str, Any, Any, str]] = []
        for sid in self._data_service.list_series():
            entry = self._data_service.get(
                DataKey(sid, data_group, "mean"))
            if entry is None:
                continue
            meta = self._data_service.get(DataKey(sid, "_meta", "grouping"))
            dtype = (meta.metadata.get("data_type", "sample")
                     if meta else "sample")
            raw.append((sid, entry.x, entry.y, dtype))

        if not raw:
            logger.warning("No %s/mean traces found for span session", data_group)
            on_cancel()
            return

        # References first, then samples, both sorted by series_id
        order = {"reference": 0, "sample": 1}
        raw.sort(key=lambda r: (order.get(r[3], 1), r[0]))
        traces = [(sid, t, y) for sid, t, y, _ in raw]

        def _on_complete(results: dict) -> None:
            self._span_session = None
            results_json = json.dumps(results)
            on_done(results_json, len(results))
            self._schedule_redraw()

        def _on_session_cancel() -> None:
            self._span_session = None
            on_cancel()
            self._schedule_redraw()

        self._span_session = SpanSelectSession(
            fig=self._fig,
            canvas=self._canvas,
            traces=traces,
            controls_frame=controls_frame,
            on_complete=_on_complete,
            on_cancel=_on_session_cancel,
        )
        self._span_session.start()

    # -----------------------------------------------------------------
    # Pipeline interaction callbacks
    # -----------------------------------------------------------------

    def _on_param_changed(self, step_id: str, param_name: str,
                          value: Any) -> None:
        """Widget value changed — push to pipeline."""
        if self._active_pipeline is not None:
            try:
                self._active_pipeline.set_param(step_id, param_name, value)
            except KeyError:
                pass
        if self._pipeline_view is not None:
            self._pipeline_view.refresh()

    def _on_run_step(self, step_id: str) -> None:
        """Run exactly one pipeline step for all active series.

        Used by the per-step '▶ Run' button in the parameter panel.
        Does not cascade into downstream steps.
        """
        if self._active_pipeline is None:
            return
        for series_id in self._series_names:
            if self._series_vars.get(series_id, tk.BooleanVar(value=False)).get():
                self._active_pipeline.run_step(
                    step_id, self._data_service, series_id)
        self._schedule_redraw()

    def _on_run_from(self, step_id: str) -> None:
        """Run the pipeline from the given step through all downstream steps.

        Used by 'Run from here' in the pipeline view.
        """
        if self._active_pipeline is None:
            return
        for series_id in self._series_names:
            if self._series_vars.get(series_id, tk.BooleanVar(value=False)).get():
                self._active_pipeline.run_from(
                    step_id, self._data_service, series_id)
        self._schedule_redraw()

    def _on_pipeline_event(self, event: str, step_id: str) -> None:
        """React to pipeline step_start / step_done / rewind events."""
        if self._active_pipeline is None:
            return
        step = self._active_pipeline.get_step(step_id) if step_id else None
        name = step.name if step else step_id

        if event == "step_start":
            self._show_status(f"Running \u2018{name}\u2019\u2026", level="info")
        elif event == "step_done":
            self._show_status(f"\u2713 \u2018{name}\u2019 completed", level="success")
            if step_id == "group_files":
                self._file_panel.refresh_grouping(self._data_service)
        elif event == "rewind":
            n = self._active_pipeline.history_length
            self._show_status(f"Pipeline rewound (history: {n})", level="info")

    def _show_status(self, message: str, level: str = "info") -> None:
        """Display a transient notification in the status bar."""
        self._status_bar.show(message, level=level)

    def _on_step_selected(self, step_id: str) -> None:
        logger.debug(f"Pipeline step selected: {step_id}")

    def _on_rewind(self, history_index: int) -> None:
        if self._active_pipeline is not None:
            self._active_pipeline.rewind(history_index)
            if self._pipeline_view is not None:
                self._pipeline_view.refresh()

    # -----------------------------------------------------------------
    # File panel callback
    # -----------------------------------------------------------------

    def _on_files_loaded(self, file_paths: list[str]) -> None:
        """Called by FilePanel after files are successfully ingested.

        Pushes paths into the pipeline's load_from_files params and runs
        the full pipeline (load → group → downstream steps).
        """
        if self._active_pipeline is None:
            return

        # Ensure loader modules are imported
        import matchbook.io.loaders  # noqa: F401

        path_str = "\n".join(file_paths)

        # Derive a series ID from the common parent directory.
        # Using dirname first avoids os.path.commonpath returning a file path
        # when only a single file is selected.
        parent_dirs = [os.path.dirname(p) for p in file_paths]
        common_dir = os.path.commonpath(parent_dirs) if parent_dirs else ""
        series_id = os.path.basename(common_dir) or "loaded"

        # Run only the load step — downstream steps are left for the user
        # to run manually once they are satisfied with the loaded data.
        self._active_pipeline.set_param(
            "load_from_files", "file_paths", path_str)
        self._active_pipeline.run_step(
            "load_from_files", self._data_service, series_id)
        # Clear downstream caches so the pipeline view shows them as pending
        self._active_pipeline.invalidate_downstream("load_from_files")

        # Refresh pipeline view
        if self._pipeline_view is not None:
            self._pipeline_view.refresh()
        self._schedule_redraw()

    # -----------------------------------------------------------------
    # Database callbacks
    # -----------------------------------------------------------------

    def _on_db_search(self, query_text: str) -> None:
        """Search the database and show results in the file panel."""
        if self._database is None:
            self._file_panel.show_db_results([])
            return

        criteria: dict[str, str] = {}
        if query_text:
            criteria["text_search"] = query_text
        results = self._database.search(**criteria)
        self._file_panel.show_db_results(results)

    def _on_db_load(self, series_ids: list[str]) -> None:
        """Load selected series from the database into the DataService."""
        if self._database is None:
            return

        for sid in series_ids:
            self._database.load_series(sid, self._data_service)

        self._schedule_redraw()

    # -----------------------------------------------------------------
    # Debug console
    # -----------------------------------------------------------------

    def _toggle_debug_console(self) -> None:
        """Show or hide the debug REPL console."""
        import numpy as np
        if self._debug_console is None:
            self._debug_console = DebugConsole(
                parent=self.root,
                namespace={
                    "app":      self,
                    "ds":       self._data_service,
                    "registry": self._registry,
                    "pipeline": self._active_pipeline,
                    "fig":      self._fig,
                    "canvas":   self._canvas,
                    "np":       np,
                    "calls":      self._echo_calls,   # for toggling call logging from the console itself
                    "data":       self._data_service.report_data,  # alias for convenience
                    "datfiles":       self._data_service.remove_redundant_dat
                },
            )
        self._debug_console.toggle()

    # -----------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------

    def run(self) -> None:
        """Start the tkinter event loop."""
        self.root.mainloop()
