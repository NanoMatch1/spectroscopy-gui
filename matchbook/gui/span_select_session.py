"""Interactive per-trace span selection session.

Manages a guided workflow in the main plot area:
- Shows one trace at a time (references first).
- The user drags a SpanSelector to define the search window for peak
  detection in that trace.
- Selections can be re-dragged freely before confirming.
- Navigation buttons (Prev / Next / Finish / Cancel) live inside a
  caller-supplied Tkinter frame in the parameter panel.
- On completion, calls ``on_complete`` with a mapping of
  ``{series_id: (t_start, t_end)}``.

This module has no knowledge of specific analysis modules.
Peak visualisation uses only numpy and scipy for the GUI preview;
the authoritative peak search runs inside the pipeline step function.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import tkinter as tk
from tkinter import ttk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector

try:
    from scipy.signal import savgol_filter as _savgol
    def _smooth(y: np.ndarray, window: int = 11) -> np.ndarray:
        if len(y) > window:
            return _savgol(y, window_length=window, polyorder=3)
        return y.copy()
except ImportError:  # pragma: no cover
    def _smooth(y: np.ndarray, window: int = 11) -> np.ndarray:
        return y.copy()


logger = logging.getLogger(__name__)


def _find_peak_in_range(
    t: np.ndarray,
    y_smooth: np.ndarray,
    xmin: float,
    xmax: float,
) -> float | None:
    """Return the time of the largest |amplitude| between xmin and xmax."""
    mask = (t >= xmin) & (t <= xmax)
    if not np.any(mask):
        return None
    ts = t[mask]
    ys = y_smooth[mask]
    return float(ts[int(np.argmax(np.abs(ys)))])


class SpanSelectSession:
    """Guided per-trace span selection session.

    Parameters
    ----------
    fig :
        The matplotlib Figure owned by the main app.
    canvas :
        The FigureCanvasTkAgg that embeds *fig*.
    traces :
        Ordered list of ``(series_id, t_array, y_array)`` tuples.
        References should appear before samples.
    controls_frame :
        A Tkinter frame (inside the parameter panel) where navigation
        buttons are rendered during the session.
    on_complete :
        Called with ``{series_id: (t_start, t_end)}`` when all traces
        are confirmed.
    on_cancel :
        Called with no arguments when the user cancels.
    """

    def __init__(
        self,
        fig: Figure,
        canvas: FigureCanvasTkAgg,
        traces: list[tuple[str, np.ndarray, np.ndarray]],
        controls_frame: ttk.Frame,
        on_complete: Callable[[dict[str, tuple[float, float]]], None],
        on_cancel: Callable[[], None],
    ) -> None:
        if not traces:
            raise ValueError("SpanSelectSession requires at least one trace.")

        self._fig = fig
        self._canvas = canvas
        self._traces = traces
        self._controls_frame = controls_frame
        self._on_complete_cb = on_complete
        self._on_cancel_cb = on_cancel

        self._idx: int = 0
        self._results: dict[str, tuple[float, float]] = {}
        self._tentative: tuple[float, float] | None = None

        # Matplotlib artists updated in-place during a session
        self._span_selector: SpanSelector | None = None
        self._peak_marker = None   # Line2D
        self._peak_text = None     # Text artist
        self._t_arr: np.ndarray | None = None
        self._y_smooth: np.ndarray | None = None

        # Tkinter nav widgets — created by _build_nav_ui()
        self._lbl_progress: ttk.Label | None = None
        self._lbl_series: ttk.Label | None = None
        self._btn_prev: ttk.Button | None = None
        self._btn_next: ttk.Button | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Build navigation UI and show the first trace."""
        self._build_nav_ui()
        self._show_trace(0)

    def draw_current_trace(self) -> None:
        """Re-render the current trace.

        Called by ``app._do_redraw`` when a session is active, so that
        normal plot_area rendering is bypassed.
        """
        self._render_trace(self._idx)

    def teardown(self) -> None:
        """Remove session UI and release matplotlib resources."""
        self._detach_span_selector()
        for widget in self._controls_frame.winfo_children():
            widget.destroy()

    # ------------------------------------------------------------------
    # Navigation callbacks (bound to Tkinter buttons)
    # ------------------------------------------------------------------

    def _on_next(self) -> None:
        """Confirm current trace's selection and advance (or finish)."""
        series_id = self._traces[self._idx][0]

        if self._tentative is not None:
            self._results[series_id] = self._tentative
        elif series_id not in self._results:
            # Nothing selected and no prior result — prompt user
            if self._lbl_progress is not None:
                self._lbl_progress.configure(
                    text=f"Drag to select a range first! ({self._idx + 1}/{len(self._traces)})")
            return

        self._tentative = None

        if self._idx >= len(self._traces) - 1:
            self._finish()
        else:
            self._show_trace(self._idx + 1)

    def _on_prev(self) -> None:
        """Step back to the previous trace."""
        if self._idx > 0:
            self._tentative = None
            self._show_trace(self._idx - 1)

    def _handle_cancel(self) -> None:
        self.teardown()
        self._on_cancel_cb()

    def _finish(self) -> None:
        self.teardown()
        self._on_complete_cb(dict(self._results))

    # ------------------------------------------------------------------
    # Trace rendering
    # ------------------------------------------------------------------

    def _show_trace(self, idx: int) -> None:
        """Navigate to trace at *idx* and render it."""
        self._idx = idx
        self._detach_span_selector()
        self._update_nav_state()
        self._render_trace(idx)

    def _render_trace(self, idx: int) -> None:
        """Draw the trace for *idx* and attach a fresh SpanSelector."""
        self._detach_span_selector()

        series_id, t_arr, y_arr = self._traces[idx]
        y_smooth = _smooth(y_arr)
        self._t_arr = t_arr
        self._y_smooth = y_smooth

        print(y_arr.shape)

        self._fig.clear()
        ax = self._fig.add_subplot(111)
        self._fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.10)

        ax.plot(t_arr, y_arr, color="steelblue", linewidth=1.2,
                label="averaged")
        ax.plot(t_arr, y_smooth, color="orange", linewidth=1.2, alpha=0.5, label="smoothed") # for helping users find the peak in noisy traces, visual only

        # Show previously confirmed range if revisiting this trace
        if series_id in self._results:
            t_s, t_e = self._results[series_id]
            ax.axvspan(t_s, t_e, alpha=0.08, color="green",
                       label="prior selection")

        # Peak marker — updated in-place by _on_span
        peak_x = np.nan
        peak_y = np.nan
        peak_label = ""

        # If we have a tentative range (e.g. coming back via Prev), show it
        if self._tentative is not None:
            t_s, t_e = self._tentative
            peak_t = _find_peak_in_range(t_arr, y_smooth, t_s, t_e)
            if peak_t is not None:
                mask = np.abs(t_arr - peak_t) < 1e-9
                if np.any(mask):
                    peak_x = peak_t
                    peak_y = float(y_smooth[mask][0])
                    peak_label = f"Peak @ {peak_t:.3f} ps"

        self._peak_marker, = ax.plot(
            [peak_x], [peak_y], "ro", ms=8, zorder=5, label=peak_label or "_nolegend_")
        self._peak_text = ax.text(
            0.5, 0.96, peak_label,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color="red",
            bbox=dict(boxstyle="round,pad=0.2", alpha=0.15, fc="white"))

        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Amplitude (a.u.)")
        short_name = series_id.split("/")[-1]
        ax.set_title(f"Align on peak  —  {short_name}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        ax.text(0.01, 0.99,
                "Drag to select region  ·  re-drag to adjust  ·  Next / Finish to confirm",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", alpha=0.15))

        self._span_selector = SpanSelector(
            ax,
            self._on_span,
            direction="horizontal",
            useblit=True,
            interactive=True,
            props=dict(alpha=0.25, facecolor="orange"),
            minspan=0.0,
        )

        self._canvas.draw_idle()

    def _on_span(self, xmin: float, xmax: float) -> None:
        """SpanSelector callback — update peak marker without re-rendering axes."""
        if xmax < xmin:
            xmin, xmax = xmax, xmin

        self._tentative = (xmin, xmax)

        if self._t_arr is None or self._y_smooth is None:
            return

        peak_t = _find_peak_in_range(self._t_arr, self._y_smooth, xmin, xmax)
        if peak_t is None:
            return

        # Find amplitude at peak (nearest sample)
        idx_pk = int(np.argmin(np.abs(self._t_arr - peak_t)))
        peak_y = float(self._y_smooth[idx_pk])

        if self._peak_marker is not None:
            self._peak_marker.set_data([peak_t], [peak_y])
        if self._peak_text is not None:
            self._peak_text.set_text(f"Peak @ {peak_t:.3f} ps")

        self._canvas.draw_idle()

    def _detach_span_selector(self) -> None:
        if self._span_selector is not None:
            self._span_selector.disconnect_events()
            self._span_selector = None

    # ------------------------------------------------------------------
    # Navigation UI
    # ------------------------------------------------------------------

    def _build_nav_ui(self) -> None:
        """Populate controls_frame with progress labels and nav buttons."""
        for widget in self._controls_frame.winfo_children():
            widget.destroy()

        self._lbl_progress = ttk.Label(
            self._controls_frame, text="", anchor="w",
            font=("TkDefaultFont", 8))
        self._lbl_progress.pack(fill=tk.X, padx=2, pady=(2, 0))

        self._lbl_series = ttk.Label(
            self._controls_frame, text="", anchor="w",
            font=("TkDefaultFont", 8, "italic"), wraplength=210)
        self._lbl_series.pack(fill=tk.X, padx=2)

        btn_row = ttk.Frame(self._controls_frame)
        btn_row.pack(fill=tk.X, pady=(4, 2))

        self._btn_prev = ttk.Button(
            btn_row, text="<< Prev", command=self._on_prev, width=8)
        self._btn_prev.pack(side=tk.LEFT, padx=2)

        self._btn_next = ttk.Button(
            btn_row, text="Next >>", command=self._on_next, width=9)
        self._btn_next.pack(side=tk.LEFT, padx=2)

        ttk.Button(
            btn_row, text="Cancel", command=self._handle_cancel, width=7,
        ).pack(side=tk.RIGHT, padx=2)

        self._update_nav_state()

    def _update_nav_state(self) -> None:
        n = len(self._traces)
        series_id = self._traces[self._idx][0]
        confirmed = len(self._results)

        if self._lbl_progress is not None:
            self._lbl_progress.configure(
                text=f"Trace {self._idx + 1} of {n}  —  {confirmed} confirmed")

        if self._lbl_series is not None:
            self._lbl_series.configure(text=series_id)

        if self._btn_prev is not None:
            self._btn_prev.configure(
                state=tk.NORMAL if self._idx > 0 else tk.DISABLED)

        if self._btn_next is not None:
            label = "Finish" if self._idx == n - 1 else "Next >>"
            self._btn_next.configure(text=label)
