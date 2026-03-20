"""Generic plot area — renders traces from the DataService.

No module-specific knowledge.  All plot content is driven by the active
series, active data groups, trace visibility flags, and data looked up
from the DataService.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure

from matchbook.core.data_service import DataKey, DataService
from matchbook.core.module_base import DataGroupDescriptor
from matchbook.gui import theme


def redraw(
    fig: Figure,
    data_service: DataService,
    active_series: list[str],
    all_series: list[str],
    data_groups: list[DataGroupDescriptor],
    trace_visibility: dict[str, bool],
    autoscale: dict[str, bool],
    manual_x_enabled: dict[str, bool],
    manual_y_enabled: dict[str, bool],
    manual_x_min: dict[str, str],
    manual_x_max: dict[str, str],
    manual_y_min: dict[str, str],
    manual_y_max: dict[str, str],
    saved_xlim: dict[str, tuple[float, float]],
    saved_ylim: dict[str, tuple[float, float]],
) -> None:
    """Clear the figure and redraw all active traces.

    This function is module-agnostic: it reads trace data from the
    DataService using keys declared in each ``DataGroupDescriptor``.
    """
    fig.clear()

    # Determine which groups have at least one visible trace
    visible_groups = [
        g for g in sorted(data_groups, key=lambda g: g.order)
        if any(trace_visibility.get(t.key, False) for t in g.traces)
    ]

    # Fixed subplot margins keep axes boundaries stable across redraws.
    # Without this, tick-label width changes shift the left edge on every toggle.
    _ADJUST = dict(left=0.10, right=0.97, top=0.94, bottom=0.09, hspace=0.45)

    if not visible_groups or not active_series:
        ax = fig.add_subplot(111)
        fig.subplots_adjust(**_ADJUST)
        ax.text(0.5, 0.5, "Select at least one series and one trace",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color="grey")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return

    axes = fig.subplots(len(visible_groups), 1, squeeze=False)
    fig.subplots_adjust(**_ADJUST)

    for row_idx, group_desc in enumerate(visible_groups):
        ax = axes[row_idx, 0]
        gkey = group_desc.key

        active_traces = [
            t for t in group_desc.traces
            if trace_visibility.get(t.key, False)
        ]

        y_labels_used: set[str] = set()
        has_any_data = False
        missing_series: list[str] = []

        for series_id in active_series:
            colour = theme.series_colour_by_id(series_id, all_series)
            series_has_data = False

            for trace_i, trace_desc in enumerate(active_traces):
                entry = data_service.get(
                    DataKey(series_id, trace_desc.group, trace_desc.name)
                )
                if entry is None:
                    continue

                x, y = entry.x, entry.y
                mask = np.isfinite(x) & np.isfinite(y)
                if not np.any(mask):
                    continue

                series_has_data = True
                has_any_data = True
                y_labels_used.add(trace_desc.y_label or group_desc.label)

                ls = (trace_desc.line_style
                      or theme.trace_line_style(trace_i))
                ax.plot(
                    x[mask], y[mask],
                    color=colour,
                    linestyle=ls,
                    linewidth=1.2,
                    label=f"{series_id}: {trace_desc.label}",
                    alpha=0.85,
                )

            if not series_has_data and active_traces:
                missing_series.append(series_id)

        # Labels
        ax.set_xlabel(group_desc.x_label)
        if len(y_labels_used) == 1:
            ax.set_ylabel(y_labels_used.pop())
        else:
            ax.set_ylabel(group_desc.label)
        ax.set_title(group_desc.label, fontsize=theme.TITLE_FONT_SIZE,
                     fontweight="bold")

        if has_any_data:
            ax.legend(fontsize=theme.LEGEND_FONT_SIZE, loc="best",
                      ncol=max(1, len(active_series)),
                      framealpha=0.8)

        # Missing-data annotation
        if missing_series:
            note = "No data: " + ", ".join(missing_series)
            ax.annotate(
                note, xy=(0.99, 0.01), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=8,
                color="red", fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", ec="red", alpha=0.7),
            )

        ax.grid(True, alpha=0.3)

        # -- Axis limits -----------------------------------------------
        x_set = y_set = False

        if manual_x_enabled.get(gkey, False):
            try:
                xlo = float(manual_x_min.get(gkey, ""))
                xhi = float(manual_x_max.get(gkey, ""))
                if xlo < xhi:
                    ax.set_xlim(xlo, xhi)
                    x_set = True
            except (ValueError, TypeError):
                pass

        if manual_y_enabled.get(gkey, False):
            try:
                ylo = float(manual_y_min.get(gkey, ""))
                yhi = float(manual_y_max.get(gkey, ""))
                if ylo < yhi:
                    ax.set_ylim(ylo, yhi)
                    y_set = True
            except (ValueError, TypeError):
                pass

        if not autoscale.get(gkey, True):
            if not x_set and gkey in saved_xlim:
                ax.set_xlim(saved_xlim[gkey])
            if not y_set and gkey in saved_ylim:
                ax.set_ylim(saved_ylim[gkey])

        # Save current limits for next non-autoscale redraw
        saved_xlim[gkey] = ax.get_xlim()
        saved_ylim[gkey] = ax.get_ylim()


