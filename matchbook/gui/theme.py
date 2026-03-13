"""Palette, fonts, and style constants for the GUI."""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Colour palette — consistent across all series
# ---------------------------------------------------------------------------

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]


def series_colour(index: int) -> str:
    """Return a colour for the *index*-th series, cycling the palette."""
    return PALETTE[index % len(PALETTE)]


def series_colour_by_id(series_id: str, all_series: list[str]) -> str:
    """Return a colour for *series_id* based on its sorted position."""
    try:
        idx = sorted(all_series).index(series_id)
    except ValueError:
        idx = hash(series_id)
    return series_colour(idx)


# ---------------------------------------------------------------------------
# Line styles — cycle through for traces within the same series
# ---------------------------------------------------------------------------

LINE_STYLES = ["-", "--", "-.", ":"]


def trace_line_style(trace_index: int) -> str:
    """Return a matplotlib line style for the *trace_index*-th trace."""
    return LINE_STYLES[trace_index % len(LINE_STYLES)]


# ---------------------------------------------------------------------------
# Font / layout constants
# ---------------------------------------------------------------------------

SIDEBAR_WIDTH = 260
SIDEBAR_FONT_SIZE = 10
SECTION_HEADER_FONT = ("TkDefaultFont", 11, "bold")
AXIS_CONTROL_FONT = ("TkDefaultFont", 9, "italic")
LEGEND_FONT_SIZE = 7
TITLE_FONT_SIZE = 11
