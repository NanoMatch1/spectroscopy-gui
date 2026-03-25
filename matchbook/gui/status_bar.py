"""Thin status bar pinned to the bottom of the root window.

Shows transient one-line notifications from the pipeline and other app
events.  Messages auto-clear after a configurable timeout.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class StatusBar:
    """Bottom-of-window notification strip.

    Usage::

        bar = StatusBar(root)          # packs itself side=BOTTOM (call first!)
        bar.show("Done", level="success")
        bar.show("Error", level="error", timeout_ms=10_000)
    """

    _COLOURS = {
        "info":    "#888888",
        "success": "#2e7d32",
        "warning": "#e65100",
        "error":   "#c62828",
    }

    def __init__(self, parent: tk.Widget) -> None:
        self._parent = parent
        self._after_id: str | None = None

        # Must be packed before other widgets so it reserves the bottom strip.
        self.frame = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
        self.frame.pack(side=tk.BOTTOM, fill=tk.X)

        self._label = tk.Label(
            self.frame, text="", anchor="e",
            font=("TkDefaultFont", 8),
            fg=self._COLOURS["info"],
        )
        self._label.pack(side=tk.RIGHT, padx=10, pady=2)

    def show(
        self,
        message: str,
        level: str = "info",
        timeout_ms: int = 6_000,
    ) -> None:
        """Display *message* for *timeout_ms* milliseconds then clear."""
        if self._after_id is not None:
            self._parent.after_cancel(self._after_id)
        colour = self._COLOURS.get(level, self._COLOURS["info"])
        self._label.configure(text=message, fg=colour)
        self._after_id = self._parent.after(timeout_ms, self._clear)

    def _clear(self) -> None:
        self._label.configure(text="")
        self._after_id = None
