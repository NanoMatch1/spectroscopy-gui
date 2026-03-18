"""Embedded REPL debug console.

Provides an interactive Python interpreter inside a ``Toplevel`` window.
The namespace is pre-loaded with live references to the application's
key objects so that any internal state can be inspected or mutated
without restarting the GUI.

Usage (from app.py)::

    console = DebugConsole(root, namespace={
        'app':      self,
        'ds':       self._data_service,
        'registry': self._registry,
        'pipeline': self._active_pipeline,
        'fig':      self._fig,
        'canvas':   self._canvas,
        'np':       numpy,
    })
    console.toggle()          # show / hide
    root.bind('<Control-`>',  lambda _e: console.toggle())
"""

from __future__ import annotations

import code
import sys
import tkinter as tk
from tkinter import ttk, font as tkfont
from typing import Any


# ---------------------------------------------------------------------------
# I/O redirector
# ---------------------------------------------------------------------------

class _WidgetWriter:
    """Redirects ``write`` calls to a Tkinter Text widget with a colour tag."""

    def __init__(self, text_widget: tk.Text, tag: str) -> None:
        self._widget = text_widget
        self._tag = tag

    def write(self, text: str) -> None:
        self._widget.configure(state="normal")
        self._widget.insert("end", text, self._tag)
        self._widget.configure(state="disabled")
        self._widget.see("end")

    def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Console window
# ---------------------------------------------------------------------------

class DebugConsole:
    """Interactive Python REPL in a detached Toplevel window.

    Parameters
    ----------
    parent :
        The root Tk window (used as the Toplevel's parent).
    namespace :
        Dict of names pre-injected into the interpreter.  Typically
        includes ``app``, ``ds``, ``pipeline``, ``fig``, ``np``, etc.
    title :
        Window title.
    """

    _PROMPT_PS1 = ">>> "
    _PROMPT_PS2 = "... "

    def __init__(
        self,
        parent: tk.Tk,
        namespace: dict[str, Any],
        title: str = "Debug Console",
    ) -> None:
        self._parent = parent
        self._namespace = namespace
        self._title = title

        self._window: tk.Toplevel | None = None
        self._history: list[str] = []
        self._history_idx: int = -1
        self._pending_lines: list[str] = []   # accumulates multi-line input
        self._console: code.InteractiveConsole | None = None
        self._echo_calls = False   # for debugging: set to True to log all calls to the console

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def toggle(self) -> None:
        """Show the console if hidden; hide it if visible."""
        if self._window is None or not self._window.winfo_exists():
            self._build_window()
        elif self._window.winfo_viewable():
            self._window.withdraw()
        else:
            self._window.deiconify()
            self._window.lift()

    def show(self) -> None:
        if self._window is None or not self._window.winfo_exists():
            self._build_window()
        else:
            self._window.deiconify()
            self._window.lift()

    def hide(self) -> None:
        if self._window and self._window.winfo_exists():
            self._window.withdraw()

    def echo_calls(self) -> None:
        """When enabled, every call to the console will be logged to stdout."""
        self._parent._echo_calls = not self._parent._echo_calls
        print(f"DebugConsole: echo_calls set to {self._parent._echo_calls}")
        

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------

    def _build_window(self) -> None:
        self._window = tk.Toplevel(self._parent)
        self._window.title(self._title)
        self._window.geometry("760x400")
        self._window.minsize(500, 200)

        # Use a monospace font
        mono = tkfont.Font(family="Consolas", size=10)
        if "Consolas" not in tkfont.families():
            mono = tkfont.Font(family="Courier New", size=10)

        # Output area
        out_frame = ttk.Frame(self._window)
        out_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 0))

        self._output = tk.Text(
            out_frame,
            state="disabled",
            wrap="word",
            font=mono,
            background="#1e1e1e",
            foreground="#d4d4d4",
            insertbackground="white",
            relief="flat",
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(out_frame, command=self._output.yview)
        self._output.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Colour tags
        self._output.tag_configure("stdout", foreground="#d4d4d4")
        self._output.tag_configure("stderr", foreground="#f48771")
        self._output.tag_configure("prompt", foreground="#569cd6")
        self._output.tag_configure("result", foreground="#9cdcfe")

        # Separator
        ttk.Separator(self._window, orient=tk.HORIZONTAL).pack(
            fill=tk.X, padx=4, pady=2)

        # Input row
        input_frame = ttk.Frame(self._window)
        input_frame.pack(fill=tk.X, padx=4, pady=(0, 4))

        self._prompt_lbl = ttk.Label(
            input_frame, text=self._PROMPT_PS1, font=mono,
            foreground="#569cd6", background="#1e1e1e", width=4)
        self._prompt_lbl.pack(side=tk.LEFT)

        self._entry = tk.Entry(
            input_frame,
            font=mono,
            background="#252526",
            foreground="#d4d4d4",
            insertbackground="white",
            relief="flat",
        )
        self._entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._entry.focus_set()

        # Key bindings on entry
        self._entry.bind("<Return>", self._on_return)
        self._entry.bind("<Up>",     self._on_history_up)
        self._entry.bind("<Down>",   self._on_history_down)
        self._entry.bind("<Tab>",    self._on_tab)

        # Closing the window just hides it
        self._window.protocol("WM_DELETE_WINDOW", self._window.withdraw)

        # Build interpreter and print banner
        self._console = code.InteractiveConsole(locals=self._namespace)
        self._write_banner()

    def _write_banner(self) -> None:
        keys = ", ".join(sorted(self._namespace.keys()))
        banner = (
            f"Matchbook debug console  —  Python {sys.version.split()[0]}\n"
            f"Available names: {keys}\n"
            f"Type help(obj) or dir(obj) to inspect.  Ctrl+` to toggle.\n"
        )
        self._write(banner, "result")
        self._write_prompt()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _on_return(self, _event: Any) -> str:
        line = self._entry.get()
        self._entry.delete(0, tk.END)

        # Echo input with prompt
        prompt = self._PROMPT_PS2 if self._pending_lines else self._PROMPT_PS1
        self._write(prompt + line + "\n", "prompt")

        # History
        if line.strip():
            self._history.append(line)
        self._history_idx = -1

        # Push to interpreter
        self._pending_lines.append(line)
        source = "\n".join(self._pending_lines)

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = _WidgetWriter(self._output, "stdout")
        sys.stderr = _WidgetWriter(self._output, "stderr")
        try:
            needs_more = self._console.push(source)  # type: ignore[union-attr]
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        if needs_more:
            self._prompt_lbl.configure(text=self._PROMPT_PS2)
        else:
            self._pending_lines.clear()
            self._prompt_lbl.configure(text=self._PROMPT_PS1)

        self._write_prompt()
        return "break"

    def _on_history_up(self, _event: Any) -> str:
        if not self._history:
            return "break"
        self._history_idx = max(0, (
            len(self._history) - 1
            if self._history_idx == -1
            else self._history_idx - 1
        ))
        self._set_entry(self._history[self._history_idx])
        return "break"

    def _on_history_down(self, _event: Any) -> str:
        if self._history_idx == -1:
            return "break"
        self._history_idx += 1
        if self._history_idx >= len(self._history):
            self._history_idx = -1
            self._set_entry("")
        else:
            self._set_entry(self._history[self._history_idx])
        return "break"

    def _on_tab(self, _event: Any) -> str:
        """Insert four spaces (basic indent support for multi-line input)."""
        self._entry.insert(tk.INSERT, "    ")
        return "break"

    def _set_entry(self, text: str) -> None:
        self._entry.delete(0, tk.END)
        self._entry.insert(0, text)

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _write(self, text: str, tag: str = "stdout") -> None:
        self._output.configure(state="normal")
        self._output.insert("end", text, tag)
        self._output.configure(state="disabled")
        self._output.see("end")

    def _write_prompt(self) -> None:
        # The prompt is shown in the label, not the output area — nothing to do
        pass
