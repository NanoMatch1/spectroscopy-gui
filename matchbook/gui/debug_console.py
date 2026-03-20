"""Embedded REPL debug console.

Provides an interactive Python interpreter plus a Script editor inside a
``Toplevel`` window.  The namespace is pre-loaded with live references to
the application's key objects and developer shortcuts.

Pre-loaded names
----------------
app, ds, registry, pipeline, fig, canvas, np
    Live application objects.
scripts  (:class:`~matchbook.gui.dev_kit.DevKit`)
    Workflow shortcuts — type ``scripts.help()`` for a full list.
tracer   (:class:`~matchbook.gui.dev_kit.CallTracer`)
    Wraps methods/functions to echo their calls to this window,
    even when triggered from Tkinter events outside the REPL.
DataKey
    Shortcut for ``matchbook.core.data_service.DataKey``.

Keyboard shortcuts
------------------
Ctrl+\`   toggle show/hide
Up/Down   cycle command history (REPL tab)
Tab       insert 4 spaces (REPL tab)
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
    """Interactive Python REPL + Script editor in a detached Toplevel window.

    Parameters
    ----------
    parent :
        The root Tk window.
    namespace :
        Names pre-injected into the interpreter.  ``scripts``, ``tracer``,
        and ``DataKey`` are added automatically.
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
        self._title = title

        # Output widget — None until window is first built
        self._output: tk.Text | None = None

        # Extend namespace with dev-kit objects
        from matchbook.core.data_service import DataKey
        from matchbook.gui.dev_kit import CallTracer, DevKit

        tracer = CallTracer(write_fn=self._write_trace)
        scripts = DevKit(
            app=namespace.get("app"),
            data_service=namespace.get("ds"),
            pipeline=namespace.get("pipeline"),
            namespace=namespace,
        )
        namespace.update({
            "scripts": scripts,
            "tracer":  tracer,
            "DataKey": DataKey,
        })
        self._namespace = namespace

        self._window: tk.Toplevel | None = None
        self._history: list[str] = []
        self._history_idx: int = -1
        self._pending_lines: list[str] = []
        self._console: code.InteractiveConsole | None = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def toggle(self) -> None:
        """Show if hidden, hide if visible."""
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

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------

    def _build_window(self) -> None:
        self._window = tk.Toplevel(self._parent)
        self._window.title(self._title)
        self._window.geometry("820x520")
        self._window.minsize(500, 300)

        mono = self._monospace_font()

        notebook = ttk.Notebook(self._window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # -- Tab 1: REPL -----------------------------------------------
        repl_frame = ttk.Frame(notebook)
        notebook.add(repl_frame, text="  REPL  ")
        self._build_repl_tab(repl_frame, mono)

        # -- Tab 2: Script editor --------------------------------------
        script_frame = ttk.Frame(notebook)
        notebook.add(script_frame, text="  Script  ")
        self._build_script_tab(script_frame, mono)

        self._window.protocol("WM_DELETE_WINDOW", self._window.withdraw)
        self._console = code.InteractiveConsole(locals=self._namespace)
        self._write_banner()

    def _build_repl_tab(self, parent: ttk.Frame, mono: Any) -> None:
        # Output pane
        out_frame = ttk.Frame(parent)
        out_frame.pack(fill=tk.BOTH, expand=True)

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

        self._output.tag_configure("stdout",  foreground="#d4d4d4")
        self._output.tag_configure("stderr",  foreground="#f48771")
        self._output.tag_configure("prompt",  foreground="#569cd6")
        self._output.tag_configure("result",  foreground="#9cdcfe")
        self._output.tag_configure("trace",   foreground="#4ec9b0")

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)

        # Input row
        input_row = ttk.Frame(parent)
        input_row.pack(fill=tk.X, padx=4, pady=(0, 4))

        self._prompt_lbl = ttk.Label(
            input_row, text=self._PROMPT_PS1, font=mono,
            foreground="#569cd6", background="#1e1e1e", width=4)
        self._prompt_lbl.pack(side=tk.LEFT)

        self._entry = tk.Entry(
            input_row,
            font=mono,
            background="#252526",
            foreground="#d4d4d4",
            insertbackground="white",
            relief="flat",
        )
        self._entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._entry.focus_set()

        self._entry.bind("<Return>", self._on_return)
        self._entry.bind("<Up>",     self._on_history_up)
        self._entry.bind("<Down>",   self._on_history_down)
        self._entry.bind("<Tab>",    self._on_tab)

    def _build_script_tab(self, parent: ttk.Frame, mono: Any) -> None:
        """Multi-line editor for paste-and-run scripts."""
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, padx=4, pady=(4, 2))

        ttk.Label(toolbar,
                  text="Paste or type a multi-line script, then click Run.",
                  foreground="grey").pack(side=tk.LEFT)

        ttk.Button(toolbar, text="▶  Run Script",
                   command=self._run_script).pack(side=tk.RIGHT)
        ttk.Button(toolbar, text="Clear",
                   command=self._clear_script).pack(side=tk.RIGHT, padx=(0, 4))

        editor_frame = ttk.Frame(parent)
        editor_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        self._script_editor = tk.Text(
            editor_frame,
            font=mono,
            background="#1e1e1e",
            foreground="#d4d4d4",
            insertbackground="white",
            relief="flat",
            borderwidth=0,
            undo=True,
        )
        sc_scroll = ttk.Scrollbar(editor_frame, command=self._script_editor.yview)
        self._script_editor.configure(yscrollcommand=sc_scroll.set)
        sc_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._script_editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._script_editor.bind("<Tab>", lambda e: (
            self._script_editor.insert(tk.INSERT, "    "), "break")[1])

        # Pre-populate with a starter comment
        starter = (
            "# Script tab — runs in the same namespace as the REPL.\n"
            "# Output appears in the REPL tab.\n"
            "# Example:\n"
            "#   scripts.load()\n"
            "#   scripts.run('align_on_peak')\n"
            "#   scripts.series()\n"
        )
        self._script_editor.insert("1.0", starter)

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------

    def _write_banner(self) -> None:
        keys = ", ".join(sorted(self._namespace.keys()))
        banner = (
            f"Matchbook debug console  —  Python {sys.version.split()[0]}\n"
            f"Names: {keys}\n"
            "Type  scripts.help()  for workflow shortcuts.\n"
            "Type  help(obj)  or  dir(obj)  to inspect anything.\n"
            "Ctrl+`  toggles this window.\n"
        )
        self._write(banner, "result")

    # ------------------------------------------------------------------
    # REPL input handling
    # ------------------------------------------------------------------

    def _on_return(self, _event: Any) -> str:
        line = self._entry.get()
        self._entry.delete(0, tk.END)

        prompt = self._PROMPT_PS2 if self._pending_lines else self._PROMPT_PS1
        self._write(prompt + line + "\n", "prompt")

        if line.strip():
            self._history.append(line)
        self._history_idx = -1

        self._pending_lines.append(line)
        source = "\n".join(self._pending_lines)

        self._exec_with_redirect(lambda: self._console.push(source),  # type: ignore[union-attr]
                                  returns_needs_more=True)
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
        self._entry.insert(tk.INSERT, "    ")
        return "break"

    def _set_entry(self, text: str) -> None:
        self._entry.delete(0, tk.END)
        self._entry.insert(0, text)

    # ------------------------------------------------------------------
    # Script tab actions
    # ------------------------------------------------------------------

    def _run_script(self) -> None:
        source = self._script_editor.get("1.0", tk.END).strip()
        if not source:
            return
        self._write("# --- running script ---\n", "prompt")

        def _exec():
            exec(compile(source, "<script>", "exec"), self._namespace)  # noqa: S102
            return False  # never needs_more

        self._exec_with_redirect(_exec, returns_needs_more=False)
        self._write("# --- done ---\n", "prompt")

    def _clear_script(self) -> None:
        self._script_editor.delete("1.0", tk.END)

    # ------------------------------------------------------------------
    # Execution helper
    # ------------------------------------------------------------------

    def _exec_with_redirect(self, fn: Any, *, returns_needs_more: bool) -> None:
        """Execute *fn* with stdout/stderr redirected to the output widget."""
        old_out, old_err = sys.stdout, sys.stderr
        assert self._output is not None
        sys.stdout = _WidgetWriter(self._output, "stdout")
        sys.stderr = _WidgetWriter(self._output, "stderr")
        try:
            result = fn()
            if returns_needs_more:
                needs_more: bool = result
            else:
                needs_more = False
        finally:
            sys.stdout = old_out
            sys.stderr = old_err

        if needs_more:
            self._prompt_lbl.configure(text=self._PROMPT_PS2)
        else:
            self._pending_lines.clear()
            self._prompt_lbl.configure(text=self._PROMPT_PS1)

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _write(self, text: str, tag: str = "stdout") -> None:
        """Write to the output pane.  Safe to call before window is built."""
        if self._output is None:
            return
        self._output.configure(state="normal")
        self._output.insert("end", text, tag)
        self._output.configure(state="disabled")
        self._output.see("end")

    def _write_trace(self, text: str) -> None:
        """Write callback used by CallTracer — always routes to the widget."""
        self._write(text, "trace")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _monospace_font() -> tkfont.Font:
        families = tkfont.families()
        for name in ("Consolas", "Cascadia Code", "Courier New", "Courier"):
            if name in families:
                return tkfont.Font(family=name, size=10)
        return tkfont.Font(family="TkFixedFont", size=10)
