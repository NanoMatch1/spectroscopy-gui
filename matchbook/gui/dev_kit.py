"""Developer toolkit for the debug console.

Exposes two objects into the console namespace:

``scripts``  — :class:`DevKit`
    Workflow shortcuts so you don't have to replay GUI events to reach a
    desired state.  Type ``scripts.help()`` at the console prompt.

``tracer``   — :class:`CallTracer`
    Wraps live methods or module-level functions and prints a line to the
    console window every time they are called.  Works even when calls
    happen from Tkinter events (outside the REPL's stdout redirect).
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Callable
from pathlib import Path


# ---------------------------------------------------------------------------
# CallTracer
# ---------------------------------------------------------------------------

class CallTracer:
    """Wrap live methods/functions to echo calls to the debug console.

    All output goes directly to the console Text widget so it appears
    whether the call originates inside the REPL or from a Tkinter event.

    Parameters
    ----------
    write_fn :
        Callable(text: str) that writes to the console output widget.

    Examples
    --------
    >>> tracer.watch(app, '_on_run_from', '_on_param_changed')
    >>> tracer.watch(pipeline, 'run_from', 'set_param')
    >>> tracer.watch_module('matchbook.modules.thz.adapter', 'step_align_on_peak')
    >>> tracer.list()
    >>> tracer.clear()
    """

    def __init__(self, write_fn: Callable[[str], None]) -> None:
        self._write = write_fn
        # Stores (obj_or_module, attr_name) -> original callable
        self._watched: dict[tuple[int, str], tuple[Any, str, Any]] = {}
        #                                           obj, label, original

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def watch(self, obj: Any, *names: str) -> None:
        """Wrap one or more methods on *obj*."""
        for name in names:
            self._wrap(obj, name, f"{type(obj).__name__}.{name}")

    def watch_all(self, obj: Any) -> None:
        """Wrap every non-private method on *obj*."""
        for name in dir(type(obj)):
            if name.startswith("_"):
                continue
            attr = getattr(obj, name, None)
            if callable(attr):
                self._wrap(obj, name, f"{type(obj).__name__}.{name}")

    def watch_module(self, module_path: str, *fn_names: str) -> None:
        """Wrap module-level functions by dotted module path.

        Example::

            tracer.watch_module(
                'matchbook.modules.thz.adapter',
                'step_load_from_files', 'step_align_on_peak')
        """
        mod = importlib.import_module(module_path)
        short = module_path.split(".")[-1]
        for name in fn_names:
            self._wrap(mod, name, f"{short}.{name}")

    def unwatch(self, obj: Any, *names: str) -> None:
        """Restore original callables for the named methods."""
        for name in names:
            key = (id(obj), name)
            if key in self._watched:
                _, _, original = self._watched.pop(key)
                setattr(obj, name, original)
                self._write(f"[TRACER] unwatched {name}\n")

    def clear(self) -> None:
        """Restore all wrapped callables."""
        for (_, name), (obj, label, original) in list(self._watched.items()):
            try:
                setattr(obj, name, original)
            except Exception:
                pass
        self._watched.clear()
        self._write("[TRACER] all traces cleared\n")

    def list(self) -> None:
        """Print what is currently being traced."""
        if not self._watched:
            self._write("[TRACER] nothing being traced\n")
            return
        self._write("[TRACER] active traces:\n")
        for (_, _), (_, label, _) in self._watched.items():
            self._write(f"  {label}\n")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _wrap(self, obj: Any, name: str, label: str) -> None:
        key = (id(obj), name)
        if key in self._watched:
            return  # already watching

        try:
            original = getattr(obj, name)
        except AttributeError:
            self._write(f"[TRACER] {label} not found — skipping\n")
            return

        if not callable(original):
            self._write(f"[TRACER] {label} is not callable — skipping\n")
            return

        write = self._write

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            parts = [_short_repr(a) for a in args]
            parts += [f"{k}={_short_repr(v)}" for k, v in kwargs.items()]
            write(f"[CALL] {label}({', '.join(parts)})\n")
            return original(*args, **kwargs)

        self._watched[key] = (obj, label, original)
        try:
            setattr(obj, name, wrapper)
        except (AttributeError, TypeError) as exc:
            del self._watched[key]
            self._write(f"[TRACER] cannot wrap {label}: {exc}\n")


def _short_repr(v: Any, max_len: int = 60) -> str:
    r = repr(v)
    if len(r) > max_len:
        r = r[:max_len] + "…"
    return r


# ---------------------------------------------------------------------------
# DevKit
# ---------------------------------------------------------------------------

class DevKit:
    """Workflow shortcuts for the debug console.

    Type ``scripts.help()`` to list all methods.

    Parameters
    ----------
    app :
        The live ``MatchbookApp`` instance.
    data_service :
        The ``DataService`` instance.
    pipeline :
        The active ``Pipeline`` instance (may be None).
    namespace :
        The console's live namespace dict — ``exec_file`` injects into it.
    """

    def __init__(
        self,
        app: Any,
        data_service: Any,
        pipeline: Any,
        namespace: dict[str, Any],
    ) -> None:
        self._app = app
        self._ds = data_service
        self._pipeline = pipeline
        self._ns = namespace
        self.script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.dev_script = os.path.join(self.script_dir.parent, "dev_script.py")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load(self, *paths: str) -> None:
        """Load files into the pipeline.

        With no arguments, loads the built-in example .acc files.

        Examples
        --------
        >>> scripts.load()
        >>> scripts.load('C:/data/ref.acc', 'C:/data/sam.acc')
        """
        if not paths:
            example_dir = (
                Path(__file__).parent.parent
                / "modules" / "thz" / "example_data"
            )
            paths = tuple(str(p) for p in sorted(example_dir.glob("*.acc")))
            if not paths:
                print(f"No .acc files found in {example_dir}")
                return

        self._app._on_files_loaded(list(paths))
        print(f"Loaded {len(paths)} file(s).")

    # ------------------------------------------------------------------
    # Pipeline control
    # ------------------------------------------------------------------

    def run(self, step_id: str = "load_from_files", series_id: str | None = None) -> None:
        """Run the pipeline from *step_id* for all (or one) active series.

        Examples
        --------
        >>> scripts.run()                        # from the first step
        >>> scripts.run('align_on_peak')
        >>> scripts.run('align_on_peak', 'my/series')
        """
        if self._pipeline is None:
            print("No active pipeline.")
            return
        series = [series_id] if series_id else list(self._app._series_names)
        for sid in series:
            self._pipeline.run_from(step_id, self._ds, sid)
        self._app._schedule_redraw()
        print(f"Ran from '{step_id}' for {len(series)} series.")

    def reset(self) -> None:
        """Reset pipeline staleness flags (does not clear loaded data).

        To fully reset including data, use  ds._data.clear()  directly,
        then call  app._schedule_redraw().
        """
        if self._pipeline is not None:
            for step in self._pipeline.steps:
                step.mark_stale()
        self._app._schedule_redraw()
        print("Pipeline steps marked stale.  Data still in DataService.")

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def series(self) -> None:
        """Print all series and their data groups."""
        all_series = sorted(self._ds.list_series())
        if not all_series:
            print("(DataService is empty)")
            return
        for sid in all_series:
            groups = sorted(self._ds.list_groups(sid))
            print(f"{sid}")
            for g in groups:
                keys = sorted(self._ds.list_names(sid, g))
                print(f"  {g}: {', '.join(keys)}")

    def data(self, series_id: str, group: str = "time_domain",
             name: str = "mean") -> Any:
        """Fetch and print a DataEntry summary.  Returns the entry.

        Examples
        --------
        >>> e = scripts.data('my/series')
        >>> e = scripts.data('my/series', 'raw', 'compiled')
        >>> e.x, e.y
        """
        from matchbook.core.data_service import DataKey
        entry = self._ds.get(DataKey(series_id, group, name))
        if entry is None:
            print(f"No entry for ({series_id!r}, {group!r}, {name!r})")
            return None
        print(f"x: shape={entry.x.shape}  range=[{entry.x[0]:.4g}, {entry.x[-1]:.4g}]")
        print(f"y: shape={entry.y.shape}  range=[{entry.y.min():.4g}, {entry.y.max():.4g}]")
        if entry.metadata:
            for k, v in entry.metadata.items():
                print(f"  meta.{k} = {v!r}")
        return entry

    def params(self, step_id: str | None = None) -> None:
        """Print current pipeline parameter values.

        Examples
        --------
        >>> scripts.params()
        >>> scripts.params('align_on_peak')
        """
        if self._pipeline is None:
            print("No active pipeline.")
            return
        steps = self._pipeline.steps
        for step in steps:
            if step_id and step.id != step_id:
                continue
            print(f"\n[{step.id}]  {step.name}")
            for k, v in step.params.items():
                print(f"  {k} = {v!r}")

    # ------------------------------------------------------------------
    # Scripting
    # ------------------------------------------------------------------

    def run_dev_script(self) -> None:
        """Run the default dev script (dev_script.py in the same folder as this file)."""
        self.exec_file(self.dev_script)

    def exec_file(self, path: str) -> None:
        """Execute a Python script file in the console namespace.

        Tip: keep a ``dev_script.py`` alongside your project for common
        setup sequences and call ``scripts.exec_file('dev_script.py')``.

        Examples
        --------
        >>> scripts.exec_file('C:/dev/my_setup.py')
        """
        path = os.path.expanduser(path)
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
        exec(compile(source, path, "exec"), self._ns)  # noqa: S102
        print(f"Executed {path}")

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    def help(self) -> None:
        """Print a summary of all DevKit methods."""
        lines = [
            "",
            "  scripts.load([path, ...])           load files (no args = example data)",
            "  scripts.run([step_id[, series_id]]) run pipeline from step",
            "  scripts.reset()                     clear DataService + pipeline state",
            "  scripts.series()                    list all series and their data groups",
            "  scripts.data(series, [group, name]) inspect a DataEntry",
            "  scripts.params([step_id])            show pipeline parameter values",
            "  scripts.exec_file(path)             execute a .py script in console ns",
            "",
            "  tracer.watch(obj, 'method', ...)    trace method calls on an object",
            "  tracer.watch_all(obj)               trace all public methods",
            "  tracer.watch_module(mod, 'fn', ...) trace module-level functions",
            "  tracer.unwatch(obj, 'method')       stop tracing a method",
            "  tracer.clear()                      stop all tracing",
            "  tracer.list()                       show active traces",
            "",
            "  DataKey(series, group, name)        build a DataService lookup key",
            "",
        ]
        print("\n".join(lines))
