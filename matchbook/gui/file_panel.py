"""File browser panel — browse, preview, load, and group data files.

Provides a Treeview file list with browse/remove controls and a
'Load & Group' button that runs the FileIngestor + pipeline step.
Completely generic — no domain-specific knowledge.
"""

from __future__ import annotations

import logging
import os
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Any, Callable

from matchbook.gui import theme
from matchbook.io.file_ingestor import FileIngestor, IngestReport
from matchbook.io.loaders.registry import registered_extensions

logger = logging.getLogger(__name__)


class FilePanel:
    """Left-side file browser panel for selecting, previewing, and loading files.

    Parameters
    ----------
    parent:
        Parent widget (typically the root window).
    on_load_requested:
        Called as ``on_load_requested(file_paths)`` when the user clicks
        'Load & Group'.  The callback receives a list of absolute paths.
    """

    def __init__(
        self,
        parent: tk.Widget,
        on_load_requested: Callable[[list[str]], None],
        *,
        width: int = 300,
    ) -> None:
        self._parent = parent
        self._on_load_requested = on_load_requested
        self._file_paths: list[str] = []

        # -- Outer frame ------------------------------------------------
        self.frame = ttk.Frame(parent, width=width)
        self.frame.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0), pady=6)
        self.frame.pack_propagate(False)

        self._build_ui()

    # -----------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------

    def _build_ui(self) -> None:
        # Header
        hdr = ttk.Label(self.frame, text="Files", font=theme.SECTION_HEADER_FONT)
        hdr.pack(anchor="w", padx=6, pady=(6, 2))

        # Button bar
        btn_bar = ttk.Frame(self.frame)
        btn_bar.pack(fill=tk.X, padx=6, pady=(0, 4))

        ttk.Button(btn_bar, text="Browse\u2026", command=self._browse_files
                   ).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(btn_bar, text="Add Folder\u2026", command=self._browse_folder
                   ).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(btn_bar, text="Remove", command=self._remove_selected
                   ).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(btn_bar, text="Clear", command=self._clear_all
                   ).pack(side=tk.LEFT)

        # File list (Treeview)
        tree_frame = ttk.Frame(self.frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 4))

        columns = ("ext", "status")
        self._tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings",
            selectmode="extended", height=12,
        )
        self._tree.heading("ext", text="Ext")
        self._tree.heading("status", text="Status")
        self._tree.column("ext", width=45, minwidth=35, stretch=False)
        self._tree.column("status", width=65, minwidth=50, stretch=False)

        # Let the first (implicit #0) column show the filename
        self._tree["show"] = ("tree", "headings")
        self._tree.heading("#0", text="Filename")
        self._tree.column("#0", width=170, minwidth=100)

        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL,
                            command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)

        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Tooltip label for full path on selection
        self._path_label = ttk.Label(
            self.frame, text="", font=("TkDefaultFont", 8),
            foreground="grey", wraplength=280,
        )
        self._path_label.pack(anchor="w", padx=6, pady=(0, 2))
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        ttk.Separator(self.frame, orient=tk.HORIZONTAL).pack(
            fill=tk.X, padx=4, pady=2)

        # Summary label
        self._summary_label = ttk.Label(
            self.frame, text="No files selected",
            font=("TkDefaultFont", 9),
        )
        self._summary_label.pack(anchor="w", padx=6, pady=(0, 4))

        # Grouping results (shows after load)
        self._grouping_text = tk.Text(
            self.frame, height=6, width=35, state=tk.DISABLED,
            font=("TkFixedFont", 8), wrap=tk.WORD,
        )
        self._grouping_text.pack(fill=tk.X, padx=6, pady=(0, 4))

        # Load button
        self._load_btn = ttk.Button(
            self.frame, text="\u25B6 Load && Group",
            command=self._on_load_clicked,
        )
        self._load_btn.pack(fill=tk.X, padx=6, pady=(0, 6))

    # -----------------------------------------------------------------
    # File selection actions
    # -----------------------------------------------------------------

    def _browse_files(self) -> None:
        """Open a file dialog to select one or more data files."""
        ext_map = registered_extensions()
        filetypes = [
            ("All supported", " ".join(f"*{e}" for e in ext_map)),
            *[(f"{e} files", f"*{e}") for e in sorted(ext_map)],
            ("All files", "*.*"),
        ]
        paths = filedialog.askopenfilenames(
            title="Select data files",
            filetypes=filetypes,
        )
        if paths:
            self._add_paths(list(paths))

    def _browse_folder(self) -> None:
        """Open a folder dialog and add all supported files within it."""
        folder = filedialog.askdirectory(title="Select folder containing data files")
        if not folder:
            return
        ext_set = set(registered_extensions().keys())
        found: list[str] = []
        for entry in sorted(os.listdir(folder)):
            fpath = os.path.join(folder, entry)
            if os.path.isfile(fpath):
                ext = os.path.splitext(entry)[1].lower()
                if ext in ext_set:
                    found.append(fpath)
        if found:
            self._add_paths(found)

    def _add_paths(self, paths: list[str]) -> None:
        """Add new file paths (deduplicating against existing)."""
        existing = set(self._file_paths)
        added = 0
        for p in paths:
            normed = os.path.normpath(p)
            if normed not in existing:
                self._file_paths.append(normed)
                existing.add(normed)
                self._insert_tree_row(normed, status="pending")
                added += 1
        if added:
            self._update_summary()

    def _insert_tree_row(self, fpath: str, status: str = "pending") -> str:
        """Insert a single file row into the treeview.  Returns the item ID."""
        filename = os.path.basename(fpath)
        ext = os.path.splitext(filename)[1].lower()
        iid = self._tree.insert(
            "", tk.END, text=filename,
            values=(ext, status),
        )
        # Store the full path as a tag so we can retrieve it later
        self._tree.item(iid, tags=(fpath,))
        return iid

    def _remove_selected(self) -> None:
        """Remove selected rows from the tree and the internal path list."""
        for iid in self._tree.selection():
            tags = self._tree.item(iid, "tags")
            if tags:
                fpath = tags[0]
                if fpath in self._file_paths:
                    self._file_paths.remove(fpath)
            self._tree.delete(iid)
        self._update_summary()

    def _clear_all(self) -> None:
        """Remove all files from the list."""
        self._tree.delete(*self._tree.get_children())
        self._file_paths.clear()
        self._update_summary()
        self._set_grouping_text("")

    # -----------------------------------------------------------------
    # Tree interaction
    # -----------------------------------------------------------------

    def _on_tree_select(self, _event: Any = None) -> None:
        sel = self._tree.selection()
        if sel:
            tags = self._tree.item(sel[0], "tags")
            self._path_label.configure(text=tags[0] if tags else "")
        else:
            self._path_label.configure(text="")

    # -----------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------

    def _on_load_clicked(self) -> None:
        """Run the FileIngestor and show results, then fire callback."""
        if not self._file_paths:
            return

        # Mark all rows as "loading..."
        for iid in self._tree.get_children():
            self._tree.set(iid, "status", "loading\u2026")
        self._parent.update_idletasks()

        # Run ingestor
        ingestor = FileIngestor()
        report = ingestor.load_files(self._file_paths)

        # Update tree with results
        self._apply_ingest_report(report)

        # Fire the callback with all successfully-loaded paths
        successful_paths = [ing.path for ing in report.succeeded]
        if successful_paths:
            self._on_load_requested(successful_paths)

    def _apply_ingest_report(self, report: IngestReport) -> None:
        """Update tree statuses and summary from an IngestReport."""
        # Build filename → IngestedFile lookup
        path_results = {os.path.normpath(ing.path): ing for ing in report.files}

        for iid in self._tree.get_children():
            tags = self._tree.item(iid, "tags")
            if not tags:
                continue
            fpath = os.path.normpath(tags[0])
            ing = path_results.get(fpath)
            if ing is None:
                self._tree.set(iid, "status", "?")
            elif ing.ok:
                self._tree.set(iid, "status", "\u2713 loaded")
            else:
                self._tree.set(iid, "status", "\u2717 error")

        n_ok = len(report.succeeded)
        n_fail = len(report.failed)
        self._summary_label.configure(
            text=f"Loaded {n_ok}/{len(report)} files"
                 + (f" ({n_fail} failed)" if n_fail else ""),
        )

    # -----------------------------------------------------------------
    # Grouping results display
    # -----------------------------------------------------------------

    def show_grouping_results(self, info_text: str) -> None:
        """Display grouping results (called by the app after pipeline runs)."""
        self._set_grouping_text(info_text)

    def _set_grouping_text(self, text: str) -> None:
        self._grouping_text.configure(state=tk.NORMAL)
        self._grouping_text.delete("1.0", tk.END)
        if text:
            self._grouping_text.insert("1.0", text)
        self._grouping_text.configure(state=tk.DISABLED)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _update_summary(self) -> None:
        n = len(self._file_paths)
        if n == 0:
            self._summary_label.configure(text="No files selected")
        else:
            self._summary_label.configure(text=f"{n} file(s) selected")

    @property
    def file_paths(self) -> list[str]:
        """Current list of file paths (read-only copy)."""
        return list(self._file_paths)
