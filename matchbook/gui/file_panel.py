"""File and database browser panel — browse, load, search, and manage data.

Provides a collapsible left panel with two sections:

* **Files** — browse disk, select files, and load them via the FileIngestor.
* **Database** — search saved series and load them back into the DataService.

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
    """Collapsible left-side panel for file loading and database access.

    Parameters
    ----------
    parent:
        Parent widget (typically the root window).
    on_load_requested:
        Called as ``on_load_requested(file_paths)`` when the user clicks
        'Load'.  The callback receives a list of absolute paths.
    on_db_search_requested:
        Called as ``on_db_search_requested(query_text)`` when the user
        clicks 'Search'.  The app should call ``show_db_results()`` with
        the results.
    on_db_load_requested:
        Called as ``on_db_load_requested(series_ids)`` when the user
        clicks 'Load Selected' in the database section.
    """

    _EXPANDED_WIDTH = 300
    _COLLAPSED_WIDTH = 28

    def __init__(
        self,
        parent: tk.Widget,
        on_load_requested: Callable[[list[str]], None],
        *,
        on_db_search_requested: Callable[[str], None] | None = None,
        on_db_load_requested: Callable[[list[str]], None] | None = None,
        width: int = _EXPANDED_WIDTH,
    ) -> None:
        self._parent = parent
        self._on_load_requested = on_load_requested
        self._on_db_search_requested = on_db_search_requested or (lambda _q: None)
        self._on_db_load_requested = on_db_load_requested or (lambda _ids: None)
        self._file_paths: list[str] = []
        self._expanded = True

        # -- Outer container (always visible) --
        self.frame = ttk.Frame(parent)
        self.frame.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0), pady=6)

        # -- Collapsed bar (shown when collapsed) --
        self._collapsed_bar = ttk.Frame(self.frame, width=self._COLLAPSED_WIDTH)

        self._expand_btn = ttk.Button(
            self._collapsed_bar, text="\u25B6", width=2,
            command=self._toggle,
        )
        self._expand_btn.pack(padx=2, pady=(6, 4))

        self._collapsed_label = ttk.Label(
            self._collapsed_bar, text="F\ni\nl\ne\ns",
            font=("TkDefaultFont", 9), justify=tk.CENTER,
        )
        self._collapsed_label.pack(pady=(0, 6))

        # -- Expanded content --
        self._content = ttk.Frame(self.frame, width=width)
        self._content.pack(side=tk.LEFT, fill=tk.Y)
        self._content.pack_propagate(False)

        self._build_ui(width)

    # -----------------------------------------------------------------
    # Collapse / expand
    # -----------------------------------------------------------------

    def _toggle(self) -> None:
        if self._expanded:
            self._content.pack_forget()
            self._collapsed_bar.pack(side=tk.LEFT, fill=tk.Y)
            self._expanded = False
        else:
            self._collapsed_bar.pack_forget()
            self._content.pack(side=tk.LEFT, fill=tk.Y)
            self._expanded = True

    # -----------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------

    def _build_ui(self, width: int) -> None:
        # ---- Collapse button at top of expanded panel ----
        top_bar = ttk.Frame(self._content)
        top_bar.pack(fill=tk.X, padx=6, pady=(6, 0))

        self._collapse_btn = ttk.Button(
            top_bar, text="\u25C0", width=2, command=self._toggle,
        )
        self._collapse_btn.pack(side=tk.LEFT)

        ttk.Label(
            top_bar, text="Files & Data",
            font=theme.SECTION_HEADER_FONT,
        ).pack(side=tk.LEFT, padx=(4, 0))

        # ============================================================
        # Section 1 — Files
        # ============================================================
        files_lf = ttk.LabelFrame(self._content, text="  Load from Disk  ")
        files_lf.pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 2))

        # Button bar
        btn_bar = ttk.Frame(files_lf)
        btn_bar.pack(fill=tk.X, padx=4, pady=(4, 2))

        ttk.Button(btn_bar, text="Browse\u2026", command=self._browse_files
                   ).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(btn_bar, text="Add Folder\u2026", command=self._browse_folder
                   ).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(btn_bar, text="Remove", command=self._remove_selected
                   ).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(btn_bar, text="Clear", command=self._clear_all
                   ).pack(side=tk.LEFT)

        # File list (Treeview)
        tree_frame = ttk.Frame(files_lf)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 2))

        columns = ("ext", "status")
        self._tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings",
            selectmode="extended", height=8,
        )
        self._tree.heading("ext", text="Ext")
        self._tree.heading("status", text="Status")
        self._tree.column("ext", width=45, minwidth=35, stretch=False)
        self._tree.column("status", width=65, minwidth=50, stretch=False)
        self._tree["show"] = ("tree", "headings")
        self._tree.heading("#0", text="Filename")
        self._tree.column("#0", width=150, minwidth=100)

        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL,
                            command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Path tooltip
        self._path_label = ttk.Label(
            files_lf, text="", font=("TkDefaultFont", 8),
            foreground="grey", wraplength=width - 30,
        )
        self._path_label.pack(anchor="w", padx=4, pady=(0, 2))
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Summary + Load button
        self._summary_label = ttk.Label(
            files_lf, text="No files selected", font=("TkDefaultFont", 9),
        )
        self._summary_label.pack(anchor="w", padx=4, pady=(0, 2))

        self._load_btn = ttk.Button(
            files_lf, text="\u25B6 Load Files",
            command=self._on_load_clicked,
        )
        self._load_btn.pack(fill=tk.X, padx=4, pady=(0, 6))

        # ============================================================
        # Section 2 — Database
        # ============================================================
        db_lf = ttk.LabelFrame(self._content, text="  Database  ")
        db_lf.pack(fill=tk.BOTH, expand=True, padx=6, pady=(2, 6))

        # Search bar
        search_bar = ttk.Frame(db_lf)
        search_bar.pack(fill=tk.X, padx=4, pady=(4, 2))

        self._search_var = tk.StringVar()
        self._search_entry = ttk.Entry(search_bar, textvariable=self._search_var)
        self._search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self._search_entry.bind("<Return>", lambda _e: self._on_db_search())

        ttk.Button(
            search_bar, text="\U0001F50D", width=3,
            command=self._on_db_search,
        ).pack(side=tk.LEFT)

        # Results treeview
        db_tree_frame = ttk.Frame(db_lf)
        db_tree_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 2))

        db_cols = ("module", "date")
        self._db_tree = ttk.Treeview(
            db_tree_frame, columns=db_cols, show="headings",
            selectmode="extended", height=5,
        )
        self._db_tree.heading("#0", text="Name")
        self._db_tree.column("#0", width=120, minwidth=80)
        self._db_tree.heading("module", text="Module")
        self._db_tree.column("module", width=60, minwidth=40, stretch=False)
        self._db_tree.heading("date", text="Date")
        self._db_tree.column("date", width=75, minwidth=50, stretch=False)
        self._db_tree["show"] = ("tree", "headings")

        db_vsb = ttk.Scrollbar(db_tree_frame, orient=tk.VERTICAL,
                               command=self._db_tree.yview)
        self._db_tree.configure(yscrollcommand=db_vsb.set)
        self._db_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        db_vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self._db_status = ttk.Label(
            db_lf, text="", font=("TkDefaultFont", 8), foreground="grey",
        )
        self._db_status.pack(anchor="w", padx=4, pady=(0, 2))

        self._db_load_btn = ttk.Button(
            db_lf, text="\u25B6 Load Selected",
            command=self._on_db_load,
        )
        self._db_load_btn.pack(fill=tk.X, padx=4, pady=(0, 6))

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
    # File loading
    # -----------------------------------------------------------------

    def _on_load_clicked(self) -> None:
        """Run the FileIngestor and show results, then fire callback."""
        if not self._file_paths:
            return

        for iid in self._tree.get_children():
            self._tree.set(iid, "status", "loading\u2026")
        self._parent.update_idletasks()

        ingestor = FileIngestor()
        report = ingestor.load_files(self._file_paths)
        self._apply_ingest_report(report)

        successful_paths = [ing.path for ing in report.succeeded]
        if successful_paths:
            self._on_load_requested(successful_paths)

    def _apply_ingest_report(self, report: IngestReport) -> None:
        """Update tree statuses and summary from an IngestReport."""
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
    # Database actions
    # -----------------------------------------------------------------

    def _on_db_search(self) -> None:
        """Fire the search callback with the current query text."""
        query = self._search_var.get().strip()
        self._on_db_search_requested(query)

    def _on_db_load(self) -> None:
        """Fire the load callback with selected series IDs."""
        series_ids: list[str] = []
        for iid in self._db_tree.selection():
            tags = self._db_tree.item(iid, "tags")
            if tags:
                series_ids.append(tags[0])
        if series_ids:
            self._on_db_load_requested(series_ids)

    def show_db_results(self, results: list[dict[str, Any]]) -> None:
        """Populate the database results treeview.

        Each result dict should have keys: ``id``, ``display_name``,
        ``module``, ``created_at`` (as returned by ``Database.search``).
        """
        self._db_tree.delete(*self._db_tree.get_children())
        for rec in results:
            display = rec.get("display_name") or rec.get("id", "?")
            module = rec.get("module", "")
            date = (rec.get("created_at") or "")[:10]
            iid = self._db_tree.insert(
                "", tk.END, text=display,
                values=(module, date),
            )
            self._db_tree.item(iid, tags=(rec.get("id", ""),))

        self._db_status.configure(
            text=f"{len(results)} series found" if results else "No results",
        )

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
