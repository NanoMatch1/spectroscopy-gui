# Matchbook — Project Context & Architecture

This file captures the full working context of the Matchbook project as of March 2026.
It is intended to bootstrap a new workspace or AI assistant session with everything
needed to continue development without loss of knowledge.

---

## 1. Project Overview

**Matchbook** is a modular, extensible framework for scientific data analysis, currently
used for THz time-domain spectroscopy (THz-TDS). It provides:

- A **generic core** (data storage, pipelines, module registry) that knows nothing about any specific domain.
- **Domain modules** (currently THz TDS) that plug in via adapters conforming to a Protocol.
- A **Tkinter + matplotlib GUI** for interactive use.
- **SQLite persistence** with incremental sync and full-text search.

The architecture is designed so the core framework and each module can be tested,
run, and reasoned about independently.

---

## 2. Environment & Tooling

| Item | Detail |
|------|--------|
| Python | 3.11.4 |
| Venv | `.venv\Scripts\python.exe` (Windows) |
| OS | Windows, PowerShell terminal |
| Dependencies | numpy, scipy, matplotlib, tkinter (stdlib), sqlite3 (stdlib) |
| **No pandas** | All numerical work uses numpy arrays |
| Test runner | Custom scripts (not pytest): `test_integration.py` (66 checks), `test_full_workflow.py` (138 checks) |
| Run tests | `.\.venv\Scripts\python.exe -m matchbook.scripts.test_integration` and `...test_full_workflow` |
| Launch GUI | `.\.venv\Scripts\python.exe -m matchbook.scripts.launch_gui` |

---

## 3. Package Map

```
matchbook/
├── core/                    # Framework core — no domain knowledge
│   ├── data_service.py      # DataKey, DataEntry, DataService (in-memory store + observer)
│   ├── pipeline.py          # PipelineStep, StepResult, Pipeline (rewindable, parametric)
│   ├── registry.py          # ModuleContext, RegisteredModule, Registry
│   ├── module_base.py       # Protocol + descriptor dataclasses (AnalysisModule, ParamType, etc.)
│   └── step_report.py       # Step execution reporting
│
├── io/                      # I/O layer
│   ├── database.py          # Database class — SQLite persistence (save/load/search/sync/provenance)
│   ├── search.py            # @register_filter/@register_post_filter, build_search_query, 11 SQL filters
│   ├── file_ingestor.py     # FileIngestor, IngestedFile, IngestReport — generic file loading
│   ├── recognisers.py       # @register_recogniser, @register_atomiser — module→data bridge
│   └── loaders/             # Loader registry + concrete loaders
│       ├── registry.py      # BaseLoader, @register_loader, get_loader_for_extension, registered_extensions
│       ├── acc_loader.py    # .acc files
│       ├── dat_loader.py    # .dat files
│       ├── txt_loader.py    # .txt files
│       └── csv_loader.py    # .csv files
│
├── services/                # Domain-agnostic services
│   ├── grouping.py          # GroupingService — filename parsing, reference/sample pairing
│   ├── grouping_step.py     # step_group_files + descriptor/factory — generic pipeline step
│   ├── filename_info.py     # FilenameInfo dataclass — parsed filename metadata
│   └── provenance.py        # Provenance tracking helpers
│
├── modules/thz/             # THz TDS module (only adapter.py imports framework)
│   ├── adapter.py           # THzModule class, step_load_from_files, step_compute_transfer_function
│   ├── containers.py        # THzData container
│   ├── models.py            # AnalysisDataset and domain models
│   ├── analysis.py          # Transfer function, FFT, optical constants computation
│   └── example_data/        # .acc test files (reference_air_nitrogen.acc, sample_germanium.acc)
│
├── gui/                     # Tkinter GUI — headless-testable with root.withdraw() + Agg backend
│   ├── app.py               # MatchbookApp — top-level window, orchestrates all panels
│   ├── file_panel.py        # FilePanel — collapsible, file browser + database section
│   ├── sidebar.py           # Series toggles, trace groups, axis controls
│   ├── parameter_panel.py   # Per-step parameter widgets (auto-generated from descriptors)
│   ├── pipeline_view.py     # Pipeline step list with status, run, rewind controls
│   ├── plot_area.py         # Module-agnostic matplotlib renderer
│   └── theme.py             # PALETTE, fonts, series_colour(), layout constants
│
├── scripts/                 # Entry points and test scripts
│   ├── launch_gui.py        # Creates DS + DB + Registry, registers THz, launches MatchbookApp
│   ├── run_thz.py           # Headless THz analysis script
│   ├── test_integration.py  # 66-check integration test
│   └── test_full_workflow.py # 138-check full workflow test (12 phases)
│
├── data_structures/         # Shared data structures
│   ├── helpers.py
│   └── spectrum.py
│
├── tools/                   # Utilities
│   └── acquisition_editor/
│
└── docs/
    └── wiki.md              # Living documentation (kept in sync with code)
```

---

## 4. Key Architectural Patterns

### 4.1 Protocol + Adapter (no inheritance)

Modules satisfy the `AnalysisModule` protocol structurally — they don't inherit or import it.
The protocol lives in `core/module_base.py` and declares:

```python
class AnalysisModule(Protocol):
    name: str
    display_name: str
    def data_groups() -> list[DataGroupDescriptor]: ...
    def pipeline_step_descriptors() -> list[PipelineStepDescriptor]: ...
    def create_pipeline_steps() -> list[PipelineStep]: ...
```

The THz adapter (`modules/thz/adapter.py`) is the **only** file in the THz package that
imports from the framework. Delete it and the rest of the THz package works standalone.

### 4.2 Three-Layer Storage

```
DataService (in-memory, observable)
    ↕ dirty tracking (dirty_keys, dirty_associations)
Database (SQLite — save_series, load_series, sync_dirty)
```

- `DataService` holds `DataEntry` objects keyed by `DataKey(series_id, group, name)`.
- Supports tagging (`tag_series`), associations (`associate`), and observer notifications.
- `Database.sync_dirty(ds)` only writes entries that changed since last sync.

### 4.3 Registry Decorators

| Decorator | Location | Purpose |
|-----------|----------|---------|
| `@register_loader` | `io/loaders/registry.py` | Maps file extensions to loader classes |
| `@register_filter` | `io/search.py` | Adds SQL WHERE-clause builders for DB search |
| `@register_post_filter` | `io/search.py` | Adds post-query result filters |
| `@register_recogniser` | `io/recognisers.py` | Module registers confidence-scoring function |
| `@register_atomiser` | `io/recognisers.py` | Module registers DataService writer |

### 4.4 Pipeline Step Signature

All pipeline steps share the same signature:

```python
def step_fn(data_service: DataService, series_id: str, **params) -> None:
```

They read inputs from `data_service.get()` and write results via `data_service.put()`.
This makes them identical whether run from the GUI, a script, or a test.

### 4.5 File Ingest Pipeline

Three-layer system for loading files from disk:

1. **LoaderRegistry** — `@register_loader` maps extensions to loader classes (`.acc`, `.dat`, `.txt`, `.csv`)
2. **FileIngestor** — generic file loading, returns `IngestedFile`/`IngestReport`
3. **Recogniser + Atomiser** — modules register confidence-scoring (`recognise_thz`) and DataService-writing (`atomise_thz`) functions

### 4.6 Grouping as a Separate Pipeline Step

Grouping (pairing references with samples by filename conventions) is **decoupled from file loading**.
It is a generic, module-agnostic pipeline step in `services/grouping_step.py`:

```python
from matchbook.services.grouping_step import create_grouping_step, grouping_step_descriptor
```

- Works after file loading OR after database import (operates on sub-series already in DataService).
- Any module can include it in its pipeline.
- Parameters: `grouping_keywords` (comma-separated string), `grouping_delimiter` (default `"_"`).

---

## 5. THz TDS Module — Current Pipeline

The THz module has **3 pipeline steps** in order:

| Step ID | Name | Parameters | Depends On |
|---------|------|------------|------------|
| `load_from_files` | Load Data | `file_paths` (newline-separated paths) | — |
| `group_files` | Group Files | `grouping_keywords`, `grouping_delimiter` | `load_from_files` |
| `transfer_function` | Compute Transfer Function | — | `group_files` |

**Data groups** (5): `raw`, `time_domain`, `fft`, `transfer_function`, `optical_constants`
**Total traces**: 15 (across all groups)

The adapter registers `recognise_thz` (confidence 1.0 if THzData) and `atomise_thz`
(writes `raw/compiled`, `time_domain/mean`, `time_domain/stderr` per file).

---

## 6. GUI Components

### FilePanel (`gui/file_panel.py`)
- **Collapsible**: click ◀ to collapse to a 28px strip (▶ + vertical "Files" label), ▶ to expand.
- **Two sections**:
  - **Load from Disk**: Browse Files / Add Folder / Remove / Clear buttons, Treeview file list, "Load Files" button.
  - **Database**: Search entry + button, results Treeview (name/module/date), "Load Selected" button.
- Callbacks: `on_load_requested(file_paths)`, `on_db_search_requested(query)`, `on_db_load_requested(series_ids)`.

### MatchbookApp (`gui/app.py`)
- Accepts optional `database` parameter for DB access in the FilePanel.
- Layout: FilePanel (left, collapsible) → Sidebar → ParameterPanel + PipelineView (right) → Plot area (centre).
- `_on_files_loaded`: pushes paths to pipeline, runs `run_from("load_from_files", ...)` (runs full pipeline downstream).
- `_on_db_search`: calls `database.search(text_search=query)`, populates FilePanel results.
- `_on_db_load`: calls `database.load_series(sid, data_service)` for each selected series.

### launch_gui.py
- Creates `DataService`, `Registry`, `Database(matchbook_data.db)`, registers `THzModule`, launches `MatchbookApp`.

---

## 7. Database API

`Database(db_path)` — SQLite persistence with these key methods:

| Method | Purpose |
|--------|---------|
| `save_series(ds, series_id, module, display_name, metadata)` | Persist full series |
| `load_series(series_id, ds) -> bool` | Restore into DataService |
| `list_series(module?, tags?) -> list[dict]` | List saved series |
| `search(**criteria) -> list[dict]` | Full search (11 SQL filters) |
| `delete_series(series_id)` | Remove from DB |
| `sync_dirty(ds) -> int` | Incremental write of dirty entries only |
| `save/load/list_pipeline_snapshots` | Pipeline parameter history |
| `record/list_provenance` | Provenance tracking |

**Search filters** (11): `name_contains`, `name_exact`, `id_contains`, `module`, `date_after`,
`date_before`, `has_tag`, `has_group`, `has_name`, `metadata_field`, `text_search`.

**Schema tables**: `series`, `series_tags`, `data_blobs`, `associations`, `provenance`, `pipeline_snapshots`.

---

## 8. DataService API

`DataService()` — in-memory data store with observer pattern:

| Method | Purpose |
|--------|---------|
| `put(entry)` / `get(key)` / `remove(key)` | CRUD for DataEntry |
| `list_series()` / `list_groups(sid)` / `list_names(sid, group)` | Discovery |
| `query(series_id?, group?, tags?)` | Filtered retrieval |
| `tag_series` / `untag_series` / `get_tags` / `all_tags` / `series_with_tag` | Tagging |
| `associate` / `remove_association` / `get_association` / `list_associations` | Associations |
| `dirty_keys` / `dirty_associations` / `is_dirty` / `mark_clean` | Change tracking |
| `subscribe` / `unsubscribe` | Observer notifications |

---

## 9. Test Structure

### test_integration.py (66 checks, 7 tests)
1. Loader Registry — load .acc files
2. GroupingService — pair reference/sample
3. DataService Atomisation — manual put + associations
4. Database Round-trip — save/clear/load/verify
5. sync_dirty — incremental write
6. step_load_from_files + step_group_files — registry-based loading then grouping
7. Module Registration — THzModule protocol compliance (3 pipeline steps)

### test_full_workflow.py (138 checks, 12 phases)
Simulates a full user session:
1. Setup & Module Registration
2. Load Data via Pipeline (run_all)
3. Inspect Data (sidebar/plot perspective)
4. Pipeline Rewind / Re-run / Staleness
5. Tagging & Associations
6. Persist to Database
7. Reload from Database
8. DB Search & Query (all 11 filters)
9. sync_dirty (incremental)
10. Pipeline Snapshots & Provenance
11. GUI Component Construction (headless)
12. Plot Rendering (headless)

---

## 10. Changelog (Recent)

### 2026-03-17 — Grouping Decoupled + FilePanel Overhaul
- **Created `matchbook/services/grouping_step.py`**:
  - `step_group_files()` — generic pipeline step, discovers sub-series, runs GroupingService, writes _meta/grouping + associations
  - `grouping_step_descriptor()` and `create_grouping_step()` factory for modules to import
- **Refactored `matchbook/modules/thz/adapter.py`**:
  - Removed all grouping logic from `step_load_from_files` (now load-only)
  - Pipeline changed: 2 steps → 3 steps (`load_from_files` → `group_files` → `transfer_function`)
  - Grouping params moved from load step to dedicated group_files step
- **Rewrote `matchbook/gui/file_panel.py`**:
  - Added collapse/expand toggle (◀/▶, collapses to 28px strip)
  - Added Database section: search entry, results Treeview, "Load Selected" button
  - Changed "Load & Group" → "Load Files" (grouping is now a pipeline step)
  - Removed grouping text area (grouping visible via pipeline)
- **Updated `matchbook/gui/app.py`**:
  - Accepts optional `database` parameter
  - Added `_on_db_search()` and `_on_db_load()` callbacks
  - `_on_files_loaded` now runs full pipeline (not just load step)
- **Updated `matchbook/scripts/launch_gui.py`**:
  - Creates `Database(matchbook_data.db)` and passes to MatchbookApp
- **Updated both test files** for 3-step pipeline
- All tests pass: 66/66 + 138/138

### 2026-03-17 — FilePanel GUI Component (earlier same day)
- Created initial FilePanel with Browse/Add Folder/Remove/Clear, Treeview, Load & Group
- Integrated into app.py with _on_files_loaded callback

### 2025-06-24 — FileIngestor + Recogniser/Atomiser Registries
- Created FileIngestor (io/file_ingestor.py), Recogniser/Atomiser registries (io/recognisers.py)
- THz adapter refactored to thin delegate pattern
- Wiki updated with documentation

---

## 11. Coding Conventions

- **No pandas** — numpy only for all numerical work.
- **Protocol-based** — modules satisfy protocols structurally (no base class inheritance).
- **Dependency injection** — components receive their dependencies, never hard-code paths or services.
- **Explicit data flow** — no hidden global state (except decorator registries which are the documented exception).
- **Pipeline steps are pure functions** of `(data_service, series_id, **params)`.
- **GUI is headless-testable** — `root.withdraw()` + `matplotlib.use("Agg")`.
- **Tests must all pass** before any work is considered complete — fix bugs in the same cycle.
- **Living documentation** — `matchbook/docs/wiki.md` is kept in sync with code changes.
- See `coding_style_guidelines.md` for general code style preferences.

---

## 12. User Preferences (Samuel)

- Prefers modular, testable code with dependency injection.
- Values clarity, modularity, and long-term maintainability over cleverness.
- Likes to use pdb breakpoints in example scripts for debugging.
- Expects living documentation kept in sync with code.
- Comfortable with Protocol-based design (structural subtyping).
- Works on Windows with PowerShell and Python venvs.
- Primary domain: THz time-domain spectroscopy / scientific data analysis.
- No pandas — numpy only.
- Prefers explicit data flow over hidden global state.
