# Matchbook — Platform Wiki

> **Living document.** Updated after every significant modification to the codebase.

---

## Table of Contents

1. [What is Matchbook?](#1-what-is-matchbook)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Framework](#3-core-framework)
   - 3.1 [DataService](#31-dataservice)
   - 3.2 [Pipeline Engine](#32-pipeline-engine)
   - 3.3 [Module Registry](#33-module-registry)
   - 3.4 [Module Protocol & Descriptors](#34-module-protocol--descriptors)
   - 3.5 [StepReport](#35-stepreport)
4. [Storage & Persistence](#4-storage--persistence)
   - 4.1 [Three-Layer Storage Model](#41-three-layer-storage-model)
   - 4.2 [SQLite Database](#42-sqlite-database)
   - 4.3 [Dirty Tracking & Sync](#43-dirty-tracking--sync)
5. [File I/O — Loader Registry](#5-file-io--loader-registry)
   - 5.1 [Registry Mechanism](#51-registry-mechanism)
   - 5.2 [BaseLoader Interface](#52-baseloader-interface)
   - 5.3 [Registered Loaders](#53-registered-loaders)
6. [Services](#6-services)
   - 6.1 [GroupingService](#61-groupingservice)
   - 6.2 [FilenameInfo](#62-filenameinfo)
   - 6.3 [Provenance](#63-provenance)
7. [Data Structures](#7-data-structures)
   - 7.1 [Spectrum](#71-spectrum)
   - 7.2 [Helpers (numpy utilities)](#72-helpers-numpy-utilities)
8. [THz TDS Module](#8-thz-tds-module)
   - 8.1 [Module Design](#81-module-design)
   - 8.2 [Adapter (adapter.py)](#82-adapter-adapterpy)
   - 8.3 [Models (models.py)](#83-models-modelspy)
   - 8.4 [Loaders (loaders.py)](#84-loaders-loaderspy)
   - 8.5 [Analysis (analysis.py)](#85-analysis-analysispy)
   - 8.6 [Containers (containers.py)](#86-containers-containerspy)
9. [GUI Layer](#9-gui-layer)
   - 9.1 [MatchbookApp](#91-matchbookapp)
   - 9.2 [Sidebar](#92-sidebar)
   - 9.3 [Plot Area](#93-plot-area)
   - 9.4 [Parameter Panel](#94-parameter-panel)
   - 9.5 [Pipeline View](#95-pipeline-view)
   - 9.6 [Theme](#96-theme)
10. [Tools](#10-tools)
    - 10.1 [Acquisition Editor](#101-acquisition-editor)
11. [Scripts](#11-scripts)
12. [Data Flow Walkthrough](#12-data-flow-walkthrough)
13. [File Map](#13-file-map)

---

## 1. What is Matchbook?

Matchbook is a **modular analysis and visualisation platform** written in Python. It is designed for scientific data processing workflows where:

- Multiple independent analysis modules (THz spectroscopy, UV-VIS, etc.) can be plugged in without modifying the framework.
- Data is processed through a **rewindable pipeline** — parameters can be adjusted and steps re-executed from any point.
- All data lives in a **central in-memory store** (DataService) that the GUI reads from directly, giving instant reactivity.
- Results persist to a **SQLite database** that supports save/load/filter/restore at session boundaries.
- The GUI is entirely **descriptor-driven** — it renders what modules declare without domain-specific knowledge.

### Design Goals

| Goal | How |
|---|---|
| Modularity | Protocol + Adapter pattern — modules are standalone packages |
| Testability | Pipeline step functions are pure `fn(data_service, series_id, **params)` |
| Rewindability | Pipeline snapshots parameter history; any state can be restored |
| Persistence | Three-layer storage: in-memory → dirty tracking → SQLite |
| GUI agnosticism | Modules produce descriptors; GUI interprets them generically |
| No pandas | All numerical work uses numpy arrays exclusively |

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                      GUI Layer                           │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌────────┐  │
│  │ Sidebar  │  │ Plot Area │  │  Params  │  │Pipeline│  │
│  │ (toggles)│  │(matplotlib│  │  Panel   │  │  View  │  │
│  └────┬─────┘  └─────┬─────┘  └────┬─────┘  └───┬────┘  │
│       │              │              │             │       │
│       └──────┬───────┴──────┬───────┘             │       │
│              │              │                     │       │
└──────────────┼──────────────┼─────────────────────┼───────┘
               │              │                     │
        ┌──────▼──────┐  ┌───▼────┐          ┌─────▼──────┐
        │ DataService │  │Registry│          │  Pipeline  │
        │ (in-memory) │  │        │          │  (engine)  │
        └──────┬──────┘  └───┬────┘          └─────┬──────┘
               │             │                     │
               │      ┌──────▼──────────┐          │
               │      │  Module (e.g.   │──────────┘
               │      │   THzModule)    │
               │      │                 │
               │      │  adapter.py ◄───── only file importing framework
               │      │  models.py      │
               │      │  loaders.py     │
               │      │  analysis.py    │
               │      │  containers.py  │
               │      └─────────────────┘
               │
        ┌──────▼──────┐
        │  Database   │
        │  (SQLite)   │
        └─────────────┘
```

**Key principle:** Modules are standalone Python packages. Only `adapter.py` imports from the framework. Every other file in a module can be tested independently with just numpy.

---

## 3. Core Framework

All framework code lives under `matchbook/core/`.

### 3.1 DataService

**File:** `matchbook/core/data_service.py`

The DataService is the **single source of truth** for all analysis data at runtime. Every piece of data is addressed by a three-part key:

```python
@dataclass(frozen=True)
class DataKey:
    series_id: str   # e.g. "sam_Si_2025-03"
    group: str       # e.g. "time_domain", "optical_constants"
    name: str        # e.g. "win_ref_amp", "n"
```

Each entry wraps the key with x/y numpy arrays and freeform metadata:

```python
@dataclass
class DataEntry:
    key: DataKey
    x: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any]
```

#### Core Methods

| Method | Purpose |
|---|---|
| `put(entry)` | Insert or update; marks dirty; notifies listeners |
| `remove(key)` | Delete one entry |
| `clear_series(series_id)` | Remove all entries for a series |
| `get(key) → DataEntry \| None` | Retrieve a single entry |
| `query(series_id, group, tags)` | Flexible filtered query |
| `list_series()` | All unique series IDs (sorted) |
| `list_groups(series_id)` | All groups within a series |
| `list_names(series_id, group)` | All names within a group |

#### Tags

Series can carry arbitrary string tags for filtering:

```python
ds.tag_series("chris", "person", "site_a")
ds.series_with_tag("person")  # → ["chris", ...]
```

#### Associations

Named links between series (e.g. sample → reference pairing):

```python
ds.associate("sample_Ge_300K", "reference", "ref_Ge_300K")
ds.get_association("sample_Ge_300K", "reference")  # → "ref_Ge_300K"
ds.list_associations("sample_Ge_300K")
# → [("sample_Ge_300K", "reference", "ref_Ge_300K")]
```

#### Observer Pattern

Listeners are notified on every mutation — the GUI uses this to debounce and redraw automatically:

```python
ds.subscribe(lambda event, key: print(f"{event}: {key}"))
# Events: "put", "remove", "clear"
```

#### Dirty Tracking

Every `put()` call adds the key to a `_dirty` set. The database layer reads `dirty_keys()` to sync only what changed, then calls `mark_clean()`.

---

### 3.2 Pipeline Engine

**File:** `matchbook/core/pipeline.py`

A rewindable, parametric sequence of processing steps.

#### PipelineStep

```python
@dataclass
class PipelineStep:
    id: str                              # unique identifier
    name: str                            # human-readable label
    fn: Callable[..., Any]               # fn(data_service, series_id, **params)
    params: dict[str, Any]               # current parameter values
    param_descriptors: list[ParameterDescriptor]
    depends_on: list[str]                # step IDs that must run first
```

Every step function has the **universal signature**:

```python
def step_fn(data_service: DataService, series_id: str, **params) -> None:
```

This means steps work identically whether called from the GUI, a script, or a test.

#### Pipeline Class

| Method | Purpose |
|---|---|
| `add_step(step)` | Append a step to the sequence |
| `run_from(step_id, ds, series_id)` | Re-run from a step through all downstream |
| `run_all(ds, series_id)` | Run the full pipeline |
| `set_param(step_id, name, value)` | Update a parameter without running |
| `get_params(step_id)` | Read current parameters |
| `rewind(history_index)` | Restore all parameters to a previous state |
| `is_stale(step_id)` | True if params changed since last run |
| `downstream_of(step_id)` | List step + everything after it |

#### History & Rewind

Every `run_from()` snapshots all parameter values before execution. Users can `rewind(n)` to restore any previous parameter state. The PipelineView in the GUI displays this as an undo button with a history counter.

#### Observer Events

Listeners receive `(event, step_id)` where event is one of: `step_start`, `step_done`, `rewind`, `param_changed`.

---

### 3.3 Module Registry

**File:** `matchbook/core/registry.py`

The Registry is the **glue layer** between modules and the framework.

#### Registration Flow

1. User creates a module instance (e.g. `THzModule()`).
2. `registry.register(module)` validates the `AnalysisModule` protocol.
3. Pulls descriptors: `data_groups()`, `pipeline_step_descriptors()`.
4. Calls `create_pipeline_steps()` → receives `PipelineStep` objects → builds a `Pipeline`.
5. Stores everything as a `RegisteredModule` record.
6. Optionally calls `module.on_register(ctx)` with a `ModuleContext`.
7. Notifies GUI callbacks (if wired) to build sidebar sections and parameter widgets.

#### Key Types

```python
@dataclass
class ModuleContext:
    data_service: DataService
    pipeline: Pipeline
    request_sidebar_section: Callable | None   # None when headless
    request_parameter_widget: Callable | None

@dataclass
class RegisteredModule:
    module: Any
    name: str
    display_name: str
    data_groups: list[DataGroupDescriptor]
    pipeline_step_descriptors: list[PipelineStepDescriptor]
    pipeline: Pipeline
```

#### Lookup

```python
registry.get_module("thz_tds")     # → RegisteredModule
registry.get_pipeline("thz_tds")   # → Pipeline
registry.all_data_groups            # merged + sorted from all modules
```

---

### 3.4 Module Protocol & Descriptors

**File:** `matchbook/core/module_base.py`

Modules do **not** inherit from a base class. They satisfy a `typing.Protocol` via structural typing — they just need the right attributes and methods.

#### AnalysisModule Protocol

```python
@runtime_checkable
class AnalysisModule(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def display_name(self) -> str: ...
    def data_groups(self) -> list[DataGroupDescriptor]: ...
    def pipeline_step_descriptors(self) -> list[PipelineStepDescriptor]: ...
    def create_pipeline_steps(self) -> list[Any]: ...
```

#### Descriptor Types

**ParamType** — enum of widget types:
`FLOAT`, `INT`, `STRING`, `BOOL`, `CHOICE`, `FILE_PATH`

**ParameterDescriptor** — one tunable parameter:
```python
@dataclass
class ParameterDescriptor:
    name: str           # internal key
    label: str          # GUI display label
    type: ParamType     # widget type
    default: Any        # initial value
    min/max/step: ...   # for sliders
    choices: list[str]  # for dropdowns
    tooltip: str
```

**TraceDescriptor** — one plottable line:
```python
@dataclass
class TraceDescriptor:
    key: str             # unique trace ID
    label: str           # legend label
    group: str           # DataKey group
    name: str            # DataKey name
    y_label: str         # y-axis label
    default_visible: bool
    line_style: str | None
```

**DataGroupDescriptor** — one subplot/section:
```python
@dataclass
class DataGroupDescriptor:
    key: str
    label: str
    x_label: str
    traces: list[TraceDescriptor]
    order: int          # sort order in GUI
```

**PipelineStepDescriptor** — one pipeline step's GUI representation:
```python
@dataclass
class PipelineStepDescriptor:
    id: str
    name: str
    params: list[ParameterDescriptor]
    depends_on: list[str]
```

---

### 3.5 StepReport

**File:** `matchbook/core/step_report.py`

A skeleton for per-step diagnostics. Pipeline steps can optionally return a `StepReport` to communicate what happened.

```python
@dataclass
class StepReport:
    step_id: str
    series_id: str
    diagnostics: list[Diagnostic]   # INFO / WARNING / ERROR messages
    metrics: dict[str, Any]         # arbitrary numeric stats
    entries_written: int
    entries_read: int
```

Convenience builders: `report.info(msg)`, `report.warn(msg)`, `report.error(msg)`.

Query properties: `report.ok`, `report.has_errors`, `report.has_warnings`, `report.summary()`.

---

## 4. Storage & Persistence

### 4.1 Three-Layer Storage Model

```
┌─────────────────────────────────────┐
│  Layer 1: DataService (in-memory)   │  ← runtime source of truth
│  Fast dict lookups, observer events │
└──────────────┬──────────────────────┘
               │  dirty tracking
┌──────────────▼──────────────────────┐
│  Layer 2: Dirty Tracking            │  ← knows what changed
│  _dirty set, _dirty_associations    │
└──────────────┬──────────────────────┘
               │  sync_dirty() or save_series()
┌──────────────▼──────────────────────┐
│  Layer 3: SQLite Database           │  ← persistent truth
│  Written at session boundaries      │
└─────────────────────────────────────┘
```

- **During a session:** DataService is the source of truth. All reads go through it.
- **On save:** Database reads dirty entries from DataService and writes them.
- **On load:** Database reads from SQLite and populates DataService.

### 4.2 SQLite Database

**File:** `matchbook/io/database.py`

#### Schema

| Table | Purpose | Key |
|---|---|---|
| `series` | Series identity + metadata | `id` (TEXT PK) |
| `series_tags` | Tag-to-series mapping | `(series_id, tag)` |
| `data_blobs` | X/Y data as numpy binary blobs | `(series_id, grp, name)` |
| `associations` | Named links between series | `(source_id, relationship)` |
| `provenance` | Step execution records | auto-increment |
| `pipeline_snapshots` | Parameter state snapshots | auto-increment |

#### Database Class

| Method | Purpose |
|---|---|
| `save_series(ds, series_id, ...)` | Persist all entries for a series |
| `load_series(series_id, ds)` | Load from DB into DataService |
| `delete_series(series_id)` | Remove a series and all related data |
| `list_series(module, tags)` | Query series with filters |
| `sync_dirty(ds)` | Write only dirty entries/associations |
| `save_pipeline_snapshot(...)` | Save parameter state |
| `load_pipeline_snapshot(id)` | Restore parameter state |
| `record_provenance(...)` | Log a processing step execution |
| `list_provenance(series_id)` | View provenance history |

Data arrays are stored as raw bytes via `ndarray.tobytes()` and restored with `np.frombuffer()`.

Metadata is filtered to JSON-serializable types before storage — runtime objects (like dataset references) are silently excluded.

### 4.3 Dirty Tracking & Sync

The `sync_dirty()` method performs an incremental write:

1. Iterates `data_service.dirty_keys()` — only entries modified since last clean.
2. Ensures the parent series row exists (`INSERT OR IGNORE`).
3. Writes each dirty data blob (`INSERT OR REPLACE`).
4. Writes dirty associations.
5. Syncs tags for affected series.
6. Commits and calls `data_service.mark_clean()`.

This is more efficient than `save_series()` which writes everything for a series regardless of what changed.

---

## 5. File I/O — Loader Registry

### 5.1 Registry Mechanism

**File:** `matchbook/io/loaders/registry.py`

An extension-based file loader discovery system. Each loader self-registers at import time via the `@register_loader` decorator.

```python
_REGISTRY: dict[str, Type[BaseLoader]] = {}

def register_loader(cls):
    ext = cls.extension.lower()
    _REGISTRY[ext] = cls
    return cls

def get_loader_for_extension(ext) -> Type[BaseLoader]:
    return _REGISTRY[ext.lower()]
```

The `matchbook/io/loaders/__init__.py` auto-imports all sibling modules using `pkgutil`, which triggers all `@register_loader` decorators at first import.

### 5.2 BaseLoader Interface

```python
class BaseLoader:
    extension: str          # e.g. ".acc"

    @classmethod
    def can_load(cls, ext: str) -> bool:
        return ext.lower() == cls.extension.lower()
```

Each subclass implements `__init__(self, filepath)` and `load() -> THzData | Spectrum`.

### 5.3 Registered Loaders

| Extension | Class | File | Returns | Description |
|---|---|---|---|---|
| `.acc` | `ACCLoader` | `acc_loader.py` | `THzData` | Multi-scan THz acquisition data. Splits on `%%` separator, parses `%` headers. |
| `.dat` | `DATLoader` | `dat_loader.py` | `THzData` | Single-scan `.dat` files. Columns: time, amplitude. |
| `.txt` | `TXTLoader` | `txt_loader.py` | `THzData` | Simple two-column numeric text files. |
| `.csv` | `CSVLoader` | `csv_loader.py` | `Spectrum` | Delimiter-sniffing CSV loader. Returns `Spectrum` dataclass. |

#### ACCLoader Detail

The `.acc` format is the primary raw acquisition format. File structure:

```
%title ref acc 1
%param Date and time,2026-01-29 17:54:14.995000
-4.000000e+001  7.123456e-004
-3.999000e+001  7.234567e-004
...
%%
%title ref acc 2
%param Date and time,2026-01-29 17:54:15.123000
-4.000000e+001  7.112345e-004
...
%%
```

- Lines starting with `%` are headers (title, timestamp, etc.).
- `%%` separates individual scans.
- Data rows are space-separated `[time, amplitude]` pairs.
- Each scan becomes a `BaseTHzData` object; the collection becomes `THzData`.

---

## 6. Services

Domain-independent services under `matchbook/services/`.

### 6.1 GroupingService

**File:** `matchbook/services/grouping.py`

Groups filenames by delimiter-separated keywords and pairs references to samples. Works for any data domain (THz, UV-VIS, calibrations, etc.).

#### Usage

```python
gs = GroupingService(
    keywords=["type", "series", "temp"],
    delimiter="_",
    filelist=["reference_air_nitrogen.acc", "sample_germanium.acc"],
)
gs.simple_grouping(delimiter="_", keywords=["type", "series", "temp"])
```

#### simple_grouping() Pipeline

1. `_build_fileitems()` — creates `FilenameInfo` for each filename.
2. `parse_filenames()` — splits on delimiter, maps components to keywords.
3. `_identify_global_references()` — finds air/substrate references by series name.
4. `integrity_check()` — warns if air/substrate references are missing.
5. `_pair_references()` — matches each sample to its corresponding reference based on keyword-field equality (ignoring `data_type`).

#### Properties

| Property | Returns |
|---|---|
| `references` | `dict[filename, FilenameInfo]` — all reference files |
| `samples` | `dict[filename, FilenameInfo]` — all sample files |
| `info` | Summary string |
| `elaborate` | Detailed repr of all file items |

#### Reference Lookup

```python
gs.get_reference_filename("sample_Ge_300K.acc", ref_type="substrate")
gs.is_reference("reference_air.acc")
gs.is_sample("sample_Ge_300K.acc")
```

### 6.2 FilenameInfo

**File:** `matchbook/services/filename_info.py`

A dataclass that parses structured metadata from a filename.

```python
@dataclass
class FilenameInfo:
    filename: str
    keywords: list[str]       # ["type", "series", "temp"]
    report_list: list[str]    # fields used for reference matching

    series: str | None
    data_type: str | None     # "reference" or "sample"
    temperature: str | None
    extra_details: list[str] | None
    air_reference: str | None
    substrate_reference: str | None
```

**Parsing:** `FilenameInfo.from_filename("reference_air_nitrogen.acc", delimiter="_", keywords=["type", "series", "temp"])` splits the stem on `_` and maps positional components:
- Index 0 → `data_type` (keyword `"type"`) → `"reference"`
- Index 1 → `series` (keyword `"series"`) → `"air"`
- Index 2 → `temperature` (keyword `"temp"`) → `"nitrogen"` (or merged into series if `merge_extra=True`)

### 6.3 Provenance

**File:** `matchbook/services/provenance.py`

Two provenance mechanisms:

**`@record_provenance` decorator** — for standalone functions:
```python
@record_provenance(store_return=True, note="Applied window")
def apply_window(data, width=10):
    ...
```
Appends a record to `data.history` (the first arg must have a `.history` list). Records: timestamp, function name, arguments (JSON-safe), optional return summary.

**`record_step_provenance()`** — for Pipeline integration:
Writes to the Database's `provenance` table via `db.record_provenance()`.

---

## 7. Data Structures

### 7.1 Spectrum

**File:** `matchbook/data_structures/spectrum.py`

A lightweight dataclass for parsed file data:

```python
@dataclass
class Spectrum:
    data: np.ndarray       # (N, M) array
    headers: list[str]     # column names
    filename: str
    data_type: str | None
```

Properties: `x` (column 0), `y` (column 1), `shape`, `copy()`. Auto-synthesizes an X axis (integer indices) if only Y data is provided.

### 7.2 Helpers (numpy utilities)

**File:** `matchbook/data_structures/helpers.py`

All pure numpy — no pandas.

| Function | Signature | Purpose |
|---|---|---|
| `find_maxima(data, window, mode, values)` | `(N,2) array → (x0, y0)` | Find abs/max/min in a window |
| `is_monotonic(x)` | `1D array → bool` | Check monotonicity |
| `smooth_trace_savgol(y, window_length, polyorder)` | `1D → 1D` | Savitzky-Golay smoothing via scipy |
| `interpolate_data(data, resolution, new_limits)` | `(N,M) → (N',M)` | Interpolate to regular grid |
| `interpolate_to_common_axis(*args, axis_col_name)` | `[dicts] → [dicts]` | Multi-dataset interpolation to highest-resolution common axis |
| `interpolate_to_longest_axis(*args, axis_col_name)` | `[dicts] → [dicts]` | Multi-dataset interpolation to longest axis |

The multi-dataset functions accept `{"data": ndarray, "headers": list[str]}` dicts and handle phase unwrapping automatically for columns named in `phase_col_names`.

---

## 8. THz TDS Module

The THz time-domain spectroscopy module lives under `matchbook/modules/thz/`.

### 8.1 Module Design

Only **one file** (`adapter.py`) imports from the framework. The rest are standalone:

| File | Imports framework? | Purpose |
|---|---|---|
| `adapter.py` | **Yes** | Bridges module ↔ framework |
| `models.py` | No | Dataclasses for THz data |
| `loaders.py` | No | Format-specific file parsers |
| `analysis.py` | No | Pure analysis functions |
| `containers.py` | No | Multi-scan data containers |

### 8.2 Adapter (adapter.py)

**File:** `matchbook/modules/thz/adapter.py`

The adapter provides:

#### THzModule Class

Satisfies the `AnalysisModule` protocol. Declares:

- **4 data groups:** Time Domain, FFT, Transfer Function, Optical Constants — with 20+ trace descriptors total.
- **3 pipeline steps:**
  - `load_data` — comparison workflow (loads from the analysis-comp data/ directory structure)
  - `load_from_files` — registry-based loading (accepts arbitrary file paths)  
  - `transfer_function` — computes H(f) from FFT data

#### Step Functions

**`step_load_data(ds, series_id, *, base_dir, person_name, sample_json)`**
- Loads a person's complete dataset using the original `loaders.load_person()`.
- Pushes every array into DataService: raw measurements, time domain, FFT, transfer function, optical constants.
- Uses `_publish_dataset()` helper to atomise an `AnalysisDataset` into individual traces.

**`step_load_from_files(ds, series_id, *, file_paths, grouping_keywords, grouping_delimiter)`**
- Parses files via the loader registry (`get_loader_for_extension()`).
- Runs `GroupingService` to pair samples with references.
- Atomises each `THzData` into the DataService (raw compiled, mean time-domain, stderr).
- Creates associations from grouping results.

**`step_compute_transfer_function(ds, series_id, **params)`**
- Reads FFT data from DataService.
- Calls `analysis.compute_transfer_function_amplitude()` and `analysis.compute_transfer_function_phase()`.
- Writes results back to DataService under the `transfer_function` group.

### 8.3 Models (models.py)

**File:** `matchbook/modules/thz/models.py`

Pure dataclasses — no framework imports:

| Class | Fields |
|---|---|
| `RawMeasurement` | `time_ps`, `amplitude` |
| `TimeDomainData` | ref/sample time, amplitude, std_error; optional raw pre-windowed |
| `FFTData` | ref/sample frequency, amplitude, phase; optional deltas |
| `OpticalConstants` | `frequency_thz`, `n`, `k`, `eps1`, `eps2`, `sigma_re`, `sigma_im` |
| `SampleInfo` | `thickness_m`, `resistivity_ohm_m` |
| `AnalysisDataset` | Complete dataset: name, sample_info, raw_ref, raw_sam, time_domain, fft, optical_constants |

### 8.4 Loaders (loaders.py)

**File:** `matchbook/modules/thz/loaders.py`

The original comparison-workflow loaders. Handles format variations across different analysis pipelines:
- Tab- and space-delimited files
- 1- or 2-row headers
- Frequency in Hz or THz (auto-detected, normalised to THz)
- Masked/missing values (`--`, empty, `inf`) → NaN
- 4-column and 12-column time-domain layouts
- 9- and 10-column FFT layouts
- 7- or 8+-column optical-constants layouts

Key functions:
| Function | Purpose |
|---|---|
| `load_raw_measurement(filepath)` | 2-column raw time-domain file |
| `load_time_domain(filepath)` | Multi-column processed time-domain |
| `load_fft(filepath)` | FFT output (amplitude, phase) |
| `load_optical_constants(filepath)` | n, k, ε, σ values |
| `load_person(name, dir, raw_ref, raw_sam, sample_info)` | Complete dataset for one person |
| `discover_people(data_dir)` | Find all person subdirectories |

### 8.5 Analysis (analysis.py)

**File:** `matchbook/modules/thz/analysis.py`

Pure numpy functions — arrays in, arrays out:

| Function | Signature | Purpose |
|---|---|---|
| `compute_transfer_function_amplitude(ref_freq, ref_amp, sam_freq, sam_amp)` | `→ (freq, \|H\|)` | Amplitude ratio with auto-interpolation |
| `compute_transfer_function_phase(ref_freq, ref_phase, sam_freq, sam_phase)` | `→ (freq, Δφ)` | Phase difference with auto-interpolation |

### 8.6 Containers (containers.py)

**File:** `matchbook/modules/thz/containers.py`

No pandas — pure numpy.

#### BaseTHzData

Holds a single scan. Parses `%title` and `%param Date and time` from headers.

| Attribute | Meaning |
|---|---|
| `raw_data` | `(N, 2)` numpy array [time, amplitude] |
| `headers` | List of header strings |
| `filename` | Parsed from `%title` header |
| `scan_index` | Scan number from `%title` header |
| `timestamp` | Parsed datetime from `%param Date and time` header |

#### THzData

Multi-scan container wrapping a list of `BaseTHzData` objects.

| Property | Returns |
|---|---|
| `data_list` | `list[BaseTHzData]` — individual scans |
| `raw_data` | `(N, 1+M)` compiled array — column 0 is time, columns 1..M are individual scan amplitudes |
| `data` | `(N, 3)` averaged array — `[time, mean, stderr]` |
| `time` | Time axis (column 0 of data) |
| `y_mean` | Mean amplitude (column 1 of data) |
| `y_err` | Standard error (column 2 of data) |

Methods: `average_data(limit)`, `update_data(new_data)`.

#### TimeDomainStats

Frozen dataclass produced by `calculate_time_domain_stats()`:

```python
@dataclass(frozen=True)
class TimeDomainStats:
    time: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    stderr: np.ndarray
    n_repeats: int
    baseline_sigma: float | None
    snr_from_baseline: np.ndarray | None
```

---

## 9. GUI Layer

All GUI code lives under `matchbook/gui/`. Built with **tkinter** + **matplotlib** (TkAgg backend). The GUI is entirely generic — all content is driven by module descriptors.

### 9.1 MatchbookApp

**File:** `matchbook/gui/app.py`

Top-level window orchestrating all components.

**Constructor:** `MatchbookApp(data_service, registry, title="Matchbook Analysis")`

**Lifecycle:**
1. Collects all `DataGroupDescriptor`s from the registry.
2. Creates tkinter variables for series toggles, trace toggles, and axis controls.
3. Builds layout: sidebar (left), parameter panel (right), plot area (centre).
4. Subscribes to DataService changes for automatic redraws.
5. `app.run()` starts the tkinter event loop.

**Debounced redraw:** Toggle changes are coalesced via `root.after(50ms)` to avoid redundant redraws.

**Pipeline interaction:** Parameter changes are pushed to the pipeline via `set_param()`. "Run from here" executes the pipeline for all active series.

### 9.2 Sidebar

**File:** `matchbook/gui/sidebar.py`

Renders from descriptors:
- **Series section:** Colour-coded checkboxes for each series (colours from theme palette).
- **Data group sections:** Collapsible sections, one per `DataGroupDescriptor`, with individual trace toggles.
- **Axis controls:** Per-group autoscale toggle, manual X/Y range inputs.
- Scrollable with mousewheel support.

### 9.3 Plot Area

**File:** `matchbook/gui/plot_area.py`

`redraw(fig, data_service, active_series, ...)` — clears the figure and redraws.
- One subplot per visible data group.
- For each active series × active trace, looks up `DataKey(series_id, trace.group, trace.name)` in the DataService.
- Applies series colour + trace line style.
- Handles autoscale vs manual axis limits, with saved limits for non-autoscale mode.
- Displays "No data: ..." annotation for missing series.

### 9.4 Parameter Panel

**File:** `matchbook/gui/parameter_panel.py`

Dynamically builds widgets from `PipelineStepDescriptor` and `ParameterDescriptor`:

| ParamType | Widget |
|---|---|
| `FLOAT/INT` (with min/max) | Slider + entry |
| `FLOAT/INT` (no range) | Entry |
| `STRING` | Entry |
| `BOOL` | Checkbox |
| `CHOICE` | Dropdown (Combobox) |
| `FILE_PATH` | Entry + browse button |

Features:
- Per-step collapsible sections with "Run" button.
- "Auto-rerun on change" checkbox — automatically re-runs from the changed step.
- Bidirectional sync between slider and entry for numeric types.

### 9.5 Pipeline View

**File:** `matchbook/gui/pipeline_view.py`

Visual step list with status indicators:

| Icon | Status |
|---|---|
| ✓ | Done / cached |
| ● | Selected / editing |
| ○ | Not yet run |
| ⚠ | Stale — upstream params changed |

Controls: "Run from here" button, "Undo" button, history counter.

Subscribes to pipeline events and refreshes status icons automatically.

### 9.6 Theme

**File:** `matchbook/gui/theme.py`

Constants and helpers:
- **Palette:** 10-colour cycle (`#1f77b4`, `#ff7f0e`, `#2ca02c`, ...).
- `series_colour(index)`, `series_colour_by_id(series_id, all_series)`.
- **Line styles:** `["-", "--", "-.", ":"]` cycling.
- Layout constants: `SIDEBAR_WIDTH=260`, font sizes, etc.

---

## 10. Tools

### 10.1 Acquisition Editor

**File:** `matchbook/tools/acquisition_editor/`

A standalone matplotlib-based interactive GUI for inspecting and editing raw acquisition scans.

**`edit_acquisitions(data, filename)`**
- Accepts a numpy array or `THzData`-like mapping.
- Displays individual scans with toggle checkboxes.
- Click a scan to highlight it; scans can be toggled on/off.
- Returns the modified dataset when the window is closed.

Has its own standalone `.acc` file loader (`_loaders.py`) for independence from the main loader registry.

---

## 11. Scripts

| Script | Purpose |
|---|---|
| `matchbook/scripts/run_thz.py` | Headless pipeline execution — loads all person data, runs pipeline, prints results. |
| `matchbook/scripts/launch_gui.py` | Loads data + launches MatchbookApp with THz module. |

Both scripts auto-discover the workspace root and add it to `sys.path`.

---

## 12. Data Flow Walkthrough

### Example: Loading data and running the pipeline (headless)

```python
from matchbook.core.data_service import DataKey, DataService
from matchbook.core.registry import Registry
from matchbook.modules.thz.adapter import THzModule
from matchbook.io.database import Database

# 1. Setup
ds = DataService()
registry = Registry(ds)
registry.register(THzModule())
pipeline = registry.get_pipeline("thz_tds")

# 2. Configure and run
pipeline.set_param("load_data", "base_dir", "/path/to/data")
pipeline.set_param("load_data", "person_name", "sam")
pipeline.set_param("load_data", "sample_json", "/path/to/sample_details.json")
pipeline.run_all(ds, "sam")

# 3. Read results
entry = ds.get(DataKey("sam", "transfer_function", "amplitude"))
print(entry.x[:5], entry.y[:5])  # frequency, |H(f)|

# 4. Persist
db = Database("results.db")
db.save_series(ds, "sam", module="thz_tds")
db.close()
```

### Example: Loading .acc files via the loader registry

```python
from matchbook.io.loaders.registry import get_loader_for_extension
import matchbook.io.loaders  # triggers @register_loader

loader_cls = get_loader_for_extension(".acc")
loader = loader_cls("reference_air_nitrogen.acc")
thz_data = loader.load()

print(thz_data)           # THzData with scan count
print(thz_data.time[:5])  # time axis
print(thz_data.y_mean[:5]) # averaged amplitude
```

### Data addressing convention

All data in the DataService follows the `(series_id, group, name)` triple:

```
series_id = "chris"
├── group = "time_domain"
│   ├── name = "orig_ref_amp"    → raw reference amplitude
│   ├── name = "orig_sam_amp"    → raw sample amplitude
│   ├── name = "win_ref_amp"     → windowed reference
│   └── name = "win_sam_amp"     → windowed sample
├── group = "fft"
│   ├── name = "ref_amp"         → reference FFT amplitude
│   ├── name = "sam_amp"         → sample FFT amplitude
│   ├── name = "ref_phase"       → reference FFT phase
│   └── name = "sam_phase"       → sample FFT phase
├── group = "transfer_function"
│   ├── name = "amplitude"       → |H(f)|
│   └── name = "phase"           → Δφ(f)
└── group = "optical_constants"
    ├── name = "n"               → refractive index
    ├── name = "k"               → extinction coefficient
    ├── name = "eps1"            → real dielectric
    ├── name = "eps2"            → imaginary dielectric
    ├── name = "sigma_re"        → real conductivity
    └── name = "sigma_im"        → imaginary conductivity
```

---

## 13. File Map

```
matchbook/
├── core/
│   ├── data_service.py      — Central in-memory data store + observer + dirty tracking
│   ├── pipeline.py           — Rewindable parametric pipeline engine
│   ├── registry.py           — Module discovery, validation, registration
│   ├── module_base.py        — Descriptor types + AnalysisModule Protocol
│   └── step_report.py        — Per-step diagnostic reporting skeleton
│
├── io/
│   ├── database.py           — SQLite persistence (save/load/sync/provenance)
│   └── loaders/
│       ├── __init__.py       — Auto-imports all loader modules
│       ├── registry.py       — BaseLoader + @register_loader + get_loader_for_extension
│       ├── acc_loader.py     — .acc multi-scan loader → THzData
│       ├── dat_loader.py     — .dat single-scan loader → THzData
│       ├── txt_loader.py     — .txt numeric loader → THzData
│       └── csv_loader.py     — .csv auto-detect loader → Spectrum
│
├── services/
│   ├── grouping.py           — Filename grouping + reference pairing
│   ├── filename_info.py      — Parsed filename metadata
│   └── provenance.py         — @record_provenance decorator + DB logging
│
├── data_structures/
│   ├── spectrum.py           — Spectrum dataclass
│   └── helpers.py            — Pure numpy utilities (maxima, smoothing, interpolation)
│
├── modules/
│   └── thz/
│       ├── adapter.py        — Framework bridge (ONLY framework import point)
│       ├── models.py         — THz data dataclasses
│       ├── loaders.py        — Format-specific comparison-workflow parsers
│       ├── analysis.py       — Pure analysis functions
│       ├── containers.py     — THzData / BaseTHzData / TimeDomainStats
│       └── example_data/     — Example .acc files for testing
│
├── gui/
│   ├── app.py                — MatchbookApp (top-level window)
│   ├── sidebar.py            — Series + trace toggle sidebar
│   ├── plot_area.py          — Generic matplotlib plot renderer
│   ├── parameter_panel.py    — Dynamic parameter widgets
│   ├── pipeline_view.py      — Pipeline step status display
│   ├── theme.py              — Colours, fonts, layout constants
│   └── __init__.py
│
├── tools/
│   └── acquisition_editor/
│       ├── __init__.py       — Re-exports edit_acquisitions
│       ├── _editor.py        — Matplotlib interactive scan editor
│       └── _loaders.py       — Standalone .acc loader
│
├── scripts/
│   ├── run_thz.py            — Headless pipeline script
│   └── launch_gui.py         — GUI launch script
│
└── docs/
    └── wiki.md               — This document
```

---

*Last updated: Session in which wiki was created.*
