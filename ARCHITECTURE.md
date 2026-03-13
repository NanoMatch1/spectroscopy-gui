# Matchbook — Modular Analysis & Display Platform

## 1. Design Goals

| Goal | Implication |
|------|-------------|
| **Modular** | Each analysis module (THz, Raman, XRD…) is self-contained and runnable without the GUI or other modules. |
| **Self-registering** | Modules declare their data groups, traces, parameters, and pipeline steps through descriptors — the GUI renders them generically. |
| **Parametric / rewindable** | The pipeline records every step's parameters.  Any step can be edited and everything downstream re-executed automatically. |
| **Scriptable** | The core engine (DataService + Pipeline) has zero GUI dependencies.  Scripts can run the full analysis headlessly. |
| **Presentation-ready** | Plot display must be publication / meeting quality: configurable themes, consistent colour palettes, exportable figures. |
| **Database-backed** | A lightweight local database (SQLite) stores datasets, tags, pipeline snapshots, and series metadata for persistent, filterable access. |

---

## 2. Package Layout

```
matchbook/
├── core/                        # Framework layer — zero GUI deps
│   ├── __init__.py
│   ├── data_service.py          # Central data store + observer bus
│   ├── pipeline.py              # Rewindable pipeline engine
│   ├── registry.py              # Module loader / discovery
│   └── module_base.py           # Abstract base + descriptor types
│
├── io/                          # IO module (always loaded)
│   ├── __init__.py
│   ├── database.py              # SQLite database interface
│   ├── importers.py             # Generic + format-specific file importers
│   └── exporters.py             # CSV / HDF5 / image export
│
├── modules/                     # Analysis modules (one sub-package each)
│   ├── __init__.py
│   └── thz/                     # THz-TDS module (first module)
│       ├── __init__.py
│       ├── module.py            # THzModule(ModuleBase) — registration
│       ├── analysis.py          # Pure analysis functions
│       ├── loaders.py           # THz-specific file parsing (from current loaders.py)
│       └── models.py            # THz data models (from current models.py)
│
├── gui/                         # GUI application (depends on core + tkinter)
│   ├── __init__.py
│   ├── app.py                   # MainWindow — orchestrator
│   ├── sidebar.py               # Dynamic collapsible sidebar builder
│   ├── plot_area.py             # Matplotlib-based plot manager
│   ├── parameter_panel.py       # Dynamic widget builder for module params
│   ├── pipeline_view.py         # Pipeline timeline / DAG controls
│   ├── series_browser.py        # Series list + tag filter UI
│   └── theme.py                 # Palette, fonts, style constants
│
└── scripts/                     # Headless / scripting entry points
    ├── run_thz.py               # Run THz pipeline without GUI
    └── launch_gui.py            # Start the GUI application
```

### Dependency Rules

```
modules/*  ──depends-on──►  core/module_base  (abstract base only)
                             core/data_service
                             core/pipeline

gui/*      ──depends-on──►  core/*
                             io/*
                             (discovers modules/* at runtime)

scripts/*  ──depends-on──►  core/*,  io/*,  modules/*
                             (no gui)
```

> **Modules never import `gui`.**  They receive a thin context object at registration time and call methods on it.

---

## 3. Core Layer

### 3.1 Data Service (`core/data_service.py`)

The DataService is the single source of truth for all data in the application. Every module writes to it; the GUI reads from it.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np


@dataclass
class DataKey:
    """Unique address for a piece of data."""
    series_id: str          # e.g. "sam_2025-03-12_silicon"
    group: str              # e.g. "time_domain", "optical_constants"
    name: str               # e.g. "ref_amplitude", "n"


@dataclass
class DataEntry:
    key: DataKey
    x: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata examples: {"x_label": "Time (ps)", "y_label": "Amplitude (V)",
    #                      "display_label": "Windowed reference", "unit": "V"}


class DataService:
    """Central data store with observer notifications."""

    def __init__(self):
        self._store: dict[tuple[str, str, str], DataEntry] = {}
        self._listeners: list[Callable[[str, DataKey], None]] = []
        self._series_tags: dict[str, set[str]] = {}  # series_id -> tags

    # ── Write ──
    def put(self, entry: DataEntry) -> None:
        k = (entry.key.series_id, entry.key.group, entry.key.name)
        self._store[k] = entry
        self._notify("put", entry.key)

    def remove(self, key: DataKey) -> None:
        k = (key.series_id, key.group, key.name)
        self._store.pop(k, None)
        self._notify("remove", key)

    # ── Read ──
    def get(self, key: DataKey) -> DataEntry | None:
        return self._store.get((key.series_id, key.group, key.name))

    def list_series(self) -> list[str]:
        return sorted({k[0] for k in self._store})

    def list_groups(self, series_id: str | None = None) -> list[str]:
        if series_id:
            return sorted({k[1] for k in self._store if k[0] == series_id})
        return sorted({k[1] for k in self._store})

    def list_names(self, series_id: str, group: str) -> list[str]:
        return sorted({k[2] for k in self._store if k[0] == series_id and k[1] == group})

    def query(self, series_id: str | None = None,
              group: str | None = None,
              tags: set[str] | None = None) -> list[DataEntry]:
        """Flexible query with optional filters."""
        results = []
        for (sid, grp, name), entry in self._store.items():
            if series_id and sid != series_id:
                continue
            if group and grp != group:
                continue
            if tags and not tags.issubset(self._series_tags.get(sid, set())):
                continue
            results.append(entry)
        return results

    # ── Tags ──
    def tag_series(self, series_id: str, *tags: str) -> None:
        self._series_tags.setdefault(series_id, set()).update(tags)

    def get_tags(self, series_id: str) -> set[str]:
        return self._series_tags.get(series_id, set())

    def series_with_tag(self, tag: str) -> list[str]:
        return [sid for sid, t in self._series_tags.items() if tag in t]

    # ── Observer ──
    def subscribe(self, listener: Callable[[str, DataKey], None]) -> None:
        self._listeners.append(listener)

    def _notify(self, event: str, key: DataKey) -> None:
        for fn in self._listeners:
            fn(event, key)
```

**Key points:**
- Data is stored as `(series_id, group, name)` triples.
- Series replace the current concept of "persons" — a series is any named dataset.
- Tags enable cross-cutting filters (e.g. `"silicon"`, `"2025-03"`, `"high-res"`).
- The observer pattern (`subscribe`) drives GUI updates without coupling.

### 3.2 Pipeline Engine (`core/pipeline.py`)

The pipeline engine tracks a directed chain of processing steps, their parameters, and cached outputs — enabling rewind and replay.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
import copy
import time


@dataclass
class PipelineStep:
    """One step in a processing pipeline."""
    id: str                                    # unique step identifier
    name: str                                  # human-readable
    fn: Callable[..., Any]                     # the processing function
    params: dict[str, Any]                     # current parameter values
    param_descriptors: list[ParameterDescriptor]  # UI-renderable descriptors
    depends_on: list[str] = field(default_factory=list)  # step ids


@dataclass
class StepResult:
    """Cached result of a step execution."""
    step_id: str
    params_snapshot: dict[str, Any]            # frozen copy of params used
    output: Any                                # whatever the step returns
    timestamp: float = field(default_factory=time.time)


class Pipeline:
    """
    Rewindable, parametric pipeline engine.

    Usage:
        pipe = Pipeline()
        pipe.add_step(PipelineStep(id="load", name="Load raw", fn=load_fn, params={}, ...))
        pipe.add_step(PipelineStep(id="window", name="Window", fn=window_fn, params={"width": 50}, ...))
        pipe.add_step(PipelineStep(id="fft", name="FFT", fn=fft_fn, params={"zero_pad": 1024}, ...))

        pipe.run_all(data_service, series_id)

        # User changes windowing width:
        pipe.set_param("window", "width", 80)
        pipe.run_from("window", data_service, series_id)  # re-runs window + fft
    """

    def __init__(self):
        self._steps: dict[str, PipelineStep] = {}
        self._order: list[str] = []            # topological order
        self._cache: dict[str, StepResult] = {}
        self._history: list[dict[str, dict[str, Any]]] = []  # list of full param snapshots
        self._listeners: list[Callable[[str, str], None]] = []  # (event, step_id)

    def add_step(self, step: PipelineStep) -> None:
        self._steps[step.id] = step
        if step.id not in self._order:
            self._order.append(step.id)

    def get_step(self, step_id: str) -> PipelineStep:
        return self._steps[step_id]

    @property
    def steps(self) -> list[PipelineStep]:
        return [self._steps[sid] for sid in self._order]

    def set_param(self, step_id: str, param_name: str, value: Any) -> None:
        """Update a parameter — does NOT auto-run (call run_from to propagate)."""
        self._steps[step_id].params[param_name] = value

    def run_from(self, step_id: str, data_service, series_id: str) -> None:
        """
        Re-execute from `step_id` through all downstream dependents.
        Each step.fn receives (data_service, series_id, **step.params)
        and should write its output into data_service.
        """
        self._snapshot_history()
        idx = self._order.index(step_id)
        for sid in self._order[idx:]:
            step = self._steps[sid]
            self._notify("step_start", sid)
            step.fn(data_service, series_id, **step.params)
            self._cache[sid] = StepResult(
                step_id=sid,
                params_snapshot=copy.deepcopy(step.params),
                output=None,   # output lives in data_service
            )
            self._notify("step_done", sid)

    def run_all(self, data_service, series_id: str) -> None:
        """Execute the full pipeline."""
        if self._order:
            self.run_from(self._order[0], data_service, series_id)

    def invalidate_from(self, step_id: str) -> list[str]:
        """Return step_id and all downstream steps that would re-run."""
        idx = self._order.index(step_id)
        return self._order[idx:]

    def get_cache(self, step_id: str) -> StepResult | None:
        return self._cache.get(step_id)

    def rewind(self, history_index: int) -> None:
        """Restore all parameters to a previous state."""
        snapshot = self._history[history_index]
        for step_id, params in snapshot.items():
            self._steps[step_id].params = copy.deepcopy(params)

    @property
    def history_length(self) -> int:
        return len(self._history)

    def subscribe(self, listener: Callable[[str, str], None]) -> None:
        self._listeners.append(listener)

    def _snapshot_history(self) -> None:
        snap = {sid: copy.deepcopy(s.params) for sid, s in self._steps.items()}
        self._history.append(snap)

    def _notify(self, event: str, step_id: str) -> None:
        for fn in self._listeners:
            fn(event, step_id)
```

**Pipeline features:**
- **Rewind**: full parameter history, restorable to any point.
- **Partial re-run**: `run_from(step_id)` only re-executes from the edited step onward.
- **Observer events**: GUI can show progress (step start / done).
- **Step functions are pure**: they read from + write to `DataService`, taking params as kwargs. This means they work identically with or without a GUI.

### 3.3 Module Base & Descriptors (`core/module_base.py`)

This is the interface contract that modules implement. Modules never see tkinter — they only produce **descriptors** that the GUI interprets.

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


# ── Parameter Descriptors ─────────────────────────────────────────────────────

class ParamType(Enum):
    FLOAT = "float"
    INT = "int"
    STRING = "string"
    BOOL = "bool"
    CHOICE = "choice"        # dropdown
    FILE_PATH = "file_path"  # file picker


@dataclass
class ParameterDescriptor:
    """Describes one tunable parameter for a pipeline step."""
    name: str                          # internal key
    label: str                         # human-readable label
    type: ParamType
    default: Any
    min: float | int | None = None     # for FLOAT / INT sliders
    max: float | int | None = None
    step: float | int | None = None    # slider step size
    choices: list[str] | None = None   # for CHOICE type
    tooltip: str = ""


# ── Data Group Descriptors ────────────────────────────────────────────────────

@dataclass
class TraceDescriptor:
    """Describes one plottable trace within a data group."""
    key: str                           # unique trace identifier
    label: str                         # display name
    group: str                         # data_service group key
    name: str                          # data_service name key
    y_label: str = ""
    default_visible: bool = False
    line_style: str | None = None      # override if needed


@dataclass
class DataGroupDescriptor:
    """
    Describes a collapsible group of traces in the sidebar.
    One DataGroupDescriptor → one collapsible section in the GUI.
    """
    key: str                           # unique group identifier
    label: str                         # section title
    x_label: str                       # x-axis label for this group's plots
    traces: list[TraceDescriptor] = field(default_factory=list)
    order: int = 0                     # sidebar sort order


# ── Pipeline Step Descriptors ─────────────────────────────────────────────────

@dataclass
class PipelineStepDescriptor:
    """Describes a pipeline step (enough for the GUI to build controls)."""
    id: str
    name: str
    params: list[ParameterDescriptor] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)


# ── Module Base ───────────────────────────────────────────────────────────────

class ModuleBase(ABC):
    """
    Abstract base class for all analysis modules.

    Lifecycle:
      1. __init__()           — module-internal setup (no GUI, no DataService)
      2. register(context)    — called by the framework; module declares its
                                data groups, params, pipeline steps
      3. load_series(...)     — called to load data for one series
      4. (pipeline runs)      — framework executes pipeline steps
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique module name, e.g. 'thz_tds'."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable module name, e.g. 'THz Time-Domain Spectroscopy'."""
        ...

    @abstractmethod
    def get_data_groups(self) -> list[DataGroupDescriptor]:
        """
        Return descriptors for all data groups this module can produce.
        Called once at registration time to build sidebar sections.
        """
        ...

    @abstractmethod
    def get_pipeline_steps(self) -> list[PipelineStepDescriptor]:
        """
        Return descriptors for all processing steps.
        Called once at registration time to build pipeline controls.
        """
        ...

    @abstractmethod
    def create_pipeline(self) -> list[PipelineStep]:
        """
        Return actual PipelineStep objects (with bound functions).
        These get added to the Pipeline engine.
        """
        ...

    def load_series(self, data_service, series_id: str,
                    file_paths: dict[str, str] | None = None,
                    **kwargs) -> None:
        """
        Load raw data for a series and write initial entries into
        data_service.  Override to handle module-specific file formats.
        """
        pass

    def on_register(self, context: ModuleContext) -> None:
        """
        Optional hook called after the module is registered with the
        framework.  The context provides access to the DataService,
        Pipeline, and GUI widget request API.
        """
        pass


# ── Module Context (provided by framework at registration) ────────────────────

@dataclass
class ModuleContext:
    """
    Thin façade handed to modules at registration time.
    Modules use this to interact with the framework without
    importing any framework internals.
    """
    data_service: Any            # DataService instance
    pipeline: Any                # Pipeline instance
    request_sidebar_section: Callable[[DataGroupDescriptor], None] | None = None
    request_parameter_widget: Callable[[PipelineStepDescriptor], None] | None = None
    # GUI requestors are None when running headless
```

---

## 4. IO Module (`io/`)

The IO module is always loaded. It provides:

### 4.1 Database (`io/database.py`)

A SQLite-backed store for persistent data management.

```
Tables:
  series       (id, name, module, created_at, metadata_json)
  series_tags  (series_id, tag)
  data_blobs   (series_id, group, name, x_blob, y_blob, metadata_json)
  pipeline_snapshots (id, series_id, step_params_json, created_at, label)
```

**Key operations:**
- `save_series(data_service, series_id)` — serialises all DataEntries for a series into the DB.
- `load_series(series_id) → populates DataService` — reconstitutes data.
- `list_series(tags=..., module=...) → list` — filtered browsing.
- `save_pipeline_snapshot(pipeline, series_id, label)` — stores full parameter state.
- `load_pipeline_snapshot(snapshot_id) → dict` — restores parameters.

### 4.2 Importers (`io/importers.py`)

Generic file-import framework. Each module can register file type handlers:

```python
# Module registers its own importers:
io_module.register_importer("thz_tds", "time_domain", load_time_domain_file)
io_module.register_importer("thz_tds", "fft", load_fft_file)
```

### 4.3 Exporters (`io/exporters.py`)

- CSV export (current `export.py` logic, generalised).
- HDF5 export for large datasets.
- Figure export (PNG/SVG/PDF at configurable DPI).

---

## 5. Module Registration Flow

This is the critical interaction protocol. Here's the full lifecycle:

```
                          ┌─────────────────────┐
                          │   Application Start  │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  Discover modules    │  (scan modules/ for ModuleBase subclasses,
                          │  in modules/*        │   or explicit list in config)
                          └──────────┬──────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            │                        │                        │
   ┌────────▼────────┐    ┌─────────▼────────┐    ┌─────────▼────────┐
   │  THzModule()    │    │  RamanModule()   │    │  XRDModule()     │
   │  .__init__()    │    │  .__init__()     │    │  .__init__()     │
   └────────┬────────┘    └─────────┬────────┘    └─────────┬────────┘
            │                        │                        │
            │         ┌──────────────▼──────────────┐         │
            └────────►│  Registry.register(module)  │◄────────┘
                      │                             │
                      │  1. module.get_data_groups() │  → sidebar sections
                      │  2. module.get_pipeline_steps()  → pipeline UI
                      │  3. module.create_pipeline() │  → engine steps
                      │  4. module.on_register(ctx)  │  → module init
                      └──────────────┬──────────────┘
                                     │
                      ┌──────────────▼──────────────┐
                      │   GUI renders all registered │
                      │   groups, traces, params     │
                      └─────────────────────────────┘
```

### What a module author writes (complete example)

Here is what the THz module would look like. This is the **entire integration surface** — everything else in the module is internal:

```python
# modules/thz/module.py
from core.module_base import (
    ModuleBase, ModuleContext,
    DataGroupDescriptor, TraceDescriptor,
    PipelineStepDescriptor, ParameterDescriptor, ParamType,
)
from core.pipeline import PipelineStep
from modules.thz.analysis import (
    step_load_raw, step_window, step_fft,
    step_transfer_function, step_optical_constants,
)


class THzModule(ModuleBase):

    @property
    def name(self) -> str:
        return "thz_tds"

    @property
    def display_name(self) -> str:
        return "THz Time-Domain Spectroscopy"

    def get_data_groups(self) -> list[DataGroupDescriptor]:
        return [
            DataGroupDescriptor(
                key="time_domain", label="Time Domain",
                x_label="Time (ps)", order=0,
                traces=[
                    TraceDescriptor("td_orig_ref",  "Original reference",     "time_domain", "orig_ref_amp",     y_label="Amplitude (V)"),
                    TraceDescriptor("td_orig_sam",  "Original sample",        "time_domain", "orig_sam_amp",     y_label="Amplitude (V)"),
                    TraceDescriptor("td_win_ref",   "Windowed reference",     "time_domain", "win_ref_amp",      y_label="Amplitude (V)", default_visible=True),
                    TraceDescriptor("td_win_sam",   "Windowed sample",        "time_domain", "win_sam_amp",      y_label="Amplitude (V)", default_visible=True),
                ],
            ),
            DataGroupDescriptor(
                key="fft", label="FFT",
                x_label="Frequency (THz)", order=1,
                traces=[
                    TraceDescriptor("fft_ref_amp",   "Reference amplitude",  "fft", "ref_amp",   y_label="Amplitude", default_visible=True),
                    TraceDescriptor("fft_sam_amp",   "Sample amplitude",     "fft", "sam_amp",   y_label="Amplitude", default_visible=True),
                    TraceDescriptor("fft_ref_phase", "Reference phase",      "fft", "ref_phase", y_label="Phase (rad)"),
                    TraceDescriptor("fft_sam_phase", "Sample phase",         "fft", "sam_phase", y_label="Phase (rad)"),
                ],
            ),
            DataGroupDescriptor(
                key="transfer_function", label="Transfer Function",
                x_label="Frequency (THz)", order=2,
                traces=[
                    TraceDescriptor("tf_amp",   "|H(f)| amplitude",  "transfer_function", "amplitude", y_label="|H(f)|",  default_visible=True),
                    TraceDescriptor("tf_phase", "Phase difference",  "transfer_function", "phase",     y_label="Δφ (rad)", default_visible=True),
                ],
            ),
            DataGroupDescriptor(
                key="optical_constants", label="Optical Constants",
                x_label="Frequency (THz)", order=3,
                traces=[
                    TraceDescriptor("oc_n",    "Refractive index n",   "optical_constants", "n",        y_label="n",         default_visible=True),
                    TraceDescriptor("oc_k",    "Extinction coeff k",   "optical_constants", "k",        y_label="k",         default_visible=True),
                    TraceDescriptor("oc_eps1", "ε₁ (real dielectric)", "optical_constants", "eps1",     y_label="ε₁"),
                    TraceDescriptor("oc_eps2", "ε₂ (imag dielectric)", "optical_constants", "eps2",     y_label="ε₂"),
                    TraceDescriptor("oc_sre",  "σ_Re (S/m)",          "optical_constants", "sigma_re", y_label="σ_Re (S/m)"),
                    TraceDescriptor("oc_sim",  "σ_Im (S/m)",          "optical_constants", "sigma_im", y_label="σ_Im (S/m)"),
                ],
            ),
        ]

    def get_pipeline_steps(self) -> list[PipelineStepDescriptor]:
        return [
            PipelineStepDescriptor(
                id="load_raw", name="Load Raw Data",
                params=[
                    ParameterDescriptor("ref_path", "Reference file", ParamType.FILE_PATH, ""),
                    ParameterDescriptor("sam_path", "Sample file",    ParamType.FILE_PATH, ""),
                ],
            ),
            PipelineStepDescriptor(
                id="window", name="Apply Window",
                depends_on=["load_raw"],
                params=[
                    ParameterDescriptor("window_type", "Window function", ParamType.CHOICE, "boxcar",
                                        choices=["boxcar", "hann", "hamming", "blackman", "tukey"]),
                    ParameterDescriptor("start_ps", "Window start (ps)", ParamType.FLOAT, 0.0,
                                        min=0.0, max=500.0, step=0.1),
                    ParameterDescriptor("end_ps",   "Window end (ps)",   ParamType.FLOAT, 100.0,
                                        min=0.0, max=500.0, step=0.1),
                ],
            ),
            PipelineStepDescriptor(
                id="fft", name="FFT",
                depends_on=["window"],
                params=[
                    ParameterDescriptor("zero_pad", "Zero-pad length", ParamType.INT, 0,
                                        min=0, max=65536, step=1),
                ],
            ),
            PipelineStepDescriptor(
                id="transfer_fn", name="Transfer Function",
                depends_on=["fft"],
            ),
            PipelineStepDescriptor(
                id="optical_constants", name="Extract Optical Constants",
                depends_on=["transfer_fn"],
                params=[
                    ParameterDescriptor("thickness_m", "Sample thickness (m)", ParamType.FLOAT, 315e-6,
                                        min=0.0, max=0.01, step=1e-6),
                    ParameterDescriptor("max_iter", "Max iterations", ParamType.INT, 100,
                                        min=1, max=10000, step=1),
                ],
            ),
        ]

    def create_pipeline(self) -> list[PipelineStep]:
        descriptors = {s.id: s for s in self.get_pipeline_steps()}
        return [
            PipelineStep(id="load_raw",           name="Load Raw Data",
                         fn=step_load_raw,
                         params=self._defaults("load_raw", descriptors),
                         param_descriptors=descriptors["load_raw"].params),
            PipelineStep(id="window",             name="Apply Window",
                         fn=step_window,
                         params=self._defaults("window", descriptors),
                         param_descriptors=descriptors["window"].params,
                         depends_on=["load_raw"]),
            PipelineStep(id="fft",                name="FFT",
                         fn=step_fft,
                         params=self._defaults("fft", descriptors),
                         param_descriptors=descriptors["fft"].params,
                         depends_on=["window"]),
            PipelineStep(id="transfer_fn",        name="Transfer Function",
                         fn=step_transfer_function,
                         params=self._defaults("transfer_fn", descriptors),
                         param_descriptors=descriptors["transfer_fn"].params,
                         depends_on=["fft"]),
            PipelineStep(id="optical_constants",  name="Extract Optical Constants",
                         fn=step_optical_constants,
                         params=self._defaults("optical_constants", descriptors),
                         param_descriptors=descriptors["optical_constants"].params,
                         depends_on=["transfer_fn"]),
        ]

    @staticmethod
    def _defaults(step_id, descriptors):
        return {p.name: p.default for p in descriptors[step_id].params}
```

### What a pipeline step function looks like

```python
# modules/thz/analysis.py
# Pipeline step functions — pure, no GUI imports.
# Signature: fn(data_service, series_id, **params)

from core.data_service import DataService, DataKey, DataEntry
import numpy as np


def step_window(data_service: DataService, series_id: str, **params):
    """Apply a window function to the raw time-domain data."""
    window_type = params.get("window_type", "boxcar")
    start_ps = params.get("start_ps", 0.0)
    end_ps = params.get("end_ps", 100.0)

    # Read raw data from data_service
    ref = data_service.get(DataKey(series_id, "time_domain", "orig_ref_amp"))
    sam = data_service.get(DataKey(series_id, "time_domain", "orig_sam_amp"))
    ref_t = data_service.get(DataKey(series_id, "time_domain", "orig_ref_time"))
    sam_t = data_service.get(DataKey(series_id, "time_domain", "orig_sam_time"))

    if ref is None or sam is None:
        return

    # Compute windowed data (actual window logic here)
    # ...windowed_ref, windowed_sam = apply_window(...)

    # Write windowed data back
    data_service.put(DataEntry(
        key=DataKey(series_id, "time_domain", "win_ref_amp"),
        x=ref_t.x, y=windowed_ref,
        metadata={"x_label": "Time (ps)", "y_label": "Amplitude (V)",
                  "display_label": "Windowed reference"},
    ))
    data_service.put(DataEntry(
        key=DataKey(series_id, "time_domain", "win_sam_amp"),
        x=sam_t.x, y=windowed_sam,
        metadata={"x_label": "Time (ps)", "y_label": "Amplitude (V)",
                  "display_label": "Windowed sample"},
    ))
```

---

## 6. GUI Layer

### 6.1 Main Application (`gui/app.py`)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Matchbook — [Active Module: THz TDS]                         [_][□][X]    │
├──────────┬──────────────────────────────────────────────┬───────────────────┤
│ SIDEBAR  │                                              │  PARAMETER PANEL  │
│          │              PLOT AREA                        │                   │
│ ┌──────┐ │  ┌────────────────────────────────────────┐  │  Pipeline:        │
│ │Series│ │  │                                        │  │  ┌─────────────┐  │
│ │ ☑ A  │ │  │         Time Domain                    │  │  │ Load Raw ✓  │  │
│ │ ☑ B  │ │  │                                        │  │  │ Window   ●  │  │
│ │ ☐ C  │ │  │                                        │  │  │ FFT      ○  │  │
│ └──────┘ │  ├────────────────────────────────────────┤  │  │ TF       ○  │  │
│          │  │                                        │  │  │ Opt.Const○  │  │
│ ┌──────┐ │  │         FFT                            │  │  └─────────────┘  │
│ │Time  │ │  │                                        │  │                   │
│ │Domain│ │  │                                        │  │  ── Window ──     │
│ │ ☑ ref│ │  ├────────────────────────────────────────┤  │  Type: [Hann  ▼]  │
│ │ ☑ sam│ │  │                                        │  │  Start: ===●==    │
│ │ ☐ raw│ │  │     Optical Constants                  │  │           42.0 ps │
│ └──────┘ │  │                                        │  │  End:   ====●=    │
│          │  │                                        │  │           98.0 ps │
│ ┌──────┐ │  └────────────────────────────────────────┘  │                   │
│ │FFT   │ │                                              │  [☑ Auto-rerun]   │
│ │ ☑ amp│ │  ┌─ Toolbar ─────────────────────────────┐  │  [▶ Run from here]│
│ │ ☐ φ  │ │  │ 🔍 Pan │ Zoom │ Save │  ◀ ▶ history  │  │                   │
│ └──────┘ │  └────────────────────────────────────────┘  │  ── FFT ──        │
│  ...     │                                              │  Zero-pad: [1024] │
│          │                                              │                   │
│ [Tags]   │                                              │  ── Opt. Const ── │
│ filter…  │                                              │  Thickness: ===●= │
└──────────┴──────────────────────────────────────────────┴───────────────────┘
```

**Three-panel layout:**

| Panel | Content | Source |
|-------|---------|--------|
| **Sidebar** (left) | Series browser + collapsible trace groups with toggles + tag filter | Built dynamically from registered `DataGroupDescriptor`s |
| **Plot Area** (centre) | Matplotlib subplots, one per active data group | Driven by `DataService.query()` + trace visibility state |
| **Parameter Panel** (right) | Pipeline step list + parameter widgets (sliders, dropdowns, text fields) + run controls | Built dynamically from registered `PipelineStepDescriptor`s |

### 6.2 Sidebar Builder (`gui/sidebar.py`)

The sidebar is built generically from descriptors:

```python
def build_sidebar(parent, data_service, registered_groups, series_vars, trace_vars):
    """
    Constructs the sidebar entirely from registered descriptors.
    No hard-coded domain knowledge.
    """
    # Series section — populated from data_service.list_series()
    build_series_section(parent, data_service, series_vars)

    # Data group sections — one collapsible per DataGroupDescriptor
    for group_desc in sorted(registered_groups, key=lambda g: g.order):
        build_group_section(parent, group_desc, trace_vars)

    # Tag filter section
    build_tag_filter(parent, data_service)
```

### 6.3 Parameter Panel (`gui/parameter_panel.py`)

Builds widgets from `ParameterDescriptor` objects:

| ParamType | Widget |
|-----------|--------|
| `FLOAT` | Slider + entry (if min/max given), else entry only |
| `INT` | Slider + entry (if min/max given), else entry only |
| `STRING` | Entry field |
| `BOOL` | Checkbox |
| `CHOICE` | Dropdown / Combobox |
| `FILE_PATH` | Entry + browse button |

Each widget is bound so that changes either auto-trigger `pipeline.run_from()` (if auto-rerun is enabled) or just mark the step as dirty.

### 6.4 Pipeline View (`gui/pipeline_view.py`)

Displays the pipeline as a vertical list of steps with status indicators:

- **✓** Completed (cached and up to date)
- **●** Currently selected / editing
- **○** Not yet run / stale
- **⚠** Dirty (upstream params changed, needs re-run)

Controls:
- **Click a step** → shows its parameters in the parameter panel.
- **[▶ Run from here]** → calls `pipeline.run_from(step_id)`.
- **[☑ Auto-rerun]** → any parameter change triggers immediate `run_from`.
- **[◀ ▶ History]** → step back/forward through pipeline history snapshots.

### 6.5 Plot Area (`gui/plot_area.py`)

Generalised version of current `_redraw()`:

```python
def redraw(fig, data_service, active_series, active_groups, trace_visibility, theme):
    """
    Generic redraw.  Reads all needed data from DataService.
    No module-specific knowledge.
    """
    fig.clear()
    visible_groups = [g for g in active_groups if any(
        trace_visibility.get(t.key, False) for t in g.traces
    )]

    axes = fig.subplots(len(visible_groups), 1, squeeze=False)

    for row, group_desc in enumerate(visible_groups):
        ax = axes[row, 0]
        for series_id in active_series:
            colour = theme.series_colour(series_id)
            for trace_i, trace_desc in enumerate(group_desc.traces):
                if not trace_visibility.get(trace_desc.key, False):
                    continue
                entry = data_service.get(DataKey(series_id, trace_desc.group, trace_desc.name))
                if entry is None:
                    continue
                ax.plot(entry.x, entry.y,
                        color=colour,
                        linestyle=theme.line_style(trace_i),
                        label=f"{series_id}: {trace_desc.label}")
        ax.set_xlabel(group_desc.x_label)
        ax.set_title(group_desc.label)
        # ... legend, grid, axis limits, etc.
```

---

## 7. Series Model

"Series" replaces "persons" — a series is a self-contained dataset from one experiment run.

```python
@dataclass
class Series:
    id: str                     # unique identifier: "sam_Si_2025-03"
    name: str                   # display name: "Sam — Silicon (March 2025)"
    module: str                 # which module: "thz_tds"
    tags: set[str]              # {"silicon", "room_temp", "sam"}
    metadata: dict[str, Any]    # {"thickness_m": 315e-6, "resistivity": 0.4}
    created_at: str             # ISO timestamp
```

Series are stored in the database and loaded into the DataService. Tags enable flexible grouping — the sidebar's series section supports multi-tag filtering.

---

## 8. Headless / Scripting Usage

The system works without the GUI:

```python
# scripts/run_thz.py
from core.data_service import DataService
from core.pipeline import Pipeline
from core.registry import Registry
from modules.thz.module import THzModule
from io.database import Database

# Setup
ds = DataService()
db = Database("my_data.db")
registry = Registry(ds)

# Register modules
thz = THzModule()
registry.register(thz)

# Load data for a series
thz.load_series(ds, "sam_silicon", file_paths={
    "ref_path": "data/original/reference_RT.txt",
    "sam_path": "data/original/sample_RT.txt",
})

# Configure and run pipeline
pipe = registry.get_pipeline("thz_tds")
pipe.set_param("window", "window_type", "hann")
pipe.set_param("window", "start_ps", 5.0)
pipe.set_param("window", "end_ps", 90.0)
pipe.set_param("optical_constants", "thickness_m", 315e-6)
pipe.run_all(ds, "sam_silicon")

# Access results
n = ds.get(DataKey("sam_silicon", "optical_constants", "n"))
print(f"n at 1 THz: {np.interp(1.0, n.x, n.y):.4f}")

# Save to database
db.save_series(ds, "sam_silicon")
```

---

## 9. Module Integration Checklist

To integrate an existing analysis module with matchbook, a module author needs to:

| # | Task | Files to create/modify |
|---|------|----------------------|
| 1 | Create `module.py` subclassing `ModuleBase` | `modules/mymod/module.py` |
| 2 | Implement `get_data_groups()` — declare what plots you produce | same file |
| 3 | Implement `get_pipeline_steps()` — declare processing steps + params | same file |
| 4 | Implement `create_pipeline()` — return `PipelineStep` objects with bound functions | same file |
| 5 | Wrap analysis functions as step functions: `fn(data_service, series_id, **params)` | `modules/mymod/analysis.py` |
| 6 | Each step reads inputs from `data_service.get()` and writes outputs via `data_service.put()` | same |
| 7 | (Optional) Implement `load_series()` for custom file parsers | `modules/mymod/module.py` |
| 8 | (Optional) Implement `on_register()` for additional setup | same file |

**Dependencies for the module**: only `core.module_base` and `core.data_service` — two files, no GUI, no IO, no database.

---

## 10. Migration Path from Current Codebase

| Current file | Becomes | Notes |
|---|---|---|
| `models.py` | `modules/thz/models.py` | Unchanged — internal to THz module |
| `loaders.py` | `modules/thz/loaders.py` | Unchanged — used by `load_series()` |
| `export.py` | `io/exporters.py` | Generalised with module-agnostic column builder |
| `gui.py` (TRACE_REGISTRY) | `modules/thz/module.py` `get_data_groups()` | Hard-coded extractors become data_service lookups |
| `gui.py` (ComparisonGUI) | `gui/app.py` + `gui/sidebar.py` + `gui/plot_area.py` | Split into generic components |
| `run_comparisons.py` | `scripts/run_thz.py` + `scripts/launch_gui.py` | Split headless vs GUI entry points |
| `sample_details.json` | Series metadata in database | Loaded during `load_series()` |

---

## 11. Additional Features That Align Well

These features emerge naturally from the architecture:

### 11.1 Overlay / Comparison Mode
Since series are first-class and the plot engine is generic, comparing any number of series is automatic — just toggle them on. The current "persons comparison" feature generalises to N-way comparison of any datasets.

### 11.2 Computed / Derived Traces
Modules can register "virtual" traces that compute on-the-fly from other data in the DataService (e.g. difference between two series, normalised overlay). These appear in the sidebar like any other trace.

### 11.3 Annotations & Markers
The DataService can store annotation entries (vertical lines, shaded regions, text labels) that the plot area renders. Useful for marking features like absorption peaks.

### 11.4 Plot Layout Presets
Save and restore named plot configurations (which groups/traces are visible, axis limits, theme). Helpful for meetings — set up a "presentation" preset and recall it.

### 11.5 Export Bundles
Export a complete analysis bundle: data CSVs + figures + pipeline parameters as a reproducible snapshot that can be re-imported.

### 11.6 Undo/Redo for Parameters
The pipeline history already provides this. The GUI can expose Ctrl+Z / Ctrl+Y to step through parameter history.

### 11.7 Batch Processing
Run a pipeline across multiple series with a parameter sweep (e.g. test 5 different window widths across 10 datasets). Results auto-populate as new series in the DataService.

### 11.8 Module Hot-Reload
During development, allow reloading a module without restarting the GUI. The registry re-scans, re-registers data groups and pipeline steps.

---

## 12. Technology Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| GUI toolkit | **tkinter + ttk** (current) | Zero-install, cross-platform, already working. Optionally migrate to **PySide6/Qt** later for richer widgets if needed. |
| Plotting | **Matplotlib** (current) | Excellent for publication-quality figures, already integrated. |
| Database | **SQLite** via `sqlite3` | Zero-install, single-file, fast for this scale. |
| Data serialisation | **NumPy `.npy`** in DB BLOBs, or **HDF5** for large data | Efficient binary storage. |
| Config/settings | **JSON** or **TOML** | Human-readable, simple. |
| Packaging | Standard **pyproject.toml** + pip | Modern Python packaging. |

---

## 13. Implementation Order

A suggested build sequence, each phase delivering a usable increment:

| Phase | Deliverable | Estimated scope |
|-------|-------------|-----------------|
| **Phase 1** | `core/data_service.py`, `core/module_base.py`, `core/pipeline.py`, `core/registry.py` — the framework skeleton with tests | Core framework |
| **Phase 2** | `modules/thz/` — port existing THz analysis into the module structure, run headlessly via script | Module migration |
| **Phase 3** | `gui/app.py`, `gui/sidebar.py`, `gui/plot_area.py` — generic GUI shell that renders registered data groups | Basic GUI |
| **Phase 4** | `gui/parameter_panel.py`, `gui/pipeline_view.py` — interactive parameter editing + pipeline rewind | Pipeline UI |
| **Phase 5** | `io/database.py` — persistence, series management, tag filtering | Database |
| **Phase 6** | Polish: themes, presets, export bundles, batch mode | Extras |
