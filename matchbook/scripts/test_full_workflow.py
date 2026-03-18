"""Full-workflow integration test — simulates a real user session.

Exercises every layer of Matchbook end-to-end using the example .acc files,
mirroring the exact path a user takes through the GUI:

  Phase 1: Setup & module registration
  Phase 2: Load data via pipeline (as the GUI does through the parameter panel)
  Phase 3: Inspect data in the DataService (as the sidebar/plot area do)
  Phase 4: Pipeline rewind / re-run / staleness (as the pipeline view does)
  Phase 5: Tagging and associations
  Phase 6: Persist to SQLite database
  Phase 7: Reload from database into a fresh DataService
  Phase 8: DB search & query system
  Phase 9: Incremental save via sync_dirty
  Phase 10: Pipeline snapshots & provenance
  Phase 11: GUI component construction (headless — no event loop)
  Phase 12: Plot rendering (headless — no display)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

import numpy as np

# ---------------------------------------------------------------------------
# Ensure workspace root is importable
# ---------------------------------------------------------------------------
_WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

# ---------------------------------------------------------------------------
# Imports — every package we intend to test
# ---------------------------------------------------------------------------
from matchbook.core.data_service import DataEntry, DataKey, DataService
from matchbook.core.module_base import (
    DataGroupDescriptor,
    PipelineStepDescriptor,
    TraceDescriptor,
)
from matchbook.core.pipeline import Pipeline, PipelineStep, StepResult
from matchbook.core.registry import Registry, RegisteredModule
from matchbook.io.database import Database
from matchbook.io.search import (
    build_search_query,
    registered_filters,
    registered_post_filters,
)
from matchbook.io.loaders.registry import get_loader_for_extension
from matchbook.modules.thz.adapter import (
    THzModule,
    step_load_from_files,
    step_compute_transfer_function,
)
from matchbook.modules.thz.containers import THzData
from matchbook.services.grouping import GroupingService
from matchbook.services.grouping_step import step_group_files
import matchbook.io.loaders  # trigger @register_loader


EXAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "modules", "thz", "example_data",
)
REF_FILE = os.path.join(EXAMPLE_DIR, "reference_air_nitrogen.acc")
SAM_FILE = os.path.join(EXAMPLE_DIR, "sample_germanium.acc")

PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        msg = f"  [FAIL] {label}"
        if detail:
            msg += f"  — {detail}"
        print(msg)


# ===================================================================
# Phase 1: Setup & Module Registration
# ===================================================================

def phase_1_setup():
    """Register THzModule, validate descriptors, build pipeline."""
    print("\n=== Phase 1: Setup & Module Registration ===")

    ds = DataService()
    registry = Registry(ds)
    thz = THzModule()
    record = registry.register(thz)

    check("Module registered", record is not None)
    check("Module name", record.name == "thz_tds")
    check("Display name", record.display_name == "THz Time-Domain Spectroscopy")

    # Data groups
    groups = record.data_groups
    check("5 data groups", len(groups) == 5)
    group_keys = [g.key for g in groups]
    check("Group keys correct",
          group_keys == ["raw", "time_domain", "fft", "transfer_function",
                         "optical_constants"],
          f"got {group_keys}")
    check("Groups ordered by .order",
          all(groups[i].order <= groups[i + 1].order
              for i in range(len(groups) - 1)))

    # Traces within groups
    all_traces = [t for g in groups for t in g.traces]
    check(f"Total traces: {len(all_traces)} >= 15", len(all_traces) >= 15)
    for t in all_traces:
        check(f"Trace '{t.key}' has y_label",
              t.y_label is not None and len(t.y_label) > 0)

    # Pipeline
    steps = record.pipeline_step_descriptors
    check("4 pipeline step descriptors", len(steps) == 4)
    check("Step IDs",
          [s.id for s in steps] == ["load_from_files", "group_files",
                                    "align_on_peak", "transfer_function"])
    check("group_files depends on load_from_files",
          "load_from_files" in steps[1].depends_on)
    check("align_on_peak depends on group_files",
          "group_files" in steps[2].depends_on)
    check("transfer_function depends on align_on_peak",
          "align_on_peak" in steps[3].depends_on)

    # Pipeline object
    pipeline = registry.get_pipeline("thz_tds")
    check("Pipeline has 4 steps", len(pipeline.steps) == 4)
    check("Pipeline step_ids", pipeline.step_ids == ["load_from_files",
                                                      "group_files",
                                                      "align_on_peak",
                                                      "transfer_function"])

    # Registry aggregation
    all_groups = registry.all_data_groups
    check("all_data_groups returns 5", len(all_groups) == 5)

    return ds, registry, pipeline


# ===================================================================
# Phase 2: Load Data via Pipeline
# ===================================================================

def phase_2_pipeline_load(ds, registry, pipeline):
    """Simulate what the GUI does: set file_paths param, run_all."""
    print("\n=== Phase 2: Load Data via Pipeline ===")

    series_id = "workflow_test"

    # Simulate ParameterPanel setting file_paths
    pipeline.set_param("load_from_files", "file_paths",
                       REF_FILE + "\n" + SAM_FILE)
    check("Param set", pipeline.get_params("load_from_files")["file_paths"]
          == REF_FILE + "\n" + SAM_FILE)

    # Simulate pressing "Run" — the GUI calls run_all
    # Also test pipeline observer events
    events = []
    def listener(event, step_id):
        events.append((event, step_id))
    pipeline.subscribe(listener)

    pipeline.run_all(ds, series_id)

    pipeline.unsubscribe(listener)

    check("Pipeline emitted events", len(events) >= 6)
    check("step_start for load_from_files",
          ("step_start", "load_from_files") in events)
    check("step_done for load_from_files",
          ("step_done", "load_from_files") in events)
    check("step_start for group_files",
          ("step_start", "group_files") in events)
    check("step_done for group_files",
          ("step_done", "group_files") in events)
    check("step_start for transfer_function",
          ("step_start", "transfer_function") in events)
    check("step_done for transfer_function",
          ("step_done", "transfer_function") in events)

    # Verify data landed
    series = ds.list_series()
    check(f"Series created: {len(series)} >= 2", len(series) >= 2)

    return series_id, series


# ===================================================================
# Phase 3: Inspect Data (Sidebar + Plot Area perspective)
# ===================================================================

def phase_3_inspect(ds, all_series, registry):
    """Verify the data the sidebar and plot area would display."""
    print("\n=== Phase 3: Inspect Data (Sidebar / Plot) ===")

    for sid in all_series:
        groups = ds.list_groups(sid)
        check(f"{sid} has groups", len(groups) >= 2,
              f"groups={groups}")

        # The sidebar builds trace toggles from data_groups descriptors
        for dg in registry.all_data_groups:
            for trace in dg.traces:
                entry = ds.get(DataKey(sid, trace.group, trace.name))
                # Not all traces populated yet (fft, optical_constants not
                # computed), but raw + time_domain should exist
                if dg.key in ("raw", "time_domain"):
                    # At least one should exist
                    pass  # checked below

        # Verify raw/compiled
        raw = ds.get(DataKey(sid, "raw", "compiled"))
        if raw is not None:
            check(f"{sid} raw/compiled has data",
                  raw.x.shape[0] > 10 and raw.y.shape[0] > 10)
            check(f"{sid} raw/compiled metadata has filename",
                  "filename" in raw.metadata)

        # Verify time_domain/mean
        td = ds.get(DataKey(sid, "time_domain", "mean"))
        if td is not None:
            check(f"{sid} time_domain/mean has data",
                  td.x.shape[0] > 10 and td.y.shape[0] > 10)
            check(f"{sid} time_domain/mean x/y same length",
                  td.x.shape == td.y.shape)
            check(f"{sid} time_domain/mean no NaN",
                  not np.any(np.isnan(td.y)))

        # Verify _meta/grouping
        meta = ds.get(DataKey(sid, "_meta", "grouping"))
        if meta is not None:
            check(f"{sid} has parsed data_type",
                  "data_type" in meta.metadata)

    # Verify list_names
    for sid in all_series:
        for grp in ds.list_groups(sid):
            names = ds.list_names(sid, grp)
            check(f"{sid}/{grp} has names", len(names) >= 1,
                  f"names={names}")


# ===================================================================
# Phase 4: Pipeline Rewind / Re-run / Staleness
# ===================================================================

def phase_4_pipeline_state(ds, pipeline, series_id):
    """Test pipeline caching, staleness detection, history, and rewind."""
    print("\n=== Phase 4: Pipeline State (Cache / Rewind) ===")

    # After run_all, steps should not be stale
    check("load_from_files not stale", not pipeline.is_stale("load_from_files"))
    check("transfer_function not stale",
          not pipeline.is_stale("transfer_function"))

    # Cache should exist
    cache_lf = pipeline.get_cache("load_from_files")
    check("load_from_files cached", cache_lf is not None)
    check("Cache is StepResult", isinstance(cache_lf, StepResult))
    check("Cache has timestamp", cache_lf.timestamp > 0)

    cache_tf = pipeline.get_cache("transfer_function")
    check("transfer_function cached", cache_tf is not None)

    # History should exist (1 snapshot from run_all)
    check("History length >= 1", pipeline.history_length >= 1)

    # Change a param → staleness
    old_delim = pipeline.get_params("group_files")["grouping_delimiter"]
    pipeline.set_param("group_files", "grouping_delimiter", "-")
    check("group_files now stale", pipeline.is_stale("group_files"))

    # downstream_of
    downstream = pipeline.downstream_of("load_from_files")
    check("downstream_of includes all steps",
          downstream == ["load_from_files", "group_files",
                         "align_on_peak", "transfer_function"])

    # Rewind to restore original params
    pipeline.rewind(0)
    restored = pipeline.get_params("group_files")["grouping_delimiter"]
    check("Rewind restored delimiter",
          restored == old_delim,
          f"expected '{old_delim}', got '{restored}'")

    # Re-run from a specific step
    pipeline.run_from("transfer_function", ds, series_id)
    check("Re-run from transfer_function succeeded",
          not pipeline.is_stale("transfer_function"))
    check("History grew", pipeline.history_length >= 2)


# ===================================================================
# Phase 5: Tagging & Associations
# ===================================================================

def phase_5_tags_and_associations(ds, all_series):
    """Test tagging and association features used across the app."""
    print("\n=== Phase 5: Tagging & Associations ===")

    for sid in all_series:
        ds.tag_series(sid, "thz_tds")
        ds.tag_series(sid, "workflow_test")

    check("Tags attached",
          all("thz_tds" in ds.get_tags(s) for s in all_series))

    # Query by tag
    tagged = ds.series_with_tag("thz_tds")
    check(f"series_with_tag returns {len(tagged)} series",
          len(tagged) == len(all_series))

    # all_tags
    at = ds.all_tags()
    check("all_tags includes thz_tds", "thz_tds" in at)
    check("all_tags includes workflow_test", "workflow_test" in at)

    # Tag removal
    ds.untag_series(all_series[0], "workflow_test")
    check("untag_series works",
          "workflow_test" not in ds.get_tags(all_series[0]))
    ds.tag_series(all_series[0], "workflow_test")  # restore

    # Associations
    if len(all_series) >= 2:
        ds.associate(all_series[1], "paired_reference", all_series[0])
        assoc = ds.get_association(all_series[1], "paired_reference")
        check("Association created", assoc == all_series[0])

        all_assocs = ds.list_associations()
        check("list_associations non-empty", len(all_assocs) >= 1)

        # Remove and re-add
        ds.remove_association(all_series[1], "paired_reference")
        check("Association removed",
              ds.get_association(all_series[1], "paired_reference") is None)
        ds.associate(all_series[1], "paired_reference", all_series[0])

    # Query method
    entries = ds.query(series_id=all_series[0])
    check(f"query(series_id) returns entries for {all_series[0]}",
          len(entries) >= 2)

    entries_grp = ds.query(group="time_domain")
    check("query(group='time_domain') returns entries",
          len(entries_grp) >= 1)

    entries_tag = ds.query(tags={"thz_tds"})
    check("query(tags={'thz_tds'}) returns entries",
          len(entries_tag) >= 2)


# ===================================================================
# Phase 6: Persist to Database
# ===================================================================

def phase_6_persist(ds, all_series):
    """Save everything to a temp SQLite database."""
    print("\n=== Phase 6: Persist to Database ===")

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()

    db = Database(db_path)

    for sid in all_series:
        db.save_series(ds, sid, module="thz_tds",
                       display_name=sid.split("/")[-1])

    db_list = db.list_series()
    check(f"DB has {len(db_list)} series", len(db_list) == len(all_series))

    for rec in db_list:
        check(f"DB record '{rec['id']}' has module",
              rec["module"] == "thz_tds")
        check(f"DB record '{rec['id']}' has tags",
              "thz_tds" in rec["tags"])

    # Associations in DB
    db_assocs = db.list_associations()
    check("DB associations exist", len(db_assocs) >= 1)

    print(f"\n  DB file: {db_path} ({os.path.getsize(db_path)} bytes)")

    return db, db_path


# ===================================================================
# Phase 7: Reload from Database
# ===================================================================

def phase_7_reload(db, all_series, original_ds):
    """Load series from DB into a fresh DataService, verify integrity."""
    print("\n=== Phase 7: Reload from Database ===")

    ds2 = DataService()
    for sid in all_series:
        result = db.load_series(sid, ds2)
        check(f"Loaded {sid}", result is True)

    loaded_series = ds2.list_series()
    check("All series restored", set(loaded_series) == set(all_series))

    # Data integrity — compare every entry
    mismatches = 0
    total = 0
    for sid in all_series:
        for grp in original_ds.list_groups(sid):
            for name in original_ds.list_names(sid, grp):
                total += 1
                orig = original_ds.get(DataKey(sid, grp, name))
                loaded = ds2.get(DataKey(sid, grp, name))
                if loaded is None:
                    mismatches += 1
                    print(f"    MISSING: {sid}/{grp}/{name}")
                elif not (np.allclose(orig.x, loaded.x) and
                          np.allclose(orig.y, loaded.y)):
                    mismatches += 1
                    print(f"    MISMATCH: {sid}/{grp}/{name}")

    check(f"Data integrity: {total} entries, {mismatches} mismatches",
          mismatches == 0)

    # Tags survived
    for sid in all_series:
        orig_tags = original_ds.get_tags(sid)
        loaded_tags = ds2.get_tags(sid)
        check(f"Tags for {sid} match", orig_tags == loaded_tags,
              f"orig={orig_tags}, loaded={loaded_tags}")

    # Associations survived
    orig_assocs = set(
        (a[0], a[1], a[2]) for a in original_ds.list_associations()
    )
    loaded_assocs = set(
        (a[0], a[1], a[2]) for a in ds2.list_associations()
    )
    check("Associations match after reload",
          orig_assocs == loaded_assocs,
          f"orig={orig_assocs}, loaded={loaded_assocs}")

    return ds2


# ===================================================================
# Phase 8: DB Search & Query
# ===================================================================

def phase_8_search(db, all_series):
    """Exercise every built-in search filter."""
    print("\n=== Phase 8: DB Search & Query ===")

    # List registered filters
    sql_filters = registered_filters()
    post_filters = registered_post_filters()
    check(f"SQL filters registered: {len(sql_filters)} >= 11",
          len(sql_filters) >= 11)
    expected = {"name_contains", "name_exact", "id_contains", "module",
                "date_after", "date_before", "has_tag", "has_group",
                "has_name", "metadata_field", "text_search"}
    check("All expected SQL filters present",
          expected.issubset(set(sql_filters.keys())),
          f"missing: {expected - set(sql_filters.keys())}")

    # build_search_query (unit test)
    sql, params = build_search_query({"name_contains": "test",
                                       "has_group": "raw"})
    check("build_search_query returns SQL string",
          "WHERE" in sql and "SELECT" in sql)
    check("build_search_query returns params list",
          isinstance(params, list) and len(params) >= 2)

    # Actual searches against the DB
    results = db.search(text_search="reference")
    check(f"text_search='reference' found {len(results)} result(s)",
          len(results) >= 1)

    results = db.search(has_tag="thz_tds")
    check(f"has_tag='thz_tds' found {len(results)}",
          len(results) == len(all_series))

    results = db.search(has_group="time_domain")
    check(f"has_group='time_domain' found {len(results)} >= 1",
          len(results) >= 1)

    results = db.search(has_group="raw")
    check(f"has_group='raw' found {len(results)} >= 1",
          len(results) >= 1)

    results = db.search(has_name="mean")
    check(f"has_name='mean' found {len(results)} >= 1",
          len(results) >= 1)

    results = db.search(module="thz_tds")
    check(f"module='thz_tds' found {len(results)}",
          len(results) == len(all_series))

    results = db.search(name_contains="air")
    check(f"name_contains='air' found {len(results)} >= 1",
          len(results) >= 1)

    today = time.strftime("%Y-%m-%d")
    results = db.search(date_after="2020-01-01")
    check(f"date_after='2020-01-01' found all",
          len(results) == len(all_series))

    results = db.search(date_before="2099-01-01")
    check(f"date_before='2099-01-01' found all",
          len(results) == len(all_series))

    # Combined filter
    results = db.search(has_tag="thz_tds", has_group="time_domain")
    check(f"Combined tag+group filter found {len(results)} >= 1",
          len(results) >= 1)

    # No results
    results = db.search(text_search="nonexistent_xyz_123")
    check("No-match search returns empty", len(results) == 0)

    # Empty search returns all
    results = db.search()
    check(f"Empty search returns all {len(all_series)}",
          len(results) == len(all_series))


# ===================================================================
# Phase 9: sync_dirty (incremental write)
# ===================================================================

def phase_9_sync_dirty(db, ds):
    """Add new data, sync only dirty entries."""
    print("\n=== Phase 9: sync_dirty ===")

    # Mark clean first
    ds.mark_clean()
    check("DataService clean after mark_clean", not ds.is_dirty())

    # Add a new entry
    test_sid = ds.list_series()[0]
    ds.put(DataEntry(
        key=DataKey(test_sid, "test_group", "test_trace"),
        x=np.linspace(0, 10, 50),
        y=np.sin(np.linspace(0, 10, 50)),
        metadata={"source": "sync_dirty_test"},
    ))
    check("DataService dirty after put", ds.is_dirty())
    check("1 dirty key", len(ds.dirty_keys()) == 1)

    count = db.sync_dirty(ds)
    check(f"sync_dirty wrote {count} entry", count == 1)
    check("Clean after sync", not ds.is_dirty())

    # Verify it round-trips
    ds_check = DataService()
    db.load_series(test_sid, ds_check)
    entry = ds_check.get(DataKey(test_sid, "test_group", "test_trace"))
    check("sync'd entry round-trips", entry is not None)
    if entry is not None:
        check("sync'd data correct",
              np.allclose(entry.y, np.sin(np.linspace(0, 10, 50))))

    # Clean up: remove test entry from ds so it doesn't pollute later checks
    ds.remove(DataKey(test_sid, "test_group", "test_trace"))


# ===================================================================
# Phase 10: Pipeline Snapshots & Provenance
# ===================================================================

def phase_10_provenance(db, pipeline, all_series):
    """Save pipeline snapshots, record provenance events."""
    print("\n=== Phase 10: Pipeline Snapshots & Provenance ===")

    sid = all_series[0]

    # Pipeline snapshot
    all_params = {
        step.id: dict(step.params) for step in pipeline.steps
    }
    snap_id = db.save_pipeline_snapshot(sid, all_params,
                                        label="initial_run")
    check("Pipeline snapshot saved", snap_id is not None and snap_id >= 0)

    snaps = db.list_pipeline_snapshots(sid)
    check("Snapshot listed", len(snaps) >= 1)
    if snaps:
        check("Snapshot has label", snaps[0]["label"] == "initial_run")

    loaded_snap = db.load_pipeline_snapshot(snap_id)
    check("Snapshot round-trips",
          set(loaded_snap.keys()) == set(all_params.keys()))

    # Provenance
    prov_id = db.record_provenance(
        sid,
        step_id="load_from_files",
        inputs=[REF_FILE, SAM_FILE],
        params=pipeline.get_params("load_from_files"),
        outputs=["raw/compiled", "time_domain/mean"],
        note="Integration test provenance",
    )
    check("Provenance recorded", prov_id is not None and prov_id >= 0)

    prov_list = db.list_provenance(sid)
    check("Provenance listed", len(prov_list) >= 1)
    if prov_list:
        check("Provenance has step_id",
              prov_list[0]["step_id"] == "load_from_files")
        check("Provenance has note",
              prov_list[0]["note"] == "Integration test provenance")


# ===================================================================
# Phase 11: GUI Component Construction (headless)
# ===================================================================

def phase_11_gui_headless(ds, registry, pipeline):
    """Build all GUI components without starting an event loop.

    This verifies that the GUI code can construct itself from the current
    data_service / registry / pipeline state without crashing.
    """
    print("\n=== Phase 11: GUI Component Construction (headless) ===")

    import tkinter as tk
    root = tk.Tk()
    root.withdraw()  # never show the window

    try:
        from matchbook.gui.sidebar import build_sidebar
        from matchbook.gui.parameter_panel import ParameterPanel
        from matchbook.gui.pipeline_view import PipelineView
        from matchbook.gui import theme

        # Theme exports
        check("Theme has PALETTE", len(theme.PALETTE) >= 5)
        check("series_colour works", theme.series_colour(0).startswith("#"))
        all_series = ds.list_series()
        check("series_colour_by_id works",
              theme.series_colour_by_id(all_series[0], all_series).startswith("#"))
        check("trace_line_style works",
              theme.trace_line_style(0) in ["-", "--", "-.", ":"])

        data_groups = registry.all_data_groups

        # Build trace/series variables (exactly what MatchbookApp.__init__ does)
        series_vars = {s: tk.BooleanVar(value=True) for s in all_series}
        trace_vars = {}
        autoscale_vars = {}
        manual_x_enabled = {}
        manual_y_enabled = {}
        manual_x_min = {}
        manual_x_max = {}
        manual_y_min = {}
        manual_y_max = {}

        for dg in data_groups:
            autoscale_vars[dg.key] = tk.BooleanVar(value=True)
            manual_x_enabled[dg.key] = tk.BooleanVar(value=False)
            manual_y_enabled[dg.key] = tk.BooleanVar(value=False)
            manual_x_min[dg.key] = tk.StringVar(value="")
            manual_x_max[dg.key] = tk.StringVar(value="")
            manual_y_min[dg.key] = tk.StringVar(value="")
            manual_y_max[dg.key] = tk.StringVar(value="")
            for t in dg.traces:
                trace_vars[t.key] = tk.BooleanVar(value=t.default_visible)

        redraws = []

        # --- Sidebar ---
        sidebar_frame = build_sidebar(
            parent=root,
            series_names=all_series,
            series_vars=series_vars,
            data_groups=data_groups,
            trace_vars=trace_vars,
            autoscale_vars=autoscale_vars,
            manual_x_enabled=manual_x_enabled,
            manual_y_enabled=manual_y_enabled,
            manual_x_min=manual_x_min,
            manual_x_max=manual_x_max,
            manual_y_min=manual_y_min,
            manual_y_max=manual_y_max,
            schedule_redraw=lambda: redraws.append(1),
        )
        check("Sidebar built without error", sidebar_frame is not None)

        # --- Parameter Panel ---
        param_changes = []
        run_requests = []
        step_descs = [
            sd for rm in registry.registered_modules.values()
            for sd in rm.pipeline_step_descriptors
        ]
        param_panel = ParameterPanel(
            parent=root,
            step_descriptors=step_descs,
            on_param_changed=lambda s, p, v: param_changes.append((s, p, v)),
            on_run_from=lambda s: run_requests.append(s),
        )
        check("ParameterPanel built", param_panel.frame is not None)
        check("ParameterPanel has auto_rerun var",
              hasattr(param_panel, "auto_rerun"))

        # Read param values back
        vals = param_panel.get_param_values("load_from_files")
        check("get_param_values returns dict", isinstance(vals, dict))

        # --- Pipeline View ---
        step_selected = []
        pv = PipelineView(
            parent=root,
            pipeline=pipeline,
            on_step_selected=lambda s: step_selected.append(s),
            on_run_from=lambda s: run_requests.append(s),
            on_rewind=lambda i: None,
        )
        check("PipelineView built", pv.frame is not None)

        # Refresh (tests _get_status which uses get_cache)
        pv.refresh()
        check("PipelineView.refresh() succeeded", True)

    finally:
        root.destroy()


# ===================================================================
# Phase 12: Plot Rendering (headless)
# ===================================================================

def phase_12_plot_headless(ds, registry):
    """Render plots into a matplotlib figure without display."""
    print("\n=== Phase 12: Plot Rendering (headless) ===")

    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    from matchbook.gui.plot_area import redraw
    from matchbook.gui import theme

    fig = plt.figure(figsize=(12, 8))
    data_groups = registry.all_data_groups
    all_series = ds.list_series()

    # Build visibility dicts — enable raw + time_domain traces
    trace_vis = {}
    autoscale = {}
    mx_en = {}
    my_en = {}
    mx_min = {}
    mx_max = {}
    my_min = {}
    my_max = {}
    saved_xlim = {}
    saved_ylim = {}

    for dg in data_groups:
        autoscale[dg.key] = True
        mx_en[dg.key] = False
        my_en[dg.key] = False
        mx_min[dg.key] = ""
        mx_max[dg.key] = ""
        my_min[dg.key] = ""
        my_max[dg.key] = ""
        for t in dg.traces:
            trace_vis[t.key] = t.default_visible

    redraw(
        fig=fig,
        data_service=ds,
        active_series=all_series,
        all_series=all_series,
        data_groups=data_groups,
        trace_visibility=trace_vis,
        autoscale=autoscale,
        manual_x_enabled=mx_en,
        manual_y_enabled=my_en,
        manual_x_min=mx_min,
        manual_x_max=mx_max,
        manual_y_min=my_min,
        manual_y_max=my_max,
        saved_xlim=saved_xlim,
        saved_ylim=saved_ylim,
    )

    n_axes = len(fig.axes)
    check(f"Figure has {n_axes} subplot(s)", n_axes >= 1)
    check("At least one axis has lines",
          any(len(ax.lines) > 0 for ax in fig.axes))

    # Save to verify it's a valid figure
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        fig.savefig(tmp_path, dpi=72)
        check(f"Figure saved ({os.path.getsize(tmp_path)} bytes)",
              os.path.getsize(tmp_path) > 1000)
    finally:
        os.unlink(tmp_path)

    plt.close(fig)


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 66)
    print("  MATCHBOOK — FULL WORKFLOW INTEGRATION TEST")
    print("  Simulates real user session with every subsystem")
    print("=" * 66)

    assert os.path.isfile(REF_FILE), f"Reference not found: {REF_FILE}"
    assert os.path.isfile(SAM_FILE), f"Sample not found: {SAM_FILE}"

    # Phase 1
    ds, registry, pipeline = phase_1_setup()

    # Phase 2
    series_id, all_series = phase_2_pipeline_load(ds, registry, pipeline)

    # Phase 3
    phase_3_inspect(ds, all_series, registry)

    # Phase 4
    phase_4_pipeline_state(ds, pipeline, series_id)

    # Phase 5
    phase_5_tags_and_associations(ds, all_series)

    # Phase 6
    db, db_path = phase_6_persist(ds, all_series)

    try:
        # Phase 7
        phase_7_reload(db, all_series, ds)

        # Phase 8
        phase_8_search(db, all_series)

        # Phase 9
        phase_9_sync_dirty(db, ds)

        # Phase 10
        phase_10_provenance(db, pipeline, all_series)

        db.close()
    finally:
        os.unlink(db_path)

    # Phase 11
    phase_11_gui_headless(ds, registry, pipeline)

    # Phase 12
    phase_12_plot_headless(ds, registry)

    # === Summary ===
    print("\n" + "=" * 66)
    total = PASS + FAIL
    print(f"  RESULTS: {PASS}/{total} passed, {FAIL} failed")
    if FAIL == 0:
        print("  ALL TESTS PASSED")
    else:
        print(f"  {FAIL} FAILURE(S) — see [FAIL] lines above")
    print("=" * 66)

    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
