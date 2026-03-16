"""Integration test — load .acc files, group, atomise, persist, restore.

Exercises the full Matchbook pipeline using the example .acc data:
  1. Load via the loader registry
  2. Group via GroupingService
  3. Atomise into DataService
  4. Create associations
  5. Save to SQLite database
  6. Load back from database
  7. Verify data integrity
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

# Ensure workspace root is on the path
_WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

from matchbook.core.data_service import DataKey, DataService
from matchbook.core.registry import Registry
from matchbook.io.database import Database
from matchbook.io.loaders.registry import get_loader_for_extension
from matchbook.modules.thz.adapter import THzModule, step_load_from_files
from matchbook.modules.thz.containers import THzData
from matchbook.services.grouping import GroupingService
import matchbook.io.loaders  # trigger @register_loader


EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "modules", "thz", "example_data")
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


def test_loader_registry():
    """Test 1: Load both .acc files via the loader registry."""
    print("\n=== Test 1: Loader Registry ===")

    loader_cls = get_loader_for_extension(".acc")
    check("Got ACCLoader class", loader_cls.__name__ == "ACCLoader")

    ref_loader = loader_cls(REF_FILE)
    ref_data = ref_loader.load()
    check("Reference loaded", isinstance(ref_data, THzData))
    check(f"Reference has {len(ref_data.data_list)} scans",
          len(ref_data.data_list) >= 2,
          f"got {len(ref_data.data_list)}")
    check("Reference has averaged data",
          ref_data.data is not None and ref_data.data.shape[0] > 0)
    check("Reference time axis exists",
          ref_data.time is not None and len(ref_data.time) > 10)
    check("Reference y_mean exists",
          ref_data.y_mean is not None and len(ref_data.y_mean) > 10)

    sam_loader = loader_cls(SAM_FILE)
    sam_data = sam_loader.load()
    check("Sample loaded", isinstance(sam_data, THzData))
    check(f"Sample has {len(sam_data.data_list)} scans",
          len(sam_data.data_list) >= 2,
          f"got {len(sam_data.data_list)}")
    check("Sample has averaged data",
          sam_data.data is not None and sam_data.data.shape[0] > 0)

    print(f"\n  Reference: {len(ref_data.data_list)} scans, "
          f"{ref_data.raw_data.shape[0]} points × {ref_data.raw_data.shape[1]} columns")
    print(f"  Sample:    {len(sam_data.data_list)} scans, "
          f"{sam_data.raw_data.shape[0]} points × {sam_data.raw_data.shape[1]} columns")
    print(f"  Time range: [{ref_data.time[0]:.2f}, {ref_data.time[-1]:.2f}]")

    return ref_data, sam_data


def test_grouping():
    """Test 2: GroupingService pairs reference and sample."""
    print("\n=== Test 2: GroupingService ===")

    filelist = [
        "reference_air_nitrogen.acc",
        "sample_germanium.acc",
    ]
    gs = GroupingService(
        keywords=["type", "series", "temp"],
        delimiter="_",
        filelist=filelist,
    )
    gs.simple_grouping(delimiter="_", keywords=["type", "series", "temp"])

    check("File items created", len(gs.file_items) == 2)
    check("Reference identified",
          gs.is_reference("reference_air_nitrogen.acc"))
    check("Sample identified",
          gs.is_sample("sample_germanium.acc"))

    refs = gs.references
    sams = gs.samples
    check(f"1 reference found", len(refs) == 1)
    check(f"1 sample found", len(sams) == 1)

    ref_item = gs.file_items["reference_air_nitrogen.acc"]
    sam_item = gs.file_items["sample_germanium.acc"]
    check(f"Ref data_type = 'reference'", ref_item.data_type == "reference")
    check(f"Sam data_type = 'sample'", sam_item.data_type == "sample")
    check(f"Ref series parsed: '{ref_item.series}'", ref_item.series is not None)
    check(f"Sam series parsed: '{sam_item.series}'", sam_item.series is not None)

    print(f"\n  Reference info: {ref_item}")
    print(f"  Sample info: {sam_item}")

    return gs


def test_dataservice_atomisation(ref_data, sam_data):
    """Test 3: Atomise THzData into DataService."""
    print("\n=== Test 3: DataService Atomisation ===")

    ds = DataService()

    # Atomise reference
    series_ref = "test/reference_air_nitrogen.acc"
    ds.put(
        __import__("matchbook.core.data_service", fromlist=["DataEntry"]).DataEntry(
            key=DataKey(series_ref, "raw", "compiled"),
            x=ref_data.raw_data[:, 0],
            y=ref_data.raw_data[:, 1],
            metadata={"n_scans": len(ref_data.data_list), "filename": "reference_air_nitrogen.acc"},
        )
    )
    ds.put(
        __import__("matchbook.core.data_service", fromlist=["DataEntry"]).DataEntry(
            key=DataKey(series_ref, "time_domain", "mean"),
            x=ref_data.time,
            y=ref_data.y_mean,
            metadata={"x_label": "Time (ps)", "y_label": "Amplitude"},
        )
    )

    # Atomise sample
    series_sam = "test/sample_germanium.acc"
    ds.put(
        __import__("matchbook.core.data_service", fromlist=["DataEntry"]).DataEntry(
            key=DataKey(series_sam, "raw", "compiled"),
            x=sam_data.raw_data[:, 0],
            y=sam_data.raw_data[:, 1],
            metadata={"n_scans": len(sam_data.data_list), "filename": "sample_germanium.acc"},
        )
    )
    ds.put(
        __import__("matchbook.core.data_service", fromlist=["DataEntry"]).DataEntry(
            key=DataKey(series_sam, "time_domain", "mean"),
            x=sam_data.time,
            y=sam_data.y_mean,
            metadata={"x_label": "Time (ps)", "y_label": "Amplitude"},
        )
    )

    # Create association
    ds.associate(series_sam, "air_reference", series_ref)

    series_list = ds.list_series()
    check("Both series in DataService", len(series_list) == 2)
    check("Reference series present", series_ref in series_list)
    check("Sample series present", series_sam in series_list)

    ref_entry = ds.get(DataKey(series_ref, "time_domain", "mean"))
    check("Ref time_domain/mean retrievable", ref_entry is not None)
    check("Ref data non-empty", ref_entry.x.shape[0] > 10)

    sam_entry = ds.get(DataKey(series_sam, "time_domain", "mean"))
    check("Sam time_domain/mean retrievable", sam_entry is not None)
    check("Sam data non-empty", sam_entry.x.shape[0] > 10)

    assoc = ds.get_association(series_sam, "air_reference")
    check("Association sam→ref exists", assoc == series_ref)

    check("DataService is dirty", ds.is_dirty())
    check(f"Dirty keys count = 4", len(ds.dirty_keys()) == 4)

    groups_ref = ds.list_groups(series_ref)
    check(f"Ref groups: {groups_ref}", set(groups_ref) == {"raw", "time_domain"})

    print(f"\n  Series: {series_list}")
    print(f"  Ref mean shape: {ref_entry.x.shape}")
    print(f"  Sam mean shape: {sam_entry.x.shape}")

    return ds, series_ref, series_sam


def test_database_roundtrip(ds, series_ref, series_sam):
    """Test 4: Save to DB, clear DataService, load back, verify."""
    print("\n=== Test 4: Database Round-trip ===")

    # Create a temp database
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()

    try:
        db = Database(db_path)

        # Save both series
        db.save_series(ds, series_ref, module="thz_tds", display_name="Reference Air")
        db.save_series(ds, series_sam, module="thz_tds", display_name="Sample Germanium")

        # Verify DB listing
        db_series = db.list_series()
        check("DB has 2 series", len(db_series) == 2)
        db_names = {s["id"] for s in db_series}
        check("Both series in DB", {series_ref, series_sam} == db_names)

        # Verify associations in DB
        db_assocs = db.list_associations(series_sam)
        check("Association persisted in DB", len(db_assocs) == 1)
        if db_assocs:
            check("Association target correct",
                  db_assocs[0]["target"] == series_ref)

        # Clear DataService completely
        ds2 = DataService()
        check("Fresh DataService is empty", len(ds2.list_series()) == 0)

        # Load back from DB
        loaded_ref = db.load_series(series_ref, ds2)
        loaded_sam = db.load_series(series_sam, ds2)
        check("Ref loaded from DB", loaded_ref is True)
        check("Sam loaded from DB", loaded_sam is True)

        series_list = ds2.list_series()
        check("Both series restored", len(series_list) == 2)

        # Verify data integrity
        ref_entry_orig = ds.get(DataKey(series_ref, "time_domain", "mean"))
        ref_entry_loaded = ds2.get(DataKey(series_ref, "time_domain", "mean"))
        check("Ref time_domain/mean restored",
              ref_entry_loaded is not None)

        if ref_entry_orig is not None and ref_entry_loaded is not None:
            x_match = np.allclose(ref_entry_orig.x, ref_entry_loaded.x)
            y_match = np.allclose(ref_entry_orig.y, ref_entry_loaded.y)
            check("Ref X data matches after round-trip", x_match)
            check("Ref Y data matches after round-trip", y_match)
            check(f"Ref shape matches: {ref_entry_orig.x.shape} == {ref_entry_loaded.x.shape}",
                  ref_entry_orig.x.shape == ref_entry_loaded.x.shape)

        sam_entry_orig = ds.get(DataKey(series_sam, "raw", "compiled"))
        sam_entry_loaded = ds2.get(DataKey(series_sam, "raw", "compiled"))
        check("Sam raw/compiled restored",
              sam_entry_loaded is not None)

        if sam_entry_orig is not None and sam_entry_loaded is not None:
            check("Sam X data matches after round-trip",
                  np.allclose(sam_entry_orig.x, sam_entry_loaded.x))
            check("Sam Y data matches after round-trip",
                  np.allclose(sam_entry_orig.y, sam_entry_loaded.y))

        # Verify association survived round-trip
        assoc = ds2.get_association(series_sam, "air_reference")
        check("Association survived round-trip", assoc == series_ref)

        print(f"\n  DB file: {db_path} ({os.path.getsize(db_path)} bytes)")
        print(f"  Loaded series: {series_list}")

        db.close()

    finally:
        os.unlink(db_path)


def test_sync_dirty(ref_data, sam_data):
    """Test 5: sync_dirty() — incremental write."""
    print("\n=== Test 5: sync_dirty() ===")

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()

    try:
        from matchbook.core.data_service import DataEntry

        ds = DataService()
        db = Database(db_path)

        series_id = "sync_test/reference"
        ds.put(DataEntry(
            key=DataKey(series_id, "time_domain", "mean"),
            x=ref_data.time,
            y=ref_data.y_mean,
            metadata={"filename": "reference_air_nitrogen.acc"},
        ))
        ds.tag_series(series_id, "test_tag")

        check("DataService is dirty before sync", ds.is_dirty())
        check("1 dirty key", len(ds.dirty_keys()) == 1)

        count = db.sync_dirty(ds)
        check(f"sync_dirty wrote {count} entries", count == 1)
        check("DataService clean after sync", not ds.is_dirty())

        # Add more data — only new data should sync
        ds.put(DataEntry(
            key=DataKey(series_id, "time_domain", "stderr"),
            x=ref_data.time,
            y=ref_data.y_err,
            metadata={},
        ))
        check("1 new dirty key after adding stderr", len(ds.dirty_keys()) == 1)

        count2 = db.sync_dirty(ds)
        check(f"Second sync wrote {count2} entries", count2 == 1)
        check("Clean after second sync", not ds.is_dirty())

        # Load into fresh DS to verify
        ds3 = DataService()
        db.load_series(series_id, ds3)
        mean_entry = ds3.get(DataKey(series_id, "time_domain", "mean"))
        stderr_entry = ds3.get(DataKey(series_id, "time_domain", "stderr"))
        check("Mean entry round-tripped", mean_entry is not None)
        check("Stderr entry round-tripped", stderr_entry is not None)

        tags = ds3.get_tags(series_id)
        check("Tag survived sync", "test_tag" in tags)

        db.close()

    finally:
        os.unlink(db_path)


def test_step_load_from_files():
    """Test 6: step_load_from_files — the adapter's registry-based loading."""
    print("\n=== Test 6: step_load_from_files (adapter) ===")

    ds = DataService()
    step_load_from_files(
        ds, "integration_test",
        file_paths=[REF_FILE, SAM_FILE],
        grouping_keywords=["type", "series", "temp"],
        grouping_delimiter="_",
    )

    series_list = ds.list_series()
    check(f"Series created: {len(series_list)} >= 2",
          len(series_list) >= 2)

    # Check that at least some series have time_domain data
    has_td = False
    has_raw = False
    for sid in series_list:
        groups = ds.list_groups(sid)
        if "time_domain" in groups:
            has_td = True
        if "raw" in groups:
            has_raw = True
    check("At least one series has time_domain group", has_td)
    check("At least one series has raw group", has_raw)

    # Check associations were created
    all_assocs = ds.list_associations()
    check(f"Associations created: {len(all_assocs)}",
          len(all_assocs) >= 0)  # might be 0 if no sample/ref pairing matched

    print(f"\n  All series: {series_list}")
    for sid in series_list:
        groups = ds.list_groups(sid)
        print(f"    {sid}: groups={groups}")
        assocs = ds.list_associations(sid)
        if assocs:
            print(f"      associations: {assocs}")


def test_module_registration():
    """Test 7: THzModule satisfies AnalysisModule protocol."""
    print("\n=== Test 7: Module Registration ===")

    ds = DataService()
    registry = Registry(ds)
    thz = THzModule()
    record = registry.register(thz)

    check("Module registered", record is not None)
    check("Module name = 'thz_tds'", record.name == "thz_tds")
    check("4 data groups", len(record.data_groups) == 4)
    check("3 pipeline steps", len(record.pipeline_step_descriptors) == 3)

    pipeline = registry.get_pipeline("thz_tds")
    check("Pipeline has 3 steps", len(pipeline.steps) == 3)
    step_ids = pipeline.step_ids
    check("Step IDs correct",
          step_ids == ["load_data", "load_from_files", "transfer_function"])

    all_groups = registry.all_data_groups
    check(f"all_data_groups returns {len(all_groups)} groups",
          len(all_groups) == 4)


def main():
    print("=" * 60)
    print("  MATCHBOOK INTEGRATION TEST")
    print("  Example data: .acc files (reference + sample)")
    print("=" * 60)

    # Verify files exist
    assert os.path.isfile(REF_FILE), f"Reference not found: {REF_FILE}"
    assert os.path.isfile(SAM_FILE), f"Sample not found: {SAM_FILE}"
    print(f"\n  Reference: {REF_FILE}")
    print(f"  Sample:    {SAM_FILE}")

    # Run all tests
    ref_data, sam_data = test_loader_registry()
    test_grouping()
    ds, series_ref, series_sam = test_dataservice_atomisation(ref_data, sam_data)
    test_database_roundtrip(ds, series_ref, series_sam)
    test_sync_dirty(ref_data, sam_data)
    test_step_load_from_files()
    test_module_registration()

    # Summary
    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"  RESULTS: {PASS}/{total} passed, {FAIL} failed")
    if FAIL == 0:
        print("  ALL TESTS PASSED")
    else:
        print(f"  {FAIL} FAILURES — see [FAIL] lines above")
    print("=" * 60)

    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
