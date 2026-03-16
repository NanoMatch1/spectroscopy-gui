"""Headless THz workflow script — no GUI, no tkinter.

Demonstrates the generic Matchbook workflow: load files via the registry,
auto-group, push to DataService, persist to database, and search/query.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile

import numpy as np

# Ensure the workspace root (parent of matchbook/) is on the path
_WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

from matchbook.core.data_service import DataKey, DataService
from matchbook.core.registry import Registry
from matchbook.io.database import Database
from matchbook.modules.thz.adapter import THzModule


def run_interactively() -> None:
    """Full interactive example — drop into pdb at interesting points.

    Workflow
    -------
    1. Set up DataService + register THz module
    2. Load .acc example files via the loader-registry pipeline step
    3. Inspect what was created in the DataService
    4. Persist to a temporary SQLite database
    5. Clear memory and reload from DB
    6. Search the database with various filter criteria
    7. Drop into pdb so you can poke around
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

    # -- locate example data --
    example_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "modules", "thz", "example_data",
    )
    ref_file = os.path.join(example_dir, "reference_air_nitrogen.acc")
    sam_file = os.path.join(example_dir, "sample_germanium.acc")

    if not os.path.isfile(ref_file):
        raise FileNotFoundError(f"Example reference file not found: {ref_file}")
    if not os.path.isfile(sam_file):
        raise FileNotFoundError(f"Example sample file not found: {sam_file}")

    # ── 1. Framework setup ────────────────────────────────────────────────
    ds = DataService()
    registry = Registry(ds)
    registry.register(THzModule())
    pipeline = registry.get_pipeline("thz_tds")

    logger.info("Pipeline steps: %s", pipeline.step_ids)

    # ── 2. Load example .acc files via pipeline ───────────────────────────
    series_id = "germanium_test"

    pipeline.set_param("load_from_files", "file_paths", f"{ref_file}\n{sam_file}")
    pipeline.set_param("load_from_files", "grouping_keywords", ["type", "series", "temp"])
    pipeline.set_param("load_from_files", "grouping_delimiter", "_")
    pipeline.run_all(ds, series_id)

    # Tag all sub-series for later searching
    for sid in ds.list_series():
        ds.tag_series(sid, "thz_tds")
        ds.tag_series(sid, "germanium")

    # ── 3. Inspect in-memory state ────────────────────────────────────────
    all_series = ds.list_series()
    logger.info(f"Series in DataService: {all_series}")

    for sid in all_series:
        groups = ds.list_groups(sid)
        logger.info(f"  {sid}: groups = {groups}")
        for grp in groups:
            names = ds.list_names(sid, grp)
            logger.info(f"    {grp}: {names}")

    # Look at a specific entry
    for sid in all_series:
        entries = ds.query(series_id=sid)
        for entry in entries:
            if entry.key.group != "_meta":
                logger.info(
                    f"  [{entry.key.series_id}] {entry.key.group}/{entry.key.name}: "
                    f"x.shape={entry.x.shape}, y.shape={entry.y.shape}, "
                    f"meta keys={list(entry.metadata.keys())}"
                )

    # Check associations
    for sid in all_series:
        assocs = ds.list_associations(sid)
        if assocs:
            logger.info(f"  Associations for {sid}: {assocs}")

    # ── 4. Persist to database ────────────────────────────────────────────
    db_path = os.path.join(tempfile.gettempdir(), "matchbook_example.db")
    db = Database(db_path)
    logger.info(f"Database path: {db_path}")

    # Save all series
    for sid in all_series:
        db.save_series(ds, sid, module="thz_tds", display_name=sid)
    logger.info("All series saved to database")

    # ── 5. Clear memory and reload ────────────────────────────────────────
    ds_fresh = DataService()
    for sid in all_series:
        found = db.load_series(sid, ds_fresh)
        logger.info(f"  Loaded '{sid}' from DB: {found}")

    # Verify round-trip integrity
    for sid in all_series:
        original_entries = ds.query(series_id=sid)
        reloaded_entries = ds_fresh.query(series_id=sid)
        logger.info(
            f"  {sid}: original={len(original_entries)} entries, "
            f"reloaded={len(reloaded_entries)} entries"
        )
        for orig in original_entries:
            reloaded = ds_fresh.get(orig.key)
            if reloaded is None:
                logger.warning(f"    MISSING after reload: {orig.key}")
            elif not (np.array_equal(orig.x, reloaded.x)
                      and np.array_equal(orig.y, reloaded.y)):
                logger.warning(f"    DATA MISMATCH: {orig.key}")

    # ── 6. Search the database ────────────────────────────────────────────
    logger.info("--- Database search examples ---")

    results = db.search(text_search="germanium")
    logger.info(f"text_search='germanium': {len(results)} results")
    for r in results:
        logger.info(f"  {r['id']} — {r['name']} (tags={r['tags']})")

    results = db.search(has_tag="thz_tds")
    logger.info(f"has_tag='thz_tds': {len(results)} results")

    results = db.search(has_group="time_domain")
    logger.info(f"has_group='time_domain': {len(results)} results")

    results = db.search(has_group="raw")
    logger.info(f"has_group='raw': {len(results)} results")

    all_db = db.search()
    logger.info(f"All series in DB: {len(all_db)}")
    for r in all_db:
        logger.info(f"  {r['id']}: module={r['module']}, tags={r['tags']}")

    logger.info("--- Done. Dropping into pdb ---")
    logger.info("Try: ds.list_series(), ds.query(series_id=...), db.search(...)")

    # Drop into interactive debugger so you can explore
    breakpoint()

    db.close()


if __name__ == "__main__":
    run_interactively()
