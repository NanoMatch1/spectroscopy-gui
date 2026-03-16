"""SQLite-backed database for persistent series storage.

Stores series metadata, tags, data blobs (as numpy binary), and pipeline
parameter snapshots — enabling save / load / filter / restore / **search**
workflows.

Searches use the extensible filter registry in ``matchbook.io.search``.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

import numpy as np

from matchbook.core.data_service import DataEntry, DataKey, DataService
from matchbook.io.search import (
    build_search_query,
    registered_post_filters,
)

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS series (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    module      TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS series_tags (
    series_id   TEXT NOT NULL,
    tag         TEXT NOT NULL,
    PRIMARY KEY (series_id, tag),
    FOREIGN KEY (series_id) REFERENCES series(id)
);

CREATE TABLE IF NOT EXISTS data_blobs (
    series_id   TEXT NOT NULL,
    grp         TEXT NOT NULL,
    name        TEXT NOT NULL,
    x_blob      BLOB NOT NULL,
    y_blob      BLOB NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (series_id, grp, name),
    FOREIGN KEY (series_id) REFERENCES series(id)
);

CREATE TABLE IF NOT EXISTS associations (
    source_id       TEXT NOT NULL,
    relationship    TEXT NOT NULL,
    target_id       TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (source_id, relationship),
    FOREIGN KEY (source_id) REFERENCES series(id),
    FOREIGN KEY (target_id) REFERENCES series(id)
);

CREATE TABLE IF NOT EXISTS provenance (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id       TEXT NOT NULL,
    step_id         TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    inputs_json     TEXT NOT NULL DEFAULT '[]',
    params_json     TEXT NOT NULL DEFAULT '{}',
    outputs_json    TEXT NOT NULL DEFAULT '[]',
    note            TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (series_id) REFERENCES series(id)
);

CREATE TABLE IF NOT EXISTS pipeline_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id   TEXT NOT NULL,
    params_json TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    label       TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (series_id) REFERENCES series(id)
);

CREATE INDEX IF NOT EXISTS idx_series_name       ON series(name);
CREATE INDEX IF NOT EXISTS idx_series_created_at ON series(created_at);
CREATE INDEX IF NOT EXISTS idx_data_blobs_grp    ON data_blobs(grp);
CREATE INDEX IF NOT EXISTS idx_data_blobs_name   ON data_blobs(name);
CREATE INDEX IF NOT EXISTS idx_series_tags_tag   ON series_tags(tag);
"""


class Database:
    """Lightweight SQLite interface for persistent data management."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info(f"Database opened: {db_path}")

    def close(self) -> None:
        self._conn.close()

    # -- Series CRUD -------------------------------------------------------

    def save_series(
        self,
        data_service: DataService,
        series_id: str,
        module: str = "",
        display_name: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist all DataEntries for *series_id* from the DataService."""
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata or {})
        display_name = display_name or series_id

        self._conn.execute(
            "INSERT OR REPLACE INTO series (id, name, module, created_at, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (series_id, display_name, module, now, meta_json),
        )

        # Tags
        tags = data_service.get_tags(series_id)
        self._conn.execute(
            "DELETE FROM series_tags WHERE series_id = ?", (series_id,))
        for tag in tags:
            self._conn.execute(
                "INSERT INTO series_tags (series_id, tag) VALUES (?, ?)",
                (series_id, tag),
            )

        # Associations (source = this series)
        self._conn.execute(
            "DELETE FROM associations WHERE source_id = ?", (series_id,))
        for sid, rel, target in data_service.list_associations(series_id):
            self._conn.execute(
                "INSERT INTO associations (source_id, relationship, target_id, created_at) "
                "VALUES (?, ?, ?, ?)",
                (sid, rel, target, now),
            )

        # Data blobs
        self._conn.execute(
            "DELETE FROM data_blobs WHERE series_id = ?", (series_id,))
        entries = data_service.query(series_id=series_id)
        for entry in entries:
            # Only persist JSON-serializable metadata; skip runtime objects
            try:
                meta_str = json.dumps(entry.metadata)
            except (TypeError, ValueError):
                serializable = {
                    k: v for k, v in entry.metadata.items()
                    if isinstance(v, (str, int, float, bool, list, dict, type(None)))
                }
                meta_str = json.dumps(serializable)
            self._conn.execute(
                "INSERT INTO data_blobs (series_id, grp, name, x_blob, y_blob, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    series_id,
                    entry.key.group,
                    entry.key.name,
                    entry.x.tobytes(),
                    entry.y.tobytes(),
                    meta_str,
                ),
            )

        self._conn.commit()
        logger.info(f"Saved series '{series_id}' ({len(entries)} entries)")

    def load_series(self, series_id: str, data_service: DataService) -> bool:
        """Load a series from the database into the DataService.

        Returns True if the series was found, False otherwise.
        """
        row = self._conn.execute(
            "SELECT id, name, module, metadata FROM series WHERE id = ?",
            (series_id,),
        ).fetchone()
        if row is None:
            return False

        # Tags
        tag_rows = self._conn.execute(
            "SELECT tag FROM series_tags WHERE series_id = ?",
            (series_id,),
        ).fetchall()
        for (tag,) in tag_rows:
            data_service.tag_series(series_id, tag)

        # Associations
        assoc_rows = self._conn.execute(
            "SELECT relationship, target_id FROM associations WHERE source_id = ?",
            (series_id,),
        ).fetchall()
        for rel, target in assoc_rows:
            data_service.associate(series_id, rel, target)

        # Data blobs
        blob_rows = self._conn.execute(
            "SELECT grp, name, x_blob, y_blob, metadata FROM data_blobs "
            "WHERE series_id = ?",
            (series_id,),
        ).fetchall()
        for grp, name, x_blob, y_blob, meta_json in blob_rows:
            x = np.frombuffer(x_blob, dtype=np.float64)
            y = np.frombuffer(y_blob, dtype=np.float64)
            meta = json.loads(meta_json) if meta_json else {}
            data_service.put(DataEntry(
                key=DataKey(series_id, grp, name),
                x=x.copy(), y=y.copy(),
                metadata=meta,
            ))

        logger.info(f"Loaded series '{series_id}' ({len(blob_rows)} entries)")
        return True

    def list_series(
        self,
        module: str | None = None,
        tags: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """List series with optional filters.  Returns dicts with id, name, module, tags."""
        query = "SELECT id, name, module, created_at, metadata FROM series"
        params: list[Any] = []
        conditions: list[str] = []

        if module:
            conditions.append("module = ?")
            params.append(module)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC"

        rows = self._conn.execute(query, params).fetchall()
        results = []
        for sid, name, mod, created, meta_json in rows:
            tag_rows = self._conn.execute(
                "SELECT tag FROM series_tags WHERE series_id = ?", (sid,)
            ).fetchall()
            series_tags = {t for (t,) in tag_rows}

            if tags and not tags.issubset(series_tags):
                continue

            results.append({
                "id": sid,
                "name": name,
                "module": mod,
                "created_at": created,
                "tags": series_tags,
                "metadata": json.loads(meta_json) if meta_json else {},
            })

        return results

    # -- Extensible search -------------------------------------------------

    def search(self, **criteria: Any) -> list[dict[str, Any]]:
        """Search for series using the registered filter system.

        Keyword arguments are ``{filter_name: value}`` pairs.  Only filter
        names from ``matchbook.io.search`` are accepted.

        Examples::

            db.search(name_contains="germanium")
            db.search(has_group="fft", date_after="2026-01-01")
            db.search(text_search="ref", has_tag="thz_tds")
            db.search(metadata_field=("scan_count", ">=", 5))
        """
        sql, params = build_search_query(criteria)
        rows = self._conn.execute(sql, params).fetchall()

        results = []
        for sid, name, mod, created, meta_json in rows:
            tag_rows = self._conn.execute(
                "SELECT tag FROM series_tags WHERE series_id = ?", (sid,)
            ).fetchall()
            results.append({
                "id": sid,
                "name": name,
                "module": mod,
                "created_at": created,
                "tags": {t for (t,) in tag_rows},
                "metadata": json.loads(meta_json) if meta_json else {},
            })

        # Apply post-filters
        post_filters = registered_post_filters()
        for key in list(criteria):
            if key in post_filters:
                results = post_filters[key](results, criteria[key], self._conn)

        return results

    def delete_series(self, series_id: str) -> None:
        """Remove a series and all its data from the database."""
        self._conn.execute("DELETE FROM data_blobs WHERE series_id = ?", (series_id,))
        self._conn.execute("DELETE FROM series_tags WHERE series_id = ?", (series_id,))
        self._conn.execute("DELETE FROM associations WHERE source_id = ?", (series_id,))
        self._conn.execute("DELETE FROM provenance WHERE series_id = ?", (series_id,))
        self._conn.execute("DELETE FROM pipeline_snapshots WHERE series_id = ?", (series_id,))
        self._conn.execute("DELETE FROM series WHERE id = ?", (series_id,))
        self._conn.commit()
        logger.info(f"Deleted series '{series_id}'")

    # -- Pipeline snapshots ------------------------------------------------

    def save_pipeline_snapshot(
        self,
        series_id: str,
        params: dict[str, dict[str, Any]],
        label: str = "",
    ) -> int:
        """Save a pipeline parameter snapshot.  Returns the snapshot id."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            "INSERT INTO pipeline_snapshots (series_id, params_json, created_at, label) "
            "VALUES (?, ?, ?, ?)",
            (series_id, json.dumps(params), now, label),
        )
        self._conn.commit()
        return cursor.lastrowid

    def load_pipeline_snapshot(self, snapshot_id: int) -> dict[str, dict[str, Any]]:
        """Load a pipeline parameter snapshot by id."""
        row = self._conn.execute(
            "SELECT params_json FROM pipeline_snapshots WHERE id = ?",
            (snapshot_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Snapshot {snapshot_id} not found")
        return json.loads(row[0])

    def list_pipeline_snapshots(
        self, series_id: str
    ) -> list[dict[str, Any]]:
        """List pipeline snapshots for a series."""
        rows = self._conn.execute(
            "SELECT id, created_at, label FROM pipeline_snapshots "
            "WHERE series_id = ? ORDER BY created_at DESC",
            (series_id,),
        ).fetchall()
        return [
            {"id": r[0], "created_at": r[1], "label": r[2]}
            for r in rows
        ]

    # -- Provenance --------------------------------------------------------

    def record_provenance(
        self,
        series_id: str,
        step_id: str,
        inputs: list[str] | None = None,
        params: dict[str, Any] | None = None,
        outputs: list[str] | None = None,
        note: str = "",
    ) -> int:
        """Record a provenance entry for a processing step.

        Parameters
        ----------
        series_id:
            The series being processed.
        step_id:
            Pipeline step identifier.
        inputs:
            List of DataKey descriptions or series IDs consumed.
        params:
            Parameter snapshot for this step execution.
        outputs:
            List of DataKey descriptions or series IDs produced.
        note:
            Optional human-readable annotation.

        Returns the row id.
        """
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            "INSERT INTO provenance "
            "(series_id, step_id, timestamp, inputs_json, params_json, outputs_json, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                series_id,
                step_id,
                now,
                json.dumps(inputs or []),
                json.dumps(params or {}),
                json.dumps(outputs or []),
                note,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def list_provenance(
        self, series_id: str,
    ) -> list[dict[str, Any]]:
        """Return provenance records for a series, newest first."""
        rows = self._conn.execute(
            "SELECT id, step_id, timestamp, inputs_json, params_json, "
            "outputs_json, note FROM provenance "
            "WHERE series_id = ? ORDER BY timestamp DESC",
            (series_id,),
        ).fetchall()
        return [
            {
                "id": r[0],
                "step_id": r[1],
                "timestamp": r[2],
                "inputs": json.loads(r[3]),
                "params": json.loads(r[4]),
                "outputs": json.loads(r[5]),
                "note": r[6],
            }
            for r in rows
        ]

    # -- Associations (direct DB access) -----------------------------------

    def list_associations(
        self, series_id: str | None = None,
    ) -> list[dict[str, str]]:
        """Return associations as dicts with source, relationship, target."""
        if series_id:
            rows = self._conn.execute(
                "SELECT source_id, relationship, target_id FROM associations "
                "WHERE source_id = ?", (series_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT source_id, relationship, target_id FROM associations"
            ).fetchall()
        return [
            {"source": r[0], "relationship": r[1], "target": r[2]}
            for r in rows
        ]

    # -- Dirty sync --------------------------------------------------------

    def sync_dirty(self, data_service: DataService) -> int:
        """Write only dirty entries and associations to the database.

        Returns the number of entries written.
        """
        now = datetime.now(timezone.utc).isoformat()
        count = 0

        # Sync dirty data entries
        for key_tuple in data_service.dirty_keys():
            entry = data_service.get(DataKey(*key_tuple))
            if entry is None:
                continue

            series_id = key_tuple[0]
            # Ensure series row exists
            self._conn.execute(
                "INSERT OR IGNORE INTO series (id, name, module, created_at) "
                "VALUES (?, ?, '', ?)",
                (series_id, series_id, now),
            )

            try:
                meta_str = json.dumps(entry.metadata)
            except (TypeError, ValueError):
                serializable = {
                    k: v for k, v in entry.metadata.items()
                    if isinstance(v, (str, int, float, bool, list, dict, type(None)))
                }
                meta_str = json.dumps(serializable)

            self._conn.execute(
                "INSERT OR REPLACE INTO data_blobs "
                "(series_id, grp, name, x_blob, y_blob, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    entry.key.series_id,
                    entry.key.group,
                    entry.key.name,
                    entry.x.tobytes(),
                    entry.y.tobytes(),
                    meta_str,
                ),
            )
            count += 1

        # Sync dirty associations
        for assoc_key in data_service.dirty_associations():
            sid, rel = assoc_key
            target = data_service.get_association(sid, rel)
            if target is not None:
                self._conn.execute(
                    "INSERT OR REPLACE INTO associations "
                    "(source_id, relationship, target_id, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (sid, rel, target, now),
                )

        # Sync tags for affected series
        affected_series = {k[0] for k in data_service.dirty_keys()}
        for series_id in affected_series:
            tags = data_service.get_tags(series_id)
            self._conn.execute(
                "DELETE FROM series_tags WHERE series_id = ?", (series_id,))
            for tag in tags:
                self._conn.execute(
                    "INSERT INTO series_tags (series_id, tag) VALUES (?, ?)",
                    (series_id, tag),
                )

        self._conn.commit()
        data_service.mark_clean()

        if count:
            logger.info(f"Synced {count} dirty entries to database")
        return count
