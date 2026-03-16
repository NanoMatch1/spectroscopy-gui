"""Central data store with observer notifications.

The DataService is the single source of truth for all analysis data.
Modules write to it via ``put()``; the GUI and scripts read from it via
``get()`` / ``query()``.  Listeners are notified on every mutation so
the GUI can redraw without polling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Data addressing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataKey:
    """Unique address for a piece of data.

    Parameters
    ----------
    series_id:
        Identifies one complete experiment run, e.g. ``"sam_Si_2025-03"``.
    group:
        Logical grouping, e.g. ``"time_domain"``, ``"optical_constants"``.
    name:
        Specific quantity, e.g. ``"win_ref_amp"``, ``"n"``.
    """
    series_id: str
    group: str
    name: str


@dataclass
class DataEntry:
    """One plottable dataset stored in the DataService.

    Parameters
    ----------
    key:
        Unique address (series, group, name).
    x, y:
        Coordinate arrays of equal length.
    metadata:
        Free-form dict — conventionally holds ``x_label``, ``y_label``,
        ``display_label``, ``unit``, etc.
    """
    key: DataKey
    x: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Observer event type
# ---------------------------------------------------------------------------

# Listener signature:  (event: str, key: DataKey) -> None
#   event is one of  "put", "remove", "clear"
DataListener = Callable[[str, DataKey], None]


# ---------------------------------------------------------------------------
# DataService
# ---------------------------------------------------------------------------

class DataService:
    """Central data store with observer notifications.

    Tracks dirty state for database sync, and holds in-memory associations
    (e.g. sample → reference pairing) that mirror the persistent DB layer.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str, str], DataEntry] = {}
        self._listeners: list[DataListener] = []
        self._series_tags: dict[str, set[str]] = {}
        self._dirty: set[tuple[str, str, str]] = set()
        # Associations: (series_id, relationship) → target series_id
        # e.g. ("sample_Ge_300K", "reference") → "reference_Ge_300K"
        self._associations: dict[tuple[str, str], str] = {}
        self._dirty_associations: set[tuple[str, str]] = set()

    # -- write -----------------------------------------------------------------

    def put(self, entry: DataEntry) -> None:
        """Insert or update a data entry and notify listeners."""
        key_tuple = (entry.key.series_id, entry.key.group, entry.key.name)
        self._store[key_tuple] = entry
        self._dirty.add(key_tuple)
        self._notify("put", entry.key)

    def remove(self, key: DataKey) -> None:
        """Remove a data entry if it exists and notify listeners."""
        key_tuple = (key.series_id, key.group, key.name)
        if key_tuple in self._store:
            del self._store[key_tuple]
            self._dirty.discard(key_tuple)
            self._notify("remove", key)

    def clear_series(self, series_id: str) -> None:
        """Remove every entry belonging to *series_id*."""
        to_remove = [k for k in self._store if k[0] == series_id]
        for k in to_remove:
            del self._store[k]
            self._dirty.discard(k)
            self._notify("clear", DataKey(*k))
        # Clear associations where this series is the source
        assoc_keys = [ak for ak in self._associations if ak[0] == series_id]
        for ak in assoc_keys:
            del self._associations[ak]
            self._dirty_associations.discard(ak)

    # -- read ------------------------------------------------------------------

    def get(self, key: DataKey) -> DataEntry | None:
        """Return a single entry or ``None``."""
        return self._store.get((key.series_id, key.group, key.name))

    def list_series(self) -> list[str]:
        """Return sorted unique series identifiers."""
        return sorted({k[0] for k in self._store})

    def list_groups(self, series_id: str | None = None) -> list[str]:
        """Return sorted unique group names, optionally filtered by series."""
        if series_id is not None:
            return sorted({k[1] for k in self._store if k[0] == series_id})
        return sorted({k[1] for k in self._store})

    def list_names(self, series_id: str, group: str) -> list[str]:
        """Return sorted names within a (series, group) pair."""
        return sorted(
            k[2] for k in self._store
            if k[0] == series_id and k[1] == group
        )

    def query(
        self,
        series_id: str | None = None,
        group: str | None = None,
        tags: set[str] | None = None,
    ) -> list[DataEntry]:
        """Flexible query with optional filters."""
        results: list[DataEntry] = []
        for (sid, grp, _name), entry in self._store.items():
            if series_id is not None and sid != series_id:
                continue
            if group is not None and grp != group:
                continue
            if tags is not None and not tags.issubset(self._series_tags.get(sid, set())):
                continue
            results.append(entry)
        return results

    # -- tags ------------------------------------------------------------------

    def tag_series(self, series_id: str, *tags: str) -> None:
        """Attach one or more tags to a series."""
        self._series_tags.setdefault(series_id, set()).update(tags)

    def untag_series(self, series_id: str, *tags: str) -> None:
        """Remove tags from a series."""
        if series_id in self._series_tags:
            self._series_tags[series_id] -= set(tags)

    def get_tags(self, series_id: str) -> set[str]:
        """Return the tag set for a series (empty set if none)."""
        return set(self._series_tags.get(series_id, set()))

    def all_tags(self) -> set[str]:
        """Return the union of all tags across every series."""
        result: set[str] = set()
        for tag_set in self._series_tags.values():
            result |= tag_set
        return result

    def series_with_tag(self, tag: str) -> list[str]:
        """Return sorted series ids that carry *tag*."""
        return sorted(
            sid for sid, tags in self._series_tags.items() if tag in tags
        )

    # -- observer --------------------------------------------------------------

    def subscribe(self, listener: DataListener) -> None:
        """Register a listener that is called on every mutation."""
        self._listeners.append(listener)

    def unsubscribe(self, listener: DataListener) -> None:
        """Remove a previously registered listener."""
        self._listeners = [fn for fn in self._listeners if fn is not listener]

    def _notify(self, event: str, key: DataKey) -> None:
        for fn in self._listeners:
            fn(event, key)

    # -- associations ----------------------------------------------------------

    def associate(
        self, series_id: str, relationship: str, target_series_id: str,
    ) -> None:
        """Create a named link between two series.

        Example::

            ds.associate("sample_Ge_300K", "reference", "ref_Ge_300K")
            ds.get_association("sample_Ge_300K", "reference")
            # → "ref_Ge_300K"
        """
        key = (series_id, relationship)
        self._associations[key] = target_series_id
        self._dirty_associations.add(key)

    def remove_association(self, series_id: str, relationship: str) -> None:
        """Remove an association if it exists."""
        key = (series_id, relationship)
        self._associations.pop(key, None)
        self._dirty_associations.discard(key)

    def get_association(self, series_id: str, relationship: str) -> str | None:
        """Return the target series for the given relationship, or None."""
        return self._associations.get((series_id, relationship))

    def list_associations(
        self, series_id: str | None = None,
    ) -> list[tuple[str, str, str]]:
        """Return associations as (series_id, relationship, target) tuples."""
        results = []
        for (sid, rel), target in self._associations.items():
            if series_id is not None and sid != series_id:
                continue
            results.append((sid, rel, target))
        return results

    # -- dirty tracking --------------------------------------------------------

    def dirty_keys(self) -> set[tuple[str, str, str]]:
        """Return data entry keys that were modified since last mark_clean."""
        return set(self._dirty)

    def dirty_associations(self) -> set[tuple[str, str]]:
        """Return association keys modified since last mark_clean."""
        return set(self._dirty_associations)

    def is_dirty(self) -> bool:
        """True if any entries or associations have been modified."""
        return bool(self._dirty or self._dirty_associations)

    def mark_clean(self) -> None:
        """Clear all dirty flags (called after database sync)."""
        self._dirty.clear()
        self._dirty_associations.clear()
