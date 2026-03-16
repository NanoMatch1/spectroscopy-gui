"""Extensible search-filter registry for database queries.

Register new SQL-level filters with ``@register_filter``.  Each filter
is a callable that receives a user-supplied value and returns a
``(sql_fragment, params_list)`` tuple that will be composed into the
WHERE clause of the series search query.

For richer post-retrieval filtering (e.g. numpy checks on loaded blobs),
use ``@register_post_filter`` — these run in Python on the result list
after the SQL query has executed.

Example — adding a new SQL filter::

    @register_filter("min_rows")
    def _min_rows(value: int) -> tuple[str, list]:
        # Only keep series that have at least *value* data rows somewhere.
        return (
            "s.id IN ("
            "  SELECT series_id FROM data_blobs "
            "  GROUP BY series_id "
            "  HAVING SUM(length(y_blob) / 8) >= ?"
            ")",
            [value],
        )

Example — adding a post-filter::

    @register_post_filter("has_positive_n")
    def _positive_n(series_list, value, db):
        # Keep only series whose 'n' trace has no negatives.
        ...
        return filtered_list
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# SQL filter:  value -> (WHERE clause fragment referencing alias 's', param list)
SQLFilterFn = Callable[[Any], tuple[str, list[Any]]]

# Post filter: (result_list, value, db_conn) -> filtered result_list
PostFilterFn = Callable[[list[dict], Any, Any], list[dict]]


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

_SQL_FILTERS: dict[str, SQLFilterFn] = {}
_POST_FILTERS: dict[str, PostFilterFn] = {}


def register_filter(name: str) -> Callable[[SQLFilterFn], SQLFilterFn]:
    """Decorator that registers a SQL-level search filter."""
    def _decorator(fn: SQLFilterFn) -> SQLFilterFn:
        if name in _SQL_FILTERS:
            raise ValueError(f"SQL filter '{name}' already registered")
        _SQL_FILTERS[name] = fn
        return fn
    return _decorator


def register_post_filter(name: str) -> Callable[[PostFilterFn], PostFilterFn]:
    """Decorator that registers a Python post-filter."""
    def _decorator(fn: PostFilterFn) -> PostFilterFn:
        if name in _POST_FILTERS:
            raise ValueError(f"Post-filter '{name}' already registered")
        _POST_FILTERS[name] = fn
        return fn
    return _decorator


def registered_filters() -> dict[str, SQLFilterFn]:
    """Return a copy of the SQL filter registry."""
    return dict(_SQL_FILTERS)


def registered_post_filters() -> dict[str, PostFilterFn]:
    """Return a copy of the post-filter registry."""
    return dict(_POST_FILTERS)


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def build_search_query(
    criteria: dict[str, Any],
) -> tuple[str, list[Any]]:
    """Compose a SELECT query from registered SQL filters.

    Parameters
    ----------
    criteria:
        ``{filter_name: value}`` pairs.  Only keys matching registered
        filter names are applied; unknown keys raise ``KeyError``.

    Returns
    -------
    (sql, params):
        Ready to pass to ``cursor.execute(sql, params)``.
    """
    base = (
        "SELECT s.id, s.name, s.module, s.created_at, s.metadata "
        "FROM series s"
    )
    conditions: list[str] = []
    params: list[Any] = []

    for key, value in criteria.items():
        fn = _SQL_FILTERS.get(key)
        if fn is None:
            raise KeyError(
                f"Unknown search filter '{key}'.  "
                f"Registered: {sorted(_SQL_FILTERS)}"
            )
        clause, clause_params = fn(value)
        conditions.append(clause)
        params.extend(clause_params)

    if conditions:
        base += " WHERE " + " AND ".join(conditions)
    base += " ORDER BY s.created_at DESC"

    return base, params


# ---------------------------------------------------------------------------
# Built-in SQL filters
# ---------------------------------------------------------------------------

@register_filter("name_contains")
def _name_contains(value: str) -> tuple[str, list]:
    """Series name contains *value* (case-insensitive substring)."""
    return "s.name LIKE ? COLLATE NOCASE", [f"%{value}%"]


@register_filter("name_exact")
def _name_exact(value: str) -> tuple[str, list]:
    """Series name exactly equals *value*."""
    return "s.name = ?", [value]


@register_filter("id_contains")
def _id_contains(value: str) -> tuple[str, list]:
    """Series id contains *value* (substring)."""
    return "s.id LIKE ?", [f"%{value}%"]


@register_filter("module")
def _module(value: str) -> tuple[str, list]:
    """Exact module name match."""
    return "s.module = ?", [value]


@register_filter("date_after")
def _date_after(value: str) -> tuple[str, list]:
    """Series created at or after ISO date string."""
    return "s.created_at >= ?", [value]


@register_filter("date_before")
def _date_before(value: str) -> tuple[str, list]:
    """Series created before ISO date string."""
    return "s.created_at < ?", [value]


@register_filter("has_tag")
def _has_tag(value: str) -> tuple[str, list]:
    """Series must have this tag."""
    return (
        "s.id IN (SELECT series_id FROM series_tags WHERE tag = ?)",
        [value],
    )


@register_filter("has_group")
def _has_group(value: str) -> tuple[str, list]:
    """Series must contain a data blob with this group name."""
    return (
        "s.id IN (SELECT series_id FROM data_blobs WHERE grp = ?)",
        [value],
    )


@register_filter("has_name")
def _has_name(value: str) -> tuple[str, list]:
    """Series must contain a data blob with this trace name."""
    return (
        "s.id IN (SELECT series_id FROM data_blobs WHERE name = ?)",
        [value],
    )


@register_filter("metadata_field")
def _metadata_field(value: tuple[str, str, Any]) -> tuple[str, list]:
    """Filter on a JSON metadata field.

    *value* is a 3-tuple ``(field_name, operator, compare_value)``
    where operator is one of ``=``, ``!=``, ``>``, ``>=``, ``<``, ``<=``.

    Example: ``metadata_field=("scan_count", ">=", 5)``
    """
    field, op, cmp_val = value
    allowed_ops = {"=", "!=", ">", ">=", "<", "<="}
    if op not in allowed_ops:
        raise ValueError(f"Operator '{op}' not in {allowed_ops}")
    return (
        f"json_extract(s.metadata, '$.{field}') {op} ?",
        [cmp_val],
    )


@register_filter("text_search")
def _text_search(value: str) -> tuple[str, list]:
    """Broad text search across id, name, and metadata JSON."""
    pattern = f"%{value}%"
    return (
        "(s.id LIKE ? OR s.name LIKE ? COLLATE NOCASE OR s.metadata LIKE ?)",
        [pattern, pattern, pattern],
    )
