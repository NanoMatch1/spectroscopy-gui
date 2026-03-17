"""Recogniser and Atomiser registries — module-agnostic data classification.

Modules register two kinds of callables:

* **Recognisers** score how confidently a module can handle an
  :class:`~matchbook.io.file_ingestor.IngestedFile`.
* **Atomisers** write the ingested data into a
  :class:`~matchbook.core.data_service.DataService`, following the
  module's own data model.

Both registries use a ``module_name`` key (e.g. ``"thz_tds"``) to
namespace entries.  Registration happens at import time via decorators.

Example (inside a module's adapter.py)::

    from matchbook.io.recognisers import register_recogniser, register_atomiser

    @register_recogniser("thz_tds")
    def recognise_thz(ingested: IngestedFile) -> float:
        from matchbook.modules.thz.containers import THzData
        return 1.0 if isinstance(ingested.data, THzData) else 0.0

    @register_atomiser("thz_tds")
    def atomise_thz(
        ingested: IngestedFile,
        data_service: DataService,
        series_id: str,
    ) -> None:
        ...  # write entries into data_service
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from matchbook.core.data_service import DataService
    from matchbook.io.file_ingestor import IngestedFile


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

class RecogniserFn(Protocol):
    """Callable that scores how well a module can handle an IngestedFile."""

    def __call__(self, ingested: IngestedFile) -> float: ...


class AtomiserFn(Protocol):
    """Callable that writes an IngestedFile into a DataService."""

    def __call__(
        self,
        ingested: IngestedFile,
        data_service: DataService,
        series_id: str,
    ) -> None: ...


# ---------------------------------------------------------------------------
# Internal storage
# ---------------------------------------------------------------------------

_RECOGNISERS: dict[str, RecogniserFn] = {}
_ATOMISERS: dict[str, AtomiserFn] = {}


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def register_recogniser(
    module_name: str,
) -> Callable[[RecogniserFn], RecogniserFn]:
    """Decorator that registers a recogniser function for *module_name*."""

    def _decorator(fn: RecogniserFn) -> RecogniserFn:
        if module_name in _RECOGNISERS and _RECOGNISERS[module_name] is not fn:
            raise ValueError(
                f"Duplicate recogniser for module '{module_name}': "
                f"{_RECOGNISERS[module_name]} vs {fn}"
            )
        _RECOGNISERS[module_name] = fn
        return fn

    return _decorator


def register_atomiser(
    module_name: str,
) -> Callable[[AtomiserFn], AtomiserFn]:
    """Decorator that registers an atomiser function for *module_name*."""

    def _decorator(fn: AtomiserFn) -> AtomiserFn:
        if module_name in _ATOMISERS and _ATOMISERS[module_name] is not fn:
            raise ValueError(
                f"Duplicate atomiser for module '{module_name}': "
                f"{_ATOMISERS[module_name]} vs {fn}"
            )
        _ATOMISERS[module_name] = fn
        return fn

    return _decorator


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def recognise(ingested: IngestedFile) -> list[tuple[str, float]]:
    """Run all recognisers against *ingested* and return scored matches.

    Returns a list of ``(module_name, confidence)`` pairs, sorted
    highest-confidence first.  Only modules with confidence > 0 are
    included.
    """
    results: list[tuple[str, float]] = []
    for module_name, fn in _RECOGNISERS.items():
        score = fn(ingested)
        if score > 0:
            results.append((module_name, score))
    results.sort(key=lambda pair: pair[1], reverse=True)
    return results


def get_atomiser(module_name: str) -> AtomiserFn:
    """Return the registered atomiser for *module_name*, or raise."""
    try:
        return _ATOMISERS[module_name]
    except KeyError:
        available = list(_ATOMISERS.keys())
        raise KeyError(
            f"No atomiser registered for module '{module_name}'. "
            f"Available: {available}"
        ) from None


def registered_recognisers() -> dict[str, RecogniserFn]:
    """Return a snapshot of the recogniser registry."""
    return dict(_RECOGNISERS)


def registered_atomisers() -> dict[str, AtomiserFn]:
    """Return a snapshot of the atomiser registry."""
    return dict(_ATOMISERS)
