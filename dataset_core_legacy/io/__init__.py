# dataset_core/io/__init__.py
from __future__ import annotations

import pkgutil
import importlib

# Import everything in this package so decorators run and registry is populated.
for m in pkgutil.iter_modules(__path__):
    # Optionally restrict to loader modules
    if m.name.endswith("_loader"):
        importlib.import_module(f"{__name__}.{m.name}")

# Export the public API
from .loaders.registry import BaseLoader, get_loader_for_extension, registered_extensions

__all__ = ["BaseLoader", "get_loader_for_extension", "registered_extensions"]
