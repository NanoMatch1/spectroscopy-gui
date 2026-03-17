"""FileIngestor — generic file loading independent of any analysis module.

Wraps the LoaderRegistry to load files by extension, producing
:class:`IngestedFile` objects that carry the raw parsed data and metadata.
Modules consume IngestedFiles via registered atomisers (see recognisers.py).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from matchbook.io.loaders.registry import (
    LoaderError,
    get_loader_for_extension,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestedFile:
    """Result of loading a single file through the loader registry.

    Attributes:
        path:        Absolute path to the source file.
        filename:    Basename of the file (e.g. ``"sam_Si_300K.txt"``).
        extension:   Lowercase extension including the dot (e.g. ``".txt"``).
        data:        The parsed object returned by the loader, or ``None``
                     on failure.
        loader_name: Name of the loader class that handled the file.
        load_error:  Error message if loading failed, else empty string.
    """

    path: str
    filename: str
    extension: str
    data: Any = None
    loader_name: str = ""
    load_error: str = ""

    @property
    def ok(self) -> bool:
        """True when the file was loaded without error."""
        return self.data is not None and not self.load_error


@dataclass
class IngestReport:
    """Summary returned by :meth:`FileIngestor.load_files`."""

    files: list[IngestedFile] = field(default_factory=list)

    @property
    def succeeded(self) -> list[IngestedFile]:
        return [f for f in self.files if f.ok]

    @property
    def failed(self) -> list[IngestedFile]:
        return [f for f in self.files if not f.ok]

    def __len__(self) -> int:
        return len(self.files)


class FileIngestor:
    """Load files via the extension-based loader registry.

    Usage::

        import matchbook.io.loaders          # trigger @register_loader
        ingestor = FileIngestor()
        report = ingestor.load_files(file_paths)

        for ingested in report.succeeded:
            print(ingested.filename, type(ingested.data))
    """

    def load_files(
        self,
        file_paths: list[str] | str,
    ) -> IngestReport:
        """Load each path through the appropriate loader.

        Parameters:
            file_paths: A list of absolute paths, or a single
                        newline-separated string of paths.

        Returns:
            An :class:`IngestReport` with one :class:`IngestedFile` per path.
        """
        if isinstance(file_paths, str):
            file_paths = [p.strip() for p in file_paths.splitlines() if p.strip()]

        report = IngestReport()
        for fpath in file_paths:
            report.files.append(self._load_one(fpath))
        return report

    # ------------------------------------------------------------------

    @staticmethod
    def _load_one(fpath: str) -> IngestedFile:
        filename = os.path.basename(fpath)
        ext = os.path.splitext(fpath)[1].lower()

        try:
            loader_cls = get_loader_for_extension(ext)
        except LoaderError as exc:
            logger.warning("No loader for %s, skipping: %s", ext, fpath)
            return IngestedFile(
                path=fpath,
                filename=filename,
                extension=ext,
                load_error=str(exc),
            )

        try:
            loader = loader_cls(fpath)
            data = loader.load()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Loader %s failed on %s: %s", loader_cls.__name__, fpath, exc)
            return IngestedFile(
                path=fpath,
                filename=filename,
                extension=ext,
                loader_name=loader_cls.__name__,
                load_error=str(exc),
            )

        return IngestedFile(
            path=fpath,
            filename=filename,
            extension=ext,
            data=data,
            loader_name=loader_cls.__name__,
        )
