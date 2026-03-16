"""Parsed metadata derived from a filename.

Lightweight and decoupled from service logic.  Matching/pairing
behaviour remains in :class:`~matchbook.services.grouping.GroupingService`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FilenameInfo:
    """Structured metadata parsed from a filename by delimiter + keyword order."""

    filename: str
    keywords: list[str] = field(
        default_factory=lambda: ["type", "series", "temp"]
    )
    report_list: list[str] = field(
        default_factory=lambda: ["data_type", "series"]
    )

    series: str | None = None
    data_type: str | None = None
    temperature: str | None = None

    extra_details: list[str] | None = None
    air_reference: str | None = None
    substrate_reference: str | None = None

    @classmethod
    def from_filename(
        cls,
        filename: str,
        *,
        delimiter: str = "_",
        keywords: list[str] | None = None,
        merge_extra: bool = False,
        **kwargs: Any,
    ) -> FilenameInfo:
        selected = keywords if keywords is not None else ["type", "series", "temp"]
        item = cls(filename=filename, keywords=list(selected))
        item.parse(
            delimiter=delimiter,
            keywords=item.keywords,
            merge_extra=merge_extra,
            **kwargs,
        )
        return item

    def parse(
        self,
        *,
        delimiter: str = "_",
        keywords: list[str] | None = None,
        merge_extra: bool = False,
        **kwargs: Any,
    ) -> None:
        """Parse this filename into fields based on delimiter + keyword ordering."""
        active_keywords = keywords if keywords is not None else self.keywords
        self.keywords = list(active_keywords)

        self.report_list = ["data_type", "series"]
        self.series = None
        self.data_type = None
        self.temperature = None
        self.extra_details = None

        # Strip file extension
        details = ".".join(self.filename.split(".")[:-1])
        components = [
            comp.strip()
            for comp in details.split(delimiter)
            if comp.strip()
        ]

        for index, key in enumerate(self.keywords):
            if index >= len(components):
                break

            value = components[index]
            if key == "type":
                self.data_type = value
            elif key == "series":
                if "." in value:
                    indices = [i for i, ch in enumerate(value) if ch == "."]
                    value = value[: indices[-1]]
                self.series = value
            elif key == "temp":
                self.temperature = value
                if "temperature" not in self.report_list:
                    self.report_list.append("temperature")
            else:
                self.__dict__[key] = value
                if key not in self.report_list:
                    self.report_list.append(key)

        if len(components) > len(self.keywords):
            extras = components[len(self.keywords):]
            if merge_extra:
                self.series = (
                    delimiter.join([self.series, *extras])
                    if self.series
                    else delimiter.join(extras)
                )
            self.extra_details = extras

    def __repr__(self) -> str:
        lines = [
            f" -> {key}: {self.__dict__.get(key, None)}"
            for key in self.report_list
        ]
        for key in ("extra_details", "air_reference", "substrate_reference"):
            lines.append(f" -> {key}: {self.__dict__.get(key, None)}")
        return f"Filename: {self.filename}\n" + "\n".join(lines)

    def __str__(self) -> str:
        return self.filename
