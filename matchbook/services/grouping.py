"""Service for grouping of filenames based on strings, keywords, and delimiters.

Used to correlate reference and sample across variable datasets such as
temperature series.  Independent of any specific analysis domain — works
for any data where pairing of data types is necessary (THz, UV-VIS,
calibrations, etc.).
"""

from __future__ import annotations

from typing import Any

from matchbook.services.filename_info import FilenameInfo


# Backwards-compatible alias
FilenameItem = FilenameInfo


class GroupingService:
    """Groups filenames by delimiter-separated keywords and pairs references to samples.

    Tracks the current working file list via ``_current_data_list``.
    """

    def __init__(
        self,
        keywords: list[str] | None = None,
        delimiter: str = "_",
        filelist: list[str] | None = None,
    ) -> None:
        self.filelist: list[str] = filelist if filelist is not None else []
        self.file_items: dict[str, FilenameInfo] = {}
        self.keywords = keywords if keywords is not None else ["type", "series", "temp"]
        self.filename_groups: list = []
        self.global_reference: dict[str, str] = {}
        self.delimiter = delimiter

        self._current_data_list: list[str] = []

    # -- Queries --

    def is_reference(self, filename: str) -> bool:
        item = self.file_items.get(filename)
        return item is not None and item.data_type == "reference"

    def is_sample(self, filename: str) -> bool:
        item = self.file_items.get(filename)
        return item is not None and item.data_type == "sample"

    @property
    def info(self) -> str:
        return "\n".join([
            f"GroupingService with {len(self.filelist)} files.",
            f"Current grouping keywords: {self.keywords}",
            f"Current delimiter: '{self.delimiter}'",
            f"Number of filename groups: {len(self.filename_groups)}",
        ])

    @property
    def elaborate(self) -> str:
        return "\n".join(
            repr(item) for item in self.file_items.values()
        )

    @property
    def references(self) -> dict[str, FilenameInfo]:
        return {
            fn: item
            for fn, item in self.file_items.items()
            if item.data_type == "reference"
        }

    @property
    def samples(self) -> dict[str, FilenameInfo]:
        return {
            fn: item
            for fn, item in self.file_items.items()
            if item.data_type == "sample"
        }

    # -- Mutation --

    def set_grouping_keywords(self, new_keywords: list[str]) -> None:
        self.keywords = new_keywords

    def get_current_data_list(self) -> list[str]:
        return self._current_data_list

    def set_current_data_list(self, new_list: list[str]) -> None:
        self._current_data_list = new_list

    def update(self, filelist: list[str] | None = None) -> None:
        """Append new filenames and rebuild the current data list."""
        for filename in (filelist or []):
            self.filelist.append(filename)
        self._current_data_list = self.filelist

    # -- Callable lookup --

    def __call__(self, filename: str, object_type: str | None = None) -> Any:
        file_obj = self.file_items.get(filename)
        if file_obj is None:
            return None
        if object_type is None:
            return file_obj
        return getattr(file_obj, object_type, None)

    def get_reference_filename(
        self, filename: str, ref_type: str = "substrate",
    ) -> str | None:
        """Find the corresponding reference filename for a given sample."""
        if ref_type not in ("substrate", "air"):
            raise ValueError("ref_type must be either 'substrate' or 'air'.")

        group_info = self(filename)
        if group_info is None:
            return None

        if group_info.data_type != "sample":
            return None

        if ref_type == "air":
            return group_info.air_reference
        return group_info.substrate_reference

    # -- Grouping pipeline --

    def simple_grouping(
        self,
        delimiter: str = "_",
        keywords: list[str] | None = None,
    ) -> dict[str, FilenameInfo]:
        """Group data by slicing the filename on delimiter + keyword order.

        Builds file items, parses filenames, identifies global references,
        runs integrity check, and pairs references to samples.
        """
        selected = keywords if keywords is not None else self.keywords

        self._build_fileitems(delimiter=delimiter, keywords=selected, merge_extra=True)
        self.keywords = selected
        self.parse_filenames()
        self._identify_global_references()
        self.integrity_check()
        self._pair_references()

        return self.file_items

    def _build_fileitems(self, **kwargs: Any) -> None:
        for filename in self.filelist:
            self.file_items[filename] = FilenameInfo.from_filename(
                filename, **kwargs,
            )

    def parse_filenames(self, **kwargs: Any) -> None:
        if kwargs.get("keywords") is None:
            kwargs["keywords"] = self.keywords
        for item in self.file_items.values():
            item.parse(**kwargs)

    def _group_by_keyword(self, keyword: str) -> dict[str, dict[str, FilenameInfo]]:
        grouping: dict[str, dict[str, FilenameInfo]] = {}
        for filename, fileitem in self.file_items.items():
            key_value = getattr(fileitem, keyword, None)
            if key_value is None:
                continue
            grouping.setdefault(key_value, {})[filename] = fileitem
        return grouping

    def _identify_global_references(self) -> None:
        for filename, item in self.file_items.items():
            if item.data_type != "reference":
                continue
            series_lower = (item.series or "").lower()
            if "air" in series_lower:
                self.global_reference["air"] = item.filename
            elif "substrate" in series_lower:
                self.global_reference["substrate"] = item.filename

    def integrity_check(self) -> list[str]:
        """Return a list of warning strings (empty if all OK)."""
        warnings: list[str] = []
        if "air" not in self.global_reference:
            warnings.append("Warning: No air reference found in global references.")
        if "substrate" not in self.global_reference:
            warnings.append("Warning: No substrate reference found in global references.")
        return warnings

    def _pair_references(self) -> None:
        references = {
            fn: item
            for fn, item in self.file_items.items()
            if item.data_type == "reference"
        }
        samples = {
            fn: item
            for fn, item in self.file_items.items()
            if item.data_type == "sample"
        }

        for filename, fileitem in samples.items():
            keywords = fileitem.report_list.copy()
            if "data_type" in keywords:
                keywords.remove("data_type")
            match_criteria = {
                key: getattr(fileitem, key, None) for key in keywords
            }

            if not match_criteria:
                fileitem.substrate_reference = self.global_reference.get("substrate")
                continue

            for ref_filename, ref_item in references.items():
                match = all(
                    getattr(ref_item, key, None) == value
                    for key, value in match_criteria.items()
                )
                if match and ref_item.data_type == "reference":
                    fileitem.substrate_reference = ref_filename
                    fileitem.air_reference = self.global_reference.get("air")

    def _separate_by_delimiters(
        self, delimiter: str | None = None,
    ) -> list[str] | None:
        delimiter = delimiter or self.delimiter
        for filename in self.filelist:
            components = filename.split(delimiter)
            return [c.strip() for c in components if c.strip()]
        return None
