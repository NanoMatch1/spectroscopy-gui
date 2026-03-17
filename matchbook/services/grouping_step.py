"""Generic grouping pipeline step — pairs references with samples.

Any module can import ``create_grouping_step`` and
``grouping_step_descriptor`` to add file-grouping to its pipeline.
The step operates on whatever sub-series already exist under a given
*series_id* in the DataService.

This means it works identically whether data was loaded from disk via
the FileIngestor or loaded from a saved database.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from matchbook.core.data_service import DataEntry, DataKey, DataService
from matchbook.core.module_base import (
    ParameterDescriptor,
    ParamType,
    PipelineStepDescriptor,
)
from matchbook.core.pipeline import PipelineStep
from matchbook.services.grouping import GroupingService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline step function
# ---------------------------------------------------------------------------

def step_group_files(
    data_service: DataService,
    series_id: str,
    *,
    grouping_keywords: list[str] | str | None = None,
    grouping_delimiter: str = "_",
) -> None:
    """Group sub-series by filename conventions and create associations.

    Discovers all sub-series of *series_id* (i.e. series whose id starts
    with ``series_id/``), extracts the filename part, runs
    ``GroupingService.simple_grouping``, and stores ``_meta/grouping``
    entries and reference associations back into the DataService.

    Parameters
    ----------
    grouping_keywords:
        Keywords to parse from filenames (e.g. ``["type", "series", "temp"]``).
        Accepts a comma-separated string from the GUI.
    grouping_delimiter:
        Character separating tokens in the filename (default ``"_"``).
    """
    # Normalise keywords from string or list
    if isinstance(grouping_keywords, str):
        keywords = [
            k.strip()
            for k in grouping_keywords.replace("\n", ",").split(",")
            if k.strip()
        ]
    elif grouping_keywords is not None:
        keywords = list(grouping_keywords)
    else:
        keywords = ["type", "series", "temp"]

    # Discover sub-series under this series_id
    prefix = series_id + "/"
    all_series = data_service.list_series()
    sub_series = [s for s in all_series if s.startswith(prefix)]

    if not sub_series:
        return

    # Extract filename from each sub-series id
    filenames = [s[len(prefix):] for s in sub_series]

    # Run grouping
    gs = GroupingService(
        keywords=keywords,
        delimiter=grouping_delimiter,
        filelist=filenames,
    )
    gs.simple_grouping(delimiter=grouping_delimiter, keywords=keywords)

    # Write _meta/grouping entries for each file
    for filename in filenames:
        file_series = f"{series_id}/{filename}"
        info = gs(filename)
        if info is not None:
            data_service.put(DataEntry(
                key=DataKey(file_series, "_meta", "grouping"),
                x=np.array([0.0]),
                y=np.array([0.0]),
                metadata={
                    "data_type": info.data_type,
                    "series": info.series,
                    "temperature": info.temperature,
                },
            ))

    # Create associations from grouping results
    for filename, info in gs.file_items.items():
        file_series = f"{series_id}/{filename}"
        if info.substrate_reference:
            ref_series = f"{series_id}/{info.substrate_reference}"
            data_service.associate(
                file_series, "substrate_reference", ref_series)
        if info.air_reference:
            air_series = f"{series_id}/{info.air_reference}"
            data_service.associate(
                file_series, "air_reference", air_series)


# ---------------------------------------------------------------------------
# Descriptor and factory for module pipelines
# ---------------------------------------------------------------------------

def grouping_step_descriptor(
    depends_on: list[str] | None = None,
) -> PipelineStepDescriptor:
    """Return a ``PipelineStepDescriptor`` for the grouping step.

    Modules include this in their ``pipeline_step_descriptors()`` list.
    """
    return PipelineStepDescriptor(
        id="group_files",
        name="Group Files",
        params=[
            ParameterDescriptor(
                name="grouping_keywords",
                label="Grouping keywords",
                type=ParamType.STRING,
                default="type, series, temp",
                tooltip="Comma-separated keywords parsed from filenames",
            ),
            ParameterDescriptor(
                name="grouping_delimiter",
                label="Filename delimiter",
                type=ParamType.STRING,
                default="_",
                tooltip="Character separating tokens in filenames",
            ),
        ],
        depends_on=depends_on or [],
    )


def create_grouping_step(
    depends_on: list[str] | None = None,
) -> PipelineStep:
    """Return a ``PipelineStep`` for grouping.

    Modules include this in their ``create_pipeline_steps()`` list.
    """
    desc = grouping_step_descriptor(depends_on)
    return PipelineStep(
        id=desc.id,
        name=desc.name,
        fn=step_group_files,
        params={
            "grouping_keywords": "type, series, temp",
            "grouping_delimiter": "_",
        },
        param_descriptors=desc.params,
        depends_on=desc.depends_on,
    )
