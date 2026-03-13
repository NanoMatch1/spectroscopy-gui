"""Headless THz analysis script — no GUI, no tkinter.

Demonstrates running the full THz pipeline from command line or notebook.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np

# Ensure the workspace root (parent of matchbook/) is on the path
_WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

from matchbook.core.data_service import DataKey, DataService
from matchbook.core.registry import Registry
from matchbook.modules.thz.adapter import THzModule


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

    base_dir = os.path.dirname(_WORKSPACE)  # analysis-comp/
    sample_json = os.path.join(base_dir, "sample_details.json")

    if not os.path.isfile(sample_json):
        # Try the workspace itself
        base_dir = _WORKSPACE
        sample_json = os.path.join(base_dir, "sample_details.json")

    # --- Setup ---
    data_service = DataService()
    registry = Registry(data_service)

    thz_module = THzModule()
    registry.register(thz_module)

    pipeline = registry.get_pipeline("thz_tds")

    # --- Load each person's data ---
    from matchbook.modules.thz.loaders import discover_people

    data_dir = os.path.join(base_dir, "data")
    people = discover_people(data_dir)

    for person in people:
        series_id = person
        pipeline.set_param("load_data", "base_dir", base_dir)
        pipeline.set_param("load_data", "person_name", person)
        pipeline.set_param("load_data", "sample_json", sample_json)
        pipeline.run_all(data_service, series_id)

        # Tag with the person's name
        data_service.tag_series(series_id, person)

        logger.info(f"Series '{series_id}': "
                     f"{len(data_service.list_groups(series_id))} groups loaded")

        # Example: read a specific value
        n_entry = data_service.get(DataKey(series_id, "optical_constants", "n"))
        if n_entry is not None:
            valid = np.isfinite(n_entry.y)
            if np.any(valid):
                idx_1thz = np.argmin(np.abs(n_entry.x[valid] - 1.0))
                logger.info(f"  n at ~1 THz: {n_entry.y[valid][idx_1thz]:.4f}")

    logger.info(f"All series: {data_service.list_series()}")


if __name__ == "__main__":
    main()
