"""Launch the Matchbook GUI with the THz TDS module.

This script:
  1. Creates a DataService and Registry.
  2. Registers the THz module.
  3. Loads data for all discovered series.
  4. Opens the GUI.
"""

from __future__ import annotations

import logging
import os
import sys

# Ensure the workspace root (parent of matchbook/) is on the path
_WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

from matchbook.core.data_service import DataService
from matchbook.core.registry import Registry
from matchbook.gui.app import MatchbookApp
from matchbook.modules.thz.adapter import THzModule
from matchbook.modules.thz.loaders import discover_people


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    base_dir = os.path.dirname(_WORKSPACE)
    sample_json = os.path.join(base_dir, "sample_details.json")

    if not os.path.isfile(sample_json):
        base_dir = _WORKSPACE
        sample_json = os.path.join(base_dir, "sample_details.json")

    # --- Core setup ---
    data_service = DataService()
    registry = Registry(data_service)

    thz_module = THzModule()
    registry.register(thz_module)

    pipeline = registry.get_pipeline("thz_tds")

    # --- Load data ---
    data_dir = os.path.join(base_dir, "data")
    if os.path.isdir(data_dir):
        people = discover_people(data_dir)
        for person in people:
            pipeline.set_param("load_data", "base_dir", base_dir)
            pipeline.set_param("load_data", "person_name", person)
            pipeline.set_param("load_data", "sample_json", sample_json)
            pipeline.run_all(data_service, person)
            data_service.tag_series(person, person)

    # --- Launch GUI ---
    app = MatchbookApp(data_service, registry, title="Matchbook — THz TDS")
    app.run()


if __name__ == "__main__":
    main()
