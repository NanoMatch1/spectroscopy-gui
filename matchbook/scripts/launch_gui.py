"""Launch the Matchbook GUI with the THz TDS module.

This script:
  1. Creates a DataService, Database, and Registry.
  2. Registers the THz module.
  3. Opens the GUI with database access.
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
from matchbook.io.database import Database
from matchbook.modules.thz.adapter import THzModule


_DEFAULT_DB = os.path.join(_WORKSPACE, "matchbook_data.db")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # --- Core setup ---
    data_service = DataService()
    registry = Registry(data_service)

    thz_module = THzModule()
    registry.register(thz_module)

    # --- Database ---
    db = Database(_DEFAULT_DB)

    # --- Launch GUI ---
    app = MatchbookApp(
        data_service, registry,
        title="Matchbook — THz TDS",
        database=db,
    )
    app.run()


if __name__ == "__main__":
    main()
