from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_dumps(obj: Any) -> str:
    # Safe JSON dump for metadata (handles numpy scalars via default=str)
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def _json_loads(s: str) -> Any:
    return json.loads(s) if s else {}


def _safe_hdf5_key(name: str) -> str:
    """
    Make a safe HDF5 group key from filenames/labels.
    HDF5 allows many characters, but keeping it conservative avoids surprises.
    """
    name = name.strip()
    # Replace path separators and weird whitespace
    name = name.replace(os.sep, "_").replace("/", "_")
    name = re.sub(r"\s+", " ", name)
    # Remove characters that can be awkward in paths
    name = re.sub(r"[^a-zA-Z0-9._\- ()\[\]]+", "_", name)
    return name[:200] if len(name) > 200 else name
