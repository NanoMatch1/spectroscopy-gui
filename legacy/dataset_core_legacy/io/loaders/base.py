from __future__ import annotations

from typing import Optional

import numpy as np

from dataset_core.data_structures.common import FileObject, DataObject as BaseData


class BaseStructureParser:
    """Protocol-like base for structure parsers.

    Subclasses should provide:
      - name: str (human-friendly)
      - priority: int (default 0)
      - sniff(file_obj) -> float  # confidence [0..1]
      - parse(file_obj) -> BaseData
    """

    name: str = "base"
    priority: int = 0

    @classmethod
    def sniff(cls, file_obj: FileObject) -> float:  # pragma: no cover - abstract
        return 0.0

    @classmethod
    def parse(cls, file_obj: FileObject) -> BaseData:  # pragma: no cover - abstract
        raise NotImplementedError


def monotonic(v: np.ndarray) -> bool:
    if v.ndim != 1 or v.size < 2:
        return False
    d = np.diff(v)
    return bool(np.all(d > 0) or np.all(d < 0))
