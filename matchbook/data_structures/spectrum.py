"""Simple container for 1D spectral / XY datasets.

Domain-agnostic: works for any columnar numeric data with optional headers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Spectrum:
    """Simple container for 1D spectral / XY datasets.

    - ``data`` is a numpy array with shape (N, M).  If a single column is
      provided it will be interpreted as the Y axis and a simple integer
      X axis will be synthesised.
    - ``headers`` stores lines from the file that could not be parsed as
      numeric rows (comments, metadata, etc.).
    """

    data: np.ndarray
    headers: Optional[List[str]] = None
    filename: Optional[str] = None
    data_type: Optional[str] = None

    def __post_init__(self) -> None:
        if self.data is None:
            self.data = np.empty((0, 2), dtype=float)

        self.data = np.asarray(self.data)

        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)

        # If only Y-values given (Nx1), synthesise X = arange(N)
        if self.data.ndim == 2 and self.data.shape[1] == 1:
            x = np.arange(self.data.shape[0], dtype=float)
            self.data = np.column_stack((x, self.data[:, 0].astype(float)))

    @property
    def x(self) -> np.ndarray:
        return self.data[:, 0]

    @property
    def y(self) -> np.ndarray:
        if self.data.shape[1] >= 2:
            return self.data[:, 1]
        return np.array([])

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def copy(self) -> Spectrum:
        return Spectrum(
            data=self.data.copy(),
            headers=None if self.headers is None else list(self.headers),
            filename=self.filename,
            data_type=self.data_type,
        )

    def __repr__(self) -> str:
        return (
            f"<Spectrum:{self.filename or 'unnamed'}, "
            f"shape={self.data.shape}, type={self.data_type}>"
        )
