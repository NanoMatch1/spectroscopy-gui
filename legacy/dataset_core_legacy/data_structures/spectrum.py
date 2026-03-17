from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Spectrum:
    """Simple container for 1D spectral / XY datasets.

    - `data` is a numpy array with shape (N, M). If a single column is
      provided it will be interpreted as the Y axis and a simple integer
      X axis will be synthesized.
    - `headers` stores lines from the file that couldn't be parsed as
      numeric rows (comments, metadata, etc.).
    """

    data: np.ndarray
    headers: Optional[List[str]] = None
    filename: Optional[str] = None
    data_type: Optional[str] = None

    def __post_init__(self):
        if self.data is None:
            self.data = np.empty((0, 2), dtype=float)

        self.data = np.asarray(self.data)

        # Normalize one-dimensional input to column format
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)

        # If only Y-values given (Nx1), synthesize X = arange(N)
        if self.data.ndim == 2 and self.data.shape[1] == 1:
            x = np.arange(self.data.shape[0], dtype=float)
            self.data = np.column_stack((x, self.data[:, 0].astype(float)))

    @property
    def x(self) -> np.ndarray:
        return self.data[:, 0]

    @property
    def y(self) -> np.ndarray:
        # Return second column if present, otherwise empty
        if self.data.shape[1] >= 2:
            return self.data[:, 1]
        return np.array([])

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def copy(self) -> "Spectrum":
        return Spectrum(data=self.data.copy(), headers=None if self.headers is None else list(self.headers), filename=self.filename, data_type=self.data_type)

    def __repr__(self) -> str:
        return f"<Spectrum:{self.filename or 'unnamed'}, shape={self.data.shape}, type={self.data_type}>"
    
    def plot_current(self, figure_obj=None, **kwargs):
        import matplotlib.pyplot as plt
        # Acquire axis: prefer provided FigureObject, otherwise create a temporary
        show_plot = False
        if figure_obj is None:
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
            show_plot = True
        else:
            ax = getattr(figure_obj, 'ax', None)
            if ax is None:
                # fallback to creating a new figure if the object is malformed
                fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
                show_plot = True

        ax.plot(self.x, self.y, label=self.filename)

        if show_plot:
            plt.show()
