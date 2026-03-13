"""Standalone THz TDS analysis functions.

Every function in this module is a **pure** analysis operation — numpy
arrays in, numpy arrays out.  No framework, no GUI, no DataService.
These can be called directly from scripts, notebooks, or tests.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Transfer function
# ---------------------------------------------------------------------------

def compute_transfer_function_amplitude(
    ref_frequency: np.ndarray,
    ref_amplitude: np.ndarray,
    sample_frequency: np.ndarray,
    sample_amplitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute |H(f)| = sample_amplitude / reference_amplitude.

    If the frequency grids differ, the sample is interpolated onto the
    reference grid.

    Returns
    -------
    frequency, amplitude_ratio
    """
    if np.array_equal(ref_frequency, sample_frequency):
        ratio = np.where(ref_amplitude != 0,
                         sample_amplitude / ref_amplitude, np.nan)
        return ref_frequency.copy(), ratio

    sam_interp = np.interp(ref_frequency, sample_frequency, sample_amplitude)
    ratio = np.where(ref_amplitude != 0,
                     sam_interp / ref_amplitude, np.nan)
    return ref_frequency.copy(), ratio


def compute_transfer_function_phase(
    ref_frequency: np.ndarray,
    ref_phase: np.ndarray,
    sample_frequency: np.ndarray,
    sample_phase: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute phase difference = sample_phase - reference_phase.

    If the frequency grids differ, the sample is interpolated onto the
    reference grid.

    Returns
    -------
    frequency, phase_difference
    """
    if np.array_equal(ref_frequency, sample_frequency):
        return ref_frequency.copy(), sample_phase - ref_phase

    sam_interp = np.interp(ref_frequency, sample_frequency, sample_phase)
    return ref_frequency.copy(), sam_interp - ref_phase
