"""Dataclasses for THz time-domain spectroscopy data.

This module has **no framework dependencies** — it uses only stdlib and numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RawMeasurement:
    """Original raw time-domain measurement (e.g. from original/ folder)."""
    time_ps: np.ndarray
    amplitude: np.ndarray


@dataclass
class TimeDomainData:
    """Processed time-domain data.

    Primary fields hold the windowed / final data (always present).
    Optional ``raw_*`` fields hold pre-windowed data when available.
    """
    ref_time_ps: np.ndarray
    ref_amplitude: np.ndarray
    sample_time_ps: np.ndarray
    sample_amplitude: np.ndarray
    ref_std_error: Optional[np.ndarray] = None
    sample_std_error: Optional[np.ndarray] = None
    raw_ref_time_ps: Optional[np.ndarray] = None
    raw_ref_amplitude: Optional[np.ndarray] = None
    raw_ref_std_error: Optional[np.ndarray] = None
    raw_sample_time_ps: Optional[np.ndarray] = None
    raw_sample_amplitude: Optional[np.ndarray] = None
    raw_sample_std_error: Optional[np.ndarray] = None


@dataclass
class FFTData:
    """Frequency-domain FFT output.  All frequencies normalised to THz."""
    ref_frequency_thz: np.ndarray
    ref_amplitude: np.ndarray
    ref_phase: np.ndarray
    sample_frequency_thz: np.ndarray
    sample_amplitude: np.ndarray
    sample_phase: np.ndarray
    ref_delta_amplitude: Optional[np.ndarray] = None
    ref_delta_phase: Optional[np.ndarray] = None
    sample_delta_amplitude: Optional[np.ndarray] = None
    sample_delta_phase: Optional[np.ndarray] = None
    phase_offset: Optional[np.ndarray] = None
    phase_diff: Optional[np.ndarray] = None


@dataclass
class OpticalConstants:
    """Extracted optical constants.  All frequencies normalised to THz."""
    frequency_thz: np.ndarray
    n: np.ndarray
    k: np.ndarray
    eps1: np.ndarray
    eps2: np.ndarray
    sigma_re: np.ndarray
    sigma_im: np.ndarray
    eps_infty: Optional[float] = None


@dataclass
class SampleInfo:
    """Sample physical properties."""
    thickness_m: float
    resistivity_ohm_m: float


@dataclass
class AnalysisDataset:
    """Complete dataset for one analysis pipeline run."""
    name: str
    sample_info: SampleInfo
    raw_reference: RawMeasurement
    raw_sample: RawMeasurement
    time_domain: Optional[TimeDomainData] = None
    fft: Optional[FFTData] = None
    optical_constants: Optional[OpticalConstants] = None
