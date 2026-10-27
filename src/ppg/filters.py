"""Signal conditioning: filtering, normalization, windowing."""
from __future__ import annotations

from typing import Iterator

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass(x: np.ndarray, fs: float, low: float = 0.5, high: float = 8.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass.

    Second-order sections (sos) are numerically safer than (b, a) coefficients,
    and sosfiltfilt runs the filter forward and backward so peaks don't shift in time.
    """
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=float))


def normalize(x: np.ndarray) -> np.ndarray:
    """Z-score normalize. Returns zeros for a flat signal instead of dividing by zero."""
    x = np.asarray(x, dtype=float)
    sd = x.std()
    if sd == 0 or not np.isfinite(sd):
        return np.zeros_like(x)
    return (x - x.mean()) / sd


def windows(n_samples: int, fs: float, win_s: float, step_s: float) -> Iterator[tuple[int, int]]:
    """Yield (start, end) sample indices for sliding windows."""
    win, step = int(round(win_s * fs)), int(round(step_s * fs))
    for start in range(0, n_samples - win + 1, step):
        yield start, start + win
