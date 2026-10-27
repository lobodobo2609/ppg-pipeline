"""Beat detection for PPG and reference R-peak detection for ECG."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .filters import bandpass, normalize


def detect_ppg_peaks(ppg_filtered: np.ndarray, fs: float, min_distance_s: float = 0.33,
                     prominence: float = 0.3) -> np.ndarray:
    """Systolic peak indices in a band-passed, z-scored PPG.

    min_distance_s=0.33 caps detection at ~180 bpm, which stops the diastolic
    bump from being counted as a second beat.
    """
    x = normalize(ppg_filtered)
    peaks, _ = find_peaks(x, distance=max(1, int(min_distance_s * fs)), prominence=prominence)
    return peaks


def detect_ecg_rpeaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Simplified Pan-Tompkins: band-pass, differentiate, square, integrate, threshold.

    Good enough as a reference on clean ICU ECG. Cross-check against NeuroKit2
    if you need something more robust.
    """
    filt = bandpass(ecg, fs, low=5.0, high=min(15.0, 0.45 * fs), order=2)
    energy = np.gradient(filt) ** 2
    win = max(1, int(0.15 * fs))
    integrated = np.convolve(energy, np.ones(win) / win, mode="same")
    candidates, _ = find_peaks(integrated, distance=int(0.3 * fs),
                               height=0.3 * np.percentile(integrated, 99))
    # Refine each candidate to the largest |filtered ECG| within +/-75 ms.
    half = int(0.075 * fs)
    refined = []
    for c in candidates:
        lo, hi = max(0, c - half), min(len(filt), c + half + 1)
        refined.append(lo + int(np.argmax(np.abs(filt[lo:hi]))))
    return np.unique(np.array(refined, dtype=int))


def heart_rate(peaks: np.ndarray, fs: float) -> float:
    """Heart rate in bpm from the median interbeat interval (robust to one missed beat)."""
    if len(peaks) < 2:
        return float("nan")
    ibi = np.diff(peaks) / fs
    return float(60.0 / np.median(ibi))
