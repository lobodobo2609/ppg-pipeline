"""Per-window PPG features."""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

SPECTRAL_BANDS = {"power_0_5_1_5hz": (0.5, 1.5), "power_1_5_3hz": (1.5, 3.0), "power_3_8hz": (3.0, 8.0)}


def _feet(x: np.ndarray, peaks: np.ndarray, fs: float, lookback_s: float = 0.4) -> np.ndarray:
    """Pulse onset ('foot') = minimum in the lookback window before each peak."""
    lb = int(lookback_s * fs)
    feet = []
    for p in peaks:
        lo = max(0, p - lb)
        feet.append(lo + int(np.argmin(x[lo:p + 1])))
    return np.array(feet, dtype=int)


def window_features(x: np.ndarray, peaks: np.ndarray, fs: float) -> dict[str, float]:
    """Beat-to-beat, morphology, and spectral features for one filtered PPG window."""
    nan = float("nan")
    out: dict[str, float] = {"n_beats": float(len(peaks))}

    # Interbeat intervals -> heart rate and HRV
    if len(peaks) >= 2:
        ibi_ms = np.diff(peaks) / fs * 1000.0
        out["hr_bpm"] = 60000.0 / float(np.median(ibi_ms))
        out["ibi_mean_ms"] = float(ibi_ms.mean())
        out["sdnn_ms"] = float(ibi_ms.std(ddof=1)) if len(ibi_ms) > 1 else nan
        out["rmssd_ms"] = float(np.sqrt(np.mean(np.diff(ibi_ms) ** 2))) if len(ibi_ms) > 1 else nan
    else:
        out.update(hr_bpm=nan, ibi_mean_ms=nan, sdnn_ms=nan, rmssd_ms=nan)

    # Morphology: amplitude (foot to peak) and rise time
    if len(peaks) >= 1:
        feet = _feet(x, peaks, fs)
        out["amplitude"] = float(np.median(x[peaks] - x[feet]))
        out["rise_time_ms"] = float(np.median((peaks - feet) / fs * 1000.0))
        # Max slope of the upstroke, from the first derivative (velocity PPG)
        d1 = np.gradient(x) * fs
        slopes = [d1[f:p + 1].max() for f, p in zip(feet, peaks) if p > f]
        out["max_upslope"] = float(np.median(slopes)) if slopes else nan
    else:
        out.update(amplitude=nan, rise_time_ms=nan, max_upslope=nan)

    # Spectral power in a few bands
    freqs, psd = welch(x, fs=fs, nperseg=min(len(x), int(4 * fs)))
    total = np.trapezoid(psd, freqs) or nan
    for name, (lo, hi) in SPECTRAL_BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        out[name] = float(np.trapezoid(psd[m], freqs[m]) / total) if m.any() else nan
    return out
