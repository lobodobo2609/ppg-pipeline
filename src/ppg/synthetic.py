"""Synthetic PPG/ECG with known ground truth, used by the tests.

If the pipeline can't recover 72 bpm from a fake 72 bpm signal, it won't work on real data.
"""
from __future__ import annotations

import numpy as np


def beat_times(hr_bpm: float, duration_s: float, hrv_std_s: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    period, t, times = 60.0 / hr_bpm, 0.5, []
    while t < duration_s - 0.5:
        times.append(t)
        t += period + rng.normal(0, hrv_std_s)
    return np.array(times)


def make_ppg(hr_bpm: float = 72, fs: float = 125, duration_s: float = 60, noise_std: float = 0.02,
             drift_amp: float = 0.0, hrv_std_s: float = 0.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Each beat = systolic Gaussian + smaller, later diastolic Gaussian.

    Returns (signal, true_beat_times_s)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * fs)) / fs
    times = beat_times(hr_bpm, duration_s, hrv_std_s, seed)
    x = np.zeros_like(t)
    for bt in times:
        x += 1.0 * np.exp(-((t - bt) ** 2) / (2 * 0.06 ** 2))          # systolic
        x += 0.35 * np.exp(-((t - bt - 0.28) ** 2) / (2 * 0.08 ** 2))  # diastolic
    x += drift_amp * np.sin(2 * np.pi * 0.1 * t)                        # baseline wander
    x += rng.normal(0, noise_std, size=t.shape)
    return x, times


def make_ecg(beat_times_s: np.ndarray, fs: float = 125, duration_s: float = 60,
             delay_s: float = -0.2, seed: int = 1) -> np.ndarray:
    """Crude ECG: sharp R spikes. delay_s<0 because the R-peak comes before the pulse
    reaches the finger (pulse transit time)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * fs)) / fs
    x = np.zeros_like(t)
    for bt in beat_times_s + delay_s:
        x += np.exp(-((t - bt) ** 2) / (2 * 0.012 ** 2))
    return x + rng.normal(0, 0.01, size=t.shape)
