"""Tests on synthetic signals with known ground truth."""
import numpy as np
import pytest

from ppg.features import window_features
from ppg.filters import bandpass, normalize, windows
from ppg.peaks import detect_ecg_rpeaks, detect_ppg_peaks, heart_rate
from ppg.pipeline import process_record, summarize
from ppg.sqi import skewness_sqi, template_correlation_sqi
from ppg.synthetic import make_ecg, make_ppg

FS = 125


def power_at(x, freq):
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / FS)
    return spec[np.argmin(np.abs(freqs - freq))]


def test_bandpass_removes_drift_keeps_pulse():
    t = np.arange(60 * FS) / FS
    x = np.sin(2 * np.pi * 0.05 * t) + np.sin(2 * np.pi * 1.2 * t)
    y = bandpass(x, FS)
    assert power_at(y, 0.05) < 0.1 * power_at(x, 0.05)
    assert power_at(y, 1.2) > 0.9 * power_at(x, 1.2)


def test_normalize_handles_flat_signal():
    assert np.all(normalize(np.ones(100)) == 0)


def test_windows_cover_signal():
    w = list(windows(1000, 100, win_s=2, step_s=1))
    assert w[0] == (0, 200) and w[-1][1] <= 1000 and len(w) == 9


@pytest.mark.parametrize("hr", [50, 72, 120])
def test_recovers_known_heart_rate(hr):
    ppg, _ = make_ppg(hr_bpm=hr, noise_std=0.05, drift_amp=0.5)
    peaks = detect_ppg_peaks(normalize(bandpass(ppg, FS)), FS)
    assert heart_rate(peaks, FS) == pytest.approx(hr, abs=1.0)


def test_does_not_double_count_diastolic_peak():
    ppg, beats = make_ppg(hr_bpm=60, duration_s=30)
    peaks = detect_ppg_peaks(normalize(bandpass(ppg, FS)), FS)
    assert abs(len(peaks) - len(beats)) <= 1


def test_ecg_rpeaks_match_beats():
    _, beats = make_ppg(hr_bpm=80, duration_s=30)
    rpeaks = detect_ecg_rpeaks(make_ecg(beats, duration_s=30), FS)
    assert abs(len(rpeaks) - len(beats)) <= 1


def test_sqi_ranks_clean_above_noise():
    clean = normalize(bandpass(make_ppg(noise_std=0.02, duration_s=8)[0], FS))
    noise = normalize(bandpass(np.random.default_rng(0).normal(size=8 * FS), FS))
    pk_c, pk_n = detect_ppg_peaks(clean, FS), detect_ppg_peaks(noise, FS)
    assert skewness_sqi(clean) > skewness_sqi(noise)
    assert template_correlation_sqi(clean, pk_c, FS) > template_correlation_sqi(noise, pk_n, FS)


def test_features_are_finite_on_clean_signal():
    seg = normalize(bandpass(make_ppg(duration_s=8, hrv_std_s=0.02)[0], FS))
    feats = window_features(seg, detect_ppg_peaks(seg, FS), FS)
    assert all(np.isfinite(v) for v in feats.values()), feats


def test_end_to_end_against_ecg_reference():
    ppg, beats = make_ppg(hr_bpm=72, noise_std=0.05, drift_amp=0.5, hrv_std_s=0.02)
    df = process_record(ppg, FS, make_ecg(beats), "synthetic")
    s = summarize(df)
    assert s["pct_good_quality"] > 90
    assert s["mae_all_bpm"] < 2.0
