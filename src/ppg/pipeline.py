"""End-to-end: raw PPG (+ optional ECG reference) -> per-window feature table."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .features import window_features
from .filters import bandpass, normalize, windows
from .peaks import detect_ecg_rpeaks, detect_ppg_peaks, heart_rate
from .sqi import is_good_quality, skewness_sqi, template_correlation_sqi

DEFAULT_CONFIG: dict[str, Any] = {
    "filter": {"low_hz": 0.5, "high_hz": 8.0, "order": 4},
    "window": {"length_s": 8.0, "step_s": 2.0},
    "peaks": {"min_distance_s": 0.33, "prominence": 0.3},
    "sqi": {"min_skew": 0.0, "min_template_corr": 0.85},
}


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return DEFAULT_CONFIG
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    return {k: {**v, **user.get(k, {})} for k, v in DEFAULT_CONFIG.items()}


def process_record(ppg: np.ndarray, fs: float, ecg: np.ndarray | None = None,
                   record_id: str = "record", config: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = config or DEFAULT_CONFIG
    f, w, p, q = cfg["filter"], cfg["window"], cfg["peaks"], cfg["sqi"]

    ppg = np.nan_to_num(np.asarray(ppg, dtype=float))
    filt = normalize(bandpass(ppg, fs, f["low_hz"], f["high_hz"], f["order"]))
    rpeaks = detect_ecg_rpeaks(ecg, fs) if ecg is not None else None

    rows = []
    for start, end in windows(len(filt), fs, w["length_s"], w["step_s"]):
        seg = filt[start:end]
        pk = detect_ppg_peaks(seg, fs, p["min_distance_s"], p["prominence"])
        skw = skewness_sqi(seg)
        corr = template_correlation_sqi(seg, pk, fs)
        row = {
            "record_id": record_id,
            "t_start_s": start / fs,
            "t_end_s": end / fs,
            "sqi_skewness": skw,
            "sqi_template_corr": corr,
            "good_quality": is_good_quality(skw, corr, q["min_skew"], q["min_template_corr"]),
            **window_features(seg, pk, fs),
        }
        if rpeaks is not None:
            r_in = rpeaks[(rpeaks >= start) & (rpeaks < end)]
            row["hr_ref_bpm"] = heart_rate(r_in, fs)
            row["hr_abs_err_bpm"] = abs(row["hr_bpm"] - row["hr_ref_bpm"])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict[str, float]:
    """Headline numbers for the README: error on all windows vs. good-quality windows."""
    out = {"n_windows": len(df), "pct_good_quality": 100 * df["good_quality"].mean()}
    if "hr_abs_err_bpm" in df:
        out["mae_all_bpm"] = df["hr_abs_err_bpm"].mean()
        out["mae_good_bpm"] = df.loc[df["good_quality"], "hr_abs_err_bpm"].mean()
    return {k: round(float(v), 3) for k, v in out.items()}
