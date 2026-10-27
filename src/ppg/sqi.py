"""Signal quality indices (SQIs) for PPG windows."""
from __future__ import annotations

import numpy as np
from scipy.stats import skew


def skewness_sqi(x: np.ndarray) -> float:
    """Skewness of the window. Clean PPG is asymmetric (sharp upstroke, slow decay),
    so it skews positive; noise and artifacts push it toward zero or negative.
    See Elgendi (2016) on optimal PPG signal quality indices."""
    return float(skew(x)) if np.std(x) > 0 else float("nan")


def template_correlation_sqi(x: np.ndarray, peaks: np.ndarray, fs: float,
                             pre_s: float = 0.25, post_s: float = 0.45) -> float:
    """Mean correlation of each beat with the window's average beat.

    Near 1.0 means beats look alike (good); low values mean artifacts or
    missed/extra detections."""
    pre, post = int(pre_s * fs), int(post_s * fs)
    beats = [x[p - pre:p + post] for p in peaks if p - pre >= 0 and p + post <= len(x)]
    if len(beats) < 3:
        return float("nan")
    beats = np.vstack(beats)
    template = beats.mean(axis=0)
    corrs = [np.corrcoef(b, template)[0, 1] for b in beats]
    return float(np.nanmean(corrs))


def is_good_quality(skewness: float, template_corr: float,
                    min_skew: float = 0.0, min_corr: float = 0.85) -> bool:
    if not (np.isfinite(skewness) and np.isfinite(template_corr)):
        return False
    return skewness >= min_skew and template_corr >= min_corr
