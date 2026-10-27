"""Loaders for public PPG datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Recording:
    record_id: str
    fs: float
    ppg: np.ndarray
    ecg: np.ndarray | None


def _clean(name: str) -> str:
    # BIDMC channel names carry trailing commas, e.g. "PLETH,"
    return name.strip().rstrip(",").upper()


def load_bidmc(record_id: str = "bidmc01", local_dir: str | Path | None = None) -> Recording:
    """Load one BIDMC record (PhysioNet 'bidmc' database).

    By default streams it from PhysioNet. Pass local_dir after downloading
    with `wfdb.dl_database('bidmc', 'data/bidmc')` to work offline.
    """
    import wfdb

    if local_dir:
        rec = wfdb.rdrecord(str(Path(local_dir) / record_id))
    else:
        rec = wfdb.rdrecord(record_id, pn_dir="bidmc")
    names = [_clean(n) for n in rec.sig_name]
    sig = rec.p_signal
    ppg = sig[:, names.index("PLETH")]
    ecg = sig[:, names.index("II")] if "II" in names else None
    return Recording(record_id, float(rec.fs), ppg, ecg)
