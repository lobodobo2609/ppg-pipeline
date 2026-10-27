"""Command-line entry point.

    python -m ppg demo                       # synthetic signal, no download needed
    python -m ppg run --record bidmc01       # one BIDMC record from PhysioNet
    python -m ppg run --all                  # all 53 BIDMC records
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .pipeline import load_config, process_record, summarize
from .synthetic import make_ecg, make_ppg


def main() -> None:
    ap = argparse.ArgumentParser(prog="ppg")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="run on a synthetic signal")
    run = sub.add_parser("run", help="run on BIDMC records")
    run.add_argument("--record", default="bidmc01")
    run.add_argument("--all", action="store_true")
    run.add_argument("--local-dir", default=None, help="folder with downloaded BIDMC files")
    for p in sub.choices.values():
        p.add_argument("--config", default="config.yaml")
        p.add_argument("--out", default="outputs")
    args = ap.parse_args()

    cfg = load_config(args.config if Path(args.config).exists() else None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.cmd == "demo":
        ppg, beats = make_ppg(hr_bpm=72, noise_std=0.05, drift_amp=0.5, hrv_std_s=0.02)
        df = process_record(ppg, 125, make_ecg(beats), "synthetic", cfg)
    else:
        from .load import load_bidmc
        ids = [f"bidmc{i:02d}" for i in range(1, 54)] if args.all else [args.record]
        frames = []
        for rid in ids:
            rec = load_bidmc(rid, args.local_dir)
            frames.append(process_record(rec.ppg, rec.fs, rec.ecg, rid, cfg))
            print(f"processed {rid}")
        df = pd.concat(frames, ignore_index=True)

    path = out / "features.csv"
    df.to_csv(path, index=False)
    print(json.dumps(summarize(df), indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
