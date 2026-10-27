# ppg-pipeline

A reproducible Python pipeline that turns raw photoplethysmography (PPG) waveforms into a clean, quality-scored feature table, and validates its heart rate estimates against a simultaneous ECG reference.

**Status:** in progress. Core pipeline and tests complete; BIDMC results and figures coming next.

## What it does

1. **Preprocess:** zero-phase Butterworth band-pass (0.5–8 Hz) to remove baseline drift and noise, then z-score normalization and 8 s sliding windows.
2. **Detect beats:** systolic peak detection with a refractory distance so the diastolic bump isn't counted twice.
3. **Validate:** ECG R-peaks (simplified Pan-Tompkins) give a reference heart rate for every window.
4. **Score signal quality:** skewness and beat-template correlation flag windows corrupted by noise or artifacts.
5. **Extract features:** heart rate, interbeat interval, HRV (SDNN, RMSSD), pulse amplitude, rise time, max upslope, and relative spectral power, all in one tidy table.

## Results

<!-- Fill in after running `python -m ppg run --all` -->
| Windows | MAE vs ECG (all) | MAE vs ECG (good quality only) | % good quality |
|---|---|---|---|
| TBD | TBD bpm | TBD bpm | TBD |

<!-- Figures: raw vs filtered, detected peaks, Bland-Altman, SQI vs error -->

## Quick start

```bash
pip install -e ".[dev]"
pytest                          # unit tests on synthetic signals with known ground truth
python -m ppg demo              # synthetic run, no download needed
python -m ppg run --record bidmc01
python -m ppg run --all         # all 53 BIDMC records -> outputs/features.csv
```

Or with Docker (tests run during the build):

```bash
docker build -t ppg-pipeline .
docker run --rm ppg-pipeline demo
```

## Data

[BIDMC PPG and Respiration Dataset](https://physionet.org/content/bidmc/) (PhysioNet): 53 eight-minute ICU recordings at 125 Hz with simultaneous PPG and ECG. Records stream from PhysioNet by default; to work offline:

```python
import wfdb; wfdb.dl_database("bidmc", "data/bidmc")
```
then `python -m ppg run --all --local-dir data/bidmc`.

## Project layout

```
src/ppg/
  load.py       dataset loaders
  filters.py    band-pass, normalization, windowing
  peaks.py      PPG beat detection, ECG R-peaks, heart rate
  sqi.py        signal quality indices
  features.py   per-window features
  synthetic.py  synthetic PPG/ECG with known ground truth (for tests)
  pipeline.py   end-to-end processing + summary
tests/          pytest suite
config.yaml     all tunable parameters
```

## Testing approach

Every stage is tested on synthetic signals where the truth is known: the filter must remove drift while keeping the pulse, peak detection must recover 50/72/120 bpm within 1 bpm without double-counting, and quality scores must rank clean signals above noise. CI runs the suite on every push.

## Limitations and next steps

- [ ] Run on all BIDMC records; add figures
- [ ] Cross-check peaks and HR against NeuroKit2
- [ ] Motion artifacts: PPG-DaLiA (wrist PPG during activity) with accelerometer-based artifact flags
- [ ] Tune SQI thresholds against measured error rather than fixed values

## References

- Elgendi, M. (2012). On the analysis of fingertip photoplethysmogram signals. *Current Cardiology Reviews.*
- Elgendi, M. (2016). Optimal signal quality index for photoplethysmogram signals. *Bioengineering.*
- Pimentel et al. (2016). Toward a robust estimation of respiratory rate from pulse oximeters. *IEEE TBME* (BIDMC dataset).
