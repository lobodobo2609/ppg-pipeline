# Notebooks

Start here with `01_explore_bidmc.ipynb`:

1. `from ppg.load import load_bidmc; rec = load_bidmc("bidmc01")`
2. Plot 10 seconds of raw `rec.ppg` and `rec.ecg` on stacked axes.
3. Plot raw vs `bandpass(rec.ppg, rec.fs)` and look for drift, clipping and flat spots.
4. Overlay `detect_ppg_peaks` on the filtered signal and `detect_ecg_rpeaks` on the ECG.
5. Write down what you notice; it becomes the README's data-inspection section.

Keep notebooks for exploration and figures. Anything reusable belongs in `src/ppg/`.
