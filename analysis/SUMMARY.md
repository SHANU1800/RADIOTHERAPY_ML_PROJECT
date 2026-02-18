# Project Summary (Feb 16, 2026)

## What the dataset contains

- Patient-specific folders under `dataset/`.
- Time-series breathing data in two main formats:
  - CSV files (e.g., `breath_hold_data.csv`)
  - TXT curve data files with header metadata and semicolon-separated values
- Core signals across files:
  - Session Time (seconds, ~0.02s step at ~50 Hz)
  - Volume (liters)
  - Balloon Valve Status
  - Patient Switch
  - Gating Mode / Status / Relay State (often manual or blank)

## Files inspected (samples)

- `dataset/Abdul Rehaman/breath_hold_data.csv`
- `dataset/ABDUL REHAMAN N_._7132024/ABDUL REHAMAN N_1.txt`
- `dataset/BENNY MARTIS_._5482024/BENNY MARTIS_ct.txt`

## Clinical/ML framing discussed

- Each patient file is an independent time-series recording; there is no direct dependency between patients.
- Multiple patients can be combined for ML training, but splits must be done by patient to avoid data leakage.
- Example ML tasks:
  - Classification: breath-hold vs normal, gating OK vs not
  - Regression: predict next volume values
  - Segmentation: label inhale/exhale cycles or stable windows

## Analysis artifacts created

- `analysis/analyze_dataset.py`
  - Scans all patient folders
  - Parses CSV and TXT formats
  - Computes per-file stats (rows, duration, volume min/mean/max/std)
  - Counts status fields (balloon valve, patient switch, gating mode)
  - Outputs summaries in `analysis/output/`

- `analysis/README.md`
  - Instructions for running the analysis

## Analysis outputs generated (run completed)

- `analysis/output/file_summary.csv` (3 file summaries)
- `analysis/output/patient_summary.csv` (3 patient summaries)
- `analysis/output/other_files.csv` (184 non-CSV/TXT files listed)

## Notes

- Some volume values can be negative (expected in respiratory flow cycles).
- Many rows contain `Manual Overide` in gating mode fields or blank placeholders.
- Large blocks of zeros appear in some files, likely idle/hold segments.
