# Dataset Analysis

This folder contains a lightweight, **no-dependency** analysis script that summarizes the respiratory time‑series data under `dataset/`.

## What it produces

When you run the script, it writes CSV outputs to `analysis/output/`:

- `file_summary.csv` — per‑file statistics (row counts, time range, volume stats, status counts)
- `patient_summary.csv` — per‑patient rollup (files, total rows, total duration)
- `other_files.csv` — non‑CSV/TXT files (extension + size)

## How it works

- `.txt` files are parsed by reading header metadata until `HeaderEnd`, then semicolon‑separated curve data.
- `.csv` files are parsed with standard CSV parsing.
- The script computes basic descriptive stats without loading entire files into memory.

## Run it

From the workspace root, run the script (Python 3.8+):

- `python analysis/analyze_dataset.py`

You can re‑run any time; it overwrites the output CSVs.

## Notes

- If you want plots later, we can add an optional `matplotlib` visualization script.
- If you want ML feature extraction, we can add windowed feature exports next.
