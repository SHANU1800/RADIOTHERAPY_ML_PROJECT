# ML Pipeline: Breathing Patterns (Radiation Gating)

Classical ML (no deep learning) for breathing-curve data: classification and optional regression with **patient-based** train/test splits.

## Setup

```bash
pip install -r requirements.txt
```

## Data

- **Location**: `dataset/` — one folder per patient containing `.dat`, `.txt`, or `.csv` curve files.
- **Formats**: `.dat` and `.txt` use a header (until `HeaderEnd`) then semicolon-separated columns: Session Time, Volume (liters), Balloon Valve Status, Patient Switch, Gating Mode, Gating Status, Relay State. `.csv` uses the same column names.
- **Summaries**: Run the analyzer to regenerate file/patient summaries (includes `.dat`):

  ```bash
  python analysis/analyze_dataset.py
  ```
  Outputs: `analysis/output/file_summary.csv`, `patient_summary.csv`, `other_files.csv`.

## Unified loader

```python
from src.load_data import load_patient_file, load_all_patients

# Single file
df = load_patient_file("dataset/SomePatient/session1.dat")

# All patients (returns one DataFrame with patient_id, file_id)
df, session_meta = load_all_patients("dataset", include_session_ini=True)
```

## ML tasks and labels

- **Classification A — Breath-hold vs free-breathing**: From `Balloon Valve Status`: 4 = inflated = breath-hold (label 1), 1 = deflated = free-breathing (label 0). See `src/labels.py`.
- **Classification B — Gating OK vs not OK**: From `Gating Mode`: "Automated" = OK (1), "Manual Overide" etc. = not OK (0).
- **Regression** (optional): Next-step volume or window stability; extend `src/features.py` and add a regressor in `src/train.py` if needed.

Labels are computed per **window** (e.g. 2 s) in `src/features.py` when building windows.

## Features

Window-level (default 2 s at 50 Hz):

- Volume: mean, std, min, max, range, change, rolling mean
- Balloon: fraction inflated, fraction deflated
- Patient Switch: fraction on
- Gating: fraction automated

See `src/features.py` (`build_windows`, `get_X_y`).

## Training

Patient-based split (no patient in both train and test):

```bash
python -m src.train --task breath_hold
python -m src.train --task gating_ok --dataset path/to/dataset --window-sec 2.0 --train-ratio 0.7 --seed 42
```

- **Models**: Random Forest, Logistic Regression, SVM; XGBoost if installed.
- **Metrics**: Accuracy, balanced accuracy, F1, confusion matrix (saved under `models/`).
- **Output**: `models/metrics.json`, `models/metrics_models.json`, `models/best_model.pkl` (includes scaler and task name).

## Config

`config.py`: `DATASET_DIR`, `RANDOM_STATE`, `TRAIN_RATIO`/`VAL_RATIO`/`TEST_RATIO`, `WINDOW_SEC`, `SAMPLE_RATE_HZ`, `MIN_WINDOW_ROWS`, and label constants.

## Project layout

- `dataset/` — raw data (unchanged)
- `analysis/` — `analyze_dataset.py` (supports .dat, .txt, .csv), `output/`
- `src/` — `load_data.py`, `labels.py`, `features.py`, `train.py`
- `notebooks/` — optional EDA (`eda_breathing_dataset.ipynb`)
- `config.py` — paths and hyperparameters
- `models/` — saved metrics and best model (created by training)

## Reproducibility

- Fix `RANDOM_STATE` in `config.py` and use `--seed 42` in `src.train`.
- Splits are by **patient_id** (folder name); document any exclusion rules (e.g. very short files) in code or this README.
