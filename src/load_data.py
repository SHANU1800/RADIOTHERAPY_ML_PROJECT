"""
Unified data loader for breathing-curve files (.dat, .txt, .csv).
Returns DataFrames with patient_id and optional session metadata.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

# Standard column names for curve data
CURVE_COLUMNS = [
    "Session Time",
    "Volume (liters)",
    "Balloon Valve Status",
    "Patient Switch",
    "Gating Mode",
    "Gating Status",
    "Relay State",
]


def _parse_header_semicolon(lines: List[str]) -> Tuple[Dict[str, str], int]:
    """Parse header until HeaderEnd; return (meta dict, index of first data line)."""
    meta: Dict[str, str] = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "HeaderEnd":
            return meta, i + 1
        if ":" in stripped and not stripped.startswith("["):
            key, value = stripped.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, len(lines)


def _parse_semicolon_data(lines: List[str], start_index: int) -> pd.DataFrame:
    """Parse semicolon-separated data lines into DataFrame."""
    if pd is None:
        raise ImportError("pandas is required for load_data")
    rows = []
    for line in lines[start_index:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("Session Time"):
            continue
        parts = [p.strip() for p in stripped.split(";")]
        if len(parts) < 2:
            continue
        # Pad to 7 columns if needed
        while len(parts) < 7:
            parts.append("")
        rows.append(parts[:7])
    if not rows:
        return pd.DataFrame(columns=CURVE_COLUMNS)
    df = pd.DataFrame(rows, columns=CURVE_COLUMNS)
    # Numeric columns
    for col in ["Session Time", "Volume (liters)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Session Time", "Volume (liters)"])
    return df


def load_patient_file(
    file_path: Union[str, Path],
    patient_id: Optional[str] = None,
    file_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a single breathing-curve file (.dat, .txt, or .csv) into a DataFrame.

    Returns DataFrame with columns: Session Time, Volume (liters), Balloon Valve Status,
    Patient Switch, Gating Mode, Gating Status, Relay State.
    If patient_id (and optionally file_id) are provided, they are added as columns.
    """
    if pd is None:
        raise ImportError("pandas is required for load_data")
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if patient_id is None:
        patient_id = path.parent.name
    if file_id is None:
        file_id = path.stem

    if suffix == ".csv":
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        # Normalize column names to match CURVE_COLUMNS if possible
        cols = list(df.columns)
        if "Session Time" not in cols and "Volume (liters)" not in cols:
            # Try first two columns as time and volume
            if len(cols) >= 2:
                df = df.rename(columns={cols[0]: "Session Time", cols[1]: "Volume (liters)"})
    elif suffix in (".dat", ".txt"):
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        _, start = _parse_header_semicolon(lines)
        df = _parse_semicolon_data(lines, start)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}. Use .dat, .txt, or .csv.")

    df["patient_id"] = patient_id
    df["file_id"] = file_id
    return df


def load_all_patients(
    dataset_dir: Union[str, Path],
    include_session_ini: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, str]]]]:
    """
    Load all breathing-curve files from dataset directory.

    dataset_dir: path to folder containing one subfolder per patient.
    include_session_ini: if True, parse session.ini per patient and return as second element.

    Returns:
        - Single concatenated DataFrame with columns from load_patient_file plus patient_id, file_id.
        - Dict[patient_id, list of session.ini blocks] if include_session_ini else empty dict.
    """
    if pd is None:
        raise ImportError("pandas is required for load_data")
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    frames: List[pd.DataFrame] = []
    session_metadata: Dict[str, List[Dict[str, str]]] = {}

    for patient_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        patient_id = patient_dir.name
        if include_session_ini:
            ini_path = patient_dir / "session.ini"
            if ini_path.exists():
                session_metadata[patient_id] = _read_session_ini(ini_path)

        for file_path in sorted(patient_dir.rglob("*")):
            if not file_path.is_file():
                continue
            suffix = file_path.suffix.lower()
            if suffix not in (".dat", ".txt", ".csv"):
                continue
            try:
                df = load_patient_file(file_path, patient_id=patient_id, file_id=file_path.stem)
                if len(df) > 0:
                    frames.append(df)
            except Exception:
                continue

    if not frames:
        empty = pd.DataFrame(columns=CURVE_COLUMNS + ["patient_id", "file_id"])
        return empty, session_metadata
    combined = pd.concat(frames, ignore_index=True)
    return combined, session_metadata


def _read_session_ini(ini_path: Path) -> List[Dict[str, str]]:
    """Parse session.ini into list of session blocks (each block is a dict)."""
    blocks: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    with ini_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("[********************") and "SESSION" in line.upper():
                if current:
                    blocks.append(current)
                current = {}
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                current[key.strip()] = value.strip()
        if current:
            blocks.append(current)
    return blocks


def get_patient_file_list(dataset_dir: Union[str, Path]) -> List[Tuple[str, Path]]:
    """Return list of (patient_id, file_path) for all .dat, .txt, .csv under dataset_dir."""
    dataset_dir = Path(dataset_dir)
    out: List[Tuple[str, Path]] = []
    for patient_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        for file_path in sorted(patient_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in (".dat", ".txt", ".csv"):
                out.append((patient_dir.name, file_path))
    return out
