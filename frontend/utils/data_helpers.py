"""
Data loading helpers for frontend.
"""
from pathlib import Path
from typing import List, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.load_data import load_all_patients, get_patient_file_list
import config as cfg


def get_patient_list() -> List[str]:
    """Get list of all patient IDs from dataset directory."""
    dataset_dir = cfg.DATASET_DIR
    if not dataset_dir.exists():
        return []
    patients = sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])
    return patients


def get_file_list(patient_id: str) -> List[Tuple[str, Path]]:
    """
    Get list of (file_id, file_path) for a patient.
    Returns list of tuples: (file_id, Path).
    """
    dataset_dir = cfg.DATASET_DIR
    patient_dir = dataset_dir / patient_id
    if not patient_dir.exists():
        return []
    files = []
    for file_path in sorted(patient_dir.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in (".dat", ".txt", ".csv"):
            files.append((file_path.stem, file_path))
    return files


def load_summary_stats() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load summary CSVs from analysis/output.
    Returns (file_summary, patient_summary, other_files).
    """
    if pd is None:
        raise ImportError("pandas required")
    analysis_out = PROJECT_ROOT / "analysis" / "output"
    file_summary = pd.DataFrame()
    patient_summary = pd.DataFrame()
    other_files = pd.DataFrame()
    
    if (analysis_out / "file_summary.csv").exists():
        file_summary = pd.read_csv(analysis_out / "file_summary.csv")
    if (analysis_out / "patient_summary.csv").exists():
        patient_summary = pd.read_csv(analysis_out / "patient_summary.csv")
    if (analysis_out / "other_files.csv").exists():
        other_files = pd.read_csv(analysis_out / "other_files.csv")
    
    return file_summary, patient_summary, other_files
