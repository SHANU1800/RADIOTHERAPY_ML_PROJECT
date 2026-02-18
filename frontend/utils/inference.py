"""
Inference functions for predictions.
"""
import pickle
from pathlib import Path
from typing import Dict, Tuple

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.load_data import load_patient_file
from src.features import build_windows, get_X_y
import config as cfg


_model_cache: dict = {}  # In-memory cache so we don't reload pickle every time


def load_model(task: str = "breath_hold") -> Tuple[any, any, str]:
    """
    Load best model from models/best_model.pkl (cached in memory).
    Returns (model, scaler, model_name).
    """
    if task in _model_cache:
        return _model_cache[task]
    
    model_path = cfg.MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train a model first.")
    
    with model_path.open("rb") as f:
        data = pickle.load(f)
    
    model = data["model"]
    scaler = data.get("scaler")
    saved_task = data.get("task", task)
    
    if saved_task != task:
        raise ValueError(f"Model trained for '{saved_task}' but requested '{task}'")
    
    _model_cache[task] = (model, scaler, type(model).__name__)
    return model, scaler, type(model).__name__


def predict_breathing_pattern(
    file_path: Path,
    task: str = "breath_hold",
    window_sec: float = 2.0,
    patient_id: str = "uploaded",
    file_id: str = "uploaded",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load file, build windows, run model, return predictions + metadata.
    
    Returns:
        windows_df: DataFrame with features + predictions + confidence
        metadata: Dict with model_name, task, num_windows, etc.
    """
    if pd is None or np is None:
        raise ImportError("pandas and numpy required")
    
    # Load file
    df = load_patient_file(file_path, patient_id=patient_id, file_id=file_id)
    if len(df) == 0:
        raise ValueError("Empty file or parsing error")
    
    # Build windows
    windows = build_windows(
        df,
        window_sec=window_sec,
        sample_rate_hz=cfg.SAMPLE_RATE_HZ,
        min_rows=cfg.MIN_WINDOW_ROWS,
    )
    if len(windows) == 0:
        raise ValueError("No windows produced (file too short?)")
    
    # Get features and labels
    X, y_true, _ = get_X_y(windows, task=task)
    X = X.fillna(0)
    
    # Load model
    model, scaler, model_name = load_model(task)
    
    # Scale features
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values
    
    # Predict
    predictions = model.predict(X_scaled)
    
    # Get probabilities if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)
        confidence = np.max(proba, axis=1)
    else:
        confidence = np.ones(len(predictions))
    
    # Add predictions to windows DataFrame
    windows = windows.copy()
    windows["prediction"] = predictions
    windows["confidence"] = confidence
    if hasattr(model, "predict_proba"):
        windows["prob_class_0"] = proba[:, 0]
        windows["prob_class_1"] = proba[:, 1]
    
    metadata = {
        "model_name": model_name,
        "task": task,
        "num_windows": len(windows),
        "num_rows": len(df),
        "window_sec": window_sec,
    }
    
    return windows, metadata
