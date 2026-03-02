"""
Inference functions for predictions (classical ML and deep learning).
Supports multi-channel DL, Grad-CAM interpretability, and signal analysis.
"""
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


_model_cache: dict = {}
_dl_model_cache: dict = {}


def load_model(task: str = "breath_hold") -> Tuple[any, any, str]:
    """
    Load best model from models/best_model.pkl (cached in memory).
    Returns (model, scaler, model_name).
    """
    if task not in ("breath_hold", "gating_ok"):
        raise ValueError(f"task must be 'breath_hold' or 'gating_ok', got: {task}")
    if task in _model_cache:
        return _model_cache[task]

    model_path = cfg.MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train a model first (python -m src.train).")

    try:
        with model_path.open("rb") as f:
            data = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, Exception) as e:
        raise ValueError(f"Model file is corrupted or invalid: {e}") from e

    if not isinstance(data, dict) or "model" not in data:
        raise ValueError("Model file must contain a dict with 'model' key.")

    model = data["model"]
    scaler = data.get("scaler")
    saved_task = data.get("task", task)

    if saved_task != task:
        raise ValueError(f"Model was trained for task '{saved_task}' but you requested '{task}'.")

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
    Load file, build windows, run classical ML model, return predictions + metadata.
    """
    if pd is None or np is None:
        raise ImportError("pandas and numpy required")
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        df = load_patient_file(file_path, patient_id=patient_id, file_id=file_id)
    except Exception as e:
        raise ValueError(f"Could not load file: {e}") from e
    if len(df) == 0:
        raise ValueError("File is empty or could not be parsed.")

    try:
        windows = build_windows(df, window_sec=window_sec, sample_rate_hz=cfg.SAMPLE_RATE_HZ, min_rows=cfg.MIN_WINDOW_ROWS)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not build windows: {e}") from e
    if len(windows) == 0:
        raise ValueError("No windows produced. File may be too short for the chosen window size.")

    try:
        X, y_true, _ = get_X_y(windows, task=task)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not extract features: {e}") from e
    X = X.fillna(0)

    try:
        model, scaler, model_name = load_model(task)
    except (FileNotFoundError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Could not load model: {e}") from e

    try:
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values
        predictions = model.predict(X_scaled)
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}") from e

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)
        confidence = np.max(proba, axis=1)
    else:
        confidence = np.ones(len(predictions))

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


# ── Deep Learning inference ──────────────────────────────────────────


def is_dl_available() -> bool:
    """Return True if TensorFlow is installed and DL inference can be used."""
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


def get_available_dl_models(task: str = "breath_hold") -> List[str]:
    """Return list of available DL model names for the given task."""
    available = []
    for name in cfg.DL_MODEL_NAMES:
        prefix = f"dl_{name.lower()}_{task}"
        model_path = cfg.DL_MODELS_DIR / f"{prefix}_model.keras"
        if model_path.exists():
            available.append(name)
    return available


def _detect_model_channels(model_name: str, task: str) -> Optional[List[str]]:
    """
    Detect the channel configuration a model was trained with by reading the
    summary JSON or the model's input shape.
    """
    summary_path = cfg.DL_MODELS_DIR / f"dl_summary_{task}.json"
    if summary_path.exists():
        try:
            import json
            with summary_path.open() as f:
                summary = json.load(f)
            channels = summary.get("channels")
            if channels:
                return channels
        except Exception:
            pass

    metrics_path = cfg.DL_MODELS_DIR / f"dl_{model_name.lower()}_{task}_metrics.json"
    if metrics_path.exists():
        try:
            import json
            with metrics_path.open() as f:
                m = json.load(f)
            n_ch = m.get("n_channels", 1)
            if n_ch == 5:
                from src.dl_features import MULTI_CHANNELS
                return list(MULTI_CHANNELS)
            elif n_ch == 1:
                return ["Volume (liters)"]
        except Exception:
            pass
    return None


def load_dl_model(model_name: str, task: str = "breath_hold"):
    """Load a trained Keras DL model. Cached in memory."""
    cache_key = f"{model_name}_{task}"
    if cache_key in _dl_model_cache:
        return _dl_model_cache[cache_key]

    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow is required for DL inference.")

    prefix = f"dl_{model_name.lower()}_{task}"
    model_path = cfg.DL_MODELS_DIR / f"{prefix}_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"DL model not found: {model_path}. "
            f"Train it first: python -m src.dl_train --model {model_name} --task {task}"
        )

    model = tf.keras.models.load_model(str(model_path))
    _dl_model_cache[cache_key] = model
    return model


def predict_breathing_pattern_dl(
    file_path: Path,
    model_name: str = "LSTM",
    task: str = "breath_hold",
    window_sec: float = 2.0,
    patient_id: str = "uploaded",
    file_id: str = "uploaded",
    channels: Optional[list] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load file, build DL windows (auto-detecting channels), run DL model,
    return predictions + metadata.
    """
    if pd is None or np is None:
        raise ImportError("pandas and numpy required")
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if channels is None:
        channels = _detect_model_channels(model_name, task)
    if channels is None:
        channels = list(cfg.DL_WINDOW_CHANNELS)

    from src.dl_features import build_dl_windows

    try:
        df = load_patient_file(file_path, patient_id=patient_id, file_id=file_id)
    except Exception as e:
        raise ValueError(f"Could not load file: {e}") from e
    if len(df) == 0:
        raise ValueError("File is empty or could not be parsed.")

    try:
        X, y_bh, y_gk, pids, meta_df = build_dl_windows(
            df, channels=channels, window_sec=window_sec,
            sample_rate_hz=cfg.SAMPLE_RATE_HZ, min_rows=cfg.MIN_WINDOW_ROWS,
            normalize=True, overlap=0.0,  # no overlap for inference — exact coverage
            compute_extras=True,
        )
    except Exception as e:
        raise ValueError(f"Could not build DL windows: {e}") from e

    if len(X) == 0:
        raise ValueError("No windows produced. File may be too short.")

    try:
        model = load_dl_model(model_name, task)
    except (FileNotFoundError, ImportError):
        raise
    except Exception as e:
        raise RuntimeError(f"Could not load DL model: {e}") from e

    try:
        y_pred_prob = model.predict(X, batch_size=cfg.DL_BATCH_SIZE).flatten()
    except Exception as e:
        raise RuntimeError(f"DL prediction failed: {e}") from e

    predictions = (y_pred_prob >= 0.5).astype(int)
    confidence = np.where(predictions == 1, y_pred_prob, 1.0 - y_pred_prob)

    result_df = meta_df.copy()
    result_df["prediction"] = predictions
    result_df["confidence"] = confidence
    result_df["prob_class_1"] = y_pred_prob
    result_df["prob_class_0"] = 1.0 - y_pred_prob
    result_df["label_breath_hold"] = y_bh
    result_df["label_gating_ok"] = y_gk

    metadata = {
        "model_name": f"DL-{model_name}",
        "model_type": "deep_learning",
        "task": task,
        "num_windows": len(result_df),
        "num_rows": len(df),
        "window_sec": window_sec,
        "channels": channels,
        "n_channels": len(channels),
    }

    return result_df, metadata


# ── Grad-CAM for temporal interpretability ───────────────────────────


def _input_gradient_fallback(model, X_tensor, target_class, window_size):
    """Fallback Grad-CAM using input gradients when layer-based approach fails."""
    import tensorflow as tf
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        preds = model(X_tensor, training=False)
        loss = preds[0, 0] if target_class == 1 else (1.0 - preds[0, 0])
    grads = tape.gradient(loss, X_tensor)
    if grads is None:
        return np.ones(window_size) * 0.5
    importance = tf.reduce_mean(tf.abs(grads[0]), axis=-1).numpy()
    importance = importance / (importance.max() + 1e-10)
    return importance


def compute_gradcam(
    model,
    X_window: "np.ndarray",
    target_class: int = 1,
) -> "np.ndarray":
    """
    Compute Grad-CAM importance scores for a single DL window.

    Uses gradient of the output w.r.t. the last convolutional (or recurrent)
    layer to produce a per-time-step importance vector.
    Falls back to input-gradient saliency if intermediate-layer approach fails
    (e.g. when the loaded model has not been called yet).

    Parameters
    ----------
    model : Keras model
    X_window : ndarray of shape (window_size, n_channels) — single window
    target_class : 1 or 0

    Returns
    -------
    importance : ndarray of shape (window_size,) — per-timestep importance (0–1)
    """
    try:
        import tensorflow as tf
    except ImportError:
        return np.ones(X_window.shape[0])

    X_batch = np.expand_dims(X_window, axis=0).astype(np.float32)
    X_tensor = tf.constant(X_batch)

    # Warm up the model so that .input / .output attributes are available.
    # This is required for models loaded from disk that have never been called.
    try:
        _ = model(X_batch, training=False)
    except Exception:
        pass

    conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv1D, tf.keras.layers.LSTM,
                              tf.keras.layers.Bidirectional, tf.keras.layers.GRU)):
            conv_layer = layer
            break

    if conv_layer is None:
        return _input_gradient_fallback(model, X_tensor, target_class, X_window.shape[0])

    try:
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[conv_layer.output, model.output],
        )
    except (AttributeError, ValueError):
        return _input_gradient_fallback(model, X_tensor, target_class, X_window.shape[0])

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(X_tensor, training=False)
        loss = preds[0, 0] if target_class == 1 else (1.0 - preds[0, 0])

    grads = tape.gradient(loss, conv_output)

    if grads is None:
        return _input_gradient_fallback(model, X_tensor, target_class, X_window.shape[0])

    weights = tf.reduce_mean(grads, axis=1, keepdims=True)
    cam = tf.reduce_sum(conv_output * weights, axis=-1).numpy()[0]
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    if len(cam) != X_window.shape[0]:
        cam = np.interp(
            np.linspace(0, 1, X_window.shape[0]),
            np.linspace(0, 1, len(cam)),
            cam,
        )

    return cam


def compute_gradcam_for_file(
    file_path: Path,
    model_name: str = "LSTM",
    task: str = "breath_hold",
    window_sec: float = 2.0,
    patient_id: str = "uploaded",
    file_id: str = "uploaded",
    channels: Optional[list] = None,
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """
    Compute Grad-CAM for all windows and stitch into a full-signal importance map.

    Returns
    -------
    time_axis : 1-D array of time values
    volume_signal : 1-D array of volume values
    importance : 1-D array of per-sample importance (0–1)
    """
    if pd is None or np is None:
        raise ImportError("pandas and numpy required")

    if channels is None:
        channels = _detect_model_channels(model_name, task)
    if channels is None:
        channels = list(cfg.DL_WINDOW_CHANNELS)

    from src.dl_features import build_dl_windows

    df = load_patient_file(file_path, patient_id=patient_id, file_id=file_id)
    X, y_bh, y_gk, pids, meta_df = build_dl_windows(
        df, channels=channels, window_sec=window_sec,
        sample_rate_hz=cfg.SAMPLE_RATE_HZ, min_rows=cfg.MIN_WINDOW_ROWS,
        normalize=True, overlap=0.0,
    )

    model = load_dl_model(model_name, task)

    time_full = pd.to_numeric(df["Session Time"], errors="coerce").fillna(0).values
    vol_full = pd.to_numeric(df["Volume (liters)"], errors="coerce").fillna(0).values
    importance_full = np.zeros(len(df), dtype=np.float32)
    counts = np.zeros(len(df), dtype=np.float32)

    window_size = X.shape[1] if len(X) > 0 else int(window_sec * cfg.SAMPLE_RATE_HZ)

    for i in range(len(X)):
        cam = compute_gradcam(model, X[i], target_class=1)
        start_time = meta_df.iloc[i]["time_start"]

        time_diffs = np.abs(time_full - start_time)
        start_idx = int(np.argmin(time_diffs))
        end_idx = min(start_idx + window_size, len(importance_full))
        seg_len = end_idx - start_idx

        if seg_len > 0 and seg_len <= len(cam):
            importance_full[start_idx:end_idx] += cam[:seg_len]
            counts[start_idx:end_idx] += 1.0

    mask = counts > 0
    importance_full[mask] /= counts[mask]

    return time_full, vol_full, importance_full


# ── Signal analysis helper ───────────────────────────────────────────


def get_signal_analysis(file_path: Path, patient_id: str = "uploaded", file_id: str = "uploaded") -> pd.DataFrame:
    """Load a file and return the raw DataFrame for signal analysis plots."""
    return load_patient_file(file_path, patient_id=patient_id, file_id=file_id)
