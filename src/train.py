"""
Train classical ML models for breathing-pattern classification with patient-based split.

Usage (from project root):
  python -m src.train [--task breath_hold|gating_ok] [--dataset path]
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
except ImportError as e:
    raise ImportError("Install scikit-learn, pandas, numpy for training.") from e

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Import from project
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.load_data import load_all_patients
from src.features import build_windows, get_X_y


def patient_split(
    X: pd.DataFrame,
    y: pd.Series,
    patient_ids: pd.Series,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split by patient so no patient appears in more than one set.
    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    """
    unique_patients = patient_ids.unique()
    n = len(unique_patients)
    if n < 2:
        # Fallback: same patient, use row split
        from sklearn.model_selection import train_test_split as tts
        X_tr, X_te, y_tr, y_te = tts(X, y, test_size=1 - train_ratio, random_state=random_state)
        X_va = pd.DataFrame()
        y_va = pd.Series(dtype=float)
        return X_tr, X_va, X_te, y_tr, y_va, y_te

    np.random.seed(random_state)
    perm = np.random.permutation(unique_patients)
    n_train = max(1, int(n * train_ratio))
    n_val = max(0, int(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train

    train_patients = set(perm[:n_train])
    val_patients = set(perm[n_train : n_train + n_val]) if n_val else set()
    test_patients = set(perm[n_train + n_val :])

    train_idx = patient_ids.isin(train_patients)
    val_idx = patient_ids.isin(val_patients)
    test_idx = patient_ids.isin(test_patients)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = (X[val_idx], y[val_idx]) if val_idx.any() else (pd.DataFrame(), pd.Series(dtype=float))
    X_test, y_test = X[test_idx], y[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    scale: bool = True,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Any, Any]:
    """
    Train RandomForest, LogisticRegression, and (if available) XGBoost.
    Optionally SVM. Return metrics dict, best model, scaler.
    """
    if scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
        if X_val is not None and len(X_val) > 0:
            X_va = scaler.transform(X_val)
        else:
            X_va = None
    else:
        scaler = None
        X_tr = X_train.values
        X_te = X_test.values
        X_va = X_val.values if X_val is not None and len(X_val) > 0 else None

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=random_state),
        "SVM": SVC(kernel="rbf", random_state=random_state, probability=True),
    }
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(n_estimators=100, random_state=random_state)

    best_score = -1
    best_name = None
    best_model = None
    all_metrics: Dict[str, Dict[str, float]] = {}

    for name, clf in models.items():
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        all_metrics[name] = {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "f1": float(f1),
            "confusion_matrix": cm.tolist(),
        }
        if bal_acc > best_score:
            best_score = bal_acc
            best_name = name
            best_model = clf

    return {
        "models": all_metrics,
        "best_model": best_name,
        "best_balanced_accuracy": float(best_score),
    }, best_model, scaler


def main() -> None:
    parser = argparse.ArgumentParser(description="Train breathing-pattern classifiers")
    parser.add_argument("--task", default="breath_hold", choices=["breath_hold", "gating_ok"])
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "models")
    args = parser.parse_args()

    try:
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        print("Loading data...")
        df, _ = load_all_patients(args.dataset, include_session_ini=False)
        if len(df) == 0:
            print("ERROR: No data loaded. Check dataset path and that it contains .dat/.txt/.csv files.")
            sys.exit(1)
        print(f"Loaded {len(df)} rows, {df['patient_id'].nunique()} patients.")

        print("Building windows and features...")
        windows = build_windows(
            df,
            window_sec=args.window_sec,
            sample_rate_hz=50,
            min_rows=50,
        )
        if len(windows) == 0:
            print("ERROR: No windows produced. Check data length and min_rows (need enough rows per window).")
            sys.exit(1)
        print(f"Built {len(windows)} windows.")

        X, y, patient_ids = get_X_y(windows, task=args.task)
        X_train, X_val, X_test, y_train, y_val, y_test = patient_split(
            X, y, patient_ids,
            train_ratio=args.train_ratio,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=args.seed,
        )
        print(f"Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)} samples.")

        if len(X_train) == 0 or len(X_test) == 0:
            print("ERROR: Train or test set is empty. Need at least 2 patients or more data.")
            sys.exit(1)

        metrics, best_model, scaler = train_and_evaluate(
            X_train, y_train, X_test, y_test,
            X_val=X_val if len(X_val) > 0 else None,
            y_val=y_val if len(y_val) > 0 else None,
            scale=True,
            random_state=args.seed,
        )

        print("\nResults:")
        for name, m in metrics["models"].items():
            print(f"  {name}: acc={m['accuracy']:.4f}, bal_acc={m['balanced_accuracy']:.4f}, f1={m['f1']:.4f}")
        print(f"Best: {metrics['best_model']} (balanced_accuracy={metrics['best_balanced_accuracy']:.4f})")

        # Save test set predictions for ROC/PR curves in frontend
        X_te = scaler.transform(X_test) if scaler is not None else X_test.values
        y_pred = best_model.predict(X_te)
        proba_class1 = None
        if hasattr(best_model, "predict_proba"):
            proba = best_model.predict_proba(X_te)
            proba_class1 = proba[:, 1].tolist()
        test_predictions = {
            "task": args.task,
            "best_model": metrics["best_model"],
            "n_test": int(len(y_test)),
            "y_true": y_test.astype(int).tolist(),
            "y_pred": y_pred.astype(int).tolist(),
            "proba_class1": proba_class1,
        }
        with (out_dir / "test_predictions.json").open("w") as f:
            json.dump(test_predictions, f, indent=2)

        # Save
        with (out_dir / "metrics.json").open("w") as f:
            json.dump({"best_model": metrics["best_model"], "best_balanced_accuracy": metrics["best_balanced_accuracy"]}, f, indent=2)
        with (out_dir / "metrics_models.json").open("w") as f:
            json.dump(metrics["models"], f, indent=2)
        with (out_dir / "best_model.pkl").open("wb") as f:
            pickle.dump({"model": best_model, "scaler": scaler, "task": args.task}, f)
        print(f"Saved metrics and model to {out_dir}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
