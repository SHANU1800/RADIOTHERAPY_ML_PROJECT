"""
Train deep learning models for breathing-pattern classification.

Usage (from project root):
  python -m src.dl_train [--task breath_hold|gating_ok] [--model LSTM|CNN1D|CNN_LSTM|BiLSTM|GRU|Attention_LSTM|ResNet1D|all]
  python -m src.dl_train --multichannel         # train with 5-channel input
  python -m src.dl_train --overlap 0.5          # 50% overlapping windows
  python -m src.dl_train --augment              # add jitter augmentation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
        precision_recall_curve,
        average_precision_score,
    )
except ImportError as e:
    raise ImportError("numpy and scikit-learn are required.") from e

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.load_data import load_all_patients
from src.dl_features import build_dl_windows, dl_patient_split, MULTI_CHANNELS
from src.dl_models import MODEL_BUILDERS, get_model


def _import_tf():
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        print("ERROR: TensorFlow is required. Install with: pip install tensorflow")
        sys.exit(1)


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights for binary labels."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {}
    for cls, cnt in zip(classes, counts):
        weights[int(cls)] = total / (len(classes) * cnt)
    return weights


def train_single_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    out_dir: Path = PROJECT_ROOT / "models",
    task: str = "breath_hold",
    use_class_weights: bool = True,
) -> Dict[str, Any]:
    """
    Train one DL model, evaluate with comprehensive metrics, and save artifacts.
    """
    tf = _import_tf()

    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"\n{'='*60}")
    print(f"Training {model_name} | input_shape={input_shape} | task={task}")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Channels: {input_shape[1]}")

    # Class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"  Class distribution (train): {dict(zip(unique.tolist(), counts.tolist()))}")
    print(f"{'='*60}")

    class_weights = compute_class_weights(y_train) if use_class_weights else None
    if class_weights:
        print(f"  Class weights: {class_weights}")

    model = get_model(model_name, input_shape, learning_rate=learning_rate)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    val_data = (X_val, y_val) if len(X_val) > 0 else None

    history = model.fit(
        X_train, y_train,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # ── Evaluation ────────────────────────────────────────────────────
    y_pred_prob = model.predict(X_test, batch_size=batch_size).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Specificity (true negative rate)
    if len(cm) == 2 and len(cm[0]) == 2:
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    else:
        specificity = 0.0
        sensitivity = rec

    try:
        auc = float(roc_auc_score(y_test, y_pred_prob))
    except ValueError:
        auc = 0.0

    # ROC curve data
    try:
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_prob)
        roc_data = {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
        }
    except ValueError:
        roc_data = None

    # PR curve data
    try:
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, y_pred_prob)
        avg_prec = float(average_precision_score(y_test, y_pred_prob))
        pr_data = {
            "precision": [float(x) for x in pr_precision],
            "recall": [float(x) for x in pr_recall],
            "average_precision": avg_prec,
        }
    except ValueError:
        pr_data = None
        avg_prec = 0.0

    metrics = {
        "model_name": model_name,
        "task": task,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "mcc": mcc,
        "roc_auc": auc,
        "average_precision": avg_prec,
        "confusion_matrix": cm,
        "epochs_trained": len(history.history.get("loss", [])),
        "input_shape": list(input_shape),
        "n_channels": input_shape[1],
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "class_weights_used": class_weights is not None,
    }

    print(f"\n  Results for {model_name}:")
    print(f"    Accuracy:          {acc:.4f}")
    print(f"    Balanced Accuracy: {bal_acc:.4f}")
    print(f"    F1 Score:          {f1:.4f}")
    print(f"    Precision:         {prec:.4f}")
    print(f"    Recall/Sensitiv.:  {rec:.4f}")
    print(f"    Specificity:       {specificity:.4f}")
    print(f"    MCC:               {mcc:.4f}")
    print(f"    ROC-AUC:           {auc:.4f}")
    print(f"    Avg Precision:     {avg_prec:.4f}")

    # ── Save artifacts ────────────────────────────────────────────────
    prefix = f"dl_{model_name.lower()}_{task}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{prefix}_model.keras"
    model.save(model_path)
    print(f"  Model saved to {model_path}")

    history_data = {}
    for k, v in history.history.items():
        history_data[k] = [float(x) for x in v]
    with (out_dir / f"{prefix}_history.json").open("w") as f:
        json.dump(history_data, f, indent=2)

    with (out_dir / f"{prefix}_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    test_preds = {
        "task": task,
        "model_name": model_name,
        "n_test": int(len(y_test)),
        "y_true": y_test.astype(int).tolist(),
        "y_pred": y_pred.tolist(),
        "y_pred_prob": y_pred_prob.tolist(),
    }
    if roc_data:
        test_preds["roc_curve"] = roc_data
    if pr_data:
        test_preds["pr_curve"] = pr_data
    with (out_dir / f"{prefix}_test_predictions.json").open("w") as f:
        json.dump(test_preds, f, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DL breathing-pattern classifiers")
    parser.add_argument("--task", default="breath_hold", choices=["breath_hold", "gating_ok"])
    parser.add_argument("--model", default="all", choices=list(MODEL_BUILDERS.keys()) + ["all"])
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument(
        "--channels", nargs="+", default=None,
        help="Column names to use as input channels. Omit to use Volume only.",
    )
    parser.add_argument(
        "--multichannel", action="store_true",
        help="Use all 5 derived channels (Volume, derivative, 2nd derivative, balloon, envelope)",
    )
    parser.add_argument(
        "--overlap", type=float, default=0.5,
        help="Window overlap fraction (0.0 = no overlap, 0.5 = 50%% overlap)",
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Enable jitter augmentation for training windows",
    )
    parser.add_argument(
        "--no-class-weights", action="store_true",
        help="Disable automatic class weight balancing",
    )
    args = parser.parse_args()

    tf = _import_tf()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    if args.multichannel:
        channels = list(MULTI_CHANNELS)
    elif args.channels:
        channels = args.channels
    else:
        channels = ["Volume (liters)"]

    print(f"Configuration:")
    print(f"  Task:        {args.task}")
    print(f"  Channels:    {channels}")
    print(f"  Window:      {args.window_sec}s, overlap={args.overlap}")
    print(f"  Augment:     {args.augment}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Class weights: {not args.no_class_weights}")
    print()

    print("Loading data...")
    df, _ = load_all_patients(args.dataset, include_session_ini=False)
    if len(df) == 0:
        print("ERROR: No data loaded.")
        sys.exit(1)
    print(f"Loaded {len(df)} rows, {df['patient_id'].nunique()} patients.")

    print("Building DL windows...")
    X, y_bh, y_gk, patient_ids, meta_df = build_dl_windows(
        df,
        channels=channels,
        window_sec=args.window_sec,
        sample_rate_hz=50,
        min_rows=50,
        normalize=True,
        overlap=args.overlap,
        augment=args.augment,
        compute_extras=True,
    )
    print(f"Built {len(X)} windows, shape per window: {X.shape[1:]}")

    if len(X) == 0:
        print("ERROR: No windows produced.")
        sys.exit(1)

    y = y_bh if args.task == "breath_hold" else y_gk

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"Label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    X_train, X_val, X_test, y_train, y_val, y_test = dl_patient_split(
        X, y, patient_ids,
        train_ratio=0.7, val_ratio=0.15, random_state=args.seed,
    )
    print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    if len(X_train) == 0 or len(X_test) == 0:
        print("ERROR: Train or test set is empty.")
        sys.exit(1)

    model_names = list(MODEL_BUILDERS.keys()) if args.model == "all" else [args.model]
    all_metrics: Dict[str, Any] = {}

    for name in model_names:
        try:
            m = train_single_model(
                name, X_train, y_train, X_val, y_val, X_test, y_test,
                epochs=args.epochs, batch_size=args.batch_size,
                learning_rate=args.lr, out_dir=args.out_dir, task=args.task,
                use_class_weights=not args.no_class_weights,
            )
            all_metrics[name] = m
        except Exception as e:
            print(f"ERROR training {name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_metrics:
        best_name = max(all_metrics, key=lambda k: all_metrics[k]["balanced_accuracy"])
        summary = {
            "task": args.task,
            "channels": channels,
            "window_sec": args.window_sec,
            "overlap": args.overlap,
            "augment": args.augment,
            "class_weights": not args.no_class_weights,
            "n_windows_total": int(len(X)),
            "models": all_metrics,
            "best_model": best_name,
            "best_balanced_accuracy": all_metrics[best_name]["balanced_accuracy"],
        }
        with (args.out_dir / f"dl_summary_{args.task}.json").open("w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"SUMMARY — Task: {args.task}")
        print(f"{'='*60}")
        header = f"{'Model':<20} {'Acc':>7} {'BalAcc':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'Spec':>7} {'MCC':>7} {'AUC':>7}"
        print(header)
        print("-" * len(header))
        for name, m in all_metrics.items():
            print(
                f"{name:<20} {m['accuracy']:>7.4f} {m['balanced_accuracy']:>7.4f} "
                f"{m['f1']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} "
                f"{m['specificity']:>7.4f} {m['mcc']:>7.4f} {m['roc_auc']:>7.4f}"
            )
        print(f"\nBest DL model: {best_name} (bal_acc={summary['best_balanced_accuracy']:.4f})")
        print(f"All results saved to {args.out_dir}")
        print(f"{'='*60}")
    else:
        print("ERROR: No models trained successfully.")
        sys.exit(1)


if __name__ == "__main__":
    main()
