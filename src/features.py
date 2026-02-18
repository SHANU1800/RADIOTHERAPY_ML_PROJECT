"""
Feature engineering for breathing-curve ML: windowed and cycle-level stats.

Produces one row per window/segment with: volume stats (mean, std, min, max, range, change),
balloon valve fractions, patient switch fraction, gating encoding.
"""
from __future__ import annotations

from typing import Optional, Tuple

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

from src.labels import window_label_breath_hold, window_label_gating_ok


def build_windows(
    df: "pd.DataFrame",
    window_sec: float = 2.0,
    sample_rate_hz: float = 50,
    min_rows: int = 50,
) -> "pd.DataFrame":
    """
    Build fixed-length windows and compute per-window features + optional labels.

    df must have columns: Session Time, Volume (liters), Balloon Valve Status,
    Patient Switch, Gating Mode, and patient_id, file_id.
    """
    if pd is None or np is None:
        raise ImportError("pandas and numpy required")
    window_size = int(round(window_sec * sample_rate_hz))
    if window_size < 10:
        window_size = 10

    rows = []
    for (pid, fid), grp in df.groupby(["patient_id", "file_id"]):
        grp = grp.sort_values("Session Time").reset_index(drop=True)
        t = grp["Session Time"].values
        v = grp["Volume (liters)"].values
        balloon = grp["Balloon Valve Status"]
        pswitch = grp["Patient Switch"]
        gating = grp["Gating Mode"]

        n = len(grp)
        for start in range(0, n - window_size + 1, window_size):
            end = start + window_size
            vw = v[start:end]
            tw = t[start:end]
            if len(vw) < min_rows:
                continue
            # Volume features
            vol_mean = float(np.nanmean(vw))
            vol_std = float(np.nanstd(vw))
            if np.isnan(vol_std):
                vol_std = 0.0
            vol_min = float(np.nanmin(vw))
            vol_max = float(np.nanmax(vw))
            vol_range = vol_max - vol_min
            vol_change = float(vw[-1] - vw[0]) if len(vw) > 1 else 0.0
            # Rolling mean over window (smooth)
            vol_rolling_mean = float(np.nanmean(vw))

            balloon_w = balloon.iloc[start:end]
            pswitch_w = pswitch.iloc[start:end]
            gating_w = gating.iloc[start:end]

            balloon_numeric = pd.to_numeric(balloon_w, errors="coerce")
            frac_inflated = float((balloon_numeric == 4).sum() / len(balloon_w))
            frac_deflated = float((balloon_numeric == 1).sum() / len(balloon_w))
            pswitch_numeric = pd.to_numeric(pswitch_w, errors="coerce")
            frac_switch_on = float((pswitch_numeric == 1).sum() / len(pswitch_w))
            gating_str = gating_w.astype(str).str.strip().str.lower()
            frac_automated = float((gating_str == "automated").sum() / len(gating_w))

            # Labels for classification
            label_breath_hold = window_label_breath_hold(balloon_w)
            label_gating_ok = window_label_gating_ok(gating_w)

            rows.append({
                "patient_id": pid,
                "file_id": fid,
                "time_start": float(tw[0]),
                "time_end": float(tw[-1]),
                "vol_mean": vol_mean,
                "vol_std": vol_std,
                "vol_min": vol_min,
                "vol_max": vol_max,
                "vol_range": vol_range,
                "vol_change": vol_change,
                "vol_rolling_mean": vol_rolling_mean,
                "frac_balloon_inflated": frac_inflated,
                "frac_balloon_deflated": frac_deflated,
                "frac_patient_switch_on": frac_switch_on,
                "frac_gating_automated": frac_automated,
                "label_breath_hold": label_breath_hold,
                "label_gating_ok": label_gating_ok,
            })
    return pd.DataFrame(rows)


def get_feature_columns(include_labels: bool = False) -> list:
    """Column names to use as model features (numeric only)."""
    feats = [
        "vol_mean", "vol_std", "vol_min", "vol_max", "vol_range", "vol_change", "vol_rolling_mean",
        "frac_balloon_inflated", "frac_balloon_deflated", "frac_patient_switch_on", "frac_gating_automated",
    ]
    if include_labels:
        feats += ["label_breath_hold", "label_gating_ok"]
    return feats


def get_X_y(
    windows_df: "pd.DataFrame",
    task: str = "breath_hold",
) -> Tuple["pd.DataFrame", "pd.Series", "pd.Series"]:
    """
    Extract X (features), y (labels), and patient_id for train/test split.

    task: "breath_hold" | "gating_ok"
    """
    if pd is None:
        raise ImportError("pandas required")
    feature_cols = [
        "vol_mean", "vol_std", "vol_min", "vol_max", "vol_range", "vol_change", "vol_rolling_mean",
        "frac_balloon_inflated", "frac_balloon_deflated", "frac_patient_switch_on", "frac_gating_automated",
    ]
    if task == "breath_hold":
        label_col = "label_breath_hold"
    else:
        label_col = "label_gating_ok"
    X = windows_df[feature_cols].copy()
    X = X.fillna(0)
    y = windows_df[label_col]
    ids = windows_df["patient_id"]
    return X, y, ids
