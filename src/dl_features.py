"""
Deep-learning feature extraction: multi-channel time-series windows for
sequence models (LSTM, CNN, Attention, etc.).

Channels produced per window (configurable):
  1. Volume (liters)              — raw breathing signal
  2. Volume derivative (dV/dt)    — rate of change (breathing speed)
  3. Volume 2nd derivative        — acceleration (rhythm changes)
  4. Balloon Valve Status         — numeric hardware signal (1-5)
  5. Volume envelope              — rolling amplitude envelope
  6. breathing_rate              — dV/dT (Session Time) — matches dataset notebooks
  7. vol_smoothed                 — rolling mean of volume (Smoothed Volume) — matches notebooks

Aligns with dataset Jupyter notebooks:
  - Abdul Rehaman: Breathing Rate = Volume.diff()/SessionTime.diff(), Smoothed Volume = rolling(5).mean()
  - Benny Martis: Volume Change = diff(), Volume MA = rolling(5).mean()
  - Afsana Sadhick: Balloon Valve Status, time-interval anomalies
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None  # type: ignore
    np = None  # type: ignore

from src.labels import window_label_breath_hold, window_label_gating_ok


DEFAULT_CHANNELS = ["Volume (liters)"]

MULTI_CHANNELS = [
    "Volume (liters)",
    "vol_derivative",
    "vol_derivative2",
    "balloon_numeric",
    "vol_envelope",
]

# Channel set that matches dataset notebooks (Abdul Rehaman, Benny Martis): Volume, Breathing Rate, Smoothed Volume
NOTEBOOK_ALIGNED_CHANNELS = [
    "Volume (liters)",
    "breathing_rate",   # dV/dT — same as notebook "Breathing Rate"
    "vol_smoothed",     # rolling mean — same as notebook "Smoothed Volume" / "Volume MA"
    "vol_derivative2",  # acceleration / "Volume Change" second order
    "balloon_numeric",
]


def _add_derived_channels(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Add derived signal channels to the DataFrame (returns copy).

    Derived channels (aligned with dataset notebooks where noted):
    - vol_derivative:     first difference of volume (per sample)
    - vol_derivative2:    second difference (Benny Martis "Volume Change" style)
    - balloon_numeric:   Balloon Valve Status as float 1-5 (Afsana Sadhick)
    - vol_envelope:       rolling std — amplitude envelope
    - breathing_rate:     Volume.diff() / SessionTime.diff() (Abdul Rehaman "Breathing Rate")
    - vol_smoothed:       rolling mean window=5 (Abdul Rehaman "Smoothed Volume", Benny "Volume MA")
    """
    df = df.copy()
    vol = pd.to_numeric(df["Volume (liters)"], errors="coerce").fillna(0.0)

    df["vol_derivative"] = vol.diff().fillna(0.0)
    df["vol_derivative2"] = df["vol_derivative"].diff().fillna(0.0)

    if "Balloon Valve Status" in df.columns:
        df["balloon_numeric"] = pd.to_numeric(
            df["Balloon Valve Status"], errors="coerce"
        ).fillna(0.0)
    else:
        df["balloon_numeric"] = 0.0

    window = 25  # 0.5 s at 50 Hz
    df["vol_envelope"] = vol.rolling(window=window, min_periods=1, center=True).std().fillna(0.0)

    # Notebook-aligned: Breathing Rate = dV/dT (Abdul Rehaman Untitled1)
    if "Session Time" in df.columns:
        st = pd.to_numeric(df["Session Time"], errors="coerce").fillna(0.0)
        dt = st.diff()
        dvol = vol.diff()
        # Avoid division by zero: where dt==0 use 0
        br = np.where(np.abs(dt) > 1e-9, dvol / dt, 0.0)
        df["breathing_rate"] = pd.Series(br, index=df.index).fillna(0.0)
    else:
        df["breathing_rate"] = df["vol_derivative"]

    # Notebook-aligned: Smoothed Volume (Abdul Rehaman rolling 5, Benny Martis "Volume MA")
    df["vol_smoothed"] = vol.rolling(window=5, min_periods=1, center=True).mean().fillna(0.0)

    return df


def compute_signal_stats(segment: "np.ndarray") -> Dict[str, float]:
    """
    Compute descriptive statistics of a 1-D signal segment.
    Useful for per-window analysis / interpretability.
    """
    if np is None:
        raise ImportError("numpy required")
    if len(segment) == 0:
        return {}
    return {
        "mean": float(np.mean(segment)),
        "std": float(np.std(segment)),
        "min": float(np.min(segment)),
        "max": float(np.max(segment)),
        "range": float(np.ptp(segment)),
        "skewness": float(_safe_skew(segment)),
        "kurtosis": float(_safe_kurtosis(segment)),
        "zero_crossing_rate": float(_zero_crossing_rate(segment)),
        "rms": float(np.sqrt(np.mean(segment ** 2))),
        "energy": float(np.sum(segment ** 2)),
    }


def _safe_skew(x):
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-10:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _safe_kurtosis(x):
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-10:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def _zero_crossing_rate(x):
    if len(x) < 2:
        return 0.0
    signs = np.sign(x - np.mean(x))
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return float(crossings / (len(x) - 1))


def compute_spectral_features(segment: "np.ndarray", sample_rate: float = 50.0) -> Dict[str, float]:
    """
    Compute frequency-domain features of a signal segment via FFT.
    """
    if np is None:
        raise ImportError("numpy required")
    n = len(segment)
    if n < 4:
        return {"dominant_freq": 0.0, "spectral_energy": 0.0, "spectral_centroid": 0.0, "bandwidth": 0.0}

    windowed = segment * np.hanning(n)
    fft_vals = np.fft.rfft(windowed)
    magnitudes = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    total_energy = float(np.sum(magnitudes ** 2))
    if total_energy < 1e-12:
        return {"dominant_freq": 0.0, "spectral_energy": 0.0, "spectral_centroid": 0.0, "bandwidth": 0.0}

    dominant_idx = np.argmax(magnitudes[1:]) + 1
    dominant_freq = float(freqs[dominant_idx])

    spectral_centroid = float(np.sum(freqs * magnitudes) / np.sum(magnitudes))
    bandwidth = float(np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitudes) / np.sum(magnitudes)))

    return {
        "dominant_freq": dominant_freq,
        "spectral_energy": total_energy,
        "spectral_centroid": spectral_centroid,
        "bandwidth": bandwidth,
    }


def build_dl_windows(
    df: "pd.DataFrame",
    channels: Optional[List[str]] = None,
    window_sec: float = 2.0,
    sample_rate_hz: float = 50,
    min_rows: int = 50,
    normalize: bool = True,
    overlap: float = 0.5,
    augment: bool = False,
    compute_extras: bool = False,
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray", "pd.DataFrame"]:
    """
    Build fixed-length windows of multi-channel time-series data for deep learning.

    Parameters
    ----------
    df : DataFrame with Session Time, Volume (liters), patient_id, file_id
    channels : list of column names to use as input channels.
               Use MULTI_CHANNELS for full multi-channel pipeline.
               Defaults to ["Volume (liters)"] for backward compat.
    window_sec : window duration in seconds.
    sample_rate_hz : expected sampling rate.
    min_rows : minimum rows for a valid window.
    normalize : per-window zero-mean unit-variance normalization per channel.
    overlap : fractional overlap between consecutive windows (0.0 = no overlap, 0.5 = 50%).
    augment : if True, add jittered copies of each window for training augmentation.
    compute_extras : if True, compute per-window signal stats and spectral features.

    Returns
    -------
    X : ndarray of shape (n_windows, window_size, n_channels)
    y_breath_hold : ndarray of shape (n_windows,) with 0/1 labels
    y_gating_ok : ndarray of shape (n_windows,) with 0/1 labels
    patient_ids : ndarray of shape (n_windows,) with patient id strings
    meta_df : DataFrame with per-window metadata
    """
    if pd is None or np is None:
        raise ImportError("pandas and numpy are required")

    if channels is None:
        channels = list(DEFAULT_CHANNELS)

    required = ["Session Time", "Volume (liters)", "patient_id", "file_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"build_dl_windows requires columns: {missing}")

    if len(df) == 0:
        n_ch = len(channels)
        ws = int(round(window_sec * sample_rate_hz))
        return (
            np.empty((0, ws, n_ch), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=object),
            pd.DataFrame(),
        )

    needs_derived = any(ch in (
        "vol_derivative", "vol_derivative2", "balloon_numeric", "vol_envelope",
        "breathing_rate", "vol_smoothed",
    ) for ch in channels)
    if needs_derived:
        df = _add_derived_channels(df)

    optional_cols = ["Balloon Valve Status", "Patient Switch", "Gating Mode"]
    missing_opt = [c for c in optional_cols if c not in df.columns]
    if missing_opt:
        df = df.copy()
        for col in missing_opt:
            df[col] = ""

    for ch in channels:
        if ch not in df.columns:
            raise ValueError(f"Channel '{ch}' not found in DataFrame columns: {list(df.columns)}")

    window_size = int(round(window_sec * sample_rate_hz))
    if window_size < 10:
        window_size = 10
    stride = max(1, int(window_size * (1.0 - overlap)))

    X_list: list = []
    y_bh_list: list = []
    y_gk_list: list = []
    pid_list: list = []
    meta_rows: list = []

    for (pid, fid), grp in df.groupby(["patient_id", "file_id"]):
        grp = grp.sort_values("Session Time").reset_index(drop=True)
        t = grp["Session Time"].values

        ch_arrays = []
        for ch in channels:
            vals = pd.to_numeric(grp[ch], errors="coerce").fillna(0.0).values.astype(np.float32)
            ch_arrays.append(vals)
        ch_matrix = np.stack(ch_arrays, axis=-1)  # (n_rows, n_channels)

        balloon = grp["Balloon Valve Status"]
        gating = grp["Gating Mode"]

        n = len(grp)
        for start in range(0, n - window_size + 1, stride):
            end_idx = start + window_size
            window_data = ch_matrix[start:end_idx].copy()

            if len(window_data) < min_rows:
                continue

            bh_label = window_label_breath_hold(balloon.iloc[start:end_idx])
            gk_label = window_label_gating_ok(gating.iloc[start:end_idx])

            row_meta = {
                "patient_id": pid,
                "file_id": fid,
                "time_start": float(t[start]),
                "time_end": float(t[end_idx - 1]),
            }

            if compute_extras:
                vol_seg = window_data[:, 0]
                sig_stats = compute_signal_stats(vol_seg)
                spec_feats = compute_spectral_features(vol_seg, sample_rate_hz)
                row_meta.update({f"sig_{k}": v for k, v in sig_stats.items()})
                row_meta.update({f"spec_{k}": v for k, v in spec_feats.items()})

            if normalize:
                mean = window_data.mean(axis=0, keepdims=True)
                std = window_data.std(axis=0, keepdims=True)
                std = np.where(std < 1e-8, 1.0, std)
                window_data = (window_data - mean) / std

            X_list.append(window_data)
            y_bh_list.append(bh_label)
            y_gk_list.append(gk_label)
            pid_list.append(pid)
            meta_rows.append(row_meta)

            if augment:
                jitter = np.random.normal(0, 0.02, window_data.shape).astype(np.float32)
                X_list.append(window_data + jitter)
                y_bh_list.append(bh_label)
                y_gk_list.append(gk_label)
                pid_list.append(pid)
                meta_rows.append(row_meta.copy())

    if not X_list:
        n_ch = len(channels)
        return (
            np.empty((0, window_size, n_ch), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=object),
            pd.DataFrame(),
        )

    X = np.stack(X_list).astype(np.float32)
    y_breath_hold = np.array(y_bh_list, dtype=np.int32)
    y_gating_ok = np.array(y_gk_list, dtype=np.int32)
    patient_ids = np.array(pid_list, dtype=object)
    meta_df = pd.DataFrame(meta_rows)

    return X, y_breath_hold, y_gating_ok, patient_ids, meta_df


def dl_patient_split(
    X: "np.ndarray",
    y: "np.ndarray",
    patient_ids: "np.ndarray",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[
    "np.ndarray", "np.ndarray", "np.ndarray",
    "np.ndarray", "np.ndarray", "np.ndarray",
]:
    """
    Split by patient so no patient appears in more than one set.
    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    """
    if np is None:
        raise ImportError("numpy is required")

    unique_patients = np.unique(patient_ids)
    n = len(unique_patients)

    if n < 2:
        n_total = len(X)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        return (
            X[:n_train], X[n_train:n_train + n_val], X[n_train + n_val:],
            y[:n_train], y[n_train:n_train + n_val], y[n_train + n_val:],
        )

    rng = np.random.RandomState(random_state)
    perm = rng.permutation(unique_patients)
    n_train = max(1, int(n * train_ratio))
    n_val = max(0, int(n * val_ratio))
    if n_train + n_val > n:
        n_val = n - n_train

    train_patients = set(perm[:n_train])
    val_patients = set(perm[n_train:n_train + n_val]) if n_val else set()
    test_patients = set(perm[n_train + n_val:])

    train_mask = np.array([p in train_patients for p in patient_ids])
    val_mask = np.array([p in val_patients for p in patient_ids])
    test_mask = np.array([p in test_patients for p in patient_ids])

    return (
        X[train_mask], X[val_mask], X[test_mask],
        y[train_mask], y[val_mask], y[test_mask],
    )
