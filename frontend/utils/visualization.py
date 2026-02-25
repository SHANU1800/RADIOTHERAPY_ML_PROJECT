"""
Visualization helpers using Plotly.
"""
from typing import Optional, List, Tuple, Dict, Any

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd
    import numpy as np
except ImportError:
    go = None
    px = None
    pd = None
    np = None


# Color scheme
COLOR_BREATH_HOLD = "#2ecc71"  # Green
COLOR_FREE_BREATHING = "#e74c3c"  # Red
COLOR_NEUTRAL = "#95a5a6"  # Gray

MAX_PLOT_POINTS = 15000  # Keep plots responsive
MAX_PREDICTION_RECTS = 150  # Too many shapes freeze the browser


def _sample_for_plot(df: "pd.DataFrame", max_points: int = MAX_PLOT_POINTS) -> "pd.DataFrame":
    """Sample dataframe to max_points for responsive plotting."""
    if pd is None:
        return df
    if len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()


def plot_breathing_curve(
    df: "pd.DataFrame",
    predictions: Optional["pd.DataFrame"] = None,
    show_balloon: bool = True,
    show_gating: bool = False,
    show_rolling_mean: bool = False,
    rolling_window: int = 50,
    height: int = 500,
    subtitle: Optional[str] = None,
) -> "go.Figure":
    """
    Plot breathing curve (time vs volume) with optional predictions and overlays.
    
    df: DataFrame with Session Time, Volume (liters), optionally Balloon Valve Status, Gating Mode
    predictions: DataFrame with time_start, time_end, prediction columns (for window coloring)
    subtitle: Optional caption (e.g. window size, file name) for research context.
    """
    if go is None or pd is None:
        raise ImportError("plotly and pandas required")
    
    plot_df = _sample_for_plot(df, MAX_PLOT_POINTS)
    
    fig = go.Figure()
    
    # Main volume curve
    fig.add_trace(go.Scatter(
        x=plot_df["Session Time"],
        y=plot_df["Volume (liters)"],
        mode="lines",
        name="Volume",
        line=dict(color="#3498db", width=1.5),
        hovertemplate="Time: %{x:.2f}s<br>Volume: %{y:.3f}L<extra></extra>",
    ))
    
    # Optional rolling mean
    if show_rolling_mean and len(plot_df) >= rolling_window:
        roll = plot_df["Volume (liters)"].rolling(window=rolling_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=plot_df["Session Time"],
            y=roll,
            mode="lines",
            name=f"Rolling mean (w={rolling_window})",
            line=dict(color="#9b59b6", width=2, dash="dash"),
            hovertemplate="Time: %{x:.2f}s<br>Rolling mean: %{y:.3f}L<extra></extra>",
        ))
    
    # Add prediction windows if provided
    if predictions is not None and len(predictions) > 0 and "prediction" in predictions.columns:
        vol_min = df["Volume (liters)"].min()
        vol_max = df["Volume (liters)"].max()
        vol_range = vol_max - vol_min if vol_max > vol_min else 1.0
        
        pred_sample = predictions if len(predictions) <= MAX_PREDICTION_RECTS else predictions.iloc[::max(1, len(predictions) // MAX_PREDICTION_RECTS)]
        for _, row in pred_sample.iterrows():
            color = COLOR_BREATH_HOLD if row["prediction"] == 1 else COLOR_FREE_BREATHING
            fig.add_shape(
                type="rect",
                x0=row.get("time_start", 0),
                y0=vol_min - 0.1 * vol_range,
                x1=row.get("time_end", df["Session Time"].max()),
                y1=vol_max + 0.1 * vol_range,
                fillcolor=color,
                opacity=0.2,
                layer="below",
                line_width=0,
            )
    
    # Balloon valve overlay (optional) - use plot_df for consistency
    if show_balloon and "Balloon Valve Status" in plot_df.columns:
        balloon_numeric = pd.to_numeric(plot_df["Balloon Valve Status"], errors="coerce")
        inflated_mask = balloon_numeric == 4
        if inflated_mask.any():
            sub = plot_df.loc[inflated_mask]
            fig.add_trace(go.Scatter(
                x=sub["Session Time"],
                y=sub["Volume (liters)"],
                mode="markers",
                name="Balloon Inflated",
                marker=dict(color=COLOR_BREATH_HOLD, size=6, symbol="diamond"),
                hovertemplate="Time: %{x:.2f}s<br>Volume: %{y:.3f}L<br>Status: Inflated<extra></extra>",
            ))
    
    title = "Breathing Pattern Over Time"
    if subtitle:
        title += f"<br><sub>{subtitle}</sub>"
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Volume (L)",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        hovermode="closest",
        height=height,
    )
    
    return fig


def plot_confusion_matrix(
    cm: List[List[int]],
    model_name: str,
    labels: List[str] = None,
    subtitle: Optional[str] = None,
) -> "go.Figure":
    """Plot confusion matrix as heatmap with optional subtitle (e.g. task, test set size)."""
    if px is None or np is None:
        raise ImportError("plotly and numpy required")
    
    if labels is None:
        labels = ["Free-breathing", "Breath-hold"]
    
    cm_array = np.array(cm)
    fig = px.imshow(
        cm_array,
        labels=dict(x="Predicted", y="Actual"),
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        text_auto="d",
    )
    title = f"Confusion Matrix - {model_name}"
    if subtitle:
        title += f"<br><sub>{subtitle}</sub>"
    fig.update_layout(
        title=title,
        height=400,
    )
    return fig


def plot_feature_importance(model, feature_names: List[str], show_pct: bool = True) -> "go.Figure":
    """Plot feature importance for tree-based models with horizontal grid and optional normalized %."""
    if go is None or np is None:
        raise ImportError("plotly and numpy required")
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not support feature importance")
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    total = sorted_importances.sum()
    pcts = (100.0 * sorted_importances / total) if total > 0 else np.zeros_like(sorted_importances)
    
    fig = go.Figure(data=[
        go.Bar(
            x=sorted_importances,
            y=sorted_features,
            orientation="h",
            marker_color="#3498db",
            text=[f"{p:.1f}%" for p in pcts] if show_pct else None,
            textposition="outside",
        )
    ])
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        height=max(400, len(feature_names) * 30),
    )
    return fig


def plot_roc_curve(fpr: List[float], tpr: List[float], auc: float, model_name: str) -> "go.Figure":
    """Plot ROC curve (FPR vs TPR) with AUC in title."""
    if go is None or np is None:
        raise ImportError("plotly and numpy required")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"AUC = {auc:.3f}",
        line=dict(color="#3498db", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(color="#95a5a6", width=1, dash="dash"),
    ))
    fig.update_layout(
        title=f"ROC Curve - {model_name} (AUC = {auc:.3f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(showgrid=True, range=[0, 1]),
        yaxis=dict(showgrid=True, range=[0, 1]),
        height=400,
    )
    return fig


def plot_precision_recall_curve(
    precision: List[float], recall: List[float], ap: float, model_name: str
) -> "go.Figure":
    """Plot Precision-Recall curve with average precision in title."""
    if go is None or np is None:
        raise ImportError("plotly and numpy required")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode="lines",
        name=f"AP = {ap:.3f}",
        line=dict(color="#e74c3c", width=2),
    ))
    fig.update_layout(
        title=f"Precision-Recall Curve - {model_name} (AP = {ap:.3f})",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(showgrid=True, range=[0, 1]),
        yaxis=dict(showgrid=True, range=[0, 1.02]),
        height=400,
    )
    return fig


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]]) -> "go.Figure":
    """Bar chart comparing Accuracy, Balanced Accuracy, F1 per model for research."""
    if go is None or np is None:
        raise ImportError("plotly and numpy required")
    models = list(metrics_dict.keys())
    metrics_names = ["accuracy", "balanced_accuracy", "f1"]
    fig = go.Figure()
    colors = ["#3498db", "#2ecc71", "#9b59b6"]
    for i, mname in enumerate(metrics_names):
        vals = [metrics_dict[m].get(mname, 0) for m in models]
        fig.add_trace(go.Bar(
            name=mname.replace("_", " ").title(),
            x=models,
            y=vals,
            marker_color=colors[i % len(colors)],
        ))
    fig.update_layout(
        title="Model Comparison (Accuracy, Balanced Accuracy, F1)",
        barmode="group",
        xaxis_title="Model",
        yaxis_title="Score",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True, range=[0, 1.02]),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_bar_counts(counts: "pd.Series", title: str, xaxis_title: str = "Category") -> "go.Figure":
    """Plotly bar chart for count data (e.g. Balloon Valve Status, Gating Mode)."""
    if go is None or pd is None:
        raise ImportError("plotly and pandas required")
    df = counts.reset_index()
    df.columns = ["category", "count"]
    fig = go.Figure(data=[
        go.Bar(x=df["category"], y=df["count"], marker_color="#3498db")
    ])
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="Count",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        height=350,
    )
    return fig


def plot_volume_histogram(volumes: "pd.Series", title: str = "Volume (L) Distribution", bins: int = 50) -> "go.Figure":
    """Histogram of volume values for research (e.g. volume distribution in a session)."""
    if go is None or np is None:
        raise ImportError("plotly and numpy required")
    v = volumes.dropna()
    if len(v) == 0:
        fig = go.Figure()
        fig.update_layout(title=title, height=350)
        return fig
    fig = go.Figure(data=[go.Histogram(x=v, nbinsx=bins, marker_color="#3498db")])
    fig.update_layout(
        title=title,
        xaxis_title="Volume (L)",
        yaxis_title="Count",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        height=350,
    )
    return fig


def plot_confidence_histogram(confidence: "pd.Series", title: str = "Confidence distribution", bins: int = 20) -> "go.Figure":
    """Histogram of prediction confidence for research."""
    if go is None or np is None:
        raise ImportError("plotly and numpy required")
    c = confidence.dropna()
    if len(c) == 0:
        fig = go.Figure()
        fig.update_layout(title=title, height=350)
        return fig
    fig = go.Figure(data=[go.Histogram(x=c, nbinsx=bins, marker_color="#9b59b6")])
    fig.update_layout(
        title=title,
        xaxis_title="Confidence",
        yaxis_title="Count",
        xaxis=dict(showgrid=True, range=[0, 1.02]),
        yaxis=dict(showgrid=True),
        height=350,
    )
    return fig


def plot_prediction_donut(
    count_1: int, count_0: int,
    label_1: str = "Breath-hold", label_0: str = "Free-breathing",
    color_1: str = COLOR_BREATH_HOLD, color_0: str = COLOR_FREE_BREATHING,
    title: str = "Prediction distribution",
) -> "go.Figure":
    """Donut chart of class distribution (easy to see at a glance)."""
    if go is None or np is None:
        raise ImportError("plotly and numpy required")
    total = count_1 + count_0
    if total == 0:
        fig = go.Figure()
        fig.update_layout(title=title, height=320)
        return fig
    fig = go.Figure(data=[go.Pie(
        labels=[label_1, label_0],
        values=[count_1, count_0],
        hole=0.55,
        marker=dict(colors=[color_1, color_0]),
        textinfo="label+percent",
        textposition="outside",
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    )])
    fig.update_layout(
        title=title,
        height=320,
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.05),
        margin=dict(t=50, b=40),
    )
    return fig


def plot_confidence_over_time(
    windows_df: "pd.DataFrame",
    title: str = "Confidence over time",
    time_col: str = "time_start",
    height: int = 320,
) -> "go.Figure":
    """Line plot of prediction confidence by window (time or index). Easy to spot low-confidence regions."""
    if go is None or pd is None:
        raise ImportError("plotly and pandas required")
    if "confidence" not in windows_df.columns or len(windows_df) == 0:
        fig = go.Figure()
        fig.update_layout(title=title, height=height)
        return fig
    x = windows_df[time_col] if time_col in windows_df.columns else windows_df.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=windows_df["confidence"],
        mode="lines+markers",
        line=dict(color="#3498db", width=2),
        marker=dict(size=4),
        name="Confidence",
        hovertemplate="%{x:.2f}s<br>Confidence: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0.7, line_dash="dash", line_color="#e74c3c", annotation_text="Low-conf threshold (0.7)")
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)" if time_col == "time_start" else "Window",
        yaxis_title="Confidence",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True, range=[0, 1.02]),
        height=height,
    )
    return fig


def plot_confidence_buckets_bar(
    high: int, medium: int, low: int,
    title: str = "Confidence levels",
    height: int = 280,
) -> "go.Figure":
    """Bar chart of High / Medium / Low confidence counts. Very easy to notice at a glance."""
    if go is None or np is None:
        raise ImportError("plotly and numpy required")
    fig = go.Figure(data=[
        go.Bar(x=["High (≥0.9)", "Medium (0.7–0.9)", "Low (<0.7)"], y=[high, medium, low],
               marker_color=["#2ecc71", "#f39c12", "#e74c3c"]),
    ])
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="Number of windows",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True),
        height=height,
    )
    return fig


def plot_prediction_timeline(
    windows_df: "pd.DataFrame",
    label_1: str = "Breath-hold",
    label_0: str = "Free-breathing",
    title: str = "Prediction timeline",
    height: int = 200,
) -> "go.Figure":
    """Horizontal colored-bar timeline showing each window's prediction across time."""
    if go is None or pd is None:
        raise ImportError("plotly and pandas required")
    if len(windows_df) == 0 or "prediction" not in windows_df.columns:
        fig = go.Figure()
        fig.update_layout(title=title, height=height)
        return fig

    sample = windows_df if len(windows_df) <= MAX_PREDICTION_RECTS else windows_df.iloc[
        :: max(1, len(windows_df) // MAX_PREDICTION_RECTS)
    ]

    fig = go.Figure()
    for pred_val, label, color in [(1, label_1, COLOR_BREATH_HOLD), (0, label_0, COLOR_FREE_BREATHING)]:
        subset = sample[sample["prediction"] == pred_val]
        if len(subset) == 0:
            continue
        starts = subset["time_start"].values if "time_start" in subset.columns else np.arange(len(subset))
        ends = subset["time_end"].values if "time_end" in subset.columns else starts + 1
        confs = subset["confidence"].values if "confidence" in subset.columns else np.ones(len(subset))
        widths = ends - starts
        fig.add_trace(go.Bar(
            x=widths,
            y=["Prediction"] * len(subset),
            base=starts,
            orientation="h",
            marker=dict(color=color, opacity=[max(0.35, float(c)) for c in confs]),
            name=label,
            showlegend=True,
            hovertemplate=[
                f"{label}<br>Time: {s:.1f}–{e:.1f} s<br>Confidence: {c:.2f}<extra></extra>"
                for s, e, c in zip(starts, ends, confs)
            ],
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis=dict(showticklabels=False),
        barmode="stack",
        height=height,
        margin=dict(l=20, r=20, t=40, b=30),
    )
    return fig


def plot_volume_features_radar(
    windows_df: "pd.DataFrame",
    title: str = "Average feature profile",
    height: int = 350,
) -> "go.Figure":
    """Radar (polar) chart of mean feature values across all windows for quick profile view."""
    if go is None or pd is None or np is None:
        raise ImportError("plotly, pandas, numpy required")
    feature_cols = ["vol_mean", "vol_std", "vol_range", "vol_change",
                    "frac_balloon_inflated", "frac_balloon_deflated"]
    available = [c for c in feature_cols if c in windows_df.columns]
    if not available or len(windows_df) == 0:
        fig = go.Figure()
        fig.update_layout(title=title, height=height)
        return fig
    means = windows_df[available].mean()
    maxv = means.abs().max()
    if maxv > 0:
        normed = means / maxv
    else:
        normed = means
    labels = [c.replace("frac_", "").replace("vol_", "vol ").replace("_", " ") for c in available]
    fig = go.Figure(data=go.Scatterpolar(
        r=normed.tolist() + [normed.tolist()[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(52,152,219,0.25)",
        line=dict(color="#3498db", width=2),
        name="Normalized mean",
    ))
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.05])),
        height=height,
        margin=dict(t=50, b=30),
    )
    return fig


def plot_state_transitions(
    windows_df: "pd.DataFrame",
    label_1: str = "Breath-hold",
    label_0: str = "Free-breathing",
    title: str = "State transitions",
    height: int = 280,
) -> "go.Figure":
    """Bar chart showing number of 0->1, 1->0, 0->0, 1->1 transitions between windows."""
    if go is None or pd is None:
        raise ImportError("plotly and pandas required")
    if len(windows_df) < 2 or "prediction" not in windows_df.columns:
        fig = go.Figure()
        fig.update_layout(title=title, height=height)
        return fig
    preds = windows_df["prediction"].values
    stay_0 = int(sum(1 for i in range(1, len(preds)) if preds[i - 1] == 0 and preds[i] == 0))
    stay_1 = int(sum(1 for i in range(1, len(preds)) if preds[i - 1] == 1 and preds[i] == 1))
    to_1 = int(sum(1 for i in range(1, len(preds)) if preds[i - 1] == 0 and preds[i] == 1))
    to_0 = int(sum(1 for i in range(1, len(preds)) if preds[i - 1] == 1 and preds[i] == 0))
    labels_bar = [
        f"{label_0} → {label_0}",
        f"{label_0} → {label_1}",
        f"{label_1} → {label_1}",
        f"{label_1} → {label_0}",
    ]
    vals = [stay_0, to_1, stay_1, to_0]
    colors = [COLOR_FREE_BREATHING, "#f39c12", COLOR_BREATH_HOLD, "#f39c12"]
    fig = go.Figure(data=[go.Bar(x=labels_bar, y=vals, marker_color=colors)])
    fig.update_layout(
        title=title,
        xaxis_title="Transition type",
        yaxis_title="Count",
        height=height,
    )
    return fig


def export_fig_png(fig: "go.Figure") -> Optional[bytes]:
    """Export Plotly figure as PNG bytes if kaleido is available; otherwise return None."""
    try:
        return fig.to_image(format="png", scale=2)
    except Exception:
        return None
