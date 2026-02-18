"""
Visualization helpers using Plotly.
"""
from typing import Optional, List, Tuple

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
) -> "go.Figure":
    """
    Plot breathing curve (time vs volume) with optional predictions and overlays.
    
    df: DataFrame with Session Time, Volume (liters), optionally Balloon Valve Status, Gating Mode
    predictions: DataFrame with time_start, time_end, prediction columns (for window coloring)
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
    
    fig.update_layout(
        title="Breathing Pattern Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Volume (liters)",
        hovermode="closest",
        height=500,
    )
    
    return fig


def plot_confusion_matrix(cm: List[List[int]], model_name: str, labels: List[str] = None) -> "go.Figure":
    """Plot confusion matrix as heatmap."""
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
        text_auto=True,
    )
    fig.update_layout(
        title=f"Confusion Matrix - {model_name}",
        height=400,
    )
    return fig


def plot_feature_importance(model, feature_names: List[str]) -> "go.Figure":
    """Plot feature importance for tree-based models."""
    if go is None:
        raise ImportError("plotly required")
    
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
    
    fig = go.Figure(data=[
        go.Bar(
            x=sorted_importances,
            y=sorted_features,
            orientation="h",
            marker_color="#3498db",
        )
    ])
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=max(400, len(feature_names) * 30),
    )
    return fig
