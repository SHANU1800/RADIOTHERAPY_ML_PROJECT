"""
Upload & Predict page — the core of the project.
Research-grade prediction interface with detailed analysis, visualizations,
glossary of terms, annotated graphs, and educational callouts.
"""
import streamlit as st
import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.utils.inference import (
    predict_breathing_pattern, predict_breathing_pattern_dl,
    get_available_dl_models, is_dl_available,
    compute_gradcam_for_file, get_signal_analysis,
)
from frontend.utils.visualization import (
    plot_breathing_curve,
    export_fig_png,
    plot_confidence_histogram,
    plot_prediction_donut,
    plot_confidence_over_time,
    plot_confidence_buckets_bar,
    plot_prediction_timeline,
    plot_volume_features_radar,
    plot_state_transitions,
    plot_signal_analysis,
    plot_frequency_spectrum,
    plot_gradcam_overlay,
)
from frontend.utils.data_helpers import prediction_to_label, confidence_bucket
from frontend.utils.error_handling import user_facing_message
from frontend.utils.icons import icon_html
import config as cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task_labels(task):
    """Return (task_label, label_1, label_0) for a given task key."""
    if task == "breath_hold":
        return "Breath-hold vs Free-breathing", "Breath-hold", "Free-breathing"
    return "Gating OK vs Not OK", "Gating OK", "Gating Not OK"


def _stability_score(conf_series):
    """
    Stability score (0-100): how consistently the model predicts the same class
    with high confidence. A perfect score means all windows have confidence >= 0.9
    and never switch class.
    """
    import numpy as np
    if len(conf_series) == 0:
        return 0.0
    mean_conf = conf_series.mean()
    std_conf = conf_series.std() if len(conf_series) > 1 else 0.0
    score = 100.0 * mean_conf * max(0, 1.0 - std_conf)
    return min(100.0, max(0.0, score))


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------

GLOSSARY = {
    "Window": (
        "A fixed-length segment of the breathing recording (e.g. 2 seconds). "
        "The model classifies each window independently."
    ),
    "Prediction": (
        "The model's classification for a single window: **1** (Breath-hold / Gating OK) or **0** (Free-breathing / Gating Not OK)."
    ),
    "Confidence": (
        "The probability the model assigns to its chosen class, ranging from 0.0 to 1.0. "
        "Higher values mean the model is more certain. Computed from `predict_proba`."
    ),
    "Low-confidence window": (
        "A window where confidence is below 0.7. These are ambiguous — the model "
        "is not sure whether it is breath-hold or free-breathing. Investigate these "
        "if clinical decisions depend on the result."
    ),
    "Confidence bucket": (
        "Windows grouped by confidence level: **High (≥ 0.9)** — model is very certain, "
        "**Medium (0.7–0.9)** — moderate certainty, **Low (< 0.7)** — weak prediction."
    ),
    "Time-in-state": (
        "The percentage of the session spent in each class (Breath-hold vs Free-breathing) "
        "according to the model's predictions."
    ),
    "Stability score": (
        "A composite score (0-100) measuring how consistently the model predicts the "
        "same class with high confidence. 100 = all windows predicted with ≥ 0.9 "
        "confidence and no class switches. Low scores suggest mixed or uncertain predictions."
    ),
    "State transition": (
        "When the model's prediction changes from one class to another between consecutive "
        "windows. Frequent transitions indicate rapid switching between breathing states."
    ),
    "Volume (L)": (
        "Lung air volume measured by the spirometer in liters. During breath-hold, "
        "volume is relatively stable; during free-breathing, it oscillates."
    ),
    "vol_mean / vol_std / vol_range": (
        "Per-window statistics of the volume signal. **vol_mean**: average volume, "
        "**vol_std**: standard deviation (variability), **vol_range**: max − min. "
        "Low std and small range usually indicate breath-hold."
    ),
    "frac_balloon_inflated": (
        "Fraction of samples in a window where the balloon valve is inflated (status = 4). "
        "High values strongly correlate with breath-hold."
    ),
    "Session Time": (
        "Elapsed time in seconds from the start of the recording."
    ),
    "Balloon Valve Status": (
        "Hardware signal: 1 = deflated (free-breathing), 2 = ready inhale, "
        "3 = ready exhale, 4 = inflated (breath-hold), 5 = fault."
    ),
    "Gating Mode": (
        "'Automated' means the radiation beam is gated automatically; 'Manual Override' "
        "means an operator took control. Gating OK = Automated."
    ),
    "Model": (
        "The trained ML classifier (e.g. RandomForest, XGBoost). Loaded from "
        "`models/best_model.pkl`. Trained on windowed features from the dataset."
    ),
    "dV/dt (derivative)": (
        "First difference of volume — approximates the rate of breathing. "
        "Positive = inhaling, negative = exhaling. Near zero during breath-hold."
    ),
    "d²V/dt² (2nd derivative)": (
        "Second difference of volume — acceleration of breathing. "
        "Peaks indicate transitions between inhale and exhale phases."
    ),
    "Envelope": (
        "Rolling standard deviation of volume over a 0.5-second window. "
        "High envelope = active breathing. Low envelope = stable / breath-hold."
    ),
    "Grad-CAM": (
        "Gradient-weighted Class Activation Mapping — a technique that visualizes "
        "which time steps the deep learning model focuses on when making its prediction. "
        "Red = high importance, meaning the model relied heavily on that region."
    ),
    "Multi-channel input": (
        "Deep learning models can use multiple signal channels simultaneously: "
        "raw volume, derivative, 2nd derivative, balloon valve status, and amplitude envelope. "
        "This gives the model richer information than single-channel (volume-only) input."
    ),
    "MCC (Matthews Correlation Coefficient)": (
        "A balanced metric for binary classification that accounts for all four confusion matrix "
        "values (TP, TN, FP, FN). Range: -1 to +1. +1 = perfect, 0 = random, -1 = inverse."
    ),
    "Specificity": (
        "True Negative Rate — the fraction of actual negatives correctly identified. "
        "For breath-hold task: fraction of free-breathing windows correctly predicted as free-breathing."
    ),
    "Sensitivity": (
        "True Positive Rate (same as Recall) — the fraction of actual positives correctly identified. "
        "For breath-hold task: fraction of breath-hold windows correctly detected."
    ),
}


# ---------------------------------------------------------------------------
# Display predictions (main analysis)
# ---------------------------------------------------------------------------

def _display_predictions(windows_df, metadata, df, task, file_id, file_name=None, chart_key="predictions_chart"):
    import pandas as pd
    import numpy as np

    task_label, label_1, label_0 = _task_labels(task)

    n = len(windows_df)
    bh_count = int((windows_df["prediction"] == 1).sum())
    fb_count = int((windows_df["prediction"] == 0).sum())
    pct_1 = 100.0 * bh_count / n if n else 0
    pct_0 = 100.0 * fb_count / n if n else 0
    conf = windows_df["confidence"]
    median_conf = float(conf.median())
    mean_conf = float(conf.mean())
    low_conf_count = int((conf < 0.7).sum())
    session_duration_s = float(df["Session Time"].max() - df["Session Time"].min()) if len(df) else 0
    stability = _stability_score(conf)

    preds_arr = windows_df["prediction"].values
    n_transitions = int(sum(1 for i in range(1, len(preds_arr)) if preds_arr[i] != preds_arr[i - 1]))

    buckets = windows_df["confidence"].apply(confidence_bucket)
    high_c = int((buckets == "High (>=0.9)").sum())
    med_c = int((buckets == "Medium (0.7-0.9)").sum())
    low_c = int((buckets == "Low (<0.7)").sum())

    # -----------------------------------------------------------------------
    # HERO RESULT — styled card (always visible, compact)
    # -----------------------------------------------------------------------
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    dominant = label_1 if pct_1 >= pct_0 else label_0
    dominant_pct = max(pct_1, pct_0)
    hero_cls = "result-panel-success" if median_conf >= 0.7 else "result-panel-warning"
    st.markdown(
        f'<div class="result-panel {hero_cls}">'
        f'<strong style="font-size:1.15rem;">Predominantly {dominant}</strong> &nbsp;'
        f'<span class="badge badge-blue">{dominant_pct:.1f}% of {n} windows</span> &nbsp;'
        f'<span class="badge badge-green">Confidence {median_conf:.2f}</span> &nbsp;'
        f'<span class="badge badge-blue">Stability {stability:.0f}/100</span><br>'
        f'<span style="color:#64748b;font-size:0.85rem;">'
        f'Task: {task_label} &middot; Window: {metadata.get("window_sec", 2)} s &middot; '
        f'Model: {metadata.get("model_name", "N/A")} &middot; File: {file_name or file_id}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # KEY INSIGHTS — auto-generated bullet points
    insights = []
    if dominant_pct > 90:
        insights.append(f"Session is strongly **{dominant}** ({dominant_pct:.0f}% of windows)")
    elif dominant_pct > 70:
        insights.append(f"Session is predominantly **{dominant}** ({dominant_pct:.0f}%)")
    else:
        insights.append(f"Session is **mixed** — {label_1}: {pct_1:.0f}%, {label_0}: {pct_0:.0f}%")

    if median_conf >= 0.9:
        insights.append("Model is **very confident** (median {:.2f})".format(median_conf))
    elif median_conf >= 0.7:
        insights.append("Model confidence is **moderate** (median {:.2f})".format(median_conf))
    else:
        insights.append("**Low confidence** — predictions may be unreliable (median {:.2f})".format(median_conf))

    if low_conf_count > 0:
        insights.append(f"**{low_conf_count}** window{'s' if low_conf_count > 1 else ''} below confidence threshold (< 0.7) — review manually")

    if n_transitions > n * 0.3:
        insights.append(f"**Frequent state changes** ({n_transitions} transitions) — rapid alternation between states")
    elif n_transitions <= 2:
        insights.append(f"Very **stable** session — only {n_transitions} transition{'s' if n_transitions != 1 else ''}")

    if stability >= 90:
        insights.append(f"Stability score: **{stability:.0f}/100** (excellent)")
    elif stability < 50:
        insights.append(f"Stability score: **{stability:.0f}/100** (poor — investigate)")

    st.markdown(
        '<div class="insights-panel"><h3>Key Insights</h3><ul>'
        + ''.join(f'<li>{i}</li>' for i in insights)
        + '</ul></div>',
        unsafe_allow_html=True,
    )

    # COMPACT METRIC ROW
    st.markdown(
        '<div class="stat-row">'
        f'<span class="stat-pill">{session_duration_s:.1f}s <small>duration</small></span>'
        f'<span class="stat-pill">{n} <small>windows</small></span>'
        f'<span class="stat-pill">{bh_count} ({pct_1:.0f}%) <small>{label_1}</small></span>'
        f'<span class="stat-pill">{fb_count} ({pct_0:.0f}%) <small>{label_0}</small></span>'
        f'<span class="stat-pill">{low_conf_count} <small>low-conf</small></span>'
        f'<span class="stat-pill">{n_transitions} <small>transitions</small></span>'
        f'<span class="stat-pill">H:{high_c} M:{med_c} L:{low_c} <small>conf buckets</small></span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # TABBED ANALYSIS (streamlined: 5 tabs + glossary expander)
    # -----------------------------------------------------------------------
    tab_curve, tab_charts, tab_analysis, tab_signal, tab_data = st.tabs([
        "Breathing Curve", "Charts", "Analysis & Clinical", "Signal & Grad-CAM", "Data & Export",
    ])

    # --- TAB 1: Breathing Curve (most important — shown first) ---
    with tab_curve:
        st.caption("Blue line = volume. Background: green = %s, red = %s." % (label_1, label_0))
        fig_curve = plot_breathing_curve(
            df,
            predictions=windows_df,
            subtitle="Window size: %s s | %s" % (metadata.get("window_sec", 2), file_name or file_id),
        )
        st.plotly_chart(fig_curve, use_container_width=True, key=chart_key)

        st.markdown("#### Prediction Timeline")
        st.caption("Each segment = one window. Green = %s, Red = %s. Opacity = confidence." % (label_1, label_0))
        fig_tl = plot_prediction_timeline(windows_df, label_1=label_1, label_0=label_0, title="Prediction timeline")
        st.plotly_chart(fig_tl, use_container_width=True, key=f"{chart_key}_timeline")

        dl1, dl2 = st.columns(2)
        with dl1:
            png_bytes = export_fig_png(fig_curve)
            if png_bytes:
                st.download_button("Download curve (PNG)", data=png_bytes, file_name="breathing_curve.png", mime="image/png", key=f"{chart_key}_dl_png")
        with dl2:
            curve_csv = df[["Session Time", "Volume (liters)"]].head(15000).to_csv(index=False)
            st.download_button("Download curve data (CSV)", data=curve_csv, file_name="curve_data.csv", mime="text/csv", key=f"{chart_key}_dl_csv")

    # --- TAB 2: Charts ---
    with tab_charts:
        st.caption("Hover over any chart for details. Green = %s, Red = %s." % (label_1, label_0))

        v1, v2 = st.columns(2)
        with v1:
            fig_donut = plot_prediction_donut(bh_count, fb_count, label_1=label_1, label_0=label_0, title="Class distribution")
            st.plotly_chart(fig_donut, use_container_width=True, key=f"{chart_key}_donut")
        with v2:
            fig_conf_time = plot_confidence_over_time(windows_df, title="Confidence over time", height=320)
            st.plotly_chart(fig_conf_time, use_container_width=True, key=f"{chart_key}_conf_time")

        v3, v4 = st.columns(2)
        with v3:
            fig_buckets = plot_confidence_buckets_bar(high_c, med_c, low_c, title="Confidence levels")
            st.plotly_chart(fig_buckets, use_container_width=True, key=f"{chart_key}_buckets")
        with v4:
            fig_hist = plot_confidence_histogram(windows_df["confidence"], title="Confidence distribution")
            st.plotly_chart(fig_hist, use_container_width=True, key=f"{chart_key}_hist")

        v5, v6 = st.columns(2)
        with v5:
            fig_trans = plot_state_transitions(windows_df, label_1=label_1, label_0=label_0, title="State transitions")
            st.plotly_chart(fig_trans, use_container_width=True, key=f"{chart_key}_trans")
        with v6:
            fig_radar = plot_volume_features_radar(windows_df, title="Average feature profile")
            st.plotly_chart(fig_radar, use_container_width=True, key=f"{chart_key}_radar")

    # --- TAB 3: Analysis & Clinical (merged Features + Clinical Zones) ---
    with tab_analysis:
        # Clinical zones at top for quick access
        st.markdown("#### Clinical Zones")
        st.caption("Key regions: longest stretches and lowest-confidence zone.")
        zones = _find_clinical_zones(windows_df, label_1, label_0)
        z1, z2, z3 = st.columns(3)
        with z1:
            st.markdown("**Longest %s**" % label_1)
            if zones["longest_1"]:
                z = zones["longest_1"]
                st.metric("Duration", "%.1f s" % z["duration"])
                st.caption("%.1f – %.1f s (%d win, conf %.2f)" % (z["start"], z["end"], z["count"], z["avg_conf"]))
            else:
                st.caption("None found")
        with z2:
            st.markdown("**Longest %s**" % label_0)
            if zones["longest_0"]:
                z = zones["longest_0"]
                st.metric("Duration", "%.1f s" % z["duration"])
                st.caption("%.1f – %.1f s (%d win, conf %.2f)" % (z["start"], z["end"], z["count"], z["avg_conf"]))
            else:
                st.caption("None found")
        with z3:
            st.markdown("**Weakest zone**")
            if zones["lowest_conf"]:
                z = zones["lowest_conf"]
                st.metric("Min conf", "%.3f" % z["min_conf"])
                st.caption("%.1f – %.1f s (%d win)" % (z["start"], z["end"], z["count"]))
            else:
                st.caption("N/A")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Feature Statistics")
        feature_cols = ["vol_mean", "vol_std", "vol_min", "vol_max", "vol_range", "vol_change",
                        "frac_balloon_inflated", "frac_balloon_deflated", "frac_patient_switch_on", "frac_gating_automated"]
        avail_feat = [c for c in feature_cols if c in windows_df.columns]
        if avail_feat:
            feat_stats = windows_df[avail_feat].describe().T
            feat_stats.index.name = "Feature"
            st.dataframe(feat_stats.style.format("{:.4f}", na_rep="–"), use_container_width=True)

            if "prediction" in windows_df.columns and n > 0:
                st.markdown("##### Feature Comparison: %s vs %s" % (label_1, label_0))
                class1 = windows_df[windows_df["prediction"] == 1][avail_feat]
                class0 = windows_df[windows_df["prediction"] == 0][avail_feat]
                compare_rows = []
                for c in avail_feat:
                    row = {"Feature": c}
                    row["%s mean" % label_1] = class1[c].mean() if len(class1) > 0 else None
                    row["%s mean" % label_0] = class0[c].mean() if len(class0) > 0 else None
                    diff = (row["%s mean" % label_1] or 0) - (row["%s mean" % label_0] or 0)
                    row["Difference"] = diff
                    compare_rows.append(row)
                compare_df = pd.DataFrame(compare_rows).set_index("Feature")
                st.dataframe(compare_df.style.format("{:.4f}", na_rep="–"), use_container_width=True)
        else:
            st.caption("No feature columns available.")

    # --- TAB 4: Signal & Grad-CAM ---
    with tab_signal:
        st.caption(
            "Multi-channel signal decomposition and model interpretability."
        )
        try:
            fig_sig = plot_signal_analysis(df, sample_rate_hz=cfg.SAMPLE_RATE_HZ)
            st.plotly_chart(fig_sig, use_container_width=True, key=f"{chart_key}_signal")
        except Exception as e:
            st.warning(f"Could not plot signal analysis: {e}")

        sig1, sig2 = st.columns(2)
        with sig1:
            st.markdown("#### Frequency Spectrum")
            try:
                vol_arr = pd.to_numeric(df["Volume (liters)"], errors="coerce").fillna(0).values
                max_samples = 10000
                if len(vol_arr) > max_samples:
                    vol_arr = vol_arr[:max_samples]
                fig_spec = plot_frequency_spectrum(vol_arr, sample_rate=cfg.SAMPLE_RATE_HZ)
                st.plotly_chart(fig_spec, use_container_width=True, key=f"{chart_key}_spectrum")
            except Exception as e:
                st.warning(f"Could not plot spectrum: {e}")
        with sig2:
            st.markdown("#### Signal Legend")
            st.markdown(
                "- **Volume**: raw breathing signal\n"
                "- **dV/dt**: breathing rate (positive = inhale)\n"
                "- **d²V/dt²**: acceleration (peaks at rhythm changes)\n"
                "- **Envelope**: rolling variability (flat = breath-hold)\n"
                "- **Dominant frequency**: primary breathing rate"
            )

        # Grad-CAM integrated here (no separate section at bottom)
        if metadata and metadata.get("model_type") == "deep_learning":
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown(
                f'<p class="page-header" style="font-size:1.2rem;">'
                f'{icon_html("chart", 20)} Grad-CAM: Where the Model Focuses</p>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Highlights which time regions the DL model considers most important. Red = high importance."
            )
            if st.button("Compute Grad-CAM", key=f"{chart_key}_gradcam_btn"):
                file_bytes = st.session_state.get("prediction_file_bytes")
                file_suffix = st.session_state.get("prediction_file_suffix", "dat")
                if file_bytes:
                    import tempfile as _tmpmod
                    with st.spinner("Computing Grad-CAM..."):
                        try:
                            with _tmpmod.NamedTemporaryFile(delete=False, suffix=f".{file_suffix}") as _tf:
                                _tf.write(file_bytes)
                                _tmp = Path(_tf.name)
                            dl_name = metadata.get("model_name", "DL-LSTM").replace("DL-", "")
                            channels = metadata.get("channels")
                            t_arr, v_arr, imp_arr = compute_gradcam_for_file(
                                _tmp, model_name=dl_name, task=task,
                                window_sec=metadata.get("window_sec", 2.0),
                                patient_id=st.session_state.get("prediction_file_id", "uploaded"),
                                channels=channels,
                            )
                            fig_gc = plot_gradcam_overlay(t_arr, v_arr, imp_arr)
                            st.plotly_chart(fig_gc, use_container_width=True, key=f"{chart_key}_gradcam_plot")
                        except Exception as e:
                            from frontend.utils.error_handling import user_facing_message as _ufm
                            st.warning(f"Could not compute Grad-CAM: {_ufm(e)}")
                        finally:
                            try:
                                _tmp.unlink(missing_ok=True)
                            except Exception:
                                pass
                else:
                    st.info("Re-upload a file and run prediction to use Grad-CAM.")

    # --- TAB 5: Data & Export ---
    with tab_data:
        # Filter controls
        filter_col1, filter_col2 = st.columns([1, 1])
        with filter_col1:
            filter_class = st.selectbox(
                "Filter by class", ["All", label_1, label_0],
                key=f"{chart_key}_filter_class"
            )
        with filter_col2:
            filter_conf = st.selectbox(
                "Filter by confidence", ["All", "High (>=0.9)", "Medium (0.7-0.9)", "Low (<0.7)"],
                key=f"{chart_key}_filter_conf"
            )

        filtered_df = windows_df.copy()
        filtered_df["window_index"] = range(1, len(filtered_df) + 1)
        filtered_df["prediction_label"] = filtered_df["prediction"].apply(lambda p: prediction_to_label(int(p), task))
        filtered_df["confidence_bucket"] = filtered_df["confidence"].apply(confidence_bucket)

        if filter_class == label_1:
            filtered_df = filtered_df[filtered_df["prediction"] == 1]
        elif filter_class == label_0:
            filtered_df = filtered_df[filtered_df["prediction"] == 0]
        if filter_conf == "High (>=0.9)":
            filtered_df = filtered_df[filtered_df["confidence"] >= 0.9]
        elif filter_conf == "Medium (0.7-0.9)":
            filtered_df = filtered_df[(filtered_df["confidence"] >= 0.7) & (filtered_df["confidence"] < 0.9)]
        elif filter_conf == "Low (<0.7)":
            filtered_df = filtered_df[filtered_df["confidence"] < 0.7]

        st.caption(f"Showing {len(filtered_df)} of {n} windows")

        display_cols = ["window_index", "time_start", "time_end", "prediction_label", "prediction",
                        "confidence", "confidence_bucket", "vol_mean", "vol_std", "vol_range",
                        "frac_balloon_inflated"]
        if "prob_class_0" in filtered_df.columns:
            display_cols.extend(["prob_class_0", "prob_class_1"])
        available_cols = [c for c in display_cols if c in filtered_df.columns]
        display_df = filtered_df[available_cols].copy()
        for col in ["time_start", "time_end", "confidence", "vol_mean", "vol_std", "vol_range"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)

        st.dataframe(display_df.head(500), use_container_width=True, height=400)
        if len(display_df) > 500:
            st.caption("Showing first 500 of %d windows. Download CSV for full data." % len(display_df))

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        export_df = windows_df.copy()
        if "prediction_label" not in export_df.columns:
            export_df["prediction_label"] = export_df["prediction"].apply(lambda p: prediction_to_label(int(p), task))
        if "confidence_bucket" not in export_df.columns:
            export_df["confidence_bucket"] = export_df["confidence"].apply(confidence_bucket)
        csv = export_df.to_csv(index=False)
        with d1:
            st.download_button(
                label="Download predictions (CSV)",
                data=csv,
                file_name="predictions_%s_%s.csv" % (task, file_id),
                mime="text/csv",
                key=f"{chart_key}_dl_pred_csv",
            )
        summary_text = _build_summary_text(
            file_name or file_id, task_label, metadata, session_duration_s, n,
            label_1, bh_count, pct_1, label_0, fb_count, pct_0,
            median_conf, low_conf_count, stability, n_transitions,
        )
        with d2:
            st.download_button(
                label="Download report (TXT)",
                data=summary_text,
                file_name="prediction_report_%s_%s.txt" % (task, file_id),
                mime="text/plain",
                key=f"{chart_key}_dl_report_txt",
            )
        with d3:
            _curve_png = export_fig_png(fig_curve)
            if _curve_png:
                st.download_button(
                    label="Download curve PNG",
                    data=_curve_png,
                    file_name="curve_%s.png" % file_id,
                    mime="image/png",
                    key=f"{chart_key}_dl_curve_png2",
                )

    # Glossary as expander (always accessible, doesn't waste a tab)
    with st.expander("Glossary of terms"):
        for term, definition in GLOSSARY.items():
            st.markdown("**%s** — %s" % (term, definition))


def _find_clinical_zones(windows_df, label_1, label_0):
    """Find longest breath-hold stretch, longest free-breathing stretch, and lowest-confidence zone."""
    result = {"longest_1": None, "longest_0": None, "lowest_conf": None}
    if len(windows_df) == 0 or "prediction" not in windows_df.columns:
        return result
    preds = windows_df["prediction"].values
    times_start = windows_df["time_start"].values if "time_start" in windows_df.columns else list(range(len(preds)))
    times_end = windows_df["time_end"].values if "time_end" in windows_df.columns else list(range(len(preds)))
    confs = windows_df["confidence"].values if "confidence" in windows_df.columns else [0.5] * len(preds)

    for target, key in [(1, "longest_1"), (0, "longest_0")]:
        best_start = best_end = best_len = 0
        cur_start = cur_len = 0
        for i in range(len(preds)):
            if preds[i] == target:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_len = cur_len
                    best_start = cur_start
                    best_end = i
            else:
                cur_len = 0
        if best_len > 0:
            seg_confs = confs[best_start:best_end + 1]
            result[key] = {
                "start": float(times_start[best_start]),
                "end": float(times_end[best_end]),
                "duration": float(times_end[best_end]) - float(times_start[best_start]),
                "count": best_len,
                "avg_conf": float(seg_confs.mean()) if hasattr(seg_confs, "mean") else 0,
            }

    # Lowest-confidence zone: sliding window of 3 with lowest mean confidence
    if len(confs) >= 3:
        import numpy as np
        win = 3
        best_mean = 1.0
        best_idx = 0
        for i in range(len(confs) - win + 1):
            m = float(np.mean(confs[i:i + win]))
            if m < best_mean:
                best_mean = m
                best_idx = i
        result["lowest_conf"] = {
            "start": float(times_start[best_idx]),
            "end": float(times_end[min(best_idx + win - 1, len(times_end) - 1)]),
            "count": win,
            "min_conf": best_mean,
        }
    return result


def _build_summary_text(file_name, task_label, metadata, duration, n,
                        label_1, bh_count, pct_1, label_0, fb_count, pct_0,
                        median_conf, low_conf_count, stability, n_transitions):
    lines = [
        "=== PREDICTION REPORT ===",
        "",
        "File: %s" % file_name,
        "Task: %s" % task_label,
        "Model: %s" % metadata.get("model_name", "N/A"),
        "Window size: %s s" % metadata.get("window_sec", 2),
        "",
        "Session duration: %.1f s" % duration,
        "Total windows: %d" % n,
        "Raw data rows: %d" % metadata.get("num_rows", 0),
        "",
        "--- Classification ---",
        "%s: %d (%.1f%%)" % (label_1, bh_count, pct_1),
        "%s: %d (%.1f%%)" % (label_0, fb_count, pct_0),
        "",
        "--- Confidence ---",
        "Median confidence: %.3f" % median_conf,
        "Low-confidence windows (<0.7): %d" % low_conf_count,
        "Stability score: %.0f / 100" % stability,
        "",
        "--- Transitions ---",
        "State transitions: %d" % n_transitions,
        "",
        "=== END OF REPORT ===",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main show() function
# ---------------------------------------------------------------------------

def show():
    st.markdown(
        f'<p class="page-header">{icon_html("upload", 28)} Upload & Predict</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="page-subtitle">'
        'Upload a breathing curve file for window-by-window classification with '
        'confidence scores, annotated charts, clinical zone analysis, and exportable reports.</p>',
        unsafe_allow_html=True,
    )

    if "prediction_results" not in st.session_state:
        st.session_state.prediction_results = None
    if "prediction_metadata" not in st.session_state:
        st.session_state.prediction_metadata = None
    if "prediction_task" not in st.session_state:
        st.session_state.prediction_task = None
    if "show_explanation" not in st.session_state:
        st.session_state.show_explanation = False

    # --- Input controls (compact layout) ---
    dl_available = is_dl_available()

    c1, c2, c3, c4 = st.columns([1.5, 1, 1.5, 1])
    with c1:
        task = st.selectbox(
            "Task",
            ["breath_hold", "gating_ok"],
            index=0,
            help="breath_hold: Classify breath-hold vs free-breathing. gating_ok: Classify gating OK vs not OK.",
        )
    with c2:
        window_sec = st.number_input(
            "Window (s)", 1.0, 5.0, cfg.WINDOW_SEC, 0.5,
            help="Smaller = more granular. Larger = smoother predictions.",
        )
    with c3:
        model_options = ["Classical ML"]
        if dl_available:
            model_options.append("Deep Learning")
        model_type = st.selectbox(
            "Model Type",
            model_options,
            index=1 if dl_available else 0,
            help="Classical ML uses hand-crafted features. Deep Learning uses raw time-series."
        )
    dl_model_name = None
    with c4:
        if model_type == "Deep Learning" and dl_available:
            available_dl = get_available_dl_models(task)
            if available_dl:
                dl_model_name = st.selectbox("Architecture", available_dl)
            else:
                st.warning(f"No DL models for '{task}'")
        else:
            st.text_input("Architecture", value="Best Model", disabled=True)

    uploaded_file = st.file_uploader(
        "Upload breathing curve file (.dat, .txt, .csv)",
        type=["dat", "txt", "csv"],
        help="Must have columns: Session Time, Volume, Balloon Valve Status, Patient Switch, Gating Mode.",
    )

    id1, id2 = st.columns(2)
    with id1:
        patient_id = st.text_input("Patient ID (optional)", value="uploaded", help="Identifier for the patient.")
    with id2:
        file_id = st.text_input("File ID (optional)", value="uploaded", help="Identifier for this file.")

    if uploaded_file is not None:
        run_disabled = model_type == "Deep Learning" and not dl_model_name
        if st.button("Run Prediction", type="primary", disabled=run_disabled):
            tmp_path = None
            with st.spinner("Processing file and running predictions..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".%s" % uploaded_file.name.split(".")[-1]) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = Path(tmp.name)

                    if model_type == "Deep Learning" and dl_model_name:
                        try:
                            windows_df, metadata = predict_breathing_pattern_dl(
                                tmp_path, model_name=dl_model_name, task=task,
                                window_sec=window_sec, patient_id=patient_id, file_id=file_id,
                            )
                        except ImportError:
                            st.error(
                                "TensorFlow is not available in this Python environment. "
                                "Run the app with Python 3.12 and TensorFlow installed: "
                                "`py -3.12 -m pip install tensorflow` then "
                                "`py -3.12 -m streamlit run run_frontend.py`"
                            )
                            st.stop()
                    else:
                        windows_df, metadata = predict_breathing_pattern(
                            tmp_path, task=task, window_sec=window_sec,
                            patient_id=patient_id, file_id=file_id,
                        )

                    from src.load_data import load_patient_file
                    df = load_patient_file(tmp_path, patient_id=patient_id, file_id=file_id)

                    st.session_state.prediction_results = windows_df
                    st.session_state.prediction_metadata = metadata
                    st.session_state.prediction_task = task
                    st.session_state.prediction_df = df
                    st.session_state.prediction_file_id = file_id
                    st.session_state.prediction_file_name = uploaded_file.name
                    st.session_state.prediction_file_bytes = uploaded_file.getvalue()
                    st.session_state.prediction_file_suffix = uploaded_file.name.split(".")[-1]
                    st.session_state.show_explanation = False
                    if "ai_explanation" in st.session_state:
                        del st.session_state.ai_explanation
                    st.rerun()
                except Exception as e:
                    st.error("Error: %s" % user_facing_message(e))
                    st.exception(e)
                    st.session_state.prediction_results = None
                    st.session_state.prediction_metadata = None
                    st.session_state.prediction_task = None
                    st.session_state.prediction_df = None
                finally:
                    if tmp_path is not None:
                        try:
                            tmp_path.unlink(missing_ok=True)
                        except OSError:
                            pass

    # --- Render results from session state ---
    if st.session_state.prediction_results is not None:
        windows_df = st.session_state.prediction_results
        metadata = st.session_state.prediction_metadata
        df = st.session_state.prediction_df
        task = st.session_state.prediction_task
        file_id = st.session_state.get("prediction_file_id", "uploaded")
        file_name = st.session_state.get("prediction_file_name")

        if metadata is None or df is None:
            st.error("Prediction data is incomplete. Please re-upload and run prediction.")
        else:
            try:
                _display_predictions(
                    windows_df, metadata, df, task, file_id,
                    file_name=file_name, chart_key="upload_predict_curve",
                )
            except Exception as e:
                st.error("Error displaying predictions: %s" % user_facing_message(e))
                st.exception(e)

        # AI Explanation
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown(
            f'<p class="page-header" style="font-size:1.3rem;">'
            f'{icon_html("robot", 22)} AI Explanation</p>',
            unsafe_allow_html=True,
        )
        if st.button("Explain Predictions with AI", help="Get AI-powered explanation", key="explain_btn"):
            try:
                from frontend.utils.llm_helper import explain_prediction, is_llm_available
                if is_llm_available():
                    with st.spinner("Generating AI explanation..."):
                        model_info = {"model_name": metadata.get("model_name", "Best Model"), "task": task}
                        explanation = explain_prediction(windows_df, model_info, task)
                        st.session_state.ai_explanation = explanation
                        st.session_state.show_explanation = True
                        st.rerun()
                else:
                    st.info("AI features require Ollama API. Go to AI Assistant page for setup instructions.")
            except Exception as e:
                st.error("Could not generate AI explanation: %s" % user_facing_message(e))
                st.exception(e)

        if "ai_explanation" in st.session_state and st.session_state.get("show_explanation", False):
            st.markdown("### Explanation")
            st.markdown(st.session_state.ai_explanation)
