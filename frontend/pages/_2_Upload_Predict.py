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

from frontend.utils.inference import predict_breathing_pattern
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
    # HERO RESULT — styled card
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

    # KEY METRICS
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Duration", "%.1f s" % session_duration_s, help="Total session length")
    m2.metric("Windows", n, help="Number of analysis windows")
    m3.metric(label_1, "%d (%.1f%%)" % (bh_count, pct_1), help="Windows classified as " + label_1)
    m4.metric(label_0, "%d (%.1f%%)" % (fb_count, pct_0), help="Windows classified as " + label_0)
    m5.metric("Low-conf", int(low_conf_count), help="Windows with confidence < 0.7")
    m6.metric("Transitions", n_transitions, help="Times the prediction switches class")

    st.caption(
        "Confidence — min: %.3f, max: %.3f, mean: %.3f, median: %.3f, std: %.3f  |  "
        "Buckets — High (>=0.9): %d, Medium (0.7-0.9): %d, Low (<0.7): %d"
        % (conf.min(), conf.max(), mean_conf, median_conf,
           conf.std() if len(conf) > 1 else 0.0, high_c, med_c, low_c)
    )

    # -----------------------------------------------------------------------
    # TABBED ANALYSIS
    # -----------------------------------------------------------------------
    tab_charts, tab_curve, tab_features, tab_clinical, tab_data, tab_glossary = st.tabs([
        "Charts", "Breathing Curve", "Features", "Clinical Zones", "Data Table", "Glossary",
    ])

    # --- TAB: Charts ---
    with tab_charts:
        st.caption("Hover over any chart for details. Green = %s, Red = %s." % (label_1, label_0))

        v1, v2 = st.columns(2)
        with v1:
            fig_donut = plot_prediction_donut(bh_count, fb_count, label_1=label_1, label_0=label_0, title="Class distribution")
            st.plotly_chart(fig_donut, use_container_width=True, key=f"{chart_key}_donut")
            st.info(
                "**Reading this chart:** The donut shows the proportion of windows in each class. "
                "A dominant slice (> 80%%) means the session is strongly one pattern."
            )
        with v2:
            fig_conf_time = plot_confidence_over_time(windows_df, title="Confidence over time", height=320)
            st.plotly_chart(fig_conf_time, use_container_width=True, key=f"{chart_key}_conf_time")
            st.info(
                "**Reading this chart:** Each dot is one window's confidence. "
                "The red dashed line at 0.7 is the low-confidence threshold. "
                "Dips below 0.7 mark uncertain windows."
            )

        v3, v4 = st.columns(2)
        with v3:
            fig_buckets = plot_confidence_buckets_bar(high_c, med_c, low_c, title="Confidence levels")
            st.plotly_chart(fig_buckets, use_container_width=True, key=f"{chart_key}_buckets")
            st.info("Green (High >= 0.9) | Orange (Medium 0.7-0.9) | Red (Low < 0.7). Ideally most windows should be green.")
        with v4:
            fig_hist = plot_confidence_histogram(windows_df["confidence"], title="Confidence distribution")
            st.plotly_chart(fig_hist, use_container_width=True, key=f"{chart_key}_hist")
            st.info("A peak near 1.0 = mostly confident. Bimodal = good class separation.")

        v5, v6 = st.columns(2)
        with v5:
            fig_trans = plot_state_transitions(windows_df, label_1=label_1, label_0=label_0, title="State transitions")
            st.plotly_chart(fig_trans, use_container_width=True, key=f"{chart_key}_trans")
            st.info("Many transitions (orange) = rapid alternation. Few = stable session.")
        with v6:
            fig_radar = plot_volume_features_radar(windows_df, title="Average feature profile")
            st.plotly_chart(fig_radar, use_container_width=True, key=f"{chart_key}_radar")
            st.info("Radar shows normalized mean feature values. High 'balloon inflated' = breath-hold dominant.")

        st.markdown("#### Prediction timeline")
        st.caption(
            "Each segment is one window. Green = %s, Red = %s. Opacity reflects confidence."
            % (label_1, label_0)
        )
        fig_tl = plot_prediction_timeline(windows_df, label_1=label_1, label_0=label_0, title="Prediction timeline")
        st.plotly_chart(fig_tl, use_container_width=True, key=f"{chart_key}_timeline")
        st.info("Scan left-to-right. Long green stretches = sustained breath-hold. Faded = uncertain.")

    # --- TAB: Breathing Curve ---
    with tab_curve:
        st.caption("Blue line = volume. Background: green = %s, red = %s." % (label_1, label_0))
        fig_curve = plot_breathing_curve(
            df,
            predictions=windows_df,
            subtitle="Window size: %s s | %s" % (metadata.get("window_sec", 2), file_name or file_id),
        )
        st.plotly_chart(fig_curve, use_container_width=True, key=chart_key)
        st.info(
            "**Flat, stable regions** = breath-hold (green). **Oscillating** = free-breathing (red). "
            "Diamond markers = balloon valve inflated. Use zoom and hover for precise values."
        )
        dl1, dl2 = st.columns(2)
        with dl1:
            png_bytes = export_fig_png(fig_curve)
            if png_bytes:
                st.download_button("Download curve (PNG)", data=png_bytes, file_name="breathing_curve.png", mime="image/png", key=f"{chart_key}_dl_png")
        with dl2:
            curve_csv = df[["Session Time", "Volume (liters)"]].head(15000).to_csv(index=False)
            st.download_button("Download curve data (CSV)", data=curve_csv, file_name="curve_data.csv", mime="text/csv", key=f"{chart_key}_dl_csv")

    # --- TAB: Features ---
    with tab_features:
        st.caption("Statistics of the features the model used to make predictions.")
        feature_cols = ["vol_mean", "vol_std", "vol_min", "vol_max", "vol_range", "vol_change",
                        "frac_balloon_inflated", "frac_balloon_deflated", "frac_patient_switch_on", "frac_gating_automated"]
        avail_feat = [c for c in feature_cols if c in windows_df.columns]
        if avail_feat:
            feat_stats = windows_df[avail_feat].describe().T
            feat_stats.index.name = "Feature"
            st.dataframe(feat_stats.style.format("{:.4f}", na_rep="–"), use_container_width=True)
            st.info(
                "Each row is a feature. **mean** = average across windows, **std** = variability. "
                "High `frac_balloon_inflated` confirms breath-hold. High `vol_std` = lots of volume variation."
            )
        else:
            st.caption("No feature columns available.")

        if avail_feat and "prediction" in windows_df.columns and n > 0:
            with st.expander("Feature comparison: %s vs %s" % (label_1, label_0)):
                class1 = windows_df[windows_df["prediction"] == 1][avail_feat]
                class0 = windows_df[windows_df["prediction"] == 0][avail_feat]
                compare_rows = []
                for c in avail_feat:
                    row = {"Feature": c}
                    if len(class1) > 0:
                        row["%s mean" % label_1] = class1[c].mean()
                        row["%s std" % label_1] = class1[c].std()
                    else:
                        row["%s mean" % label_1] = None
                        row["%s std" % label_1] = None
                    if len(class0) > 0:
                        row["%s mean" % label_0] = class0[c].mean()
                        row["%s std" % label_0] = class0[c].std()
                    else:
                        row["%s mean" % label_0] = None
                        row["%s std" % label_0] = None
                    compare_rows.append(row)
                compare_df = pd.DataFrame(compare_rows).set_index("Feature")
                st.dataframe(compare_df.style.format("{:.4f}", na_rep="–"), use_container_width=True)
                st.info(
                    "Large differences (e.g. `frac_balloon_inflated` high for %s but low for %s) "
                    "show which features the model relies on." % (label_1, label_0)
                )

    # --- TAB: Clinical Zones ---
    with tab_clinical:
        st.caption(
            "Identifies important regions: longest breath-hold, longest free-breathing, "
            "and lowest-confidence zone."
        )
        zones = _find_clinical_zones(windows_df, label_1, label_0)
        z1, z2, z3 = st.columns(3)
        with z1:
            st.markdown("**Longest %s stretch**" % label_1)
            if zones["longest_1"]:
                z = zones["longest_1"]
                st.metric("Duration", "%.1f s" % z["duration"])
                st.caption("%.1f – %.1f s (%d windows, avg conf %.2f)" % (z["start"], z["end"], z["count"], z["avg_conf"]))
            else:
                st.caption("No %s windows." % label_1)
        with z2:
            st.markdown("**Longest %s stretch**" % label_0)
            if zones["longest_0"]:
                z = zones["longest_0"]
                st.metric("Duration", "%.1f s" % z["duration"])
                st.caption("%.1f – %.1f s (%d windows, avg conf %.2f)" % (z["start"], z["end"], z["count"], z["avg_conf"]))
            else:
                st.caption("No %s windows." % label_0)
        with z3:
            st.markdown("**Lowest-confidence zone**")
            if zones["lowest_conf"]:
                z = zones["lowest_conf"]
                st.metric("Min confidence", "%.3f" % z["min_conf"])
                st.caption("%.1f – %.1f s (%d windows)" % (z["start"], z["end"], z["count"]))
            else:
                st.caption("N/A")
        st.info(
            "**Why this matters:** Longest breath-hold is clinically relevant for treatment planning "
            "(longer holds = more radiation delivery). Lowest-confidence zone may need manual review."
        )

    # --- TAB: Data Table ---
    with tab_data:
        windows_df = windows_df.copy()
        windows_df["window_index"] = range(1, len(windows_df) + 1)
        windows_df["prediction_label"] = windows_df["prediction"].apply(lambda p: prediction_to_label(int(p), task))
        windows_df["confidence_bucket"] = windows_df["confidence"].apply(confidence_bucket)

        display_cols = ["window_index", "time_start", "time_end", "prediction_label", "prediction",
                        "confidence", "confidence_bucket", "vol_mean", "vol_std", "vol_range",
                        "frac_balloon_inflated"]
        if "prob_class_0" in windows_df.columns:
            display_cols.extend(["prob_class_0", "prob_class_1"])
        available_cols = [c for c in display_cols if c in windows_df.columns]
        display_df = windows_df[available_cols].copy()
        for col in ["time_start", "time_end", "confidence", "vol_mean", "vol_std", "vol_range"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)

        display_rows = display_df.head(500)
        st.dataframe(display_rows, use_container_width=True, height=400)
        if len(display_df) > 500:
            st.caption("Showing first 500 of %d windows. Download CSV for full data." % len(display_df))

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Export")
        export_df = windows_df.copy()
        if "prediction_label" not in export_df.columns:
            export_df["prediction_label"] = export_df["prediction"].apply(lambda p: prediction_to_label(int(p), task))
        if "confidence_bucket" not in export_df.columns:
            export_df["confidence_bucket"] = export_df["confidence"].apply(confidence_bucket)
        csv = export_df.to_csv(index=False)
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                label="Download predictions (CSV)",
                data=csv,
                file_name="predictions_%s_%s.csv" % (task, file_id),
                mime="text/csv",
                key=f"{chart_key}_dl_pred_csv",
            )
        with d2:
            summary_text = _build_summary_text(
                file_name or file_id, task_label, metadata, session_duration_s, n,
                label_1, bh_count, pct_1, label_0, fb_count, pct_0,
                median_conf, low_conf_count, stability, n_transitions,
            )
            st.download_button(
                label="Download report (TXT)",
                data=summary_text,
                file_name="prediction_report_%s_%s.txt" % (task, file_id),
                mime="text/plain",
                key=f"{chart_key}_dl_report_txt",
            )

        with st.expander("Copy summary for report"):
            st.text(summary_text)

    # --- TAB: Glossary ---
    with tab_glossary:
        st.caption("Definitions of all terms, columns, and metrics used on this page.")
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

    # --- Input controls ---
    col1, col2 = st.columns(2)
    with col1:
        task = st.selectbox(
            "Task",
            ["breath_hold", "gating_ok"],
            index=0,
            help="breath_hold: Classify breath-hold vs free-breathing. gating_ok: Classify gating OK vs not OK.",
        )
        task_label, _, _ = _task_labels(task)
        st.caption("Classify: %s" % task_label)
    with col2:
        window_sec = st.slider(
            "Window Size (seconds)", 1.0, 5.0, cfg.WINDOW_SEC, 0.1,
            help="Smaller windows = more granular predictions. Larger windows = smoother predictions.",
        )
        st.caption("Size of time windows for feature extraction")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["dat", "txt", "csv"],
        help="Upload a breathing curve file (.dat, .txt, or .csv). Must have columns: Session Time, Volume, Balloon Valve Status, Patient Switch, Gating Mode.",
    )

    patient_id = st.text_input("Patient ID (optional)", value="uploaded", help="Identifier for the patient.")
    file_id = st.text_input("File ID (optional)", value="uploaded", help="Identifier for this file.")

    if uploaded_file is not None:
        if st.button("Run Prediction", type="primary"):
            tmp_path = None
            with st.spinner("Processing file and running predictions..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".%s" % uploaded_file.name.split(".")[-1]) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = Path(tmp.name)

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
