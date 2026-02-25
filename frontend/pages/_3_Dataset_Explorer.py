"""
Dataset Explorer page.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.utils.data_helpers import get_patient_list, get_file_list, load_summary_stats
from frontend.utils.error_handling import user_facing_message
from frontend.utils.visualization import (
    plot_breathing_curve,
    plot_bar_counts,
    plot_volume_histogram,
    export_fig_png,
)
from frontend.utils.icons import icon_html
from src.load_data import load_patient_file, get_patient_session_ini
from src.features import build_windows
import config as cfg


@st.cache_data(ttl=300)
def load_patient_data(patient_id: str, file_id: str, file_path_str: str):
    """Cache loaded patient data. Pass file_path as string for stable hashing."""
    from pathlib import Path
    return load_patient_file(Path(file_path_str), patient_id=patient_id, file_id=file_id)


def show():
    st.markdown(f'<p class="page-header">{icon_html("search", 28)} Dataset Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Browse and visualize existing patient breathing curve data.</p>', unsafe_allow_html=True)
    
    with st.expander("Dataset-wide stats (from analysis)"):
        try:
            file_summary, patient_summary, _ = load_summary_stats()
            if len(file_summary) > 0 or len(patient_summary) > 0:
                st.caption("**Total files:** %d | **Total duration:** %s" % (
                    len(file_summary),
                    "%.1f s" % patient_summary["duration_s"].sum() if len(patient_summary) > 0 and "duration_s" in patient_summary.columns else "N/A",
                ))
                if len(patient_summary) > 0:
                    st.dataframe(patient_summary.head(20), use_container_width=True, height=200)
                    st.caption("Patient-level summary (first 20). Run analysis/analyze_dataset.py to refresh.")
            else:
                st.caption("No summary data yet. Run analysis/analyze_dataset.py to generate file_summary.csv and patient_summary.csv.")
        except Exception as e:
            st.caption("Could not load summary stats: " + user_facing_message(e))
    
    # Patient selector
    patients = get_patient_list()
    if len(patients) == 0:
        st.warning("No patients found in dataset directory.")
        return
    
    patient_id = st.selectbox("Select Patient", patients)
    
    # File selector
    files = get_file_list(patient_id)
    if len(files) == 0:
        st.warning(f"No curve files found for patient: {patient_id}")
        return
    
    file_options = [f"{fid} ({path.name})" for fid, path in files]
    file_idx = st.selectbox("Select File", range(len(file_options)), format_func=lambda i: file_options[i])
    
    if file_idx is not None:
        file_id, file_path = files[file_idx]
        
        with st.spinner("Loading file..."):
            try:
                df = load_patient_data(patient_id, file_id, str(file_path))
                
                if len(df) == 0:
                    st.error(user_facing_message(ValueError("Empty file or parsing error")))
                    return
                
                # Statistics panel
                st.subheader("Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                duration = df["Session Time"].max() - df["Session Time"].min()
                with col1:
                    st.metric("Duration", f"{duration:.2f}s")
                with col2:
                    st.metric("Rows", len(df))
                with col3:
                    st.metric("Volume Min", f"{df['Volume (liters)'].min():.3f}L")
                with col4:
                    st.metric("Volume Max", f"{df['Volume (liters)'].max():.3f}L")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Volume Mean", f"{df['Volume (liters)'].mean():.3f}L")
                with col2:
                    st.metric("Volume Std", f"{df['Volume (liters)'].std():.3f}L")
                
                # Balloon status counts (Plotly for export and consistency)
                if "Balloon Valve Status" in df.columns:
                    st.subheader("Balloon Valve Status")
                    balloon_counts = df["Balloon Valve Status"].astype(str).value_counts()
                    fig_balloon = plot_bar_counts(balloon_counts, "Balloon Valve Status", "Status")
                    st.plotly_chart(fig_balloon, use_container_width=True, key="explorer_balloon")
                    png_b = export_fig_png(fig_balloon)
                    if png_b:
                        st.download_button("Download PNG", data=png_b, file_name="balloon_status.png", mime="image/png", key="dl_balloon_png")
                
                # Gating mode counts
                if "Gating Mode" in df.columns:
                    st.subheader("Gating Mode")
                    gating_counts = df["Gating Mode"].astype(str).value_counts()
                    fig_gating = plot_bar_counts(gating_counts, "Gating Mode", "Mode")
                    st.plotly_chart(fig_gating, use_container_width=True, key="explorer_gating")
                    png_g = export_fig_png(fig_gating)
                    if png_g:
                        st.download_button("Download PNG", data=png_g, file_name="gating_mode.png", mime="image/png", key="dl_gating_png")
                
                # Optional volume distribution (research)
                with st.expander("Volume distribution (histogram)"):
                    fig_hist = plot_volume_histogram(df["Volume (liters)"], title=f"Volume (L) - {file_path.name}", bins=50)
                    st.plotly_chart(fig_hist, use_container_width=True, key="explorer_hist")
                
                # Visualization
                st.subheader("Breathing Curve")
                show_balloon = st.checkbox("Show Balloon Valve Status", value=True)
                show_gating = st.checkbox("Show Gating Mode", value=False)
                show_rolling_mean = st.checkbox("Show rolling mean", value=False, help="Overlay rolling mean of volume")
                
                fig = plot_breathing_curve(
                    df,
                    show_balloon=show_balloon,
                    show_gating=show_gating,
                    show_rolling_mean=show_rolling_mean,
                    subtitle=file_path.name,
                )
                st.plotly_chart(fig, use_container_width=True, key="explorer_curve")
                st.caption("**Research context:** File: %s | Patient: %s | Duration: %.2f s" % (file_path.name, patient_id, duration))
                png_curve = export_fig_png(fig)
                if png_curve:
                    st.download_button("Download chart (PNG)", data=png_curve, file_name="breathing_curve.png", mime="image/png", key="dl_explorer_curve_png")
                curve_csv = df[["Session Time", "Volume (liters)"]].head(15000).to_csv(index=False)
                st.download_button("Download curve data (CSV)", data=curve_csv, file_name=f"curve_{patient_id}_{file_id}.csv", mime="text/csv", key="dl_explorer_curve_csv")
                
                session_blocks = get_patient_session_ini(cfg.DATASET_DIR, patient_id)
                if session_blocks:
                    with st.expander("Session metadata (session.ini)"):
                        for i, block in enumerate(session_blocks):
                            st.caption("Block %d" % (i + 1))
                            st.json(block)
                
                file_stats = {
                    "patient_id": [patient_id],
                    "file": [file_path.name],
                    "duration_s": [round(duration, 2)],
                    "rows": [len(df)],
                    "volume_min_L": [round(df["Volume (liters)"].min(), 4)],
                    "volume_max_L": [round(df["Volume (liters)"].max(), 4)],
                    "volume_mean_L": [round(df["Volume (liters)"].mean(), 4)],
                }
                if "Balloon Valve Status" in df.columns:
                    bc = df["Balloon Valve Status"].astype(str).value_counts()
                    file_stats["balloon_counts"] = ["; ".join("%s: %d" % (k, v) for k, v in bc.items())]
                if "Gating Mode" in df.columns:
                    gc = df["Gating Mode"].astype(str).value_counts()
                    file_stats["gating_counts"] = ["; ".join("%s: %d" % (k, v) for k, v in gc.items())]
                file_stats_df = pd.DataFrame(file_stats)
                st.download_button("Download this file's stats (CSV)", data=file_stats_df.to_csv(index=False), file_name=f"file_stats_{patient_id}_{file_id}.csv", mime="text/csv", key="dl_explorer_file_stats")
                
                # Window breakdown (ground truth labels)
                st.subheader("Window Breakdown (Ground Truth)")
                window_sec = st.slider("Window Size (seconds)", 1.0, 5.0, cfg.WINDOW_SEC, 0.1, key="explorer_window")
                
                if st.button("Compute Windows"):
                    with st.spinner("Building windows..."):
                        try:
                            windows = build_windows(
                                df,
                                window_sec=window_sec,
                                sample_rate_hz=cfg.SAMPLE_RATE_HZ,
                                min_rows=cfg.MIN_WINDOW_ROWS,
                            )
                        except Exception as e:
                            st.error("Error building windows: " + user_facing_message(e))
                            windows = None
                        
                        if windows is not None and len(windows) > 0:
                            st.info(f"Created {len(windows)} windows")
                            
                            # Plot with ground truth labels
                            fig = plot_breathing_curve(df, predictions=windows.rename(columns={"label_breath_hold": "prediction"}))
                            st.plotly_chart(fig, use_container_width=True, key="explorer_windows_curve")
                            
                            # Window summary
                            breath_hold_count = (windows["label_breath_hold"] == 1).sum()
                            free_breathing_count = (windows["label_breath_hold"] == 0).sum()
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Breath-hold Windows", breath_hold_count)
                            with col2:
                                st.metric("Free-breathing Windows", free_breathing_count)
                            
                            wdisp = windows[["time_start", "time_end", "label_breath_hold", "vol_mean", "vol_std"]].head(500)
                            st.dataframe(wdisp, use_container_width=True)
                            if len(windows) > 500:
                                st.caption(f"Showing first 500 of {len(windows)} windows.")
                        else:
                            if windows is not None:
                                st.warning("No windows produced (file too short?)")
                
            except Exception as e:
                st.error("Error loading file: " + user_facing_message(e))
                st.exception(e)
    
    # Report generation
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(f'<p class="page-header" style="font-size:1.3rem;">{icon_html("file", 22)} Generate Report</p>', unsafe_allow_html=True)
    if st.button("Generate Patient Report with AI", help="Generate AI-powered analysis report for this patient"):
        try:
            from frontend.utils.llm_helper import generate_report, is_llm_available
            if is_llm_available():
                with st.spinner("Generating report..."):
                    analysis_data = {
                        "patient_id": patient_id,
                        "files": [f[1].name for f in files],
                        "selected_file": files[file_idx][1].name if file_idx < len(files) else ""
                    }
                    report = generate_report(analysis_data, "patient_summary")
                    st.markdown("### Generated Report")
                    st.markdown(report)
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name=f"patient_report_{patient_id}.txt",
                        mime="text/plain"
                    )
            else:
                st.info("AI features require Ollama API. Go to AI Assistant page for setup instructions.")
        except Exception as e:
            st.warning(f"Could not generate report: {user_facing_message(e)}")
