"""
Dataset Explorer page.
"""
import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.utils.data_helpers import get_patient_list, get_file_list
from frontend.utils.visualization import plot_breathing_curve
from frontend.utils.icons import icon_html
from src.load_data import load_patient_file
from src.features import build_windows
import config as cfg


@st.cache_data(ttl=300)
def load_patient_data(patient_id: str, file_id: str, file_path_str: str):
    """Cache loaded patient data. Pass file_path as string for stable hashing."""
    from pathlib import Path
    return load_patient_file(Path(file_path_str), patient_id=patient_id, file_id=file_id)


def show():
    st.markdown(f'<h1 style="display: flex; align-items: center; gap: 10px;">{icon_html("search", 32)} Dataset Explorer</h1>', unsafe_allow_html=True)
    st.markdown("Browse and visualize existing patient breathing curve data.")
    
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
                    st.error("Empty file or parsing error")
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
                
                # Balloon status counts
                if "Balloon Valve Status" in df.columns:
                    st.subheader("Balloon Valve Status")
                    balloon_counts = df["Balloon Valve Status"].astype(str).value_counts()
                    st.bar_chart(balloon_counts)
                
                # Gating mode counts
                if "Gating Mode" in df.columns:
                    st.subheader("Gating Mode")
                    gating_counts = df["Gating Mode"].astype(str).value_counts()
                    st.bar_chart(gating_counts)
                
                # Visualization
                st.subheader("Breathing Curve")
                show_balloon = st.checkbox("Show Balloon Valve Status", value=True)
                show_gating = st.checkbox("Show Gating Mode", value=False)
                
                fig = plot_breathing_curve(df, show_balloon=show_balloon, show_gating=show_gating)
                st.plotly_chart(fig, use_container_width=True)
                
                # Window breakdown (ground truth labels)
                st.subheader("Window Breakdown (Ground Truth)")
                window_sec = st.slider("Window Size (seconds)", 1.0, 5.0, cfg.WINDOW_SEC, 0.1, key="explorer_window")
                
                if st.button("Compute Windows"):
                    with st.spinner("Building windows..."):
                        windows = build_windows(
                            df,
                            window_sec=window_sec,
                            sample_rate_hz=cfg.SAMPLE_RATE_HZ,
                            min_rows=cfg.MIN_WINDOW_ROWS,
                        )
                        
                        if len(windows) > 0:
                            st.info(f"Created {len(windows)} windows")
                            
                            # Plot with ground truth labels
                            fig = plot_breathing_curve(df, predictions=windows.rename(columns={"label_breath_hold": "prediction"}))
                            st.plotly_chart(fig, use_container_width=True)
                            
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
                            st.warning("No windows produced (file too short?)")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.exception(e)
    
    # Report generation option
    st.markdown("---")
    st.markdown(f'<h3 style="display: flex; align-items: center; gap: 8px;">{icon_html("file", 24)} Generate Report</h3>', unsafe_allow_html=True)
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
            st.warning(f"Could not generate report: {str(e)}")
