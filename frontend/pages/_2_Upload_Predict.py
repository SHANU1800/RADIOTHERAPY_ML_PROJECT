"""
Upload & Predict page.
"""
import streamlit as st
import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.utils.inference import predict_breathing_pattern
from frontend.utils.visualization import plot_breathing_curve
from frontend.utils.icons import icon_html
import config as cfg


def _display_predictions(windows_df, metadata, df, task, file_id, chart_key="predictions_chart"):
    """Helper function to display prediction results. chart_key ensures unique Streamlit element IDs."""
    # Display results
    st.success(f"Processed {metadata['num_rows']} rows, {metadata['num_windows']} windows")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        breath_hold_count = (windows_df["prediction"] == 1).sum()
        st.metric("Breath-hold Windows", breath_hold_count)
    with col2:
        free_breathing_count = (windows_df["prediction"] == 0).sum()
        st.metric("Free-breathing Windows", free_breathing_count)
    with col3:
        avg_confidence = windows_df["confidence"].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
    with col4:
        st.metric("Model", metadata["model_name"])
    
    # Interactive plot (unique key to avoid duplicate element ID)
    st.subheader("Breathing Curve with Predictions")
    fig = plot_breathing_curve(df, predictions=windows_df)
    st.plotly_chart(fig, use_container_width=True, key=chart_key)
    
    # Window-level table
    st.subheader("Window-Level Predictions")
    display_cols = [
        "time_start", "time_end", "prediction", "confidence",
        "vol_mean", "vol_std", "frac_balloon_inflated"
    ]
    if "prob_class_0" in windows_df.columns:
        display_cols.extend(["prob_class_0", "prob_class_1"])
    
    available_cols = [c for c in display_cols if c in windows_df.columns]
    # Format numeric columns
    display_df = windows_df[available_cols].copy()
    for col in ["time_start", "time_end", "confidence", "vol_mean", "vol_std"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
    
    # Show at most 500 rows to keep UI responsive
    display_rows = display_df.head(500)
    st.dataframe(display_rows, use_container_width=True, height=400)
    if len(display_df) > 500:
        st.caption(f"Showing first 500 of {len(display_df)} windows. Use CSV download for full data.")
    
    # Download CSV
    csv = windows_df.to_csv(index=False)
    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name=f"predictions_{task}_{file_id}.csv",
        mime="text/csv",
    )


def show():
    st.markdown(f'<h1 style="display: flex; align-items: center; gap: 10px;">{icon_html("upload", 32)} Upload & Predict</h1>', unsafe_allow_html=True)
    st.markdown("Upload a breathing curve file (.dat, .txt, or .csv) to get predictions.")
    
    # Initialize session state for predictions
    if "prediction_results" not in st.session_state:
        st.session_state.prediction_results = None
    if "prediction_metadata" not in st.session_state:
        st.session_state.prediction_metadata = None
    if "prediction_task" not in st.session_state:
        st.session_state.prediction_task = None
    if "show_explanation" not in st.session_state:
        st.session_state.show_explanation = False
    
    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        task = st.selectbox(
            "Task",
            ["breath_hold", "gating_ok"],
            index=0,
            help="breath_hold: Classify breath-hold vs free-breathing patterns. gating_ok: Classify gating OK vs not OK."
        )
        task_label = "Breath-hold vs Free-breathing" if task == "breath_hold" else "Gating OK vs Not OK"
        st.caption(f"Classify: {task_label}")
    
    with col2:
        window_sec = st.slider(
            "Window Size (seconds)",
            1.0, 5.0, cfg.WINDOW_SEC, 0.1,
            help="Size of time windows for feature extraction. Smaller windows = more granular predictions. Larger windows = smoother predictions."
        )
        st.caption("Size of time windows for feature extraction")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["dat", "txt", "csv"],
        help="Upload a breathing curve file (.dat, .txt, or .csv). File must have columns: Session Time, Volume, Balloon Valve Status, Patient Switch, Gating Mode."
    )
    
    patient_id = st.text_input("Patient ID (optional)", value="uploaded", help="Identifier for the patient. Used for tracking and organization.")
    file_id = st.text_input("File ID (optional)", value="uploaded", help="Identifier for this file. Used in output filenames.")
    
    if uploaded_file is not None:
        if st.button("Run Prediction", type="primary"):
            with st.spinner("Processing file and running predictions..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = Path(tmp.name)
                    
                    # Run prediction
                    windows_df, metadata = predict_breathing_pattern(
                        tmp_path,
                        task=task,
                        window_sec=window_sec,
                        patient_id=patient_id,
                        file_id=file_id,
                    )
                    
                    # Load original data for plotting
                    from src.load_data import load_patient_file
                    df = load_patient_file(tmp_path, patient_id=patient_id, file_id=file_id)
                    
                    # Cleanup
                    tmp_path.unlink()
                    
                    # Store in session state for persistence
                    st.session_state.prediction_results = windows_df
                    st.session_state.prediction_metadata = metadata
                    st.session_state.prediction_task = task
                    st.session_state.prediction_df = df
                    st.session_state.prediction_file_id = file_id
                    # Clear explanation when new prediction is run
                    st.session_state.show_explanation = False
                    if "ai_explanation" in st.session_state:
                        del st.session_state.ai_explanation
                    
                    # Rerun so we only render from session state block (avoids duplicate plotly_chart)
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.exception(e)
                    # Clear session state on error
                    st.session_state.prediction_results = None
                    st.session_state.prediction_metadata = None
                    st.session_state.prediction_task = None
                    st.session_state.prediction_df = None
    
    # Show predictions from session state (single place we render — avoids duplicate elements)
    if st.session_state.prediction_results is not None:
        windows_df = st.session_state.prediction_results
        metadata = st.session_state.prediction_metadata
        df = st.session_state.prediction_df
        task = st.session_state.prediction_task
        file_id = st.session_state.get("prediction_file_id", "uploaded")
        
        # Display predictions (unique key for plotly chart)
        _display_predictions(windows_df, metadata, df, task, file_id, chart_key="upload_predict_curve")
        
        # AI Explanation section
        st.markdown("---")
        st.markdown(f'<h3 style="display: flex; align-items: center; gap: 8px;">{icon_html("robot", 24)} AI Explanation</h3>', unsafe_allow_html=True)
        
        if st.button("Explain Predictions with AI", help="Get AI-powered explanation of prediction results", key="explain_btn"):
            try:
                from frontend.utils.llm_helper import explain_prediction, is_llm_available
                if is_llm_available():
                    with st.spinner("Generating AI explanation..."):
                        model_info = {
                            "model_name": metadata.get("model_name", "Best Model"),
                            "task": task
                        }
                        explanation = explain_prediction(windows_df, model_info, task)
                        st.session_state.ai_explanation = explanation
                        st.session_state.show_explanation = True
                        st.rerun()
                else:
                    st.info("AI features require Ollama API. Go to AI Assistant page for setup instructions.")
            except Exception as e:
                st.error(f"Could not generate AI explanation: {str(e)}")
                st.exception(e)
        
        # Show explanation if it exists
        if "ai_explanation" in st.session_state and st.session_state.get("show_explanation", False):
            st.markdown("### Explanation")
            st.markdown(st.session_state.ai_explanation)

