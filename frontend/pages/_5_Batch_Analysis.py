"""
Batch Analysis page.
"""
import streamlit as st
import tempfile
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.utils.inference import predict_breathing_pattern
from frontend.utils.visualization import plot_breathing_curve
from frontend.utils.icons import icon_html
import config as cfg


def show():
    st.markdown(f'<h1 style="display: flex; align-items: center; gap: 10px;">{icon_html("package", 32)} Batch Analysis</h1>', unsafe_allow_html=True)
    st.markdown("Upload multiple breathing curve files and process them in batch.")
    
    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        task = st.selectbox("Task", ["breath_hold", "gating_ok"], index=0)
        task_label = "Breath-hold vs Free-breathing" if task == "breath_hold" else "Gating OK vs Not OK"
        st.caption(f"Classify: {task_label}")
    
    with col2:
        window_sec = st.slider("Window Size (seconds)", 1.0, 5.0, cfg.WINDOW_SEC, 0.1)
        st.caption("Size of time windows for feature extraction")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["dat", "txt", "csv"],
        accept_multiple_files=True,
        help="Upload multiple breathing curve files"
    )
    
    if uploaded_files and len(uploaded_files) > 0:
        if st.button("Process All Files", type="primary"):
            results = []
            all_windows = []
            temp_files = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
                progress_bar.progress((idx + 1) / len(uploaded_files))
                
                try:
                    # Save temporarily
                    suffix = f".{uploaded_file.name.split('.')[-1]}"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = Path(tmp.name)
                        temp_files.append(tmp_path)
                    
                    # Predict
                    windows_df, metadata = predict_breathing_pattern(
                        tmp_path,
                        task=task,
                        window_sec=window_sec,
                        patient_id=f"batch_{idx}",
                        file_id=uploaded_file.name,
                    )
                    
                    # Aggregate results
                    breath_hold_count = (windows_df["prediction"] == 1).sum()
                    free_breathing_count = (windows_df["prediction"] == 0).sum()
                    avg_confidence = windows_df["confidence"].mean()
                    
                    results.append({
                        "File": uploaded_file.name,
                        "Rows": metadata["num_rows"],
                        "Windows": metadata["num_windows"],
                        "Breath-hold": breath_hold_count,
                        "Free-breathing": free_breathing_count,
                        "Avg Confidence": f"{avg_confidence:.3f}",
                        "Model": metadata["model_name"],
                    })
                    
                    # Add file_id to windows for tracking
                    windows_df["source_file"] = uploaded_file.name
                    all_windows.append(windows_df)
                    
                except Exception as e:
                    results.append({
                        "File": uploaded_file.name,
                        "Rows": "Error",
                        "Windows": "Error",
                        "Breath-hold": "N/A",
                        "Free-breathing": "N/A",
                        "Avg Confidence": "N/A",
                        "Model": f"Error: {str(e)[:50]}",
                    })
            
            # Cleanup temp files
            for tmp_path in temp_files:
                try:
                    tmp_path.unlink()
                except:
                    pass
            
            progress_bar.empty()
            status_text.empty()
            
            # Display summary table
            st.success(f"Processed {len([r for r in results if r['Windows'] != 'Error'])}/{len(uploaded_files)} files successfully")
            
            st.subheader("Summary Table")
            summary_df = pd.DataFrame(results)
            st.dataframe(summary_df, use_container_width=True)
            
            # Combined windows DataFrame
            if len(all_windows) > 0:
                combined_windows = pd.concat(all_windows, ignore_index=True)
                
                st.subheader("Combined Predictions")
                st.dataframe(combined_windows, use_container_width=True, height=400)
                
                # Download combined CSV
                csv = combined_windows.to_csv(index=False)
                st.download_button(
                    label="Download Combined Predictions CSV",
                    data=csv,
                    file_name=f"batch_predictions_{task}.csv",
                    mime="text/csv",
                )
                
                # Comparison view
                st.subheader("File Comparison")
                selected_files = st.multiselect(
                    "Select files to compare",
                    options=[r["File"] for r in results if r["Windows"] != "Error"],
                    default=[r["File"] for r in results if r["Windows"] != "Error"][:3] if len(results) > 0 else [],
                )
                
                if selected_files:
                    # Load and plot selected files
                    from src.load_data import load_patient_file
                    cols = st.columns(len(selected_files))
                    
                    for idx, filename in enumerate(selected_files):
                        file_idx = next(i for i, r in enumerate(results) if r["File"] == filename)
                        uploaded_file = uploaded_files[file_idx]
                        
                        with cols[idx]:
                            st.write(f"**{filename}**")
                            
                            # Reload file for plotting
                            try:
                                suffix = f".{filename.split('.')[-1]}"
                                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                    tmp.write(uploaded_file.getvalue())
                                    tmp_path = Path(tmp.name)
                                
                                df = load_patient_file(tmp_path, patient_id=f"batch_{file_idx}", file_id=filename)
                                file_windows = all_windows[file_idx]
                                
                                fig = plot_breathing_curve(df, predictions=file_windows)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                tmp_path.unlink()
                            except Exception as e:
                                st.error(f"Error plotting: {e}")
