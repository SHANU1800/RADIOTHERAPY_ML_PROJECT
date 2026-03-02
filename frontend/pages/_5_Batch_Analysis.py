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
from frontend.utils.visualization import plot_breathing_curve, export_fig_png
from frontend.utils.data_helpers import prediction_to_label, confidence_bucket
from frontend.utils.error_handling import user_facing_message
from frontend.utils.icons import icon_html
import config as cfg


def show():
    st.markdown(f'<p class="page-header">{icon_html("package", 28)} Batch Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Upload multiple breathing curve files and process them all at once.</p>', unsafe_allow_html=True)
    
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
                    nw = len(windows_df)
                    pred_1 = (windows_df["prediction"] == 1).sum()
                    pred_0 = (windows_df["prediction"] == 0).sum()
                    conf = windows_df["confidence"]
                    median_conf = conf.median()
                    low_conf = (conf < 0.7).sum()
                    pct_1 = 100.0 * pred_1 / nw if nw else 0
                    pct_0 = 100.0 * pred_0 / nw if nw else 0
                    label_1 = "Breath-hold" if task == "breath_hold" else "Gating OK"
                    label_0 = "Free-breathing" if task == "breath_hold" else "Gating Not OK"
                    windows_df["prediction_label"] = windows_df["prediction"].apply(lambda p: prediction_to_label(int(p), task))
                    windows_df["confidence_bucket"] = windows_df["confidence"].apply(confidence_bucket)
                    
                    results.append({
                        "File": uploaded_file.name,
                        "Rows": metadata["num_rows"],
                        "Windows": metadata["num_windows"],
                        label_1: pred_1,
                        label_0: pred_0,
                        f"% {label_1}": f"{pct_1:.1f}%",
                        f"% {label_0}": f"{pct_0:.1f}%",
                        "Median Confidence": f"{median_conf:.3f}",
                        "Low-conf (<0.7)": int(low_conf),
                        "Avg Confidence": f"{conf.mean():.3f}",
                        "Model": metadata["model_name"],
                    })
                    
                    # Add file_id to windows for tracking
                    windows_df["source_file"] = uploaded_file.name
                    all_windows.append(windows_df)
                    
                except Exception as e:
                    label_1 = "Breath-hold" if task == "breath_hold" else "Gating OK"
                    label_0 = "Free-breathing" if task == "breath_hold" else "Gating Not OK"
                    results.append({
                        "File": uploaded_file.name,
                        "Rows": "Error",
                        "Windows": "Error",
                        label_1: "N/A",
                        label_0: "N/A",
                        f"% {label_1}": "N/A",
                        f"% {label_0}": "N/A",
                        "Median Confidence": "N/A",
                        "Low-conf (<0.7)": "N/A",
                        "Avg Confidence": "N/A",
                        "Model": user_facing_message(e),
                    })
            
            # Cleanup temp files
            for tmp_path in temp_files:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            
            progress_bar.empty()
            status_text.empty()
            
            # Display summary
            n_ok = len([r for r in results if r['Windows'] != 'Error'])
            n_err = len(uploaded_files) - n_ok
            panel_cls = "result-panel-success" if n_err == 0 else "result-panel-warning"
            failed_badge = f' &nbsp;<span class="badge badge-red">{n_err} failed</span>' if n_err else ""
            st.markdown(
                f'<div class="result-panel {panel_cls}">'
                f'<strong>Processed {n_ok}/{len(uploaded_files)} files</strong> &nbsp;'
                f'<span class="badge badge-green">{n_ok} OK</span>'
                f'{failed_badge}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("#### Summary Table")
            summary_df = pd.DataFrame(results)
            st.dataframe(summary_df, use_container_width=True)
            
            # Combined windows DataFrame
            if len(all_windows) > 0:
                combined_windows = pd.concat(all_windows, ignore_index=True)
                if "prediction_label" not in combined_windows.columns:
                    combined_windows["prediction_label"] = combined_windows["prediction"].apply(lambda p: prediction_to_label(int(p), task))
                if "confidence_bucket" not in combined_windows.columns:
                    combined_windows["confidence_bucket"] = combined_windows["confidence"].apply(confidence_bucket)
                
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
                    from src.load_data import load_patient_file
                    cols = st.columns(min(len(selected_files), 3))
                    
                    for idx, filename in enumerate(selected_files):
                        file_idx = next((i for i, r in enumerate(results) if r["File"] == filename), None)
                        if file_idx is None or file_idx >= len(uploaded_files):
                            continue
                        uploaded_file = uploaded_files[file_idx]
                        file_windows = combined_windows[combined_windows["source_file"] == filename]
                        
                        with cols[idx % len(cols)]:
                            st.write(f"**{filename}**")
                            
                            tmp_path = None
                            try:
                                suffix = f".{filename.split('.')[-1]}"
                                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                    tmp.write(uploaded_file.getvalue())
                                    tmp_path = Path(tmp.name)
                                
                                df = load_patient_file(tmp_path, patient_id=f"batch_{file_idx}", file_id=filename)
                                
                                fig = plot_breathing_curve(
                                    df,
                                    predictions=file_windows,
                                    subtitle=f"{filename}, window={window_sec}s",
                                )
                                st.plotly_chart(fig, use_container_width=True, key=f"batch_compare_plot_{idx}")
                                n_f = len(file_windows)
                                if n_f > 0:
                                    p1 = (file_windows["prediction"] == 1).sum()
                                    p0 = (file_windows["prediction"] == 0).sum()
                                    mc = file_windows["confidence"].median()
                                    lab1 = "Breath-hold" if task == "breath_hold" else "Gating OK"
                                    lab0 = "Free-breathing" if task == "breath_hold" else "Gating Not OK"
                                    st.caption(f"{lab1}: {100*p1/n_f:.0f}% | {lab0}: {100*p0/n_f:.0f}% | Median confidence: {mc:.2f}")
                                png_batch = export_fig_png(fig)
                                if png_batch:
                                    st.download_button("Download PNG", data=png_batch, file_name=f"curve_{filename}.png", mime="image/png", key=f"dl_batch_png_{idx}")
                                fw_export = file_windows.copy()
                                if "prediction_label" not in fw_export.columns:
                                    fw_export["prediction_label"] = fw_export["prediction"].apply(lambda p: prediction_to_label(int(p), task))
                                if "confidence_bucket" not in fw_export.columns:
                                    fw_export["confidence_bucket"] = fw_export["confidence"].apply(confidence_bucket)
                                file_csv = fw_export.to_csv(index=False)
                                st.download_button("Download this file's predictions (CSV)", data=file_csv, file_name=f"predictions_{filename}.csv", mime="text/csv", key=f"dl_batch_csv_{idx}")
                            except Exception as e:
                                st.error("Error plotting: " + user_facing_message(e))
                            finally:
                                if tmp_path is not None:
                                    try:
                                        tmp_path.unlink(missing_ok=True)
                                    except OSError:
                                        pass
