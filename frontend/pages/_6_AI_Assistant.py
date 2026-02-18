"""
AI Assistant page - LLM-powered explanations and Q&A.
"""
import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.utils.llm_helper import (
    explain_prediction,
    answer_question,
    generate_report,
    is_llm_available,
    get_llm_status
)
from frontend.utils.data_helpers import load_summary_stats
from frontend.utils.icons import icon_html
import config as cfg


def show():
    st.markdown(f'<h1 style="display: flex; align-items: center; gap: 10px;">{icon_html("robot", 32)} AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown("Ask questions, get explanations, and generate reports using AI.")
    
    # Check LLM status
    llm_status = get_llm_status()
    
    if not llm_status["available"]:
        st.markdown(f'<div style="display: flex; align-items: center; gap: 8px;">{icon_html("alert", 20, "#ffc107")} <strong>LLM integration not available.</strong></div>', unsafe_allow_html=True)
        if not llm_status.get("requests_installed", True):
            st.markdown(f'<div style="display: flex; align-items: center; gap: 8px;">{icon_html("x", 20, "#dc3545")} <strong>requests library not installed.</strong> Install with: pip install requests</div>', unsafe_allow_html=True)
        else:
            st.info(f"""
            **To enable AI features:**
            1. Install and run Ollama: https://ollama.ai
            2. Pull the model: `ollama pull {llm_status["model"]}`
            3. Make sure Ollama is running: `ollama serve` or start Ollama application
            4. Restart the frontend
            
            **Current Configuration:**
            - API URL: {llm_status["api_url"]}
            - Model: {llm_status["model"]}
            - Status: {'Configured' if llm_status['configured'] else 'Using defaults'}
            - Error: {llm_status.get('note', 'Unknown error')}
            """)
    else:
        st.markdown(f'<div style="display: flex; align-items: center; gap: 8px;">{icon_html("check", 20, "#28a745")} <strong>LLM integration is active and ready!</strong></div>', unsafe_allow_html=True)
        st.info(f"""
        **Configuration:**
        - API URL: {llm_status["api_url"]}
        - Model: {llm_status["model"]}
        - Status: Ready to use
        
        You can now ask questions, get explanations, and generate reports!
        """)
    
    # Tabs for different AI features
    tab1, tab2, tab3 = st.tabs(["Ask Questions", "Explain Predictions", "Generate Reports"])
    
    with tab1:
        st.subheader("Ask Questions About the System")
        st.markdown("Ask questions about the dataset, models, predictions, or how to use the system.")
        
        # Get context for questions
        try:
            file_summary, patient_summary, _ = load_summary_stats()
            dataset_info = {
                "total_patients": len(patient_summary),
                "total_files": len(file_summary),
                "total_duration_hours": patient_summary["duration_s"].sum() / 3600 if "duration_s" in patient_summary.columns else 0
            }
        except:
            dataset_info = {}
        
        # Model info
        model_path = cfg.MODELS_DIR / "best_model.pkl"
        model_info = {
            "model_available": model_path.exists(),
            "models_dir": str(cfg.MODELS_DIR)
        }
        
        # Question input
        question = st.text_area(
            "Enter your question:",
            placeholder="e.g., How does the breath-hold classification work? What features are most important? How do I interpret confidence scores?",
            height=100
        )
        
        if st.button("Ask", type="primary"):
            if not question.strip():
                st.warning("Please enter a question.")
            elif not llm_status["available"]:
                st.error("LLM not available. Please configure Ollama API.")
            else:
                with st.spinner("Thinking..."):
                    context = {
                        "dataset_info": dataset_info,
                        "model_info": model_info
                    }
                    answer = answer_question(question, context)
                    st.markdown("### Answer")
                    st.markdown(answer)
    
    with tab2:
        st.subheader("Explain Prediction Results")
        st.markdown("Upload a predictions CSV or provide prediction data to get AI explanations.")
        
        # File upload for predictions
        uploaded_file = st.file_uploader(
            "Upload predictions CSV (optional)",
            type=["csv"],
            help="Upload a CSV file with predictions from Upload & Predict or Batch Analysis page"
        )
        
        # Or manual input
        st.markdown("**Or enter prediction summary manually:**")
        col1, col2 = st.columns(2)
        with col1:
            total_windows = st.number_input("Total Windows", min_value=1, value=100)
            breath_hold_windows = st.number_input("Breath-hold Windows", min_value=0, value=50)
        with col2:
            free_breathing_windows = total_windows - breath_hold_windows
            avg_confidence = st.number_input("Average Confidence", min_value=0.0, max_value=1.0, value=0.85, step=0.01)
        
        task = st.selectbox("Task", ["breath_hold", "gating_ok"], index=0)
        
        if st.button("Explain Predictions", type="primary"):
            if not llm_status["available"]:
                st.error("LLM not available. Please configure Ollama API.")
            else:
                with st.spinner("Generating explanation..."):
                    # Create mock predictions DataFrame
                    import pandas as pd
                    predictions_df = pd.DataFrame({
                        "prediction": [1] * breath_hold_windows + [0] * free_breathing_windows,
                        "confidence": [avg_confidence] * total_windows,
                        "vol_mean": [0.05] * total_windows,
                        "vol_std": [0.15] * total_windows
                    })
                    
                    model_info = {
                        "model_name": "Best Model",
                        "task": task
                    }
                    
                    explanation = explain_prediction(predictions_df, model_info, task)
                    st.markdown("### Explanation")
                    st.markdown(explanation)
    
    with tab3:
        st.subheader("Generate Analysis Reports")
        st.markdown("Generate comprehensive reports for patients, predictions, or dataset overview.")
        
        report_type = st.selectbox(
            "Report Type",
            ["patient_summary", "prediction_analysis", "dataset_overview"],
            format_func=lambda x: {
                "patient_summary": "Patient Summary",
                "prediction_analysis": "Prediction Analysis",
                "dataset_overview": "Dataset Overview"
            }[x]
        )
        
        # Get dataset info for reports
        try:
            file_summary, patient_summary, _ = load_summary_stats()
            analysis_data = {
                "total_patients": len(patient_summary),
                "total_files": len(file_summary),
                "patient_summary": patient_summary.to_dict("records")[:5] if len(patient_summary) > 0 else [],
                "file_summary": file_summary.to_dict("records")[:10] if len(file_summary) > 0 else []
            }
        except:
            analysis_data = {}
        
        if report_type == "patient_summary":
            patient_id = st.text_input("Patient ID (optional)", value="")
            if patient_id:
                analysis_data["patient_id"] = patient_id
        
        if st.button("Generate Report", type="primary"):
            if not llm_status["available"]:
                st.error("LLM not available. Please configure Ollama API.")
            else:
                with st.spinner("Generating report..."):
                    report = generate_report(analysis_data, report_type)
                    st.markdown("### Generated Report")
                    st.markdown(report)
                    
                    # Download button
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name=f"{report_type}_report.txt",
                        mime="text/plain"
                    )
    
    # Help section
    with st.expander("ℹ️ How to Use AI Assistant"):
        st.markdown("""
        **Ask Questions Tab:**
        - Ask about how the system works
        - Get help with interpreting results
        - Learn about features and models
        
        **Explain Predictions Tab:**
        - Upload a predictions CSV from Upload & Predict page
        - Or enter summary statistics manually
        - Get detailed explanations of prediction patterns
        
        **Generate Reports Tab:**
        - Create patient summaries
        - Generate prediction analyses
        - Create dataset overview reports
        
        **Note:** AI features require Ollama API to be configured and running.
        See the warning message at the top for setup instructions.
        """)
