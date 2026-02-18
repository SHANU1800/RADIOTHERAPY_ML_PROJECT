"""
Home / Overview page.
"""
import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.utils.data_helpers import load_summary_stats
from frontend.utils.icons import icon_html
import config as cfg


@st.cache_data(ttl=60)
def _cached_summary_stats():
    return load_summary_stats()


def show():
    st.markdown(f'<p class="main-header" style="display: flex; align-items: center; gap: 10px;">{icon_html("lungs", 40)} Breathing Patterns ML Dashboard</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
    ### About
    This dashboard provides tools for analyzing breathing patterns in radiation therapy gating data.
    Use classical ML models to classify breath-hold vs free-breathing patterns and evaluate gating performance.
    """)
    
    # Quick stats (cached to avoid blocking)
    try:
        file_summary, patient_summary, _ = _cached_summary_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", len(patient_summary) if len(patient_summary) > 0 else 0)
        
        with col2:
            total_files = len(file_summary) if len(file_summary) > 0 else 0
            st.metric("Total Files", total_files)
        
        with col3:
            if len(patient_summary) > 0 and "duration_s" in patient_summary.columns:
                total_duration = patient_summary["duration_s"].sum()
                st.metric("Total Duration", f"{total_duration:.1f}s")
            else:
                st.metric("Total Duration", "N/A")
        
        with col4:
            model_path = cfg.MODELS_DIR / "best_model.pkl"
            has_model_icon = icon_html("check", 20, "#28a745") if model_path.exists() else icon_html("x", 20, "#dc3545")
            st.markdown(f'<div style="display: flex; align-items: center; gap: 5px;">{has_model_icon} <span>Model Available</span></div>', unsafe_allow_html=True)
            st.metric("", "Yes" if model_path.exists() else "No")
        
        # Model performance summary
        if model_path.exists():
            st.markdown("---")
            st.subheader("Model Performance Summary")
            try:
                import json
                metrics_path = cfg.MODELS_DIR / "metrics.json"
                if metrics_path.exists():
                    with metrics_path.open() as f:
                        metrics = json.load(f)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Best Model**: {metrics.get('best_model', 'N/A')}")
                    with col2:
                        st.info(f"**Balanced Accuracy**: {metrics.get('best_balanced_accuracy', 0):.4f}")
            except Exception:
                pass
        
    except Exception as e:
        st.warning(f"Could not load summary stats: {e}")
    
    st.markdown("---")
    st.markdown("""
    ### Pages
    - **Upload & Predict**: Upload a breathing curve file and get predictions
    - **Dataset Explorer**: Browse and visualize existing patient data
    - **Model Performance**: View detailed model metrics and confusion matrices
    - **Batch Analysis**: Process multiple files at once
    """)
    
    st.markdown("---")
    st.markdown("""
    ### Documentation
    See `README_ML.md` for details on the ML pipeline, data formats, and training.
    """)
