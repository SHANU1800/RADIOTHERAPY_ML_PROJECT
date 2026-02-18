"""
Model Performance page.
"""
import streamlit as st
import json
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.utils.visualization import plot_confusion_matrix, plot_feature_importance
from frontend.utils.icons import icon_html
import config as cfg


def show():
    st.markdown(f'<h1 style="display: flex; align-items: center; gap: 10px;">{icon_html("chart", 32)} Model Performance</h1>', unsafe_allow_html=True)
    st.markdown("View detailed metrics and performance of trained models.")
    
    model_path = cfg.MODELS_DIR / "best_model.pkl"
    metrics_path = cfg.MODELS_DIR / "metrics_models.json"
    
    if not model_path.exists():
        st.warning("No trained model found. Train a model first using: `python -m src.train`")
        return
    
    if not metrics_path.exists():
        st.warning("Metrics file not found. Train a model first.")
        return
    
    # Load metrics
    with metrics_path.open() as f:
        metrics = json.load(f)
    
    # Best model summary
    best_metrics_path = cfg.MODELS_DIR / "metrics.json"
    if best_metrics_path.exists():
        with best_metrics_path.open() as f:
            best_info = json.load(f)
        st.info(f"**Best Model**: {best_info.get('best_model', 'N/A')} (Balanced Accuracy: {best_info.get('best_balanced_accuracy', 0):.4f})")
    
    st.markdown("---")
    
    # Metrics cards for each model
    st.subheader("Model Comparison")
    models = list(metrics.keys())
    
    cols = st.columns(len(models))
    for i, model_name in enumerate(models):
        with cols[i]:
            m = metrics[model_name]
            st.metric(
                model_name,
                f"{m['balanced_accuracy']:.4f}",
                delta=f"F1: {m['f1']:.4f}",
            )
    
    # Detailed metrics table
    st.subheader("Detailed Metrics")
    metrics_df = []
    for model_name, m in metrics.items():
        metrics_df.append({
            "Model": model_name,
            "Accuracy": f"{m['accuracy']:.4f}",
            "Balanced Accuracy": f"{m['balanced_accuracy']:.4f}",
            "F1 Score": f"{m['f1']:.4f}",
        })
    
    import pandas as pd
    st.dataframe(pd.DataFrame(metrics_df), use_container_width=True)
    
    st.markdown("---")
    
    # Confusion matrices
    st.subheader("Confusion Matrices")
    
    # Determine labels based on task (default to breath_hold)
    labels = ["Free-breathing", "Breath-hold"]
    
    for model_name, m in metrics.items():
        with st.expander(f"{model_name} - Confusion Matrix"):
            cm = m["confusion_matrix"]
            fig = plot_confusion_matrix(cm, model_name, labels=labels)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show numeric values
            st.write("**Matrix Values:**")
            st.write(f"True Negatives (TN): {cm[0][0]}")
            st.write(f"False Positives (FP): {cm[0][1]}")
            st.write(f"False Negatives (FN): {cm[1][0]}")
            st.write(f"True Positives (TP): {cm[1][1]}")
    
    # Feature importance
    st.markdown("---")
    st.subheader("Feature Importance")
    
    try:
        with model_path.open("rb") as f:
            model_data = pickle.load(f)
        model = model_data["model"]
        
        feature_names = [
            "vol_mean", "vol_std", "vol_min", "vol_max", "vol_range", "vol_change", "vol_rolling_mean",
            "frac_balloon_inflated", "frac_balloon_deflated", "frac_patient_switch_on", "frac_gating_automated",
        ]
        
        if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
            fig = plot_feature_importance(model, feature_names)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"{type(model).__name__} does not support feature importance visualization.")
    except Exception as e:
        st.warning(f"Could not load feature importance: {e}")
    
    # AI Assistant integration
    st.markdown("---")
    st.markdown(f'<h3 style="display: flex; align-items: center; gap: 8px;">{icon_html("robot", 24)} Ask AI About Model Performance</h3>', unsafe_allow_html=True)
    question = st.text_input(
        "Ask a question about the model:",
        placeholder="e.g., Why is Random Forest performing best? What do these confusion matrices tell us?",
        key="model_question"
    )
    if st.button("Ask AI", key="ask_model_ai"):
        try:
            from frontend.utils.llm_helper import answer_question, is_llm_available
            if not question.strip():
                st.warning("Please enter a question.")
            elif is_llm_available():
                with st.spinner("Thinking..."):
                    context = {
                        "model_info": {
                            "best_model": best_info.get('best_model', 'N/A') if best_info else 'N/A',
                            "metrics": metrics,
                            "models_available": list(metrics.keys())
                        }
                    }
                    answer = answer_question(question, context)
                    st.markdown("### AI Answer")
                    st.markdown(answer)
            else:
                st.info("AI features require Ollama API. Go to AI Assistant page for setup instructions.")
        except Exception as e:
            st.warning(f"Could not get AI answer: {str(e)}")
