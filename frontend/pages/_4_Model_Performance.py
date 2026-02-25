"""
Model Performance page.
"""
import streamlit as st
import json
import pickle
import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from frontend.utils.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_metrics_comparison,
    export_fig_png,
)
from frontend.utils.error_handling import user_facing_message
from frontend.utils.icons import icon_html
import config as cfg


def show():
    st.markdown(f'<p class="page-header">{icon_html("chart", 28)} Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Confusion matrix, ROC/PR curves, feature importance, and model comparison.</p>', unsafe_allow_html=True)
    
    model_path = cfg.MODELS_DIR / "best_model.pkl"
    metrics_path = cfg.MODELS_DIR / "metrics_models.json"
    
    if not model_path.exists():
        st.warning("No trained model found. Train a model first using: `python -m src.train`")
        return
    
    if not metrics_path.exists():
        st.warning("Metrics file not found. Train a model first.")
        return
    
    # Load metrics
    try:
        with metrics_path.open() as f:
            metrics = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        st.error(f"Could not load metrics file: {e}. Re-train the model.")
        return
    
    # Best model summary (initialize so AI block never sees undefined best_info)
    best_info = None
    best_metrics_path = cfg.MODELS_DIR / "metrics.json"
    if best_metrics_path.exists():
        try:
            with best_metrics_path.open() as f:
                best_info = json.load(f)
            bm_name = best_info.get('best_model', 'N/A')
            bm_acc = best_info.get('best_balanced_accuracy', 0)
            st.markdown(
                f'<div class="result-panel result-panel-success">'
                f'<strong style="font-size:1.1rem;">Best model: {bm_name}</strong> &nbsp;'
                f'<span class="badge badge-green">Balanced Accuracy {bm_acc:.4f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        except (json.JSONDecodeError, OSError) as e:
            st.warning(f"Could not load best model info: {e}")
            best_info = None
    
    # Task from saved model (for labels and subtitles)
    task = "breath_hold"
    try:
        with model_path.open("rb") as f:
            model_data = pickle.load(f)
        task = model_data.get("task", "breath_hold")
    except Exception:
        pass
    task_label = "Breath-hold vs Free-breathing" if task == "breath_hold" else "Gating OK vs Not OK"
    st.caption(f"**Task**: {task_label}")
    
    st.markdown("---")
    
    # --- Research focus: best model only (confusion matrix, ROC/PR, feature importance) ---
    
    best_model_name = best_info.get("best_model", list(metrics.keys())[0]) if best_info else list(metrics.keys())[0]
    best_metrics = metrics.get(best_model_name, {})
    labels = ["Free-breathing", "Breath-hold"] if task == "breath_hold" else ["Gating Not OK", "Gating OK"]
    
    # Best model confusion matrix (main research view)
    st.subheader("Confusion Matrix (Best Model)")
    cm = best_metrics.get("confusion_matrix")
    if cm is not None:
        fig_cm = plot_confusion_matrix(cm, best_model_name, labels=labels, subtitle=f"Task: {task_label}, test set")
        st.plotly_chart(fig_cm, use_container_width=True, key="cm_best")
        png_cm = export_fig_png(fig_cm)
        if png_cm:
            st.download_button("Download PNG", data=png_cm, file_name="confusion_matrix_best.png", mime="image/png", key="dl_cm_best_png")
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        st.download_button("Download CSV", data=cm_df.to_csv(), file_name="confusion_matrix_best.csv", mime="text/csv", key="dl_cm_best_csv")
        st.write("**Matrix values:** TN = %s, FP = %s, FN = %s, TP = %s" % (cm[0][0], cm[0][1], cm[1][0], cm[1][1]))
    
    # ROC and Precision-Recall curves (research)
    test_pred_path = cfg.MODELS_DIR / "test_predictions.json"
    if test_pred_path.exists():
        try:
            with test_pred_path.open() as f:
                test_pred = json.load(f)
            y_true = test_pred["y_true"]
            proba_class1 = test_pred.get("proba_class1")
            saved_best_name = test_pred.get("best_model", best_model_name)
            n_test = test_pred.get("n_test", 0)
            
            if proba_class1 is not None and len(proba_class1) == len(y_true):
                from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
                fpr, tpr, _ = roc_curve(y_true, proba_class1)
                roc_auc = auc(fpr, tpr)
                precision, recall, _ = precision_recall_curve(y_true, proba_class1)
                ap = average_precision_score(y_true, proba_class1)
                
                st.markdown("---")
                st.subheader("ROC & Precision-Recall Curves (Test Set)")
                st.caption(f"Best model: {saved_best_name}, n = {n_test} test samples.")
                col_roc, col_pr = st.columns(2)
                with col_roc:
                    fig_roc = plot_roc_curve(fpr, tpr, roc_auc, saved_best_name)
                    st.plotly_chart(fig_roc, use_container_width=True, key="roc_curve")
                    png_roc = export_fig_png(fig_roc)
                    if png_roc:
                        st.download_button("Download ROC (PNG)", data=png_roc, file_name="roc_curve.png", mime="image/png", key="dl_roc_png")
                with col_pr:
                    fig_pr = plot_precision_recall_curve(precision, recall, ap, saved_best_name)
                    st.plotly_chart(fig_pr, use_container_width=True, key="pr_curve")
                    png_pr = export_fig_png(fig_pr)
                    if png_pr:
                        st.download_button("Download PR (PNG)", data=png_pr, file_name="precision_recall_curve.png", mime="image/png", key="dl_pr_png")
            else:
                st.markdown("---")
                st.caption("ROC/PR curves require probabilities. Re-train to save test predictions with probabilities.")
        except Exception as e:
            st.caption(f"Could not load test predictions for ROC/PR: {e}")
    else:
        st.markdown("---")
        st.caption("Re-train the model to generate ROC and Precision-Recall curves (test_predictions.json).")
    
    # Feature importance (research)
    st.markdown("---")
    st.subheader("Feature Importance (Best Model)")
    
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
            st.plotly_chart(fig, use_container_width=True, key="feature_importance")
            png_fi = export_fig_png(fig)
            if png_fi:
                st.download_button("Download chart (PNG)", data=png_fi, file_name="feature_importance.png", mime="image/png", key="dl_fi_png")
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
            else:
                imp = np.abs(model.coef_[0])
            fi_df = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)
            st.download_button("Download feature importance (CSV)", data=fi_df.to_csv(index=False), file_name="feature_importance.csv", mime="text/csv", key="dl_fi_csv")
        else:
            st.info(f"{type(model).__name__} does not support feature importance visualization.")
    except Exception as e:
        st.warning(f"Could not load feature importance: {e}")
    
    # AI Assistant
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(f'<p class="page-header" style="font-size:1.3rem;">{icon_html("robot", 22)} Ask AI About Results</p>', unsafe_allow_html=True)
    question = st.text_input(
        "Ask about the model or these metrics:",
        placeholder="e.g., What do these confusion matrix values mean? Which features matter most for breath-hold?",
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
            st.warning(user_facing_message(e))
    
    # --- Optional: model comparison (collapsed by default) ---
    st.markdown("---")
    with st.expander("Compare all trained models (optional)"):
        st.caption("Detailed comparison of all models trained during the last run. Focus above is on the best model for research.")
        models = list(metrics.keys())
        cols = st.columns(min(len(models), 6))
        for i, model_name in enumerate(models):
            with cols[i % len(cols)]:
                m = metrics[model_name]
                bal_acc = m.get("balanced_accuracy", 0)
                f1_val = m.get("f1", 0)
                st.metric(model_name, f"{bal_acc:.4f}", delta=f"F1: {f1_val:.4f}")
        
        metrics_df = []
        for model_name, m in metrics.items():
            metrics_df.append({
                "Model": model_name,
                "Accuracy": f"{m.get('accuracy', 0):.4f}",
                "Balanced Accuracy": f"{m.get('balanced_accuracy', 0):.4f}",
                "F1 Score": f"{m.get('f1', 0):.4f}",
            })
        st.dataframe(pd.DataFrame(metrics_df), use_container_width=True)
        metrics_csv = pd.DataFrame(metrics_df).to_csv(index=False)
        st.download_button("Download metrics table (CSV)", data=metrics_csv, file_name="model_metrics.csv", mime="text/csv", key="dl_metrics_csv")
        
        fig_comp = plot_metrics_comparison(metrics)
        st.plotly_chart(fig_comp, use_container_width=True, key="metrics_comparison")
        png_comp = export_fig_png(fig_comp)
        if png_comp:
            st.download_button("Download comparison chart (PNG)", data=png_comp, file_name="metrics_comparison.png", mime="image/png", key="dl_metrics_png")
        
        st.subheader("Confusion matrices (all models)")
        for model_name, m in metrics.items():
            with st.expander(f"{model_name}"):
                cm = m.get("confusion_matrix")
                if cm is None:
                    st.caption("No confusion matrix available for this model.")
                    continue
                fig = plot_confusion_matrix(cm, model_name, labels=labels, subtitle=f"Task: {task_label}")
                st.plotly_chart(fig, use_container_width=True, key=f"cm_{model_name.replace(' ', '_')}")
                png_cm = export_fig_png(fig)
                if png_cm:
                    st.download_button(f"Download PNG", data=png_cm, file_name=f"confusion_matrix_{model_name.replace(' ', '_')}.png", mime="image/png", key=f"dl_cm_{model_name.replace(' ', '_')}")
                cm_df = pd.DataFrame(cm, index=labels, columns=labels)
                st.download_button(f"Download CSV", data=cm_df.to_csv(), file_name=f"confusion_matrix_{model_name.replace(' ', '_')}.csv", mime="text/csv", key=f"dl_cm_csv_{model_name.replace(' ', '_')}")
                st.write("TN = %s, FP = %s, FN = %s, TP = %s" % (cm[0][0], cm[0][1], cm[1][0], cm[1][1]))
    
    st.markdown("---")
