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


def _show_dl_performance(task: str, task_label: str, labels: list, classical_metrics: dict = None):
    """Render deep learning model metrics, ROC/PR curves, training curves, and comparison."""
    import plotly.graph_objects as go
    from frontend.utils.visualization import (
        plot_dl_roc_curves,
        plot_dl_pr_curves,
        plot_dl_vs_classical_comparison,
        plot_dl_metrics_radar,
    )

    dl_summary_path = cfg.MODELS_DIR / f"dl_summary_{task}.json"
    if not dl_summary_path.exists():
        st.markdown("---")
        st.subheader("Deep Learning Models")
        st.info(
            "No deep learning models found for this task. "
            "Train them with: `python -m src.dl_train --task " + task + "`"
        )
        return

    try:
        with dl_summary_path.open() as f:
            dl_summary = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        st.warning(f"Could not load DL summary: {e}")
        return

    dl_models = dl_summary.get("models", {})
    if not dl_models:
        return

    st.markdown("---")
    st.subheader("Deep Learning Models")

    # Training configuration info
    dl_channels = dl_summary.get("channels", ["Volume (liters)"])
    dl_overlap = dl_summary.get("overlap", 0)
    dl_n_windows = dl_summary.get("n_windows_total", "?")
    st.caption(
        f"Channels: {', '.join(dl_channels)} | "
        f"Overlap: {dl_overlap} | "
        f"Windows: {dl_n_windows} | "
        f"Class weights: {dl_summary.get('class_weights', 'N/A')}"
    )

    best_dl = dl_summary.get("best_model", "")
    best_dl_acc = dl_summary.get("best_balanced_accuracy", 0)
    best_m = dl_models.get(best_dl, {})
    st.markdown(
        f'<div class="result-panel result-panel-success">'
        f'<strong>Best DL model: {best_dl}</strong> &nbsp;'
        f'<span class="badge badge-green">Balanced Accuracy {best_dl_acc:.4f}</span> &nbsp;'
        f'<span class="badge badge-blue">AUC {best_m.get("roc_auc", 0):.4f}</span> &nbsp;'
        f'<span class="badge badge-blue">MCC {best_m.get("mcc", 0):.4f}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Metric cards
    n_cols = min(len(dl_models), 4)
    dl_cols = st.columns(n_cols)
    for i, (name, m) in enumerate(dl_models.items()):
        with dl_cols[i % n_cols]:
            st.metric(
                f"DL-{name}",
                f"Bal.Acc {m.get('balanced_accuracy', 0):.4f}",
                delta=f"AUC {m.get('roc_auc', 0):.4f}",
            )

    # Comprehensive metrics table
    dl_df_rows = []
    for name, m in dl_models.items():
        dl_df_rows.append({
            "Model": f"DL-{name}",
            "Accuracy": f"{m.get('accuracy', 0):.4f}",
            "Bal. Accuracy": f"{m.get('balanced_accuracy', 0):.4f}",
            "F1": f"{m.get('f1', 0):.4f}",
            "Precision": f"{m.get('precision', 0):.4f}",
            "Recall": f"{m.get('recall', 0):.4f}",
            "Specificity": f"{m.get('specificity', 0):.4f}",
            "MCC": f"{m.get('mcc', 0):.4f}",
            "ROC-AUC": f"{m.get('roc_auc', 0):.4f}",
            "Avg Prec.": f"{m.get('average_precision', 0):.4f}",
            "Epochs": m.get("epochs_trained", "?"),
            "Channels": m.get("n_channels", "?"),
        })
    st.dataframe(pd.DataFrame(dl_df_rows), use_container_width=True)

    # ── DL ROC and PR Curves ──────────────────────────────────────
    st.markdown("#### ROC & Precision-Recall Curves (DL models)")
    dl_roc_data = {}
    dl_pr_data = {}
    for name in dl_models:
        prefix = f"dl_{name.lower()}_{task}"
        pred_path = cfg.MODELS_DIR / f"{prefix}_test_predictions.json"
        if not pred_path.exists():
            continue
        try:
            with pred_path.open() as f:
                pred_data = json.load(f)
            if "roc_curve" in pred_data:
                dl_roc_data[f"DL-{name}"] = {
                    **pred_data["roc_curve"],
                    "auc": dl_models[name].get("roc_auc", 0),
                }
            if "pr_curve" in pred_data:
                dl_pr_data[f"DL-{name}"] = pred_data["pr_curve"]
        except Exception:
            continue

    if dl_roc_data or dl_pr_data:
        col_roc, col_pr = st.columns(2)
        if dl_roc_data:
            with col_roc:
                fig_roc = plot_dl_roc_curves(dl_roc_data)
                st.plotly_chart(fig_roc, use_container_width=True, key="dl_roc_all")
        if dl_pr_data:
            with col_pr:
                fig_pr = plot_dl_pr_curves(dl_pr_data)
                st.plotly_chart(fig_pr, use_container_width=True, key="dl_pr_all")
    else:
        st.caption("ROC/PR curve data not available. Re-train DL models to generate.")

    # ── DL vs Classical ML Comparison ─────────────────────────────
    if classical_metrics:
        st.markdown("#### Deep Learning vs Classical ML")
        fig_comp = plot_dl_vs_classical_comparison(classical_metrics, dl_models)
        st.plotly_chart(fig_comp, use_container_width=True, key="dl_vs_ml_comp")
        st.info("Comparison of balanced accuracy across all model types.")

    # ── Per-model detailed analysis (radar + training curves) ─────
    st.markdown("#### Per-Model Analysis")
    selected_dl = st.selectbox("Select DL model for detailed view", list(dl_models.keys()), key="dl_detail_select")
    if selected_dl and selected_dl in dl_models:
        m = dl_models[selected_dl]

        col_radar, col_info = st.columns([1, 1])
        with col_radar:
            fig_radar = plot_dl_metrics_radar(m, f"DL-{selected_dl}")
            st.plotly_chart(fig_radar, use_container_width=True, key=f"dl_radar_{selected_dl}")
        with col_info:
            st.markdown(f"**DL-{selected_dl}** Details")
            st.write(f"- Input shape: {m.get('input_shape', '?')}")
            st.write(f"- Channels: {m.get('n_channels', '?')}")
            st.write(f"- Train/Val/Test: {m.get('n_train', '?')}/{m.get('n_val', '?')}/{m.get('n_test', '?')}")
            st.write(f"- Epochs trained: {m.get('epochs_trained', '?')}")
            st.write(f"- Class weights: {'Yes' if m.get('class_weights_used') else 'No'}")
            st.write(f"- **Sensitivity**: {m.get('sensitivity', 0):.4f}")
            st.write(f"- **Specificity**: {m.get('specificity', 0):.4f}")
            st.write(f"- **MCC**: {m.get('mcc', 0):.4f}")

        # Training curves
        prefix = f"dl_{selected_dl.lower()}_{task}"
        hist_path = cfg.MODELS_DIR / f"{prefix}_history.json"
        if hist_path.exists():
            try:
                with hist_path.open() as f:
                    hist = json.load(f)
                loss = hist.get("loss", [])
                val_loss = hist.get("val_loss", [])
                acc_hist = hist.get("accuracy", [])
                val_acc = hist.get("val_accuracy", [])
                epochs_range = list(range(1, len(loss) + 1))

                col_l, col_a = st.columns(2)
                with col_l:
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(x=epochs_range, y=loss, name="Train Loss", line=dict(color="#3498db")))
                    if val_loss:
                        fig_loss.add_trace(go.Scatter(x=epochs_range, y=val_loss, name="Val Loss", line=dict(color="#e74c3c")))
                    fig_loss.update_layout(title=f"{selected_dl} - Loss", xaxis_title="Epoch", yaxis_title="Loss", height=350)
                    st.plotly_chart(fig_loss, use_container_width=True, key=f"dl_loss_{selected_dl}")

                with col_a:
                    if acc_hist:
                        fig_acc = go.Figure()
                        fig_acc.add_trace(go.Scatter(x=epochs_range, y=acc_hist, name="Train Acc", line=dict(color="#3498db")))
                        if val_acc:
                            fig_acc.add_trace(go.Scatter(x=epochs_range, y=val_acc, name="Val Acc", line=dict(color="#e74c3c")))
                        fig_acc.update_layout(title=f"{selected_dl} - Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy", height=350)
                        st.plotly_chart(fig_acc, use_container_width=True, key=f"dl_acc_{selected_dl}")
            except Exception:
                pass

        # Confusion matrix
        cm = m.get("confusion_matrix")
        if cm is not None:
            st.markdown(f"**Confusion Matrix — DL-{selected_dl}**")
            fig = plot_confusion_matrix(cm, f"DL-{selected_dl}", labels=labels, subtitle=f"Task: {task_label}")
            st.plotly_chart(fig, use_container_width=True, key=f"dl_cm_{selected_dl}")
            if len(cm) == 2 and len(cm[0]) == 2:
                st.write("TN = %s, FP = %s, FN = %s, TP = %s" % (cm[0][0], cm[0][1], cm[1][0], cm[1][1]))


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

    # Best model summary
    best_info = None
    best_metrics_path = cfg.MODELS_DIR / "metrics.json"
    if best_metrics_path.exists():
        try:
            with best_metrics_path.open() as f:
                best_info = json.load(f)
        except (json.JSONDecodeError, OSError):
            best_info = None

    # Task from saved model
    task = "breath_hold"
    try:
        with model_path.open("rb") as f:
            model_data = pickle.load(f)
        task = model_data.get("task", "breath_hold")
    except Exception:
        pass
    task_label = "Breath-hold vs Free-breathing" if task == "breath_hold" else "Gating OK vs Not OK"

    best_model_name = best_info.get("best_model", list(metrics.keys())[0]) if best_info else list(metrics.keys())[0]
    best_metrics = metrics.get(best_model_name, {})

    # Check for DL models
    dl_summary_path = cfg.MODELS_DIR / f"dl_summary_{task}.json"
    dl_summary = None
    if dl_summary_path.exists():
        try:
            with dl_summary_path.open() as f:
                dl_summary = json.load(f)
        except Exception:
            pass

    # ── Compact summary dashboard ─────────────────────────────────
    bm_name = best_info.get('best_model', 'N/A') if best_info else 'N/A'
    bm_acc = best_info.get('best_balanced_accuracy', 0) if best_info else 0
    dl_best = dl_summary.get("best_model", "") if dl_summary else ""
    dl_best_acc = dl_summary.get("best_balanced_accuracy", 0) if dl_summary else 0
    n_classical = len(metrics)
    n_dl = len(dl_summary.get("models", {})) if dl_summary else 0

    st.markdown(
        f'<div class="result-panel result-panel-success">'
        f'<strong>Classical ML:</strong> {bm_name} '
        f'<span class="badge badge-green">Bal.Acc {bm_acc:.4f}</span>'
        f'{" &nbsp;|&nbsp; <strong>Deep Learning:</strong> " + dl_best + " <span class=badge badge-blue>Bal.Acc " + f"{dl_best_acc:.4f}" + "</span>" if dl_best else ""}'
        f'<br><span style="color:#64748b;font-size:0.83rem;">'
        f'Task: {task_label} &middot; {n_classical} classical models &middot; {n_dl} DL models</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Quick metric comparison pills
    if n_dl > 0 and dl_best_acc > 0:
        winner = "DL" if dl_best_acc > bm_acc else "Classical ML"
        diff = abs(dl_best_acc - bm_acc) * 100
        st.markdown(
            '<div class="stat-row">'
            f'<span class="stat-pill">{winner} wins <small>by {diff:.1f}pp balanced accuracy</small></span>'
            f'<span class="stat-pill">{n_classical + n_dl} <small>total models trained</small></span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Tabbed model views ──────────────────────────────────────
    tab_overview, tab_classical, tab_dl = st.tabs([
        "Overview", "Classical ML Details", "Deep Learning Details",
    ])
    labels = ["Free-breathing", "Breath-hold"] if task == "breath_hold" else ["Gating Not OK", "Gating OK"]

    # ── TAB: Overview (side-by-side comparison) ──────────────────
    with tab_overview:
        # All models comparison table
        all_rows = []
        for model_name, m in metrics.items():
            all_rows.append({
                "Model": model_name, "Type": "Classical",
                "Accuracy": m.get("accuracy", 0), "Bal. Accuracy": m.get("balanced_accuracy", 0),
                "F1": m.get("f1", 0),
            })
        if dl_summary:
            for name, m in dl_summary.get("models", {}).items():
                all_rows.append({
                    "Model": f"DL-{name}", "Type": "Deep Learning",
                    "Accuracy": m.get("accuracy", 0), "Bal. Accuracy": m.get("balanced_accuracy", 0),
                    "F1": m.get("f1", 0),
                })
        if all_rows:
            all_df = pd.DataFrame(all_rows).sort_values("Bal. Accuracy", ascending=False)
            st.dataframe(
                all_df.style.format({"Accuracy": "{:.4f}", "Bal. Accuracy": "{:.4f}", "F1": "{:.4f}"}),
                use_container_width=True,
            )

        # Side-by-side: best classical vs best DL
        if dl_summary and dl_summary.get("models"):
            ov1, ov2 = st.columns(2)
            with ov1:
                st.markdown(f"#### Best Classical: {bm_name}")
                cm = best_metrics.get("confusion_matrix")
                if cm is not None:
                    fig_cm = plot_confusion_matrix(cm, bm_name, labels=labels, subtitle="Classical ML")
                    st.plotly_chart(fig_cm, use_container_width=True, key="ov_cm_classical")
            with ov2:
                st.markdown(f"#### Best DL: {dl_best}")
                dl_m = dl_summary["models"].get(dl_best, {})
                cm_dl = dl_m.get("confusion_matrix")
                if cm_dl is not None:
                    fig_cm_dl = plot_confusion_matrix(cm_dl, f"DL-{dl_best}", labels=labels, subtitle="Deep Learning")
                    st.plotly_chart(fig_cm_dl, use_container_width=True, key="ov_cm_dl")

        fig_comp = plot_metrics_comparison(metrics)
        st.plotly_chart(fig_comp, use_container_width=True, key="metrics_comparison")

    # ── TAB: Classical ML Details ─────────────────────────────────
    with tab_classical:
        st.subheader(f"Best Model: {best_model_name}")

        cm = best_metrics.get("confusion_matrix")
        if cm is not None:
            col_cm, col_curves = st.columns(2)
            with col_cm:
                fig_cm = plot_confusion_matrix(cm, best_model_name, labels=labels, subtitle=task_label)
                st.plotly_chart(fig_cm, use_container_width=True, key="cm_best")
                st.caption("TN=%s, FP=%s, FN=%s, TP=%s" % (cm[0][0], cm[0][1], cm[1][0], cm[1][1]))

            with col_curves:
                test_pred_path = cfg.MODELS_DIR / "test_predictions.json"
                if test_pred_path.exists():
                    try:
                        with test_pred_path.open() as f:
                            test_pred = json.load(f)
                        y_true = test_pred["y_true"]
                        proba_class1 = test_pred.get("proba_class1")
                        if proba_class1 and len(proba_class1) == len(y_true):
                            from sklearn.metrics import roc_curve, auc
                            fpr, tpr, _ = roc_curve(y_true, proba_class1)
                            roc_auc = auc(fpr, tpr)
                            fig_roc = plot_roc_curve(fpr, tpr, roc_auc, best_model_name)
                            st.plotly_chart(fig_roc, use_container_width=True, key="roc_curve")
                    except Exception:
                        st.caption("ROC curve data not available.")
                else:
                    st.caption("Train model to generate ROC curve.")

        # Feature importance
        st.markdown("#### Feature Importance")
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
            else:
                st.info(f"{type(model).__name__} does not support feature importance visualization.")
        except Exception as e:
            st.warning(f"Could not load feature importance: {e}")

        # All classical models
        with st.expander("All classical models"):
            metrics_rows = []
            for model_name, m in metrics.items():
                metrics_rows.append({
                    "Model": model_name,
                    "Accuracy": f"{m.get('accuracy', 0):.4f}",
                    "Bal. Accuracy": f"{m.get('balanced_accuracy', 0):.4f}",
                    "F1": f"{m.get('f1', 0):.4f}",
                })
            st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)
            for model_name, m in metrics.items():
                cm_m = m.get("confusion_matrix")
                if cm_m:
                    with st.expander(model_name):
                        fig = plot_confusion_matrix(cm_m, model_name, labels=labels, subtitle=task_label)
                        st.plotly_chart(fig, use_container_width=True, key=f"cm_{model_name.replace(' ', '_')}")

    # ── TAB: Deep Learning Details ────────────────────────────────
    with tab_dl:
        _show_dl_performance(task, task_label, labels, classical_metrics=metrics)

    # AI Assistant (compact)
    with st.expander("Ask AI about these results"):
        question = st.text_input(
            "Ask about the model or these metrics:",
            placeholder="e.g., What do these confusion matrix values mean?",
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
                                "best_model": bm_name,
                                "metrics": metrics,
                                "models_available": list(metrics.keys())
                            }
                        }
                        answer = answer_question(question, context)
                        st.markdown("### AI Answer")
                        st.markdown(answer)
                else:
                    st.info("AI features require Ollama API. Go to AI Assistant page for setup.")
            except Exception as e:
                st.warning(user_facing_message(e))
