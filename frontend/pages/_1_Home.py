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
from frontend.utils.error_handling import user_facing_message
from frontend.utils.icons import icon_html
import config as cfg


@st.cache_data(ttl=60)
def _cached_summary_stats():
    return load_summary_stats()


def _feature_card(icon_name, title, desc, accent="#2563eb"):
    return (
        f'<div class="feature-card">'
        f'<div class="fc-icon" style="background:{accent}15;">{icon_html(icon_name, 24, accent)}</div>'
        f'<h4>{title}</h4><p>{desc}</p></div>'
    )


def show():
    # Hero banner
    st.markdown(
        f'<div class="hero-banner">'
        f'<h1>{icon_html("activity", 36, "#fff")} Breathing Patterns ML</h1>'
        f'<p>Research-grade classification of breathing patterns in radiation therapy gating data. '
        f'Upload a file, get window-by-window predictions with confidence scores, annotated charts, '
        f'clinical zone analysis, and exportable reports.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # --- System status row ---
    model_path = cfg.MODELS_DIR / "best_model.pkl"
    model_ok = model_path.exists()
    ds_ok = cfg.DATASET_DIR.exists() and any(cfg.DATASET_DIR.iterdir()) if cfg.DATASET_DIR.exists() else False

    try:
        from frontend.utils.llm_helper import is_llm_available
        llm_ok = is_llm_available()
    except Exception:
        llm_ok = False

    s1, s2, s3 = st.columns(3)
    with s1:
        cls = "badge-green" if model_ok else "badge-yellow"
        txt = "Trained & ready" if model_ok else "Not trained yet"
        st.markdown(f'{icon_html("shield", 18)} **Model** &nbsp; <span class="badge {cls}">{txt}</span>', unsafe_allow_html=True)
    with s2:
        cls = "badge-green" if ds_ok else "badge-yellow"
        txt = "Available" if ds_ok else "Not found"
        st.markdown(f'{icon_html("layers", 18)} **Dataset** &nbsp; <span class="badge {cls}">{txt}</span>', unsafe_allow_html=True)
    with s3:
        cls = "badge-green" if llm_ok else "badge-red"
        txt = "Connected" if llm_ok else "Offline"
        st.markdown(f'{icon_html("robot", 18)} **AI / LLM** &nbsp; <span class="badge {cls}">{txt}</span>', unsafe_allow_html=True)

    # --- Dataset stats ---
    try:
        file_summary, patient_summary, _ = _cached_summary_stats()

        if len(patient_summary) > 0 or len(file_summary) > 0:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown(f'<p class="page-header">{icon_html("chart", 24)} Dataset at a glance</p>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Patients", len(patient_summary) if len(patient_summary) > 0 else 0)
            c2.metric("Files", len(file_summary) if len(file_summary) > 0 else 0)
            if len(patient_summary) > 0 and "duration_s" in patient_summary.columns:
                total_dur = patient_summary["duration_s"].sum()
                c3.metric("Total duration", "%.0f s" % total_dur)
            else:
                c3.metric("Total duration", "N/A")
            c4.metric("Model", "Yes" if model_ok else "No")

            if model_ok:
                try:
                    import json
                    mp = cfg.MODELS_DIR / "metrics.json"
                    if mp.exists():
                        with mp.open() as f:
                            m = json.load(f)
                        st.markdown(
                            f'<span class="badge badge-blue">{m.get("best_model", "N/A")}</span> &nbsp; '
                            f'Balanced accuracy **{m.get("best_balanced_accuracy", 0):.4f}**',
                            unsafe_allow_html=True,
                        )
                except Exception:
                    pass

    except Exception as e:
        st.warning(user_facing_message(e))

    # --- Feature cards ---
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(f'<p class="page-header">{icon_html("zap", 24)} Features</p>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(
            _feature_card("upload", "Upload & Predict",
                          "Upload a breathing curve file and get detailed window-by-window classification "
                          "with confidence scores, annotated charts, and clinical zone analysis.",
                          "#2563eb"),
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            _feature_card("search", "Dataset Explorer",
                          "Browse patient data, visualize breathing curves, inspect balloon valve and gating "
                          "signals, and compute ground-truth window labels.",
                          "#7c3aed"),
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            _feature_card("chart", "Model Performance",
                          "View confusion matrix, ROC/PR curves, and feature importance for the best trained "
                          "model. Compare all trained classifiers.",
                          "#0891b2"),
            unsafe_allow_html=True,
        )

    f4, f5, f6 = st.columns(3)
    with f4:
        st.markdown(
            _feature_card("package", "Batch Analysis",
                          "Process multiple files at once. Get a summary table, combined predictions CSV, "
                          "and side-by-side curve comparisons.",
                          "#c2410c"),
            unsafe_allow_html=True,
        )
    with f5:
        st.markdown(
            _feature_card("robot", "AI Assistant",
                          "Ask questions about the system, get AI-powered explanations of predictions, "
                          "and generate patient or dataset reports.",
                          "#16a34a"),
            unsafe_allow_html=True,
        )
    with f6:
        st.markdown(
            _feature_card("clipboard", "Export & Report",
                          "Download predictions as CSV, breathing curves as PNG, and auto-generated "
                          "summary reports for research and clinical review.",
                          "#9333ea"),
            unsafe_allow_html=True,
        )

    # --- Quick start ---
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(f'<p class="page-header">{icon_html("target", 24)} Quick start</p>', unsafe_allow_html=True)

    st.markdown("""
1. **Train a model** (if not done):  `python -m src.train --dataset dataset/`
2. Navigate to **Upload & Predict** in the sidebar
3. Upload a `.dat`, `.txt`, or `.csv` breathing curve file
4. Review the detailed analysis: charts, metrics, clinical zones, glossary
5. Download predictions (CSV) or a summary report (TXT) for your records
    """)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.caption("See `README_ML.md` for full documentation on the ML pipeline, data formats, and training.")
