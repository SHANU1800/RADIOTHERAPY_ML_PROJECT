"""
Entry point for Streamlit frontend.
Run with: streamlit run run_frontend.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
try:
    from frontend.utils.icons import icon_html
except Exception:
    def icon_html(name: str, size: int = 24, color: str = "currentColor") -> str:
        return ""

from frontend.utils.error_handling import user_facing_message

st.set_page_config(
    page_title="Breathing Patterns ML Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global theme CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---- colour tokens ---- */
:root {
    --accent: #2563eb;
    --accent-light: #dbeafe;
    --success: #16a34a;
    --success-light: #dcfce7;
    --warning: #d97706;
    --warning-light: #fef3c7;
    --danger: #dc2626;
    --danger-light: #fee2e2;
    --surface: #f8fafc;
    --surface-alt: #f1f5f9;
    --border: #e2e8f0;
    --text: #1e293b;
    --text-muted: #64748b;
}

/* ---- sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] hr { border-color: #334155; }

/* ---- page header class ---- */
.page-header {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
}
.page-subtitle {
    color: var(--text-muted);
    font-size: 1.05rem;
    margin-bottom: 1.2rem;
}

/* ---- hero banner ---- */
.hero-banner {
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    color: #fff;
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
}
.hero-banner h1 { color: #fff; margin: 0 0 0.5rem 0; font-size: 2.2rem; letter-spacing: -0.03em; }
.hero-banner p  { color: #e0e7ff; margin: 0; font-size: 1.05rem; line-height: 1.6; }

/* ---- cards ---- */
.card {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    transition: box-shadow .15s, transform .15s;
}
.card:hover {
    box-shadow: 0 4px 24px rgba(0,0,0,.06);
    transform: translateY(-2px);
}
.card-accent  { border-left: 4px solid var(--accent); }
.card-success { border-left: 4px solid var(--success); }
.card-warning { border-left: 4px solid var(--warning); }
.card-danger  { border-left: 4px solid var(--danger); }

.card-title {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 0.35rem;
}
.card-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.2;
}
.card-caption {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: 0.3rem;
}

/* ---- feature card (home) ---- */
.feature-card {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: box-shadow .15s, transform .15s;
    height: 100%;
}
.feature-card:hover {
    box-shadow: 0 6px 32px rgba(37,99,235,.10);
    transform: translateY(-3px);
}
.feature-card .fc-icon {
    width: 48px; height: 48px;
    background: var(--accent-light);
    border-radius: 12px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 0.75rem;
}
.feature-card h4 { margin: 0 0 0.4rem 0; color: var(--text); font-size: 1.05rem; }
.feature-card p  { margin: 0; color: var(--text-muted); font-size: 0.88rem; line-height: 1.5; }

/* ---- status badge ---- */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-green  { background: var(--success-light); color: var(--success); }
.badge-yellow { background: var(--warning-light); color: var(--warning); }
.badge-red    { background: var(--danger-light);  color: var(--danger);  }
.badge-blue   { background: var(--accent-light);  color: var(--accent);  }

/* ---- section divider ---- */
.section-divider {
    border: none;
    height: 1px;
    background: var(--border);
    margin: 1.5rem 0;
}

/* ---- result panel ---- */
.result-panel {
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin: 0.75rem 0;
}
.result-panel-success {
    background: var(--success-light);
    border: 1px solid #86efac;
}
.result-panel-warning {
    background: var(--warning-light);
    border: 1px solid #fcd34d;
}

/* ---- metric grid (tighter) ---- */
[data-testid="stMetric"] {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
[data-testid="stMetric"] label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-muted); }
[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; color: var(--text); }

/* ---- sidebar brand ---- */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.5rem 0 0.75rem 0;
}
.sidebar-brand h2 {
    color: #f1f5f9 !important;
    font-size: 1.15rem;
    margin: 0;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.sidebar-footer {
    color: #475569;
    font-size: 0.75rem;
    padding-top: 1rem;
    text-align: center;
}

/* ---- subtle tweaks ---- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
}
.stExpander { border: 1px solid var(--border); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f'<div class="sidebar-brand">{icon_html("lungs", 28, "#60a5fa")}'
    f'<h2>Breathing ML</h2></div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

_NAV_ITEMS = [
    ("Upload & Predict", "upload"),
    ("Home",             "lungs"),
    ("Dataset Explorer", "search"),
    ("Model Performance","chart"),
    ("Batch Analysis",   "package"),
    ("AI Assistant",     "robot"),
]
_nav_labels = [item[0] for item in _NAV_ITEMS]
page = st.sidebar.radio("Navigation", _nav_labels, label_visibility="collapsed")

# Status dots
import config as cfg
_model_ok = (cfg.MODELS_DIR / "best_model.pkl").exists()
_ds_ok = cfg.DATASET_DIR.exists() and any(cfg.DATASET_DIR.iterdir()) if cfg.DATASET_DIR.exists() else False
st.sidebar.markdown("---")
st.sidebar.markdown(
    f'<span class="badge {"badge-green" if _model_ok else "badge-yellow"}">'
    f'{"Model ready" if _model_ok else "No model"}</span> &nbsp;'
    f'<span class="badge {"badge-green" if _ds_ok else "badge-yellow"}">'
    f'{"Dataset found" if _ds_ok else "No dataset"}</span>',
    unsafe_allow_html=True,
)
st.sidebar.markdown('<div class="sidebar-footer">KMC Project v1.0</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------
_PAGE_MODULES = {
    "Upload & Predict":  "frontend.pages._2_Upload_Predict",
    "Home":              "frontend.pages._1_Home",
    "Dataset Explorer":  "frontend.pages._3_Dataset_Explorer",
    "Model Performance": "frontend.pages._4_Model_Performance",
    "Batch Analysis":    "frontend.pages._5_Batch_Analysis",
    "AI Assistant":      "frontend.pages._6_AI_Assistant",
}

import importlib
_mod_path = _PAGE_MODULES.get(page)
if _mod_path:
    _mod = importlib.import_module(_mod_path)
    try:
        _mod.show()
    except Exception as e:
        st.error(user_facing_message(e))
        st.exception(e)
