"""
Main Streamlit app for Breathing Patterns ML Dashboard.
Run with: streamlit run run_frontend.py (from project root)
"""
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Breathing Patterns ML Dashboard",
    page_icon=None,  # Using SVG icons instead
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #ecf0f1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
from frontend.utils.icons import icon_html
st.sidebar.markdown(f'<h2 style="display: flex; align-items: center; gap: 8px;">{icon_html("lungs", 28)} Breathing Patterns ML</h2>', unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Upload & Predict", "Dataset Explorer", "Model Performance", "Batch Analysis", "AI Assistant"],
)

# Route to pages (wrap in try/except so errors show instead of blank page)
if page == "Home":
    import frontend.pages._1_Home as page_home
    try:
        page_home.show()
    except Exception as e:
        st.error(f"Error on Home: {str(e)}")
        st.exception(e)
elif page == "Upload & Predict":
    import frontend.pages._2_Upload_Predict as page_upload
    try:
        page_upload.show()
    except Exception as e:
        st.error(f"Error on Upload & Predict: {str(e)}")
        st.exception(e)
elif page == "Dataset Explorer":
    import frontend.pages._3_Dataset_Explorer as page_explorer
    try:
        page_explorer.show()
    except Exception as e:
        st.error(f"Error on Dataset Explorer: {str(e)}")
        st.exception(e)
elif page == "Model Performance":
    import frontend.pages._4_Model_Performance as page_metrics
    try:
        page_metrics.show()
    except Exception as e:
        st.error(f"Error on Model Performance: {str(e)}")
        st.exception(e)
elif page == "Batch Analysis":
    import frontend.pages._5_Batch_Analysis as page_batch
    try:
        page_batch.show()
    except Exception as e:
        st.error(f"Error on Batch Analysis: {str(e)}")
        st.exception(e)
elif page == "AI Assistant":
    import frontend.pages._6_AI_Assistant as page_ai
    try:
        page_ai.show()
    except Exception as e:
        st.error(f"Error on AI Assistant: {str(e)}")
        st.exception(e)
