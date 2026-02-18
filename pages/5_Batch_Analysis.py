"""
Batch Analysis page — runs in isolation when user selects it (Streamlit native multipage).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
st.set_page_config(page_title="Batch Analysis | Breathing ML", page_icon="🫁", layout="wide", initial_sidebar_state="expanded")

from frontend.pages._5_Batch_Analysis import show
show()
