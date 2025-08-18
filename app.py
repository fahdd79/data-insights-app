"""
app.py

Main entry point for the Streamlit app.

Responsibilities:
  - Page configuration (title, icon, layout)
  - Sidebar "About" and usage hints
  - Data loading (delegates to data_loader.load_data)
  - High-level navigation (tabs): Profiling → EDA → Modeling
  - Passes the loaded DataFrame to each section

Modules used:
  - data_loader.py      -> load_data()
  - profiling.py        -> show_profiling(df)
  - eda.py              -> show_eda(df)
  - modeling.py         -> show_modeling(df)

Notes:
  - We stash the current Python interpreter path into session_state so the
    Modeling module can display it in the SHAP status panel (useful when
    debugging environments).
"""

from __future__ import annotations

import sys
import streamlit as st

from data_loader import load_data
from profiling import show_profiling
from eda import show_eda
from modeling import show_modeling


# ──────────────────────────────────────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoInsight — Data → EDA → ML → Report",
    page_icon="🔎",
    layout="wide",
)

# Store interpreter path for SHAP status (read in modeling.py)
st.session_state["python_exec"] = sys.executable

# Optional: light CSS polish (keeps things compact and wide)
st.markdown(
    """
    <style>
      /* tighten up the layout slightly */
      .block-container { padding-top: 1rem; padding-bottom: 2rem; }
      /* make expander labels a bit bolder */
      .streamlit-expanderHeader { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Header & Sidebar
# ──────────────────────────────────────────────────────────────────────────────
st.title("AutoInsight")
st.caption("Upload a dataset or pick a demo, explore it, train a quick model, and export a report.")

with st.sidebar:
    st.header("📥 Data")
    # The loader renders its own widgets (selectbox + file_uploader) and returns df
    # We keep the rest of the sidebar content below it.
    df = load_data()

    st.markdown("---")
    st.header("ℹ️ About")
    st.write(
        "This app performs lightweight **profiling**, interactive **EDA**, and baseline "
        "**ML** (classification or regression) with safe cross-validation. "
        "It also includes **SHAP** explainability (global + row-level) and exports a "
        "self-contained **HTML report**."
    )
    st.caption("Stack: Python · Pandas · scikit-learn · Streamlit · Matplotlib · SHAP")

# Safety: if loader halted execution (e.g., waiting for upload), Streamlit won’t reach here.
# If we do get here, `df` should be a valid DataFrame.

# ──────────────────────────────────────────────────────────────────────────────
# Tabs: Profiling → EDA → Modeling
# ──────────────────────────────────────────────────────────────────────────────
tab_profile, tab_eda, tab_model = st.tabs(
    ["🗂️ Profiling", "📊 EDA", "🤖 Modeling"]
)

with tab_profile:
    show_profiling(df)

with tab_eda:
    show_eda(df)

with tab_model:
    show_modeling(df)