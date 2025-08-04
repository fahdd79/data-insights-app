import streamlit as st
from data_loader import load_data
from profiling    import show_profiling
from eda          import show_eda
from modeling     import show_modeling

def main():
    st.set_page_config(
        page_title="AutoInsight",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ─── Global CSS ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <style>
          footer { visibility: hidden; }
          h1, h2, h3, h4 { font-family: 'Segoe UI', sans-serif; }
          .css-18e3th9 { text-align: center; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ─── SIDEBAR & NAVIGATION ──────────────────────────────────────────────────
    st.sidebar.title("🔍 AutoInsight")
    page = st.sidebar.radio(
        "Navigation",
        ["Upload & Preview", "Profiling", "EDA", "Modeling"]
    )

    # ─── LOAD DATA ONCE ────────────────────────────────────────────────────────
    df = load_data()

    # ─── PAGE ROUTER ───────────────────────────────────────────────────────────
    if page == "Upload & Preview":
        st.header("📥 Upload & Preview")
        st.write("Use the sidebar to select or upload your dataset.")
        st.dataframe(df.head(), use_container_width=True)

    elif page == "Profiling":
        show_profiling(df)

    elif page == "EDA":
        show_eda(df)

    else:  # Modeling
        show_modeling(df)


if __name__ == "__main__":
    main()
