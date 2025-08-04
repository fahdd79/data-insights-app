import streamlit as st
import pandas as pd

def show_profiling(df: pd.DataFrame):
    st.subheader("🗂️ Dataset Preview & Profiling")
    st.dataframe(df.head(), use_container_width=True)
    st.markdown(f"**Rows:** {df.shape[0]} &nbsp;&nbsp; **Columns:** {df.shape[1]}")

    with st.expander("🔎 Column Data Types", expanded=True):
        st.dataframe(pd.DataFrame(df.dtypes, columns=["Data Type"]), use_container_width=True)

    with st.expander("🚫 Missing Values Per Column"):
        st.dataframe(pd.DataFrame(df.isnull().sum(), columns=["Missing Count"]), use_container_width=True)

    with st.expander("✅ Unique Values Per Column"):
        st.dataframe(pd.DataFrame(df.nunique(), columns=["Unique Count"]), use_container_width=True)
