import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris, load_wine

def load_data():
    source = st.sidebar.selectbox(
        "Data source",
        [
            "Upload CSV/Excel",
            "Iris (classification)",
            "Wine (classification)",
        ],
        index=0,
    )

    if source == "Upload CSV/Excel":
        uploaded = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx"])
        if not uploaded:
            st.info("Please upload a CSV or Excel file to get started.")
            st.stop()
        try:
            return pd.read_csv(uploaded)
        except Exception:
            return pd.read_excel(uploaded)

    elif source == "Iris (classification)":
        data = load_iris(as_frame=True)
        return data.frame

    else:
        data = load_wine(as_frame=True)
        return data.frame
