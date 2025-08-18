"""
profiling.py

Lightweight dataset preview & basic profiling for the Streamlit app.

What this renders:
  1) A head() preview of the dataset (scrollable table).
  2) Row/column counts (+ a small memory footprint estimate).
  3) Column data types.
  4) Missing values per column (count + percent).
  5) Unique values per column.

Notes:
- We avoid heavy profiling libs to keep the app fast and portable.
- All tables use `use_container_width=True` so they adapt to page width.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


def show_profiling(df: pd.DataFrame) -> None:
    """
    Render a quick dataset preview & profiling section in Streamlit.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to profile (already loaded by the app).
    """
    st.subheader("🗂️ Dataset Preview & Profiling")

    # ────────────────────────────────────────────────────────────────────
    # Preview: show the first few rows so users can sanity-check columns.
    # You can tweak how many rows to display by changing .head(5) below.
    # ────────────────────────────────────────────────────────────────────
    st.dataframe(df.head(), use_container_width=True)

    # Basic shape + a rough memory estimate (helps explain slowness on huge files).
    mem_bytes = df.memory_usage(deep=True).sum()
    mem_mb = mem_bytes / (1024 ** 2)
    st.markdown(
        f"**Rows:** {df.shape[0]} &nbsp;&nbsp; **Columns:** {df.shape[1]} "
        f"&nbsp;&nbsp; **~Memory:** {mem_mb:.2f} MB"
    )

    # ────────────────────────────────────────────────────────────────────
    # Column data types: useful to know what will be one-hot encoded later.
    # ────────────────────────────────────────────────────────────────────
    with st.expander("🔎 Column Data Types", expanded=True):
        dtypes_df = pd.DataFrame(df.dtypes, columns=["Data Type"])
        st.dataframe(dtypes_df, use_container_width=True)

    # ────────────────────────────────────────────────────────────────────
    # Missing values per column: show both count and percentage.
    # Sorting by count makes the biggest issues float to the top.
    # ────────────────────────────────────────────────────────────────────
    with st.expander("🚫 Missing Values Per Column"):
        if df.shape[0] == 0:
            st.info("No rows available to compute missing values.")
        else:
            miss_cnt = df.isna().sum()
            miss_pct = (miss_cnt / len(df) * 100).round(2)
            missing_df = pd.DataFrame(
                {"Missing Count": miss_cnt, "Missing %": miss_pct}
            ).sort_values("Missing Count", ascending=False)
            st.dataframe(missing_df, use_container_width=True)

    # ────────────────────────────────────────────────────────────────────
    # Unique values per column: helpful for spotting ID-like columns or
    # very high-cardinality categoricals
    # ────────────────────────────────────────────────────────────────────
    with st.expander("✅ Unique Values Per Column"):
        unique_df = pd.DataFrame(df.nunique(), columns=["Unique Count"]).sort_values(
            "Unique Count", ascending=False
        )
        st.dataframe(unique_df, use_container_width=True)