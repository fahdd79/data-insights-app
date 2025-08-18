"""
eda.py

Exploratory Data Analysis (EDA) components for the Streamlit app.

What this renders:
  1) Summary statistics (numeric + categorical) in a scrollable table.
  2) Histograms for each numeric column.
  3) Correlation heatmap for numeric features.
  4) Scatter-matrix across numeric features.

Notes:
- We purposely compute a numeric-only view once and reuse it.
- We handle small/edge cases (e.g., no numeric columns, only one numeric column).
- Figures are explicitly closed after st.pyplot(...) to avoid memory/figure buildup
"""

from __future__ import annotations

import streamlit as st
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import pandas as pd


def show_eda(df: pd.DataFrame) -> None:
    """
    Render the Exploratory Data Analysis (EDA) section in Streamlit.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset provided by the user or demo loader.
        Expected to contain a mix of numeric and non-numeric columns.
    """
    st.subheader("📊 Exploratory Data Analysis (EDA)")

    # ────────────────────────────────────────────────────────────────────
    # 1) Summary Statistics
    # - include="all" computes stats for numeric AND categorical columns
    # - .T (transpose) to show one row per column (easier to scan)
    # ────────────────────────────────────────────────────────────────────
    with st.expander("📋 Summary Statistics", expanded=True):
        stats = df.describe(include="all").T
        st.dataframe(stats, use_container_width=True)

    # Build a reusable numeric-only view. We'll need it for plots below.
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_df = df[numeric_cols] if numeric_cols else pd.DataFrame()

    # ────────────────────────────────────────────────────────────────────
    # 2) Histograms
    # - One histogram per numeric column
    # - dropna() excludes missing values from the distribution
    # ────────────────────────────────────────────────────────────────────
    with st.expander("📈 Histograms"):
        if not numeric_cols:
            st.info("No numeric columns found to plot histograms.")
        else:
            for col in numeric_cols:
                st.markdown(f"**Histogram for `{col}`**")
                fig, ax = plt.subplots()
                ax.hist(df[col].dropna(), bins=30)
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                ax.grid(alpha=0.3, linestyle=":", linewidth=0.5)
                st.pyplot(fig)
                plt.close(fig)

    # ────────────────────────────────────────────────────────────────────
    # 3) Correlation Heatmap (numeric features only)
    # - Requires at least two numeric columns (otherwise correlation is trivial)
    # - We use matplotlib's matshow for a lightweight heatmap
    # ────────────────────────────────────────────────────────────────────
    with st.expander("🔗 Correlation Heatmap"):
        if len(numeric_cols) < 2:
            st.info("Need at least two numeric columns to compute a correlation heatmap.")
        else:
            corr = numeric_df.corr(numeric_only=True)
            fig, ax = plt.subplots()
            cax = ax.matshow(corr)  # simple heatmap; leaves default colormap
            fig.colorbar(cax, fraction=0.046, pad=0.04)

            # Tick labels for both axes
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.columns)

            # Light grid lines to separate cells visually
            ax.set_xticks([x - 0.5 for x in range(1, len(corr.columns))], minor=True)
            ax.set_yticks([y - 0.5 for y in range(1, len(corr.columns))], minor=True)
            ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)

            st.pyplot(fig)
            plt.close(fig)

    # ────────────────────────────────────────────────────────────────────
    # 4) Scatter Matrix (pairwise relationships between numeric features)
    # - Requires at least two numeric columns
    # - alpha controls point transparency; diagonal="hist" shows univariate hist
    # ────────────────────────────────────────────────────────────────────
    with st.expander("🔍 Scatter Matrix"):
        if len(numeric_cols) < 2:
            st.info("Need at least two numeric columns to draw a scatter matrix.")
        else:
            scatter_matrix(numeric_df, alpha=0.5, diagonal="hist", figsize=(8, 8))
            st.pyplot(plt.gcf())
            plt.close(plt.gcf())