import streamlit as st
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def show_eda(df):
    st.subheader("📊 Exploratory Data Analysis (EDA)")

    with st.expander("📋 Summary Statistics", expanded=True):
        stats = df.describe(include="all").T
        st.dataframe(stats, use_container_width=True)

    with st.expander("📈 Histograms"):
        for col in df.select_dtypes(include=["int64", "float64"]).columns:
            st.markdown(f"**Histogram for `{col}`**")
            fig, ax = plt.subplots()
            ax.hist(df[col].dropna(), bins=30)
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    with st.expander("🔗 Correlation Heatmap"):
        corr = df.select_dtypes(include=["int64", "float64"]).corr()
        fig, ax = plt.subplots()
        cax = ax.matshow(corr)
        fig.colorbar(cax)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)
        st.pyplot(fig)

    with st.expander("🔍 Scatter Matrix"):
        numeric_df = df.select_dtypes(include=["int64", "float64"])
        scatter_matrix(numeric_df, alpha=0.5, diagonal="hist", figsize=(8, 8))
        st.pyplot(plt.gcf())
