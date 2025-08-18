"""
data_loader.py

Data-loading utilities for the Streamlit app.

What this provides:
  - A single `load_data()` function that renders a sidebar control to pick a
    data source and returns a pandas DataFrame.
  - Sources:
      1) User upload (CSV / Excel .xlsx)
      2) Demo datasets: Iris (classification), Wine (classification)

Design notes:
  - CSV reading is given a couple of fallbacks (encoding, delimiter sniffing)
    to handle messy real-world files more gracefully.
  - Excel requires `openpyxl` for .xlsx files; we surface a friendly message
    if it's missing.
  - For very large files, we display a quick heads-up to the user.
  - Demo datasets are cached to avoid reloading on every rerun.
"""

from __future__ import annotations

import io
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris, load_wine


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _read_csv_with_fallbacks(raw_bytes: bytes) -> pd.DataFrame:
    """
    Try to read CSV content from raw bytes using a few sensible fallbacks.

    Order:
      1) Default pandas read_csv (fast path)
      2) UTF-8 decoding issues → try ISO-8859-1 ('latin1')
      3) Parser issues → try delimiter sniffing (sep=None with python engine)

    Parameters
    ----------
    raw_bytes : bytes
        The raw file content.

    Returns
    -------
    pd.DataFrame
        Parsed CSV as a DataFrame, or raises the last exception if all attempts fail.
    """
    # 1) Plain attempt (most files)
    try:
        return pd.read_csv(io.BytesIO(raw_bytes))
    except UnicodeDecodeError:
        pass  # try latin1 next
    except pd.errors.ParserError:
        pass  # try delimiter sniffing next

    # 2) Encoding fallback: latin1
    try:
        return pd.read_csv(io.BytesIO(raw_bytes), encoding="latin1")
    except pd.errors.ParserError:
        pass  # try delimiter sniffing next

    # 3) Delimiter sniffing (slower, but handles odd separators)
    #    sep=None triggers the Python engine's built-in sniffer.
    return pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine="python")


@st.cache_data(show_spinner=False)
def _load_demo(dataset: str) -> pd.DataFrame:
    """
    Load a small demo dataset and cache it.

    Parameters
    ----------
    dataset : {"iris", "wine"}
        Which demo to load.

    Returns
    -------
    pd.DataFrame
    """
    if dataset == "iris":
        data = load_iris(as_frame=True)
        return data.frame
    elif dataset == "wine":
        data = load_wine(as_frame=True)
        return data.frame
    else:
        raise ValueError(f"Unknown demo dataset: {dataset}")


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Render a sidebar data source picker and return the resulting DataFrame.

    UI (in the sidebar):
      - Selectbox for source (Upload / Iris / Wine)
      - File uploader if Upload is chosen

    Returns
    -------
    pd.DataFrame
        The loaded dataset ready for profiling/EDA/modeling.

    Notes
    -----
    - If no file is uploaded when "Upload CSV/Excel" is selected, the function
      calls `st.stop()` to halt execution until the user provides a file.
    """
    source = st.sidebar.selectbox(
        "Data source",
        [
            "Upload CSV/Excel",
            "Iris (classification)",
            "Wine (classification)",
        ],
        index=0,
        help="Choose a demo dataset to explore quickly, or upload your own CSV/XLSX.",
    )

    if source == "Upload CSV/Excel":
        uploaded = st.sidebar.file_uploader(
            "Choose a file",
            type=["csv", "xlsx"],
            help="CSV or Excel (.xlsx). For very large files, consider sampling first."
        )
        if not uploaded:
            st.info("Please upload a CSV or Excel file to get started.")
            st.stop()

        # Read the file content once so we can retry parsing if needed.
        raw = uploaded.read()
        size_mb = len(raw) / (1024 * 1024)
        if size_mb > 50:
            st.warning(f"Large file detected (~{size_mb:.1f} MB). This may be slow to process.")

        # Decide reader based on extension; fall back to CSV attempts otherwise.
        name_lower = (uploaded.name or "").lower()
        try:
            if name_lower.endswith(".xlsx"):
                try:
                    # Requires openpyxl to be installed
                    df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
                except ImportError:
                    st.error(
                        "Reading .xlsx requires `openpyxl`. Install it via:\n\n"
                        "```bash\npip install openpyxl\n```"
                    )
                    st.stop()
            else:
                df = _read_csv_with_fallbacks(raw)
        except Exception as e:
            st.error(f"Sorry, we couldn't parse that file as CSV/Excel.\n\n**Details:** {e}")
            st.stop()

        # Light sanity check: warn if there are no columns/rows
        if df.shape[1] == 0 or df.shape[0] == 0:
            st.warning("The uploaded file appears to be empty or has no readable columns.")
        return df

    elif source == "Iris (classification)":
        return _load_demo("iris")

    else:  # "Wine (classification)"
        return _load_demo("wine")