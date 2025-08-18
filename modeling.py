"""
modeling.py

Modeling module for the Streamlit app.

What this renders:
  - Sidebar controls to choose target column, algorithm, hyperparameters, test split, CV folds
  - Safe train/test split (with stratification for classification when possible)
  - Optional cross-validation with guards for tiny datasets / minority classes
  - Metrics (Accuracy/F1 + report & confusion matrix for classification; MSE/R² for regression)
  - Tree-model feature importances (table + barh plot)
  - SHAP explainability (global summary bar & beeswarm, optional row-level waterfall)
  - Self-contained HTML report (tables + embedded images via base64)

Design choices & guardrails:
  - Problem type heuristic: treat as classification if target dtype is object OR has < 20 unique values.
  - K-NN neighbor cap: n_neighbors ≤ training sample count to prevent scikit-learn errors.
  - CV guard: only run if ≥2 samples per class (classification); otherwise explain and skip.
  - SHAP: uses shap.Explainer (Tree/Linear are fast; KNN disabled by default for speed).
"""

from __future__ import annotations

import math
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    KFold,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score,
)

from report_utils import fig_to_base64

# Optional dependency: SHAP. Keep the app fully functional if it's missing.
try:
    import shap  # type: ignore
    shap_available = True
except Exception:
    shap = None
    shap_available = False


def show_modeling(df: pd.DataFrame) -> None:
    """
    Render the Modeling section.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to model. The user chooses the target column in the sidebar.
    """
    st.subheader("🤖 Auto-ML Modeling")

    # ──────────────────────────────────────────────────────────────────────
    # Target & problem type
    # Heuristic: If target is categorical (object) or has few unique values (<20),
    # treat it as classification; otherwise regression.
    # ──────────────────────────────────────────────────────────────────────
    target = st.sidebar.selectbox("Target column", df.columns)
    is_clf = (df[target].dtype == "object") or (df[target].nunique() < 20)

    if is_clf:
        algos = {
            "Random Forest": RandomForestClassifier,
            "Gradient Boosting": GradientBoostingClassifier,
            "Logistic Regression": LogisticRegression,
            "K-NN": KNeighborsClassifier,
        }
    else:
        algos = {
            "Random Forest": RandomForestRegressor,
            "Gradient Boosting": GradientBoostingRegressor,
            "Linear Regression": LinearRegression,
            "K-NN": KNeighborsRegressor,
        }

    algorithm = st.sidebar.selectbox("Algorithm", list(algos.keys()))
    ModelClass = algos[algorithm]

    # ──────────────────────────────────────────────────────────────────────
    # Feature matrix X / target vector y
    # - One-hot encode non-numeric columns so sklearn receives pure numeric arrays.
    # - Drop rows with missing values after encoding to avoid runtime errors
    #   (simple, deterministic behavior suitable for a demo app).
    # - Align y to X.index in case encoding dropped rows.
    # ──────────────────────────────────────────────────────────────────────
    X = pd.get_dummies(df.drop(columns=[target])).dropna()
    y = df[target].loc[X.index]

    # ──────────────────────────────────────────────────────────────────────
    # Test split size
    # Streamlit slider → percentage → fraction. We also compute counts so
    # we can constrain K-NN neighbors relative to available training data.
    # sklearn uses ceil(n_samples * test_size) for the test set.
    # ──────────────────────────────────────────────────────────────────────
    test_pct = st.sidebar.slider("Test set (%)", 10, 50, 25)
    test_size = test_pct / 100.0
    test_count = math.ceil(len(X) * test_size)
    training_count = max(len(X) - test_count, 1)  # at least 1 to avoid zero-sample issues

    # ──────────────────────────────────────────────────────────────────────
    # Hyperparameters (surface only the most impactful knobs)
    # Trees: n_estimators, max_depth (+ RF: min_samples_split, max_features;
    #        + GB: learning_rate, subsample)
    # K-NN: neighbors bounded by training_count (prevents predict errors)
    # ──────────────────────────────────────────────────────────────────────
    params: dict = {}
    if algorithm in ["Random Forest", "Gradient Boosting"]:
        params["n_estimators"] = st.sidebar.slider(
            "Trees (n_estimators)", 10, 500, 100, step=10
        )
        md = st.sidebar.slider("Max depth (0 = none)", 0, 50, 5, step=1)
        if md > 0:
            params["max_depth"] = md

        if algorithm == "Random Forest":
            params["min_samples_split"] = st.sidebar.slider(
                "Min samples split", 2, 20, 2, step=1
            )
            params["max_features"] = st.sidebar.selectbox(
                "Max features", ["auto", "sqrt", "log2"], index=1
            )

        if algorithm == "Gradient Boosting":
            params["learning_rate"] = st.sidebar.slider(
                "Learning rate", 0.01, 1.0, 0.10, step=0.01
            )
            params["subsample"] = st.sidebar.slider(
                "Subsample fraction", 0.1, 1.0, 1.0, step=0.1
            )

    elif algorithm == "K-NN":
        # Keep k within available training samples so kneighbors() never errors.
        default_k = min(5, training_count)
        params["n_neighbors"] = st.sidebar.slider(
            "Neighbors (n_neighbors)", 1, training_count, default_k
        )

    # Instantiate the model from the selected class and params
    model = ModelClass(**params)

    # ──────────────────────────────────────────────────────────────────────
    # Cross-validation (safe for tiny data)
    # - For classification, we require ≥2 samples per class to enable CV.
    # - For regression, we only require ≥2 total samples.
    # - StratifiedKFold preserves class proportions across folds.
    # ──────────────────────────────────────────────────────────────────────
    scoring = "accuracy" if is_clf else "r2"
    can_cv = len(X) >= 2 and (not is_clf or y.value_counts().min() >= 2)

    if can_cv:
        max_folds = y.value_counts().min() if is_clf else len(X)
        cv_folds = st.sidebar.slider("CV folds", 2, max_folds, min(3, max_folds))
        splitter = (
            StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            if is_clf
            else KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        )
        with st.spinner(f"Running {cv_folds}-fold CV…"):
            cv_scores = cross_val_score(model, X, y, cv=splitter, scoring=scoring)
        st.write("CV scores:", [f"{s:.2f}" for s in cv_scores])
        st.write(f"Mean {scoring}: {pd.Series(cv_scores).mean():.2f}")
    else:
        st.info("Skipping cross-validation (need ≥2 samples per class for classification).")
        cv_scores = []

    # ──────────────────────────────────────────────────────────────────────
    # Train/Test split & model fit
    # - Stratify when possible for classification to keep class ratios stable.
    # ──────────────────────────────────────────────────────────────────────
    stratify = y if (is_clf and y.value_counts().min() >= 2) else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=stratify, random_state=42
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    # ──────────────────────────────────────────────────────────────────────
    # Metrics
    # - Classification: Accuracy + weighted F1 + full report + confusion matrix.
    # - Regression: MSE + R².
    # ──────────────────────────────────────────────────────────────────────
    st.markdown("**Test Set Results**")
    if is_clf:
        st.write(f"- Accuracy: **{accuracy_score(y_te, y_pred):.2f}**")
        st.write(f"- F1 Score: **{f1_score(y_te, y_pred, average='weighted'):.2f}**")
        st.text("Classification Report:\n" + classification_report(y_te, y_pred))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_te, y_pred))
    else:
        st.write(f"- MSE: **{mean_squared_error(y_te, y_pred):.2f}**")
        st.write(f"- R²: **{r2_score(y_te, y_pred):.2f}**")

    # ──────────────────────────────────────────────────────────────────────
    # HTML report (tabular summary + metrics + feature importances)
    # We build up HTML fragments in order, then offer a download button.
    # ──────────────────────────────────────────────────────────────────────
    report_parts = [
        "<html><head><meta charset='utf-8'><title>AutoInsight Report</title></head><body>",
        "<h1>AutoInsight Report</h1>",
        "<h2>Summary Statistics</h2>",
        df.describe(include="all").T.to_html(classes="table table-striped", border=0),
        "<h2>Modeling Results</h2>",
        f"<p><strong>Problem Type:</strong> {'Classification' if is_clf else 'Regression'}</p>",
        "<h3>Cross-Validation Scores</h3>",
        pd.DataFrame(cv_scores, columns=[scoring]).to_html(border=0),
        f"<p><strong>Mean {scoring}:</strong> {pd.Series(cv_scores).mean():.2f}</p>",
        "<h3>Test Set Metrics</h3>",
    ]
    if is_clf:
        report_parts += [
            f"<p><strong>Accuracy:</strong> {accuracy_score(y_te, y_pred):.2f}</p>",
            f"<p><strong>F1 Score:</strong> {f1_score(y_te, y_pred, average='weighted'):.2f}</p>",
            "<h4>Classification Report</h4>",
            f"<pre>{classification_report(y_te, y_pred)}</pre>",
        ]
    else:
        report_parts += [
            f"<p><strong>MSE:</strong> {mean_squared_error(y_te, y_pred):.2f}</p>",
            f"<p><strong>R²:</strong> {r2_score(y_te, y_pred):.2f}</p>",
        ]

    # ──────────────────────────────────────────────────────────────────────
    # Feature importances (tree models only)
    # - We expose a table and a horizontal bar plot.
    # - The bar plot is also embedded into the downloadable HTML via base64.
    # ──────────────────────────────────────────────────────────────────────
    if hasattr(model, "feature_importances_"):
        imp_df = (
            pd.Series(model.feature_importances_, index=X.columns)
            .sort_values(ascending=False)
            .to_frame("Importance")
        )
        st.markdown("**Feature Importances**")
        fig, ax = plt.subplots()
        imp_df["Importance"].plot.barh(ax=ax)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        st.pyplot(fig)

        # Include in HTML report (table + image)
        report_parts += [
            "<h3>Feature Importances (Table)</h3>",
            imp_df.to_html(classes="table table-striped", border=0),
        ]
        img_b64 = fig_to_base64(fig)
        report_parts += [
            "<h3>Feature Importances Plot</h3>",
            f'<img src="data:image/png;base64,{img_b64}" alt="Feature Importances" width="600"/>',
        ]
        plt.close(fig)
    else:
        st.info("Feature importances not available for the selected algorithm.")

    # ──────────────────────────────────────────────────────────────────────
    # SHAP explainability (global + local)
    # - Shows a small status note.
    # - Global: summary bar & beeswarm for how features impact predictions overall.
    # - Local: row-level waterfall explaining one prediction (optional).
    # ──────────────────────────────────────────────────────────────────────
    st.markdown("**SHAP status (explainability)**")
    if shap_available:
        st.caption(
            f"Python interpreter: {st.session_state.get('python_exec', '') or ''}\n\n"
            f"shap {getattr(shap, '__version__', '?')} loaded from: {getattr(shap, '__file__', '?')}"
        )
    else:
        st.info("Install SHAP to enable explainability plots: `pip install shap`")

    if shap_available:
        show_shap = st.checkbox("Show SHAP explainability plots", value=True)
        if show_shap:
            if "K-NN" in algorithm:
                # KernelExplainer for KNN is slow; we keep UX snappy by disabling.
                st.info("SHAP for K-NN is disabled (KernelExplainer is very slow). Try a tree/linear model.")
            else:
                try:
                    # Fit the explainer on training data only to avoid test leakage.
                    explainer = shap.Explainer(model, X_tr, feature_names=X.columns)

                    # For very large test sets, subsample to keep plots responsive.
                    X_te_for_shap = X_te
                    if len(X_te_for_shap) > 1000:
                        X_te_for_shap = X_te_for_shap.sample(1000, random_state=42)

                    shap_values = explainer(X_te_for_shap)

                    # Global — summary bar (mean |impact|)
                    st.markdown("**SHAP Summary (bar)**")
                    shap.summary_plot(shap_values, X_te_for_shap, plot_type="bar", show=False)
                    st.pyplot(plt.gcf())
                    # Embed summary bar in HTML report
                    shap_bar_b64 = fig_to_base64(plt.gcf())
                    report_parts += [
                        "<h3>SHAP Summary (Mean |Impact|)</h3>",
                        f'<img src="data:image/png;base64,{shap_bar_b64}" alt="SHAP Summary" width="700"/>'
                    ]

                    # Global — beeswarm (direction + magnitude + value color)
                    st.markdown("**SHAP Summary (beeswarm)**")
                    shap.summary_plot(shap_values, X_te_for_shap, show=False)
                    st.pyplot(plt.gcf())
                    plt.close(plt.gcf())

                except Exception as e:
                    st.warning(f"SHAP couldn't compute global plots for this model: {e}")

        # Row-level / local explanation (waterfall for a single prediction)
        with st.expander("🔍 Explain a single prediction (row-level SHAP)"):
            if "K-NN" in algorithm:
                st.info("Row-level SHAP for K-NN is disabled by default (too slow).")
            elif len(X_te) == 0:
                st.info("No rows in the test set.")
            else:
                row_idx = st.number_input(
                    "Pick a row from the test set",
                    min_value=0,
                    max_value=max(len(X_te) - 1, 0),
                    value=0,
                    step=1,
                )
                try:
                    explainer = shap.Explainer(model, X_tr, feature_names=X.columns)
                    sv_all = explainer(X_te)

                    # Multiclass case: sv_all.values has shape (n_samples, n_features, n_classes).
                    # We choose the predicted class for the selected row.
                    if getattr(sv_all.values, "ndim", 0) == 3:
                        pred_label = model.predict(X_te.iloc[[row_idx]])[0]
                        class_idx = list(model.classes_).index(pred_label)
                        sv = shap.Explanation(
                            values=sv_all.values[row_idx, :, class_idx],
                            base_values=sv_all.base_values[row_idx, class_idx],
                            data=sv_all.data[row_idx],
                            feature_names=X.columns,
                        )
                    else:
                        # Regression or binary classification (2D values)
                        sv = sv_all[row_idx]

                    # Waterfall: shows base value → feature pushes → final prediction
                    shap.plots.waterfall(sv, max_display=10, show=False)
                    st.pyplot(plt.gcf())

                    # Optional: embed this specific row’s waterfall in the HTML report
                    add_row_to_report = st.checkbox("Include this waterfall in the HTML report", value=False)
                    if add_row_to_report:
                        waterfall_b64 = fig_to_base64(plt.gcf())
                        report_parts += [
                            "<h3>Row-level SHAP Waterfall</h3>",
                            f"<p><em>Row index:</em> {int(row_idx)}</p>",
                            f'<img src="data:image/png;base64,{waterfall_b64}" alt="SHAP Waterfall" width="700"/>'
                        ]
                    st.caption(
                        "Right of zero = pushes prediction up; left = pushes it down. "
                        "Bar length = strength of that push for this specific row."
                    )
                    plt.close(plt.gcf())
                except Exception as e:
                    st.warning(f"SHAP couldn't explain this prediction: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Finalize & offer HTML report download
    # We join all HTML parts, then Streamlit serves it as a downloadable file.
    # ──────────────────────────────────────────────────────────────────────
    report_parts.append("</body></html>")
    report_html = "\n".join(report_parts)

    st.download_button(
        "📥 Download HTML Report",
        report_html,
        file_name="autoinsight_report.html",
        mime="text/html",
    )