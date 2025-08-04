import math
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    KFold
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score
)

from report_utils import fig_to_base64

def show_modeling(df: pd.DataFrame):
    st.subheader("🤖 Auto-ML Modeling")

    # ─── Sidebar: Target & Algorithm ───────────────────────────────────────
    target    = st.sidebar.selectbox("Target column", df.columns)
    is_clf    = (df[target].dtype == "object") or (df[target].nunique() < 20)

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

    # ─── Prepare Features & Target ───────────────────────────────────────
    X = pd.get_dummies(df.drop(columns=[target])).dropna()
    y = df[target].loc[X.index]

    # ─── Sidebar: Test Split ─────────────────────────────────────────────
    test_pct  = st.sidebar.slider("Test set (%)", 10, 50, 25)
    test_size = test_pct / 100.0

    # Compute how many training samples you'll have
    # sklearn uses ceil(n_samples * test_size) for test, so:
    test_count     = math.ceil(len(X) * test_size)
    training_count = len(X) - test_count
    training_count = max(training_count, 1)

    # ─── Sidebar: Hyperparameters ───────────────────────────────────────
    params = {}
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
                "Learning rate", 0.01, 1.0, 0.1, step=0.01
            )
            params["subsample"] = st.sidebar.slider(
                "Subsample fraction", 0.1, 1.0, 1.0, step=0.1
            )

    elif algorithm == "K-NN":
        # Bound neighbors by training_count so we never exceed available samples
        default_k = min(5, training_count)
        params["n_neighbors"] = st.sidebar.slider(
            "Neighbors (n_neighbors)",
            1,
            training_count,
            default_k
        )

    # ─── Instantiate Model ───────────────────────────────────────────────
    model = ModelClass(**params)

    # ─── Robust Cross-Validation ────────────────────────────────────────
    scoring = "accuracy" if is_clf else "r2"
    can_cv  = len(X) >= 2 and (not is_clf or y.value_counts().min() >= 2)

    if can_cv:
        max_folds = y.value_counts().min() if is_clf else len(X)
        cv_folds  = st.sidebar.slider(
            "CV folds", 2, max_folds, min(3, max_folds)
        )
        splitter = (
            StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            if is_clf
            else
            KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        )
        with st.spinner(f"Running {cv_folds}-fold CV…"):
            cv_scores = cross_val_score(model, X, y, cv=splitter, scoring=scoring)

        st.write("CV scores:", [f"{s:.2f}" for s in cv_scores])
        st.write(f"Mean {scoring}: {pd.Series(cv_scores).mean():.2f}")
    else:
        st.info("Skipping cross-validation (need ≥2 samples per class).")
        cv_scores = []

    # ─── Train/Test Split & Fit ─────────────────────────────────────────
    stratify = y if (is_clf and y.value_counts().min() >= 2) else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=stratify, random_state=42
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    # ─── Display Test Metrics ───────────────────────────────────────────
    st.markdown("**Test Set Results**")
    if is_clf:
        st.write(f"- Accuracy: **{accuracy_score(y_te, y_pred):.2f}**")
        st.write(f"- F1 Score: **{f1_score(y_te, y_pred, average='weighted'):.2f}**")
        st.text("Classification Report:\n" + classification_report(y_te, y_pred))
        st.write("Confusion Matrix:", confusion_matrix(y_te, y_pred))
    else:
        st.write(f"- MSE: **{mean_squared_error(y_te, y_pred):.2f}**")
        st.write(f"- R²: **{r2_score(y_te, y_pred):.2f}**")

    # ─── Build HTML Report ─────────────────────────────────────────────
    report_parts = [
        "<html><head><meta charset='utf-8'><title>AutoInsight Report</title></head><body>",
        "<h1>AutoInsight Report</h1>",
        "<h2>Summary Statistics</h2>",
        df.describe(include="all").T.to_html(classes='table table-striped', border=0),
        "<h2>Modeling Results</h2>",
        f"<p><strong>Problem Type:</strong> {'Classification' if is_clf else 'Regression'}</p>",
        "<h3>CV Scores</h3>",
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

    # ─── Feature Importances ────────────────────────────────────────────
    if hasattr(model, "feature_importances_"):
        imp_df = (
            pd.Series(model.feature_importances_, index=X.columns)
              .sort_values(ascending=False)
              .to_frame("Importance")
        )
        report_parts += [
            "<h3>Feature Importances</h3>",
            imp_df.to_html(classes="table table-striped", border=0),
        ]
        fig, ax = plt.subplots()
        imp_df["Importance"].plot.barh(ax=ax)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        st.pyplot(fig)
        img_b64 = fig_to_base64(fig)
        report_parts += [
            "<h3>Feature Importances Plot</h3>",
            f'<img src="data:image/png;base64,{img_b64}" alt="Feature Importances" width="600"/>'
        ]
    else:
        st.info("Feature importances not available.")

    report_parts.append("</body></html>")
    report_html = "\n".join(report_parts)

    # ─── HTML Download ──────────────────────────────────────────────────
    st.download_button(
        "📥 Download HTML Report",
        report_html,
        file_name="autoinsight_report.html",
        mime="text/html"
    )
