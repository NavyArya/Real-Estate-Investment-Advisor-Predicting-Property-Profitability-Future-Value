# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score
)

st.set_page_config(page_title="Real Estate: Train & Predict (no .pkl needed)", layout="wide")

st.title("Real Estate Investment — Train & Predict (on upload)")

st.markdown("""
Upload your `india_housing_prices.csv` (or similar) and the app will:
- preprocess and engineer a few features (as in your notebook)
- train classification (Good_Investment_multi) and regression (Future_Price_5Y) models
- show evaluation metrics and allow single-sample prediction
""")

uploaded_file = st.file_uploader("Upload CSV file (e.g. india_housing_prices.csv)", type=["csv"])

def preprocess_and_engineer(df):
    """Apply the same preprocessing/feature engineering as in the notebook, defensively."""
    df = df.copy()
    # basic cleaning: remove exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # numeric/categorical lists detection
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Fill missing
    for col in num_cols:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    for col in cat_cols:
        if df[col].mode().shape[0] > 0:
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)
        else:
            df[col] = df[col].fillna("missing")

    # Price per SqFt
    if "Price_in_Lakhs" in df.columns and "Size_in_SqFt" in df.columns:
        df["Price_per_SqFt"] = (
            df["Price_in_Lakhs"] * 100000 /
            df["Size_in_SqFt"].replace(0, np.nan)
        )
        df["Price_per_SqFt"] = df["Price_per_SqFt"].replace([np.inf, -np.inf], np.nan)
        df["Price_per_SqFt"] = df["Price_per_SqFt"].fillna(df["Price_per_SqFt"].median())

    # School density score 0-1
    if "Nearby_Schools" in df.columns:
        schools = df["Nearby_Schools"].astype(float)
        df["School_Density_Score"] = (schools - schools.min()) / (schools.max() - schools.min() + 1e-9)

    # Good_Investment_price (price <= city median)
    if {"City", "Price_in_Lakhs"}.issubset(df.columns):
        city_median_price = df.groupby("City")["Price_in_Lakhs"].transform("median")
        df["Good_Investment_price"] = (df["Price_in_Lakhs"] <= city_median_price).astype(int)

    # Good_Investment_pps (price_per_sqft <= city median)
    if {"City", "Price_per_SqFt"}.issubset(df.columns):
        city_median_pps = df.groupby("City")["Price_per_SqFt"].transform("median")
        df["Good_Investment_pps"] = (df["Price_per_SqFt"] <= city_median_pps).astype(int)

    # investment_score (example multi-factor)
    df["investment_score"] = 0
    if "BHK" in df.columns:
        df["investment_score"] += np.where(df["BHK"].fillna(0) >= 3, 1, 0)
    if "RERA" in df.columns:
        # assume binary 0/1 or strings
        df["investment_score"] += np.where(df["RERA"].astype(str).isin(["1","True","true","Y","y"]), 1, 0)
    if "School_Density_Score" in df.columns:
        df["investment_score"] += np.where(df["School_Density_Score"] > 0.5, 1, 0)

    df["Good_Investment_multi"] = (df["investment_score"] >= 2).astype(int)

    # Future_Price_5Y approximation
    if "Price_in_Lakhs" in df.columns:
        df["Future_Price_5Y"] = df["Price_in_Lakhs"] * 1.47  # approx (1.08)^5

    return df

@st.cache_data(show_spinner=False)
def prepare_data(df):
    df_proc = preprocess_and_engineer(df)

    # Drop rows without targets
    target_class = "Good_Investment_multi"
    if target_class not in df_proc.columns:
        raise ValueError(f"Target column '{target_class}' missing after preprocessing.")

    if "Future_Price_5Y" not in df_proc.columns:
        raise ValueError("Regression target 'Future_Price_5Y' missing after preprocessing.")

    # Define features: drop IDs & targets
    exclude_cols = ["ID", "Good_Investment_multi", "Future_Price_5Y", "investment_score"]
    feature_cols = [c for c in df_proc.columns if c not in exclude_cols]

    X = df_proc[feature_cols].copy()
    y_class = df_proc["Good_Investment_multi"].copy()
    y_reg = df_proc["Future_Price_5Y"].copy()

    # Split numeric & categorical
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Convert all columns to numeric where possible (defensive)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Fill NaNs created by coercion with medians for numeric columns
    for col in X.columns:
        if X[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(0)

    # Final numeric-only filter for modeling simplicity
    # We'll one-hot encode any remaining "object" columns (unlikely after coercion)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]

    return X, y_class, y_reg, num_cols, cat_cols

@st.cache_resource(show_spinner=False)
def build_pipelines(num_cols, cat_cols):
    """Return classification and regression pipelines."""
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ],
        remainder="drop"
    )

    # Classification pipeline (RandomForest)
    clf_pipe = Pipeline([
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Regression pipeline (RandomForestRegressor)
    reg_pipe = Pipeline([
        ("preprocess", preprocessor),
        ("reg", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    return clf_pipe, reg_pipe

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Couldn't read CSV: {e}")
        st.stop()

    st.sidebar.header("Preview & Options")
    if st.sidebar.checkbox("Show top rows", value=True):
        st.write(df.head())

    # Show missing columns warning
    st.sidebar.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Prepare data (preprocess + split)
    try:
        X, y_class, y_reg, num_cols, cat_cols = prepare_data(df)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

    st.write("### Feature summary")
    st.write(f"Numeric features detected: {len(num_cols)} — {num_cols}")
    st.write(f"Categorical features detected (will be one-hot encoded): {len(cat_cols)} — {cat_cols}")

    # allow user to choose classifier/regressor or both
    st.sidebar.header("Model choices")
    do_classify = st.sidebar.checkbox("Train classification (Good_Investment_multi)", value=True)
    do_regress = st.sidebar.checkbox("Train regression (Future_Price_5Y)", value=True)

    if not (do_classify or do_regress):
        st.warning("Select at least one task in the sidebar.")
        st.stop()

    # Split - keep same split for both tasks
    try:
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
        )
    except Exception as e:
        # fallback if stratify fails (e.g., single class)
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_class, y_reg, test_size=0.2, random_state=42
        )

    st.write("Train / test shapes:", X_train.shape, X_test.shape)

    # Build pipelines
    clf_pipe, reg_pipe = build_pipelines(num_cols, cat_cols)

    if st.button("Train models"):
        with st.spinner("Training... this may take a bit depending on data size"):
            results = {}
            if do_classify:
                clf_pipe.fit(X_train, y_class_train)
                y_pred = clf_pipe.predict(X_test)
                if hasattr(clf_pipe.named_steps['clf'], "predict_proba"):
                    y_proba = clf_pipe.predict_proba(X_test)[:, 1]
                else:
                    # fallback: use decision_function if exists, else zeros
                    try:
                        y_proba = clf_pipe.decision_function(X_test)
                        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-9)
                    except Exception:
                        y_proba = np.zeros(len(y_test))

                results['classification'] = {
                    "Accuracy": accuracy_score(y_class_test, y_pred),
                    "Precision": precision_score(y_class_test, y_pred, zero_division=0),
                    "Recall": recall_score(y_class_test, y_pred, zero_division=0),
                    "F1": f1_score(y_class_test, y_pred, zero_division=0),
                    "ROC_AUC": roc_auc_score(y_class_test, y_proba) if len(np.unique(y_class_test))>1 else np.nan
                }

            if do_regress:
                reg_pipe.fit(X_train, y_reg_train)
                y_reg_pred = reg_pipe.predict(X_test)
                results['regression'] = {
                    "RMSE": mean_squared_error(y_reg_test, y_reg_pred, squared=False),
                    "R2": r2_score(y_reg_test, y_reg_pred)
                }

        st.success("Training finished.")
        st.write("### Results")
        if 'classification' in results:
            st.subheader("Classification metrics (Good_Investment_multi)")
            st.write(pd.DataFrame([results['classification']]).T.rename(columns={0:"value"}))
        if 'regression' in results:
            st.subheader("Regression metrics (Future_Price_5Y)")
            st.write(pd.DataFrame([results['regression']]).T.rename(columns={0:"value"}))

        # Simple plot: Price_in_Lakhs distribution and Price_per_SqFt if present
        st.write("### Quick plots")
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        if "Price_in_Lakhs" in df.columns:
            ax.hist(df["Price_in_Lakhs"].dropna(), bins=50)
            ax.set_title("Price_in_Lakhs distribution")
            ax.set_xlabel("Price (Lakhs)")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.write("Column `Price_in_Lakhs` not found — skipping histogram.")

        # Feature importance (classification) if RF
        if do_classify:
            try:
                clf_model = clf_pipe.named_steps['clf']
                if hasattr(clf_model, "feature_importances_"):
                    importances = clf_model.feature_importances_
                    # get feature names after preprocessing
                    pre = clf_pipe.named_steps['preprocess']
                    try:
                        feat_names = pre.get_feature_names_out()
                    except Exception:
                        # fallback: combine num and encoder feature names
                        feat_names_num = num_cols
                        if len(cat_cols)>0:
                            enc = pre.named_transformers_['cat']
                            enc_names = enc.get_feature_names_out(cat_cols)
                            feat_names = np.concatenate([feat_names_num, enc_names])
                        else:
                            feat_names = feat_names_num
                    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
                    imp_df = imp_df.sort_values("importance", ascending=False).head(15)
                    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
                    ax2.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
                    ax2.set_title("Top features (classification)")
                    st.pyplot(fig2)
                else:
                    st.info("Classifier has no feature_importances_.")
            except Exception as e:
                st.write("Could not compute feature importance:", e)

        # Save models in session state for prediction UI
        st.session_state["clf_pipe"] = clf_pipe if do_classify else None
        st.session_state["reg_pipe"] = reg_pipe if do_regress else None
        st.session_state["X_columns"] = X.columns.tolist()

    # Prediction UI (single row)
    st.write("---")
    st.subheader("Single-sample prediction (use column order below)")
    st.write("Columns used as features (order matters):")
    st.write(X.columns.tolist())

    with st.form("single_predict"):
        st.write("Enter comma-separated values matching the feature columns order (or paste a row):")
        sample_text = st.text_area("Sample row (CSV row)", height=80)
        submit = st.form_submit_button("Predict")

    if submit:
        if "X_columns" not in st.session_state:
            st.error("No trained pipelines available — train models first.")
        else:
            col_count = len(st.session_state["X_columns"])
            # parse sample
            try:
                # try reading as CSV row
                sample_vals = [v.strip() for v in sample_text.split(",")]
                if len(sample_vals) != col_count:
                    st.warning(f"You provided {len(sample_vals)} values but the model expects {col_count}.")
                # create dataframe
                sample_df = pd.DataFrame([sample_vals], columns=st.session_state["X_columns"])
                # convert to numeric where possible
                for c in sample_df.columns:
                    sample_df[c] = pd.to_numeric(sample_df[c], errors="ignore")

                # classification predict
                if st.session_state.get("clf_pipe"):
                    clf_pipe_use = st.session_state["clf_pipe"]
                    try:
                        cls_pred = clf_pipe_use.predict(sample_df)[0]
                        cls_proba = clf_pipe_use.predict_proba(sample_df)[0][1] if hasattr(clf_pipe_use.named_steps['clf'], "predict_proba") else None
                        st.write("Classification result (Good Investment?):", int(cls_pred))
                        if cls_proba is not None:
                            st.write(f"Probability (class=1): {cls_proba:.3f}")
                    except Exception as e:
                        st.error("Classification prediction failed: " + str(e))

                # regression predict
                if st.session_state.get("reg_pipe"):
                    reg_pipe_use = st.session_state["reg_pipe"]
                    try:
                        reg_pred = reg_pipe_use.predict(sample_df)[0]
                        st.write("Predicted Future_Price_5Y (Lakhs):", float(reg_pred))
                    except Exception as e:
                        st.error("Regression prediction failed: " + str(e))
            except Exception as e:
                st.error("Could not parse sample row: " + str(e))

else:
    st.info("Upload a CSV to get started. The notebook logic used as reference is included in the app.")
