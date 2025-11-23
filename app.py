# Corrected app.py with updated fairness.compute_group_metrics integration
import streamlit as st
import pandas as pd
import numpy as np

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

st.set_page_config(page_title="AI Model Bias Detector", page_icon="🔍", layout="wide")

try:
    from backend import data_processing, model_train, fairness
except ImportError as e:
    import sys
    error_msg = str(e)
    python_path = sys.executable
    st.error(f"Error: {error_msg}\nInstall missing packages.")
    st.stop()

st.title("🔍 AI Model Bias Detector")

if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'fairness_results' not in st.session_state:
    st.session_state.fairness_results = None

st.sidebar.header("📁 Dataset Upload")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.sidebar.success(f"Dataset loaded! {len(df)} rows.")
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")

if st.session_state.df is not None:
    df = st.session_state.df

    st.header("📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", len(df))
    col2.metric("Columns", len(df.columns))
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.header("⚙️ Configuration")
    col1, col2 = st.columns(2)

    with col1:
        target_col = st.selectbox("Select Target Column", df.columns.tolist())
    with col2:
        protected_col = st.selectbox("Select Protected Attribute", [c for c in df.columns if c != target_col])

    st.header("🚀 Actions")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        preprocess_btn = st.button("🔄 Preprocess")
    with col2:
        train_btn = st.button("🏋️ Train", disabled=st.session_state.preprocessed_data is None)
    with col3:
        evaluate_btn = st.button("📊 Evaluate", disabled=st.session_state.model is None)
    with col4:
        fairness_btn = st.button("⚖️ Fairness", disabled=st.session_state.model is None)

    if preprocess_btn:
        try:
            X_train, X_test, y_train, y_test, encoded_df = data_processing.preprocess_data(df, target_col)
            st.session_state.preprocessed_data = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'encoded_df': encoded_df
            }
            st.success("Preprocessing complete!")
        except Exception as e:
            st.error(f"Error: {str(e)}")

    if train_btn:
        try:
            pdata = st.session_state.preprocessed_data
            model = model_train.train_model(pdata['X_train'], pdata['y_train'])
            st.session_state.model = model
            st.success("Model trained!")
        except Exception as e:
            st.error(f"Error: {str(e)}")

    if evaluate_btn:
        try:
            pdata = st.session_state.preprocessed_data
            accuracy, report = model_train.evaluate_model(
                st.session_state.model, pdata['X_test'], pdata['y_test']
            )
            st.session_state.evaluation_results = {'accuracy': accuracy, 'report': report}
            st.success("Evaluation complete!")
        except Exception as e:
            st.error(f"Error: {str(e)}")

    if st.session_state.evaluation_results is not None:
        st.header("📊 Evaluation Results")
        res = st.session_state.evaluation_results
        st.metric("Accuracy", f"{res['accuracy']:.4f}")
        st.text_area("Classification Report", res['report'], height=300)

    # ---- FIXED FAIRNESS SECTION ----
    if fairness_btn:
        try:
            pdata = st.session_state.preprocessed_data
            y_test = pdata['y_test']
            X_test = pdata['X_test']
            model = st.session_state.model

            # Predictions
            y_pred = model.predict(X_test)

            # Protected feature values (original df aligned to test index)
            protected_test = df.loc[X_test.index, protected_col]

            dp_diff, eo_diff, metrics_df = fairness.compute_group_metrics(
                y_test, y_pred, protected_test
            )

            st.session_state.fairness_results = {
                'demographic_parity_diff': dp_diff,
                'equal_opportunity_diff': eo_diff,
                'metrics_df': metrics_df
            }

            st.success("Fairness analysis complete!")
        except Exception as e:
            st.error(f"Error during fairness analysis: {str(e)}")

    if st.session_state.fairness_results is not None:
        st.header("⚖️ Fairness Results")

        fr = st.session_state.fairness_results
        col1, col2 = st.columns(2)
        col1.metric("Demographic Parity Diff", f"{fr['demographic_parity_diff']:.4f}")
        col2.metric("Equal Opportunity Diff", f"{fr['equal_opportunity_diff']:.4f}")

        st.subheader("Per-group metrics")
        st.dataframe(fr['metrics_df'], use_container_width=True)
else:
    st.info("Upload a dataset to begin.")