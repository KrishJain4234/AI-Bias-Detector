# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px  # NEW

from backend.data_processing import prepare_dataset, apply_smote, apply_reweighing
from backend.model_train import train_and_predict, evaluate_model
from backend.fairness import compute_group_metrics

st.set_page_config(layout="wide", page_title="AI Bias Detector (SMOTE + Reweighing)")

st.title("AI Bias Detector — SMOTE & Reweighing")

# Session state
if "df" not in st.session_state:
    st.session_state.df = None
if "results" not in st.session_state:
    st.session_state.results = None
if "download_df" not in st.session_state:
    st.session_state.download_df = None
if "reweighted_df" not in st.session_state:
    st.session_state.reweighted_df = None

# Sidebar controls
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample dataset (backend/sample_adult.csv)", value=False)

if uploaded_file:
    st.session_state.df = pd.read_csv(uploaded_file)
elif use_sample:
    try:
        st.session_state.df = pd.read_csv("backend/sample_adult.csv")
    except Exception:
        st.sidebar.error("Sample dataset not found in backend/sample_adult.csv")
        st.session_state.df = None

df = st.session_state.df
if df is None:
    st.info("Upload a CSV dataset or enable the sample dataset to begin.")
    st.stop()

st.sidebar.markdown(f"Dataset: {df.shape[0]} rows × {df.shape[1]} cols")

cols = df.columns.tolist()
# Target and sensitive selectors
target_col = st.sidebar.selectbox("Target column (binary)", options=cols, index=len(cols)-1)
# sensitive attribute candidates: prefer object type columns
sensitive_candidates = [c for c in cols if df[c].dtype == 'object']
if not sensitive_candidates:
    sensitive_candidates = cols
sensitive_col = st.sidebar.selectbox("Sensitive attribute (categorical)", options=sensitive_candidates)

# Model selection
model_map = {
    "Logistic Regression": "logistic",
    "Random Forest": "random_forest",
    "SVM": "svm",
    "Decision Tree": "decision_tree"
}
model_choice_display = st.sidebar.selectbox("Select model", list(model_map.keys()))
model_choice = model_map[model_choice_display]

# Mitigation selection
mitigation_choice = st.sidebar.selectbox("Bias mitigation method", ["None", "SMOTE", "Reweighing"])

# SMOTE params
smote_k = st.sidebar.number_input("SMOTE k_neighbors", min_value=1, max_value=20, value=5)

# Train/test split
test_size = st.sidebar.slider("Test set size", 0.10, 0.40, 0.20, 0.05)

# Train button
if st.sidebar.button("Train & Analyze"):
    try:
        prepared = prepare_dataset(df, target_col=target_col, sensitive_col=sensitive_col, test_size=test_size)
    except Exception as e:
        st.error(f"Dataset preparation failed: {e}")
        st.stop()

    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]
    s_train = prepared["s_train"]
    s_test = prepared["s_test"]
    df_clean = prepared["df_clean"]

    st.session_state.df_clean = df_clean

    # default values
    sample_weights = None
    X_train_used = X_train.copy()
    y_train_used = y_train.copy()

    if mitigation_choice == "SMOTE":
        st.info("Applying SMOTE to training set...")
        X_train_used, y_train_used = apply_smote(X_train, y_train, k_neighbors=int(smote_k))
        st.session_state.download_df = pd.concat([X_train_used.reset_index(drop=True),
                                                  pd.Series(y_train_used, name=target_col).reset_index(drop=True)], axis=1)
        st.session_state.reweighted_df = None

    elif mitigation_choice == "Reweighing":
        st.info("Computing reweighing sample weights...")
        sample_weights = apply_reweighing(y_train, s_train)
        reweighted_download = pd.concat([X_train.reset_index(drop=True),
                                         pd.Series(y_train, name=target_col).reset_index(drop=True),
                                         sample_weights.reset_index(drop=True)], axis=1)
        st.session_state.reweighted_df = reweighted_download
        st.session_state.download_df = None

    else:
        st.session_state.download_df = None
        st.session_state.reweighted_df = None

    # Train model
    model, y_pred, y_proba = train_and_predict(
        model_choice, X_train_used, y_train_used, X_test, sample_weight=sample_weights
    )

    # Evaluate
    acc, report = evaluate_model(y_test, y_pred)
    metrics = compute_group_metrics(y_test, y_pred, s_test)

    st.session_state.results = {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": acc,
        "report": report,
        "fairness": metrics,
        "mitigation": mitigation_choice,
        "sample_weights": sample_weights
    }

# If no results yet
if st.session_state.results is None:
    st.info("Train a model to see metrics.")
    st.stop()

# Display results
res = st.session_state.results
st.header("Model performance")
st.write(f"Model: **{model_choice_display}**")
st.write(f"Mitigation: **{res['mitigation']}**")
st.write(f"Accuracy: **{res['accuracy']:.4f}**")

# -------------------------------
# CLEAN CLASSIFICATION REPORT TABLE
# -------------------------------
from sklearn.metrics import classification_report

report_dict = classification_report(res["y_test"], res["y_pred"], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
st.subheader("Classification Report")
st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

# -------------------------------
# FAIRNESS METRICS TABLE
# -------------------------------
st.header("Fairness metrics")
metrics = res["fairness"]
metric_df = pd.DataFrame({
    "Metric": [
        "Demographic Parity Difference",
        "Equal Opportunity Difference",
        "Equalized Odds Difference",
        "Disparate Impact Ratio",
        "Predictive Parity Difference"
    ],
    "Value": [
        metrics["demographic_parity_difference"],
        metrics["equal_opportunity_difference"],
        metrics["equalized_odds_difference"],
        metrics["disparate_impact_ratio"],
        metrics["predictive_parity_difference"]
    ]
})
st.dataframe(metric_df.style.format({"Value": "{:.4f}"}))

# -------------------------------
# GROUP METRICS TABLE
# -------------------------------
st.header("Group-level metrics")
gm = metrics["group_metrics"].copy()
st.dataframe(gm.sort_values(by="count", ascending=False).reset_index(drop=True))

# -------------------------------
# FIXED & IMPROVED GRAPH SECTION
# -------------------------------

st.header("Visualizations")

# 1️⃣ Positive Prediction Rate — Proper Scaling
y_min = gm["positive_prediction_rate"].min() * 0.8
y_max = gm["positive_prediction_rate"].max() * 1.2

fig_ppr = px.bar(
    gm,
    x="group",
    y="positive_prediction_rate",
    text="positive_prediction_rate",
    color="positive_prediction_rate",
    color_continuous_scale="Blues",
    title="Positive Prediction Rate (PPR) per group",
)
fig_ppr.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside",
    marker=dict(line=dict(width=1, color="black"))
)
fig_ppr.update_layout(yaxis=dict(range=[y_min, y_max]), height=450)
st.plotly_chart(fig_ppr, use_container_width=True)

# 2️⃣ True Positive Rate — Proper Scaling
y_min = gm["true_positive_rate"].min() * 0.8
y_max = gm["true_positive_rate"].max() * 1.2

fig_tpr = px.bar(
    gm,
    x="group",
    y="true_positive_rate",
    text="true_positive_rate",
    color="true_positive_rate",
    color_continuous_scale="Greens",
    title="True Positive Rate (TPR) per group",
)
fig_tpr.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside",
    marker=dict(line=dict(width=1, color="black"))
)
fig_tpr.update_layout(yaxis=dict(range=[y_min, y_max]), height=450)
st.plotly_chart(fig_tpr, use_container_width=True)

# 3️⃣ False Positive Rate — Proper Scaling
y_min = gm["false_positive_rate"].min() * 0.8
y_max = gm["false_positive_rate"].max() * 1.2

fig_fpr = px.bar(
    gm,
    x="group",
    y="false_positive_rate",
    text="false_positive_rate",
    color="false_positive_rate",
    color_continuous_scale="Reds",
    title="False Positive Rate (FPR) per group",
)
fig_fpr.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside",
    marker=dict(line=dict(width=1, color="black"))
)
fig_fpr.update_layout(yaxis=dict(range=[y_min, y_max]), height=450)
st.plotly_chart(fig_fpr, use_container_width=True)

# 4️⃣ Pie chart
fig_pie = px.pie(
    gm,
    names="group",
    values="count",
    title="Sensitive Group Distribution",
    hole=0.35
)
fig_pie.update_traces(textinfo="percent+label")
st.plotly_chart(fig_pie, use_container_width=True)

# Warnings
if res["accuracy"] < 0.7:
    st.warning("Low accuracy (<0.7). Fairness metrics may be unreliable.")
small_groups = gm[gm["count"] < 50]
if not small_groups.empty:
    st.warning(f"Small sensitive groups detected (count < 50): {list(small_groups['group'])}")

# Downloads
st.header("Downloads")
if st.session_state.download_df is not None:
    csv_res = st.session_state.download_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download SMOTE-resampled training CSV", data=csv_res,
                       file_name="resampled_dataset.csv", mime="text/csv")

if st.session_state.reweighted_df is not None:
    csv_rw = st.session_state.reweighted_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download reweighted training CSV (with sample_weight)", data=csv_rw,
                       file_name="reweighted_dataset.csv", mime="text/csv")

out = res["X_test"].copy()
out[target_col] = res["y_test"].values
out["y_pred"] = res["y_pred"].values
try:
    out["y_proba"] = res["y_proba"].values
except:
    pass
csv_out = out.to_csv(index=False).encode('utf-8')
st.download_button("Download test predictions CSV", data=csv_out,
                   file_name="predictions.csv", mime="text/csv")

st.success("Analysis complete.")
