import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from backend.data_processing import prepare_dataset, apply_smote, apply_reweighing
from backend.model_train import train_and_predict, evaluate_model
from backend.fairness import compute_group_metrics

st.set_page_config(
    layout="wide",
    page_title="AI Bias Detector",
    page_icon="📊"
)

st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #000000;
        color: #f3f4f6;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0.5) !important;
        border-bottom: 1px solid #1f2937;
        backdrop-filter: blur(10px);
    }

    .main {
        background-color: #000000;
        padding: 2rem;
    }

    .main h1 {
        color: #f3f4f6;
        font-size: 1.5rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }

    .main h2 {
        color: #f3f4f6;
        font-size: 1.125rem;
        font-weight: 600;
        border-bottom: 1px solid #1f2937;
        padding-bottom: 0.75rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    .main h3 {
        color: #e5e7eb;
        font-weight: 500;
        font-size: 1rem;
    }

    div[data-testid="metric-container"] {
        background-color: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: none !important;
    }

    div[data-testid="metric-container"] label {
        color: #9ca3af !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="stDataFrame"] {
        background-color: #111827 !important;
        border-radius: 12px !important;
        border: 1px solid #1f2937 !important;
        overflow: hidden;
    }

    div[data-testid="stDataFrame"] > div {
        background-color: #111827 !important;
    }

    .stButton > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 1rem 2rem !important;
        border-radius: 8px !important;
        border: none !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background-color: #f3f4f6 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(255, 255, 255, 0.1) !important;
    }

    .stDownloadButton > button {
        background-color: #111827 !important;
        color: #ffffff !important;
        border: 1px solid #1f2937 !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }

    .stDownloadButton > button:hover {
        background-color: #1f2937 !important;
        transform: translateY(-2px) !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 1px solid #1f2937 !important;
    }

    section[data-testid="stSidebar"] h2 {
        color: #f3f4f6 !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #1f2937 !important;
        padding-bottom: 0.75rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }

    section[data-testid="stSidebar"] h3 {
        color: #e5e7eb !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }

    section[data-testid="stSidebar"] label {
        color: #d1d5db !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
    }

    .stAlert {
        border-radius: 12px !important;
        border-left: 4px solid !important;
        padding: 1rem !important;
        font-weight: 500 !important;
        background-color: #111827 !important;
        border: 1px solid #1f2937 !important;
    }

    div[data-testid="stFileUploader"] {
        background-color: #111827 !important;
        border: 2px dashed #374151 !important;
        border-radius: 12px !important;
        padding: 2rem !important;
    }

    div[data-testid="stFileUploader"] label {
        color: #d1d5db !important;
        font-weight: 500 !important;
    }

    div[data-testid="stFileUploader"] > div {
        background-color: transparent !important;
    }

    input[type="text"], input[type="number"], select, textarea {
        background-color: #0a0a0a !important;
        color: #ffffff !important;
        border: 1px solid #1f2937 !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
    }

    input[type="text"]:focus, input[type="number"]:focus, select:focus, textarea:focus {
        border-color: #374151 !important;
        box-shadow: none !important;
    }

    input[type="checkbox"] {
        accent-color: #ffffff !important;
    }

    div[data-testid="stSlider"] {
        padding: 1rem 0 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid #1f2937 !important;
    }

    .stTabs [aria-selected="true"] {
        border-bottom-color: #ffffff !important;
    }

    .plot-card {
        background-color: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin-bottom: 1.5rem !important;
    }

    .legend-item {
        background-color: #0a0a0a;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .legend-term {
        color: #ffffff;
        font-weight: 600;
        font-size: 1.125rem;
        margin-bottom: 0.5rem;
    }

    .legend-definition {
        color: #d1d5db;
        margin-bottom: 0.75rem;
        line-height: 1.6;
        font-size: 0.95rem;
    }

    .legend-formula {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 0.75rem;
        color: #d1d5db;
        font-family: 'Courier New', monospace;
        font-size: 0.875rem;
    }

    .section-divider {
        border-top: 1px solid #1f2937;
        margin: 2rem 0;
    }

    .info-box {
        background-color: #111827;
        border-left: 4px solid #374151;
        border-radius: 12px;
        padding: 1.5rem;
        color: #d1d5db;
        margin: 1rem 0;
    }

    .warning-box {
        background-color: #111827;
        border-left: 4px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        color: #fbbf24;
        margin: 1rem 0;
    }

    .success-box {
        background-color: #111827;
        border-left: 4px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        color: #6ee7b7;
        margin: 1rem 0;
    }

    footer {
        border-top: 1px solid #1f2937;
        padding: 2rem;
        text-align: center;
        color: #6b7280;
        margin-top: 3rem;
    }

    .metric-label {
        color: #9ca3af;
        font-size: 0.875rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Legend data
legend_items = [
    {
        "term": "True Positive Rate (TPR)",
        "definition": "Sensitivity or Recall - the proportion of actual positive cases correctly identified by the model.",
        "formula": "TPR = TP / (TP + FN)"
    },
    {
        "term": "False Positive Rate (FPR)",
        "definition": "The proportion of actual negative cases incorrectly classified as positive.",
        "formula": "FPR = FP / (FP + TN)"
    },
    {
        "term": "Positive Prediction Rate (PPR)",
        "definition": "The proportion of samples predicted as positive by the model for a given group.",
        "formula": "PPR = Positive Predictions / Total Predictions"
    },
    {
        "term": "Demographic Parity",
        "definition": "Fairness metric ensuring all groups receive positive predictions at similar rates.",
        "formula": "Difference in PPR between groups ≈ 0"
    },
    {
        "term": "Equal Opportunity",
        "definition": "Ensures all groups have equal True Positive Rates.",
        "formula": "Difference in TPR between groups ≈ 0"
    },
    {
        "term": "Equalized Odds",
        "definition": "Ensures both TPR and FPR are equal across groups.",
        "formula": "Both TPR and FPR differences ≈ 0"
    },
    {
        "term": "Disparate Impact",
        "definition": "Ratio of positive prediction rates between groups. Values < 0.8 indicate potential discrimination.",
        "formula": "Ratio = PPR(unprivileged) / PPR(privileged)"
    },
    {
        "term": "Predictive Parity",
        "definition": "Ensures precision (positive predictive value) is equal across groups.",
        "formula": "Difference in precision between groups ≈ 0"
    },
    {
        "term": "SMOTE",
        "definition": "Synthetic Minority Over-sampling Technique - creates synthetic samples for minority class.",
        "formula": "Generates samples by interpolating between minority class neighbors"
    },
    {
        "term": "Reweighing",
        "definition": "Assigns weights to training samples to balance representation and reduce bias.",
        "formula": "Weight = (Expected / Observed) for each group-label combination"
    }
]

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = None
if "results" not in st.session_state:
    st.session_state.results = None
if "download_df" not in st.session_state:
    st.session_state.download_df = None
if "reweighted_df" not in st.session_state:
    st.session_state.reweighted_df = None
if "show_legend" not in st.session_state:
    st.session_state.show_legend = False

# Header with legend toggle
col_header_left, col_header_right = st.columns([1, 0.2])
with col_header_left:
    st.title("📊 AI Bias Detector")
    st.caption("SMOTE & Reweighing Analysis")

with col_header_right:
    if st.button("📖 ML Terms", use_container_width=True):
        st.session_state.show_legend = not st.session_state.show_legend

# Legend modal
if st.session_state.show_legend:
    st.markdown("---")
    st.subheader("Machine Learning Terms & Metrics")
    st.markdown("Learn about key fairness metrics and bias mitigation techniques.")

    for item in legend_items:
        with st.container():
            st.markdown(f"<div class='legend-item'>", unsafe_allow_html=True)
            st.markdown(f"<div class='legend-term'>{item['term']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='legend-definition'>{item['definition']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='legend-formula'>{item['formula']}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

# Sidebar controls
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("📂 Upload CSV", type=["csv"])
use_sample = st.sidebar.checkbox("📊 Use sample dataset", value=False)

if uploaded_file:
    st.session_state.df = pd.read_csv(uploaded_file)
elif use_sample:
    try:
        st.session_state.df = pd.read_csv("backend/sample_adult.csv")
    except Exception:
        st.sidebar.error("Sample dataset not found")
        st.session_state.df = None

df = st.session_state.df

if df is None:
    st.markdown("""
    <div class='info-box' style='text-align: center; padding: 3rem;'>
        <h2 style='color: #e5e7eb; margin-bottom: 1rem;'>👋 Welcome</h2>
        <p style='color: #d1d5db; font-size: 1.05rem;'>Upload a CSV dataset or enable the sample dataset to begin your fairness analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

st.sidebar.markdown(f"**Dataset:** {df.shape[0]:,} rows × {df.shape[1]} cols")
st.sidebar.markdown("---")

cols = df.columns.tolist()
target_col = st.sidebar.selectbox("🎯 Target Column", options=cols, index=len(cols)-1)
sensitive_candidates = [c for c in cols if df[c].dtype == 'object']
if not sensitive_candidates:
    sensitive_candidates = cols
sensitive_col = st.sidebar.selectbox("👥 Sensitive Attribute", options=sensitive_candidates)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Settings")

model_map = {
    "Logistic Regression": "logistic",
    "Random Forest": "random_forest",
    "SVM": "svm",
    "Decision Tree": "decision_tree"
}
model_choice_display = st.sidebar.selectbox("Select Model", list(model_map.keys()))
model_choice = model_map[model_choice_display]

mitigation_choice = st.sidebar.selectbox("🛡️ Bias Mitigation", ["None", "SMOTE", "Reweighing"])
smote_k = st.sidebar.number_input("SMOTE k_neighbors", min_value=1, max_value=20, value=5)
test_size = st.sidebar.slider("📊 Test Size", 0.10, 0.40, 0.20, 0.05)

st.sidebar.markdown("---")

if st.sidebar.button("🚀 Train & Analyze", use_container_width=True):
    with st.spinner("Processing..."):
        try:
            prepared = prepare_dataset(df, target_col=target_col, sensitive_col=sensitive_col, test_size=test_size)
        except Exception as e:
            st.error(f"Preparation failed: {e}")
            st.stop()

        X_train = prepared["X_train"]
        X_test = prepared["X_test"]
        y_train = prepared["y_train"]
        y_test = prepared["y_test"]
        s_train = prepared["s_train"]
        s_test = prepared["s_test"]
        df_clean = prepared["df_clean"]

        st.session_state.df_clean = df_clean

        sample_weights = None
        X_train_used = X_train.copy()
        y_train_used = y_train.copy()

        if mitigation_choice == "SMOTE":
            with st.spinner("Applying SMOTE..."):
                X_train_used, y_train_used = apply_smote(X_train, y_train, k_neighbors=int(smote_k))
                st.session_state.download_df = pd.concat([X_train_used.reset_index(drop=True),
                                                          pd.Series(y_train_used, name=target_col).reset_index(drop=True)], axis=1)
                st.session_state.reweighted_df = None

        elif mitigation_choice == "Reweighing":
            with st.spinner("Computing reweighting..."):
                sample_weights = apply_reweighing(y_train, s_train)
                reweighted_download = pd.concat([X_train.reset_index(drop=True),
                                                 pd.Series(y_train, name=target_col).reset_index(drop=True),
                                                 sample_weights.reset_index(drop=True)], axis=1)
                st.session_state.reweighted_df = reweighted_download
                st.session_state.download_df = None
        else:
            st.session_state.download_df = None
            st.session_state.reweighted_df = None

        with st.spinner("Training model..."):
            model, y_pred, y_proba = train_and_predict(
                model_choice, X_train_used, y_train_used, X_test, sample_weight=sample_weights
            )

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
        st.success("Analysis complete!")

if st.session_state.results is None:
    st.markdown("""
    <div class='info-box' style='text-align: center; padding: 2rem; margin-top: 2rem;'>
        <h3 style='color: #e5e7eb;'>⬅️ Configure and click "Train & Analyze"</h3>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

res = st.session_state.results

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# Results section
st.header("📊 Results")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model", model_choice_display)
with col2:
    st.metric("Mitigation", res['mitigation'])
with col3:
    st.metric("Accuracy", f"{res['accuracy']:.4f}")

st.markdown("---")

from sklearn.metrics import classification_report

report_dict = classification_report(res["y_test"], res["y_pred"], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

st.subheader("📈 Classification Report")
st.dataframe(
    report_df.style.format("{:.3f}").background_gradient(cmap='Blues_r', subset=['precision', 'recall', 'f1-score']),
    use_container_width=True
)

st.markdown("---")
st.header("⚖️ Fairness Metrics")

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

st.dataframe(
    metric_df.style.format({"Value": "{:.4f}"}).background_gradient(cmap='RdYlGn_r', subset=['Value']),
    use_container_width=True
)

st.markdown("---")
st.header("👥 Group-level Metrics")

gm = metrics["group_metrics"].copy()
st.dataframe(
    gm.sort_values(by="count", ascending=False).reset_index(drop=True)
    .style.background_gradient(cmap='Spectral_r', subset=['positive_prediction_rate', 'true_positive_rate', 'false_positive_rate']),
    use_container_width=True
)

st.markdown("---")
st.header("📉 Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
    y_min = gm["positive_prediction_rate"].min() * 0.8
    y_max = gm["positive_prediction_rate"].max() * 1.2

    fig_ppr = px.bar(
        gm,
        x="group",
        y="positive_prediction_rate",
        text="positive_prediction_rate",
        color="positive_prediction_rate",
        color_continuous_scale="Blues",
        title="Positive Prediction Rate (PPR)",
    )
    fig_ppr.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_ppr.update_layout(
        yaxis=dict(range=[y_min, y_max]),
        height=400,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color='#d1d5db', family='Arial')
    )
    st.plotly_chart(fig_ppr, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
    fig_pie = px.pie(
        gm,
        names="group",
        values="count",
        title="Sensitive Group Distribution",
        hole=0.35
    )
    fig_pie.update_traces(textinfo="percent+label", textfont_size=12)
    fig_pie.update_layout(
        height=400,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color='#d1d5db', family='Arial')
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
    y_min = gm["true_positive_rate"].min() * 0.8
    y_max = gm["true_positive_rate"].max() * 1.2

    fig_tpr = px.bar(
        gm,
        x="group",
        y="true_positive_rate",
        text="true_positive_rate",
        color="true_positive_rate",
        color_continuous_scale="Greens",
        title="True Positive Rate (TPR)",
    )
    fig_tpr.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_tpr.update_layout(
        yaxis=dict(range=[y_min, y_max]),
        height=400,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color='#d1d5db', family='Arial')
    )
    st.plotly_chart(fig_tpr, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
    y_min = gm["false_positive_rate"].min() * 0.8
    y_max = gm["false_positive_rate"].max() * 1.2

    fig_fpr = px.bar(
        gm,
        x="group",
        y="false_positive_rate",
        text="false_positive_rate",
        color="false_positive_rate",
        color_continuous_scale="Reds",
        title="False Positive Rate (FPR)",
    )
    fig_fpr.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_fpr.update_layout(
        yaxis=dict(range=[y_min, y_max]),
        height=400,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color='#d1d5db', family='Arial')
    )
    st.plotly_chart(fig_fpr, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if res["accuracy"] < 0.7:
    st.markdown("""
    <div class='warning-box'>
        ⚠️ Low accuracy (<0.7). Fairness metrics may be unreliable.
    </div>
    """, unsafe_allow_html=True)

small_groups = gm[gm["count"] < 50]
if not small_groups.empty:
    st.markdown(f"""
    <div class='warning-box'>
        ⚠️ Small groups detected (count < 50): {', '.join(list(small_groups['group']))}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.header("💾 Downloads")

col_d1, col_d2, col_d3 = st.columns(3)

with col_d1:
    if st.session_state.download_df is not None:
        csv_res = st.session_state.download_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 SMOTE Dataset", data=csv_res,
                           file_name="resampled_dataset.csv", mime="text/csv", use_container_width=True)

with col_d2:
    if st.session_state.reweighted_df is not None:
        csv_rw = st.session_state.reweighted_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Reweighted Dataset", data=csv_rw,
                           file_name="reweighted_dataset.csv", mime="text/csv", use_container_width=True)

with col_d3:
    out = res["X_test"].copy()
    out[target_col] = res["y_test"].values
    out["y_pred"] = res["y_pred"].values
    try:
        out["y_proba"] = res["y_proba"].values
    except:
        pass
    csv_out = out.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Test Predictions", data=csv_out,
                       file_name="predictions.csv", mime="text/csv", use_container_width=True)

st.markdown("""
<footer>
Built with ❤️ by team Git Commit or Die
</footer>
""", unsafe_allow_html=True)
