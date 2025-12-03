import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from backend.data_processing import prepare_dataset, apply_smote, apply_reweighing
from backend.model_train import train_and_predict, evaluate_model
from backend.fairness import compute_group_metrics

st.set_page_config(
    layout="wide",
    page_title="AI Bias Detector",
    page_icon="⚡",
    initial_sidebar_state="collapsed"
)

# Dark theme colors
theme = {
    "bg_main": "#0a0a0b",
    "bg_card": "#141417",
    "bg_elevated": "#1c1c21",
    "bg_input": "#141417",
    "text_primary": "#f4f4f5",
    "text_secondary": "#a1a1aa",
    "text_muted": "#71717a",
    "border": "#27272a",
    "accent": "#6366f1",
    "accent_hover": "#4f46e5",
    "success": "#22c55e",
    "warning": "#eab308",
    "error": "#ef4444",
}

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}

    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
        background: {theme['bg_main']} !important;
        color: {theme['text_primary']} !important;
    }}

    [data-testid="stHeader"] {{
        background: transparent !important;
    }}

    .main .block-container {{
        background: {theme['bg_main']} !important;
        padding: 1.5rem 2.5rem 3rem !important;
        max-width: 1600px !important;
    }}

    h1, h2, h3, h4, h5, h6, p, span, div, label {{
        color: {theme['text_primary']} !important;
    }}

    .main h1 {{
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.025em !important;
    }}

    .main h2 {{
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: {theme['text_secondary']} !important;
        margin: 2rem 0 1rem 0 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }}

    [data-testid="stCaptionContainer"] p {{
        color: {theme['text_muted']} !important;
        font-size: 0.875rem !important;
    }}

    /* Metric Cards */
    div[data-testid="metric-container"] {{
        background: {theme['bg_card']} !important;
        border: 1px solid {theme['border']} !important;
        border-radius: 12px !important;
        padding: 1.25rem !important;
    }}

    div[data-testid="metric-container"] label {{
        color: {theme['text_muted']} !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }}

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: {theme['text_primary']} !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }}

    /* DataFrames */
    div[data-testid="stDataFrame"] {{
        background: {theme['bg_card']} !important;
        border-radius: 12px !important;
        border: 1px solid {theme['border']} !important;
        overflow: hidden !important;
    }}

    div[data-testid="stDataFrame"] > div {{
        background: {theme['bg_card']} !important;
    }}

    /* Config Panel */
    .config-panel {{
        background: {theme['bg_card']};
        border: 1px solid {theme['border']};
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }}

    /* Buttons */
    .stButton > button {{
        background: {theme['accent']} !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        padding: 0.625rem 1.25rem !important;
        border-radius: 8px !important;
        border: none !important;
        transition: all 0.15s ease !important;
    }}

    .stButton > button:hover {{
        background: {theme['accent_hover']} !important;
        transform: translateY(-1px) !important;
    }}

    .stDownloadButton > button {{
        background: transparent !important;
        color: {theme['text_primary']} !important;
        border: 1px solid {theme['border']} !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        border-radius: 8px !important;
    }}

    .stDownloadButton > button:hover {{
        background: {theme['bg_elevated']} !important;
        border-color: {theme['accent']} !important;
    }}

    /* File Uploader */
    div[data-testid="stFileUploader"] {{
        background: {theme['bg_elevated']} !important;
        border: 1px dashed {theme['border']} !important;
        border-radius: 10px !important;
        padding: 1rem !important;
    }}

    div[data-testid="stFileUploader"]:hover {{
        border-color: {theme['accent']} !important;
    }}

    /* Form Elements */
    .stSelectbox label, .stNumberInput label, .stSlider label, .stCheckbox label {{
        color: {theme['text_muted']} !important;
        font-weight: 500 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.03em !important;
    }}

    .stSelectbox > div > div {{
        background: {theme['bg_elevated']} !important;
        border: 1px solid {theme['border']} !important;
        border-radius: 8px !important;
        color: {theme['text_primary']} !important;
    }}

    .stSelectbox > div > div:hover, .stSelectbox > div > div:focus-within {{
        border-color: {theme['accent']} !important;
    }}

    .stNumberInput > div > div > input {{
        background: {theme['bg_elevated']} !important;
        border: 1px solid {theme['border']} !important;
        border-radius: 8px !important;
        color: {theme['text_primary']} !important;
    }}

    .stCheckbox > label > span {{
        color: {theme['text_secondary']} !important;
    }}

    /* Slider */
    .stSlider > div > div > div {{
        background: {theme['border']} !important;
    }}

    .stSlider > div > div > div > div {{
        background: {theme['accent']} !important;
    }}

    /* Cards */
    .card {{
        background: {theme['bg_card']};
        border: 1px solid {theme['border']};
        border-radius: 12px;
        padding: 1.25rem;
    }}

    .card-header {{
        color: {theme['text_muted']};
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.5rem;
    }}

    .card-value {{
        color: {theme['text_primary']};
        font-size: 1.5rem;
        font-weight: 600;
    }}

    /* Legend */
    .legend-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
    }}

    .legend-item {{
        background: {theme['bg_card']};
        border: 1px solid {theme['border']};
        border-radius: 10px;
        padding: 1rem;
    }}

    .legend-term {{
        color: {theme['text_primary']};
        font-weight: 600;
        font-size: 0.875rem;
        margin-bottom: 0.375rem;
    }}

    .legend-def {{
        color: {theme['text_secondary']};
        font-size: 0.8rem;
        line-height: 1.4;
        margin-bottom: 0.5rem;
    }}

    .legend-formula {{
        background: {theme['bg_elevated']};
        border-radius: 6px;
        padding: 0.375rem 0.625rem;
        color: {theme['accent']};
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 0.7rem;
        display: inline-block;
    }}

    /* Status */
    .status {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: {theme['bg_card']};
        border: 1px solid {theme['border']};
        border-radius: 100px;
        padding: 0.375rem 0.875rem;
        font-size: 0.8rem;
        color: {theme['text_secondary']};
    }}

    .status-dot {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: {theme['success']};
    }}

    /* Alerts */
    .alert {{
        background: {theme['bg_card']};
        border: 1px solid {theme['border']};
        border-radius: 10px;
        padding: 1rem 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }}

    .alert-warning {{
        border-left: 3px solid {theme['warning']};
    }}

    .alert-warning span {{
        color: {theme['warning']} !important;
    }}

    .alert-info {{
        border-left: 3px solid {theme['accent']};
    }}

    .alert-success {{
        border-left: 3px solid {theme['success']};
    }}

    /* Section Divider */
    hr {{
        border: none !important;
        border-top: 1px solid {theme['border']} !important;
        margin: 1.5rem 0 !important;
    }}

    /* Plot Container */
    .plot-container {{
        background: {theme['bg_card']};
        border: 1px solid {theme['border']};
        border-radius: 12px;
        padding: 0.75rem;
    }}

    /* Footer */
    .footer {{
        text-align: center;
        color: {theme['text_muted']};
        font-size: 0.8rem;
        padding: 2rem 0;
        margin-top: 2rem;
        border-top: 1px solid {theme['border']};
    }}

    /* Hide sidebar */
    section[data-testid="stSidebar"] {{
        display: none !important;
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}

    ::-webkit-scrollbar-track {{
        background: {theme['bg_main']};
    }}

    ::-webkit-scrollbar-thumb {{
        background: {theme['border']};
        border-radius: 3px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: {theme['text_muted']};
    }}

    /* Expander */
    .streamlit-expanderHeader {{
        background: {theme['bg_card']} !important;
        border: 1px solid {theme['border']} !important;
        border-radius: 10px !important;
    }}

    .streamlit-expanderContent {{
        background: {theme['bg_card']} !important;
        border: 1px solid {theme['border']} !important;
        border-top: none !important;
    }}
</style>
""", unsafe_allow_html=True)

# Legend data
legend_items = [
    {"term": "True Positive Rate", "def": "Proportion of actual positives correctly identified.", "formula": "TP / (TP + FN)"},
    {"term": "False Positive Rate", "def": "Proportion of negatives incorrectly classified as positive.", "formula": "FP / (FP + TN)"},
    {"term": "Positive Prediction Rate", "def": "Proportion predicted as positive for a group.", "formula": "Pos Pred / Total"},
    {"term": "Demographic Parity", "def": "All groups receive positive predictions at similar rates.", "formula": "PPR diff ≈ 0"},
    {"term": "Equal Opportunity", "def": "All groups have equal True Positive Rates.", "formula": "TPR diff ≈ 0"},
    {"term": "Equalized Odds", "def": "Both TPR and FPR are equal across groups.", "formula": "TPR & FPR ≈ 0"},
    {"term": "Disparate Impact", "def": "Ratio of PPR between groups. <0.8 indicates bias.", "formula": "PPR(u) / PPR(p)"},
    {"term": "SMOTE", "def": "Creates synthetic samples for minority class.", "formula": "Interpolation"},
    {"term": "Reweighing", "def": "Assigns weights to balance representation.", "formula": "Exp / Obs"},
]

# Session state
for key in ["df", "results", "download_df", "reweighted_df"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "show_legend" not in st.session_state:
    st.session_state.show_legend = False

# Header
st.markdown("## ⚡ AI Bias Detector")
st.caption("Detect and mitigate bias in machine learning models")

# Config Panel
st.markdown("<div class='config-panel'>", unsafe_allow_html=True)

row1_col1, row1_col2, row1_col3, row1_col4 = st.columns([2.5, 1.5, 1.5, 1.5])

with row1_col1:
    uploaded_file = st.file_uploader("Dataset", type=["csv"], label_visibility="collapsed")
    use_sample = st.checkbox("Use sample data", value=False)

with row1_col2:
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
    elif use_sample:
        try:
            st.session_state.df = pd.read_csv("backend/sample_adult.csv")
        except:
            st.session_state.df = None

    df = st.session_state.df
    if df is not None:
        cols = df.columns.tolist()
        target_col = st.selectbox("Target Column", options=cols, index=len(cols)-1)
    else:
        target_col = None

with row1_col3:
    if df is not None:
        sensitive_candidates = [c for c in cols if df[c].dtype == 'object'] or cols
        sensitive_col = st.selectbox("Sensitive Attr", options=sensitive_candidates)
    else:
        sensitive_col = None

with row1_col4:
    model_map = {"Logistic Regression": "logistic", "Random Forest": "random_forest", "SVM": "svm", "Decision Tree": "decision_tree"}
    model_choice_display = st.selectbox("Model", list(model_map.keys()))
    model_choice = model_map[model_choice_display]

row2_col1, row2_col2, row2_col3, row2_col4 = st.columns([2.5, 1.5, 1.5, 1.5])

with row2_col1:
    mitigation_choice = st.selectbox("Mitigation Strategy", ["None", "SMOTE", "Reweighing"])

with row2_col2:
    if mitigation_choice == "SMOTE":
        smote_k = st.number_input("SMOTE k-neighbors", min_value=1, max_value=20, value=5)
    else:
        smote_k = 5

with row2_col3:
    test_size = st.slider("Test Split %", 0.10, 0.40, 0.20, 0.05)

with row2_col4:
    st.write("")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_analysis = st.button("Analyze", use_container_width=True)
    with col_btn2:
        if st.button("Glossary", use_container_width=True):
            st.session_state.show_legend = not st.session_state.show_legend

st.markdown("</div>", unsafe_allow_html=True)

# Dataset status
if df is not None:
    st.markdown(f"""
    <div class='status'>
        <div class='status-dot'></div>
        {df.shape[0]:,} rows · {df.shape[1]} columns
    </div>
    """, unsafe_allow_html=True)

# Legend
if st.session_state.show_legend:
    st.markdown("---")
    st.markdown("## Glossary")
    
    cols = st.columns(3)
    for i, item in enumerate(legend_items):
        with cols[i % 3]:
            st.markdown(f"""
            <div class='legend-item'>
                <div class='legend-term'>{item['term']}</div>
                <div class='legend-def'>{item['def']}</div>
                <div class='legend-formula'>{item['formula']}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("")

# No data
if df is None:
    st.markdown("""
    <div class='alert alert-info' style='margin-top: 2rem; justify-content: center;'>
        <span>Upload a CSV dataset or enable sample data to begin analysis</span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Run Analysis
if run_analysis:
    with st.spinner("Analyzing..."):
        try:
            prepared = prepare_dataset(df, target_col=target_col, sensitive_col=sensitive_col, test_size=test_size)
        except Exception as e:
            st.error(f"Preparation failed: {e}")
            st.stop()

        X_train, X_test = prepared["X_train"], prepared["X_test"]
        y_train, y_test = prepared["y_train"], prepared["y_test"]
        s_train, s_test = prepared["s_train"], prepared["s_test"]

        sample_weights = None
        X_train_used, y_train_used = X_train.copy(), y_train.copy()

        if mitigation_choice == "SMOTE":
            X_train_used, y_train_used = apply_smote(X_train, y_train, k_neighbors=int(smote_k))
            st.session_state.download_df = pd.concat([X_train_used.reset_index(drop=True), pd.Series(y_train_used, name=target_col).reset_index(drop=True)], axis=1)
            st.session_state.reweighted_df = None
        elif mitigation_choice == "Reweighing":
            sample_weights = apply_reweighing(y_train, s_train)
            st.session_state.reweighted_df = pd.concat([X_train.reset_index(drop=True), pd.Series(y_train, name=target_col).reset_index(drop=True), sample_weights.reset_index(drop=True)], axis=1)
            st.session_state.download_df = None
        else:
            st.session_state.download_df = None
            st.session_state.reweighted_df = None

        model, y_pred, y_proba = train_and_predict(model_choice, X_train_used, y_train_used, X_test, sample_weight=sample_weights)
        acc, report = evaluate_model(y_test, y_pred)
        metrics = compute_group_metrics(y_test, y_pred, s_test)

        st.session_state.results = {
            "model": model, "X_test": X_test, "y_test": y_test, "y_pred": y_pred, "y_proba": y_proba,
            "accuracy": acc, "report": report, "fairness": metrics, "mitigation": mitigation_choice
        }

# No results
if st.session_state.results is None:
    st.markdown("""
    <div class='alert alert-info' style='margin-top: 2rem; justify-content: center;'>
        <span>Configure settings above and click "Analyze" to start</span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

res = st.session_state.results

# Results Header
st.markdown("---")
st.markdown("## Results")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model", model_choice_display)
with col2:
    st.metric("Mitigation", res['mitigation'])
with col3:
    st.metric("Accuracy", f"{res['accuracy']:.1%}")

# Classification Report
st.markdown("## Classification Report")
from sklearn.metrics import classification_report
report_dict = classification_report(res["y_test"], res["y_pred"], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

# Fairness Metrics
st.markdown("## Fairness Metrics")
metrics = res["fairness"]

f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
with f_col1:
    st.metric("Demographic Parity", f"{metrics['demographic_parity_difference']:.4f}")
with f_col2:
    st.metric("Equal Opportunity", f"{metrics['equal_opportunity_difference']:.4f}")
with f_col3:
    st.metric("Equalized Odds", f"{metrics['equalized_odds_difference']:.4f}")
with f_col4:
    st.metric("Disparate Impact", f"{metrics['disparate_impact_ratio']:.4f}")
with f_col5:
    st.metric("Predictive Parity", f"{metrics['predictive_parity_difference']:.4f}")

# Group Metrics
st.markdown("## Group Analysis")
gm = metrics["group_metrics"].copy()
st.dataframe(gm.sort_values(by="count", ascending=False).reset_index(drop=True), use_container_width=True)

# Visualizations
st.markdown("## Visualizations")

chart_colors = ["#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd"]

col1, col2 = st.columns(2)

with col1:
    fig_ppr = px.bar(
        gm, x="group", y="positive_prediction_rate", 
        text="positive_prediction_rate",
        color_discrete_sequence=chart_colors,
        title="Positive Prediction Rate by Group"
    )
    fig_ppr.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_ppr.update_layout(
        height=350,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=theme['text_secondary']),
        title_font=dict(size=14, color=theme['text_primary']),
        xaxis=dict(gridcolor=theme['border'], title=""),
        yaxis=dict(gridcolor=theme['border'], title=""),
        margin=dict(t=50, b=40, l=40, r=20)
    )
    st.plotly_chart(fig_ppr, use_container_width=True)

with col2:
    fig_pie = px.pie(
        gm, names="group", values="count", 
        title="Group Distribution",
        hole=0.5,
        color_discrete_sequence=chart_colors
    )
    fig_pie.update_layout(
        height=350,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=theme['text_secondary']),
        title_font=dict(size=14, color=theme['text_primary']),
        margin=dict(t=50, b=40, l=40, r=20)
    )
    st.plotly_chart(fig_pie, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    fig_tpr = px.bar(
        gm, x="group", y="true_positive_rate",
        text="true_positive_rate",
        color_discrete_sequence=["#22c55e"],
        title="True Positive Rate by Group"
    )
    fig_tpr.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_tpr.update_layout(
        height=350,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=theme['text_secondary']),
        title_font=dict(size=14, color=theme['text_primary']),
        xaxis=dict(gridcolor=theme['border'], title=""),
        yaxis=dict(gridcolor=theme['border'], title=""),
        margin=dict(t=50, b=40, l=40, r=20)
    )
    st.plotly_chart(fig_tpr, use_container_width=True)

with col4:
    fig_fpr = px.bar(
        gm, x="group", y="false_positive_rate",
        text="false_positive_rate",
        color_discrete_sequence=["#ef4444"],
        title="False Positive Rate by Group"
    )
    fig_fpr.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_fpr.update_layout(
        height=350,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=theme['text_secondary']),
        title_font=dict(size=14, color=theme['text_primary']),
        xaxis=dict(gridcolor=theme['border'], title=""),
        yaxis=dict(gridcolor=theme['border'], title=""),
        margin=dict(t=50, b=40, l=40, r=20)
    )
    st.plotly_chart(fig_fpr, use_container_width=True)

# Warnings
if res["accuracy"] < 0.7:
    st.markdown(f"""
    <div class='alert alert-warning'>
        <span>⚠️ Low accuracy ({res['accuracy']:.1%}). Fairness metrics may be unreliable.</span>
    </div>
    """, unsafe_allow_html=True)

small_groups = gm[gm["count"] < 50]
if not small_groups.empty:
    st.markdown(f"""
    <div class='alert alert-warning'>
        <span>⚠️ Small sample groups: {', '.join(list(small_groups['group']))}</span>
    </div>
    """, unsafe_allow_html=True)

# Downloads
st.markdown("## Export")
col_d1, col_d2, col_d3 = st.columns(3)

with col_d1:
    if st.session_state.download_df is not None:
        st.download_button("Download SMOTE Data", st.session_state.download_df.to_csv(index=False).encode('utf-8'), "smote_dataset.csv", "text/csv", use_container_width=True)

with col_d2:
    if st.session_state.reweighted_df is not None:
        st.download_button("Download Weighted Data", st.session_state.reweighted_df.to_csv(index=False).encode('utf-8'), "reweighted_dataset.csv", "text/csv", use_container_width=True)

with col_d3:
    out = res["X_test"].copy()
    out[target_col] = res["y_test"].values
    out["y_pred"] = res["y_pred"].values
    if res["y_proba"] is not None:
        try:
            out["y_proba"] = res["y_proba"]
        except:
            pass
    st.download_button("Download Predictions", out.to_csv(index=False).encode('utf-8'), "predictions.csv", "text/csv", use_container_width=True)

st.markdown("<div class='footer'>Built by Git Commit or Die</div>", unsafe_allow_html=True)
