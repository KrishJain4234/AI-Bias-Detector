"""
Streamlit frontend for AI Model Bias Detector.

This application provides a user-friendly interface for:
1. Uploading datasets
2. Selecting target and protected attributes
3. Training models
4. Analyzing fairness metrics
"""

import streamlit as st
import pandas as pd
import numpy as np

# Try to import plotly, show error if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

# Set page configuration first
st.set_page_config(
    page_title="AI Model Bias Detector",
    page_icon="🔍",
    layout="wide"
)

# Import backend modules with error handling
try:
    from backend import data_processing, model_train, fairness
except ImportError as e:
    import sys
    error_msg = str(e)
    python_path = sys.executable
    
    st.error(f"""
    ⚠️ **Required packages are not installed!**
    
    Error: {error_msg}
    
    **Python interpreter:** {python_path}
    
    Please install the required dependencies by running:
    ```
    {python_path} -m pip install -r requirements.txt
    ```
    
    Or install individually:
    ```
    {python_path} -m pip install pandas numpy scikit-learn fairlearn plotly streamlit
    ```
    
    After installing, please restart the Streamlit app.
    """)
    st.stop()

# Show warning if plotly is not available
if not PLOTLY_AVAILABLE:
    st.warning("""
    ⚠️ **Plotly is not installed!**
    
    Charts will not be displayed. Please install the required dependencies:
    ```
    pip install -r requirements.txt
    ```
    Or install plotly directly:
    ```
    pip install plotly
    ```
    """)

# Title and description
st.title("🔍 AI Model Bias Detector")
st.markdown("""
    This application helps you detect and analyze bias in machine learning models.
    Upload your dataset, train a model, and evaluate fairness metrics.
""")

# Initialize session state
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

# Sidebar for file upload
st.sidebar.header("📁 Dataset Upload")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload a CSV file containing your dataset"
)

# Load dataset if file is uploaded
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.sidebar.success(f"Dataset loaded successfully! ({len(df)} rows, {len(df.columns)} columns)")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {str(e)}")

# Main content area
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Display dataset info and preview
    st.header("📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Display dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Display dataset statistics
    with st.expander("📈 Dataset Statistics"):
        st.dataframe(df.describe(), use_container_width=True)
    
    # Selection area
    st.header("⚙️ Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        # Target column selection
        target_col = st.selectbox(
            "Select Target Column",
            options=df.columns.tolist(),
            help="Choose the column that represents the target variable to predict"
        )
    
    with col2:
        # Protected attribute selection
        protected_col = st.selectbox(
            "Select Protected Attribute",
            options=[col for col in df.columns if col != target_col],
            help="Choose the column that represents the protected attribute for fairness analysis"
        )
    
    # Action buttons
    st.header("🚀 Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        preprocess_btn = st.button("🔄 Preprocess Data", type="primary", use_container_width=True)
    
    with col2:
        train_btn = st.button("🏋️ Train Model", type="primary", use_container_width=True, 
                             disabled=st.session_state.preprocessed_data is None)
    
    with col3:
        evaluate_btn = st.button("📊 Evaluate Model", type="primary", use_container_width=True,
                                disabled=st.session_state.model is None)
    
    # Fairness analysis button
    fairness_btn = st.button("⚖️ Analyze Fairness", type="primary", use_container_width=True,
                             disabled=st.session_state.model is None)
    
    # Preprocess data
    if preprocess_btn:
        with st.spinner("Preprocessing data..."):
            try:
                X_train, X_test, y_train, y_test, encoded_df = data_processing.preprocess_data(df, target_col)
                st.session_state.preprocessed_data = {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'encoded_df': encoded_df
                }
                st.success("✅ Data preprocessing completed successfully!")
                
                # Display preprocessing info
                st.info(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")
            except Exception as e:
                st.error(f"❌ Error during preprocessing: {str(e)}")
    
    # Train model
    if train_btn and st.session_state.preprocessed_data is not None:
        with st.spinner("Training model..."):
            try:
                preprocessed = st.session_state.preprocessed_data
                model = model_train.train_model(preprocessed['X_train'], preprocessed['y_train'])
                st.session_state.model = model
                st.success("✅ Model training completed successfully!")
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
    
    # Evaluate model
    if evaluate_btn and st.session_state.model is not None and st.session_state.preprocessed_data is not None:
        with st.spinner("Evaluating model..."):
            try:
                preprocessed = st.session_state.preprocessed_data
                accuracy, report = model_train.evaluate_model(
                    st.session_state.model,
                    preprocessed['X_test'],
                    preprocessed['y_test']
                )
                st.session_state.evaluation_results = {
                    'accuracy': accuracy,
                    'report': report
                }
                st.success("✅ Model evaluation completed successfully!")
            except Exception as e:
                st.error(f"❌ Error during evaluation: {str(e)}")
    
    # Display evaluation results
    if st.session_state.evaluation_results is not None:
        st.header("📊 Model Evaluation Results")
        results = st.session_state.evaluation_results
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{results['accuracy']:.4f}")
        
        with col2:
            st.text_area("Classification Report", results['report'], height=300)
    
    # Analyze fairness
    if fairness_btn and st.session_state.model is not None:
        with st.spinner("Analyzing fairness metrics..."):
            try:
                # Use encoded_df if available for more accurate analysis
                encoded_df = None
                if st.session_state.preprocessed_data is not None:
                    # Use original df for protected column values to preserve original categories
                    encoded_df = st.session_state.preprocessed_data.get('encoded_df')
                
                demo_parity_diff, equal_opp_diff, detailed_metrics = fairness.fairness_analysis(
                    st.session_state.model,
                    df,
                    target_col,
                    protected_col,
                    encoded_df=encoded_df
                )
                st.session_state.fairness_results = {
                    'demographic_parity_diff': demo_parity_diff,
                    'equal_opportunity_diff': equal_opp_diff,
                    'detailed_metrics': detailed_metrics
                }
                st.success("✅ Fairness analysis completed successfully!")
            except Exception as e:
                st.error(f"❌ Error during fairness analysis: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display fairness results
    if st.session_state.fairness_results is not None:
        st.header("⚖️ Fairness Analysis Results")
        results = st.session_state.fairness_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Demographic Parity Difference",
                f"{results['demographic_parity_diff']:.4f}",
                help="Difference in positive prediction rates between groups. Lower is better (0 = perfect parity)."
            )
        
        with col2:
            st.metric(
                "Equal Opportunity Difference",
                f"{results['equal_opportunity_diff']:.4f}",
                help="Difference in true positive rates between groups. Lower is better (0 = perfect equality)."
            )
        
        # Detailed metrics
        if 'detailed_metrics' in results and isinstance(results['detailed_metrics'], dict):
            detailed = results['detailed_metrics']
            
            if 'demographic_parity_rates' in detailed:
                st.subheader("Demographic Parity Rates by Group")
                demo_df = pd.DataFrame.from_dict(
                    detailed['demographic_parity_rates'],
                    orient='index',
                    columns=['Positive Prediction Rate']
                )
                demo_df.index.name = protected_col
                
                # Bar chart
                if PLOTLY_AVAILABLE and px is not None:
                    fig_demo = px.bar(
                        demo_df,
                        y='Positive Prediction Rate',
                        labels={'index': protected_col, 'value': 'Positive Prediction Rate'},
                        title="Demographic Parity by Group"
                    )
                    st.plotly_chart(fig_demo, use_container_width=True)
                else:
                    st.warning("⚠️ Plotly is not installed. Install it with: `pip install plotly` to see charts.")
                
                st.dataframe(demo_df, use_container_width=True)
            
            if 'equal_opportunity_rates' in detailed:
                st.subheader("Equal Opportunity Rates (True Positive Rates) by Group")
                opp_df = pd.DataFrame.from_dict(
                    detailed['equal_opportunity_rates'],
                    orient='index',
                    columns=['True Positive Rate']
                )
                opp_df.index.name = protected_col
                
                # Bar chart
                if PLOTLY_AVAILABLE and px is not None:
                    fig_opp = px.bar(
                        opp_df,
                        y='True Positive Rate',
                        labels={'index': protected_col, 'value': 'True Positive Rate'},
                        title="Equal Opportunity (True Positive Rate) by Group"
                    )
                    st.plotly_chart(fig_opp, use_container_width=True)
                else:
                    st.warning("⚠️ Plotly is not installed. Install it with: `pip install plotly` to see charts.")
                
                st.dataframe(opp_df, use_container_width=True)
            
            # Interpretation
            st.subheader("📝 Interpretation")
            if results['demographic_parity_diff'] < 0.1:
                st.success("✅ **Demographic Parity**: Good - Low difference indicates fair distribution of predictions across groups.")
            else:
                st.warning("⚠️ **Demographic Parity**: High difference indicates potential bias in prediction distribution.")
            
            if results['equal_opportunity_diff'] < 0.1:
                st.success("✅ **Equal Opportunity**: Good - Low difference indicates fair true positive rates across groups.")
            else:
                st.warning("⚠️ **Equal Opportunity**: High difference indicates potential bias in model performance across groups.")
else:
    # Welcome message when no dataset is loaded
    st.info("👈 Please upload a CSV file using the sidebar to get started.")
    
    # Example instructions
    with st.expander("📖 How to use this application"):
        st.markdown("""
        1. **Upload Dataset**: Click on "Choose a CSV file" in the sidebar to upload your dataset
        2. **Configure**: Select the target column (what you want to predict) and protected attribute (for fairness analysis)
        3. **Preprocess**: Click "Preprocess Data" to prepare your data for training
        4. **Train Model**: Click "Train Model" to train a logistic regression model
        5. **Evaluate**: Click "Evaluate Model" to see accuracy and classification metrics
        6. **Analyze Fairness**: Click "Analyze Fairness" to see demographic parity and equal opportunity metrics
        """)
