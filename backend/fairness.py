"""
Fairness analysis module for evaluating model fairness metrics.

This module provides functions to analyze fairness metrics including
demographic parity difference and equal opportunity difference.
"""

import pandas as pd
import numpy as np

# Import sklearn and fairlearn modules with error handling
try:
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    raise ImportError(
        "scikit-learn is not installed. Please install it using: pip install scikit-learn\n"
        "Or install all requirements: pip install -r requirements.txt"
    )


def fairness_analysis(model, df, target_col, protected_col, encoded_df=None):
    """
    Perform fairness analysis on the model predictions.
    
    This function calculates:
    1. Demographic Parity Difference: Difference in positive prediction rates
       between different groups in the protected attribute
    2. Equal Opportunity Difference: Difference in true positive rates (recall)
       between different groups in the protected attribute
    
    Args:
        model: Trained machine learning model
        df (pd.DataFrame): Dataframe containing features and target
        target_col (str): Name of the target column
        protected_col (str): Name of the protected attribute column
        encoded_df (pd.DataFrame, optional): Preprocessed dataframe with encoded columns.
                                            If provided, will be used instead of re-encoding.
        
    Returns:
        tuple: A tuple containing:
            - demographic_parity_diff (float): Demographic parity difference
            - equal_opportunity_diff (float): Equal opportunity difference
            - detailed_metrics (dict): Dictionary with detailed fairness metrics
    """
    # Always use original df for protected column to preserve original categories
    protected_values = df[protected_col].values
    
    if encoded_df is not None:
        # Use the preprocessed encoded dataframe (should already have same preprocessing as training)
        # Drop target column, keep all other columns (including protected_col if it was used as a feature)
        X_encoded = encoded_df.drop(columns=[target_col]).copy()
        
        # Ensure no NaN values remain - fill them
        for col in X_encoded.columns:
            if X_encoded[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X_encoded[col]):
                    fill_value = X_encoded[col].median()
                    if pd.isna(fill_value):  # If median is NaN, use 0
                        fill_value = 0
                    X_encoded[col].fillna(fill_value, inplace=True)
                else:
                    mode_values = X_encoded[col].mode()
                    fill_value = mode_values[0] if len(mode_values) > 0 else 0
                    X_encoded[col].fillna(fill_value, inplace=True)
        
        # Ensure the columns match what the model expects
        if hasattr(model, 'feature_names_in_'):
            # Model has feature names stored (newer sklearn versions)
            # Only keep columns that were used in training
            expected_cols = list(model.feature_names_in_)
            missing_cols = set(expected_cols) - set(X_encoded.columns)
            
            if missing_cols:
                raise ValueError(f"Missing features in data: {missing_cols}. "
                               f"Expected: {expected_cols}, Got: {list(X_encoded.columns)}")
            
            # Reorder to match model's expected order (exact column order matters)
            X_encoded = X_encoded[expected_cols]
        elif hasattr(model, 'n_features_in_'):
            # For older sklearn versions, ensure same number of features
            if X_encoded.shape[1] != model.n_features_in_:
                raise ValueError(
                    f"Feature count mismatch: Model expects {model.n_features_in_} features, "
                    f"but data has {X_encoded.shape[1]} features. "
                    f"Columns: {list(X_encoded.columns)}"
                )
    else:
        # Fallback: encode on the fly (not recommended - may not match training preprocessing)
        X = df.drop(columns=[target_col])
        
        # Encode categorical variables if needed
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        X_encoded = X.copy()
        
        # Encode categorical columns
        for col in categorical_cols:
            if col != protected_col:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        
        # Fill NaN values
        for col in numerical_cols:
            if X_encoded[col].isnull().any():
                X_encoded[col].fillna(X_encoded[col].median(), inplace=True)
        
        # This won't have standardization applied, so may not work correctly
        # Better to use encoded_df from preprocessing
    
    # Make predictions
    try:
        # Convert DataFrame to numpy array for sklearn prediction
        # Sklearn can handle DataFrames in newer versions, but arrays are safer
        X_array = X_encoded.values if isinstance(X_encoded, pd.DataFrame) else X_encoded
        
        # Final check for NaN values
        if np.isnan(X_array).any():
            # Replace any remaining NaN with 0 (shouldn't happen if preprocessing was correct)
            X_array = np.nan_to_num(X_array, nan=0.0)
        
        y_pred = model.predict(X_array)
    except Exception as e:
        error_msg = str(e)
        
        # Check if it's a NaN error
        if 'NaN' in error_msg or 'missing' in error_msg.lower():
            # Try to fill NaN values more aggressively
            if isinstance(X_encoded, pd.DataFrame):
                X_encoded = X_encoded.fillna(X_encoded.median(numeric_only=True))
                X_encoded = X_encoded.fillna(0)
                try:
                    y_pred = model.predict(X_encoded)
                except:
                    raise ValueError(f"Unable to make predictions due to data issues: {error_msg}")
            else:
                raise ValueError(f"Unable to make predictions: {error_msg}")
        else:
            raise ValueError(f"Unable to make predictions: {error_msg}")
    
    # Get true labels
    y_true = df[target_col].values
    
    # Handle binary classification (assume positive class is the higher value)
    if len(np.unique(y_true)) == 2:
        positive_class = max(np.unique(y_true))
    else:
        positive_class = 1  # Default assumption
    
    # Get unique groups in protected attribute
    unique_groups = np.unique(protected_values)
    
    if len(unique_groups) < 2:
        # Not enough groups for fairness analysis
        return 0.0, 0.0, {"error": "Not enough groups in protected attribute for fairness analysis"}
    
    # Calculate demographic parity (positive prediction rate) for each group
    demographic_parity_rates = {}
    for group in unique_groups:
        group_mask = protected_values == group
        group_pred = y_pred[group_mask]
        if len(group_pred) > 0:
            demographic_parity_rates[group] = np.mean(group_pred == positive_class)
        else:
            demographic_parity_rates[group] = 0.0
    
    # Calculate demographic parity difference (max - min)
    demographic_parity_values = list(demographic_parity_rates.values())
    demographic_parity_diff = max(demographic_parity_values) - min(demographic_parity_values)
    
    # Calculate equal opportunity (true positive rate / recall) for each group
    equal_opportunity_rates = {}
    for group in unique_groups:
        group_mask = protected_values == group
        group_true = y_true[group_mask]
        group_pred = y_pred[group_mask]
        
        if len(group_true) > 0:
            # Calculate true positive rate
            tp = np.sum((group_true == positive_class) & (group_pred == positive_class))
            fn = np.sum((group_true == positive_class) & (group_pred != positive_class))
            
            if tp + fn > 0:
                equal_opportunity_rates[group] = tp / (tp + fn)
            else:
                equal_opportunity_rates[group] = 0.0
        else:
            equal_opportunity_rates[group] = 0.0
    
    # Calculate equal opportunity difference (max - min)
    equal_opportunity_values = list(equal_opportunity_rates.values())
    equal_opportunity_diff = max(equal_opportunity_values) - min(equal_opportunity_values)
    
    # Create detailed metrics dictionary
    detailed_metrics = {
        "demographic_parity_rates": {str(k): float(v) for k, v in demographic_parity_rates.items()},
        "equal_opportunity_rates": {str(k): float(v) for k, v in equal_opportunity_rates.items()},
        "demographic_parity_diff": float(demographic_parity_diff),
        "equal_opportunity_diff": float(equal_opportunity_diff)
    }
    
    return demographic_parity_diff, equal_opportunity_diff, detailed_metrics
