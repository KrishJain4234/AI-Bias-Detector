"""
Data processing module for preprocessing datasets.

This module handles data preprocessing tasks including train-test splitting,
encoding categorical variables, and preparing data for model training.
"""

import pandas as pd
import numpy as np

# Import sklearn modules with error handling
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    raise ImportError(
        "scikit-learn is not installed. Please install it using: pip install scikit-learn\n"
        "Or install all requirements: pip install -r requirements.txt"
    )


def preprocess_data(df, target_col):
    """
    Preprocess the dataset for model training.
    
    This function performs the following operations:
    1. Separates features and target variable
    2. Encodes categorical variables
    3. Splits data into training and testing sets
    4. Standardizes numerical features
    
    Args:
        df (pd.DataFrame): Input dataframe containing features and target
        target_col (str): Name of the target column
        
    Returns:
        tuple: A tuple containing:
            - X_train (pd.DataFrame): Training features
            - X_test (pd.DataFrame): Testing features
            - y_train (pd.Series): Training target
            - y_test (pd.Series): Testing target
            - encoded_df (pd.DataFrame): Original dataframe with encoded columns
    """
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    encoded_df = df_copy.copy()
    
    # Separate features and target
    X = df_copy.drop(columns=[target_col])
    y = df_copy[target_col]
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Encode categorical variables using LabelEncoder
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoded_df[col] = X[col]
        label_encoders[col] = le
    
    # Encode target variable if it's categorical
    if y.dtype == 'object' or y.dtype.name == 'category':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))
        encoded_df[target_col] = y
    else:
        encoded_df[target_col] = y
    
    # Convert to numeric for numerical columns (handle any remaining issues)
    for col in numerical_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        # Also update encoded_df
        encoded_df[col] = X[col]
    
    # Fill NaN values with median for numerical columns
    for col in numerical_cols:
        if X[col].isna().any():
            median_val = X[col].median()
            # Handle case where all values are NaN
            if pd.isna(median_val):
                median_val = 0
            X[col].fillna(median_val, inplace=True)
            # Update encoded_df with filled values
            encoded_df[col] = X[col]
    
    # Split into train and test sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Standardize numerical features
    scaler = StandardScaler()
    if len(numerical_cols) > 0:
        # Standardize train and test sets
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
        
        # Also standardize the full dataset for fairness analysis
        # First ensure no NaN values in encoded_df
        for col in numerical_cols:
            if encoded_df[col].isnull().any():
                median_val = encoded_df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                encoded_df[col].fillna(median_val, inplace=True)
        
        # Now standardize
        encoded_df[numerical_cols] = scaler.transform(encoded_df[numerical_cols])
    
    # Final check: Ensure encoded_df has no NaN values in any column
    for col in encoded_df.columns:
        if encoded_df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(encoded_df[col]):
                fill_val = encoded_df[col].median()
                if pd.isna(fill_val):
                    fill_val = 0
                encoded_df[col].fillna(fill_val, inplace=True)
            else:
                mode_vals = encoded_df[col].mode()
                fill_val = mode_vals[0] if len(mode_vals) > 0 else 0
                encoded_df[col].fillna(fill_val, inplace=True)
    
    return X_train, X_test, y_train, y_test, encoded_df
