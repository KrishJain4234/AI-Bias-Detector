"""
Model training module for training and evaluating machine learning models.

This module handles model training using logistic regression and provides
evaluation metrics including accuracy and classification reports.
"""

import pandas as pd
import numpy as np

# Import sklearn modules with error handling
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    raise ImportError(
        "scikit-learn is not installed. Please install it using: pip install scikit-learn\n"
        "Or install all requirements: pip install -r requirements.txt"
    )


def train_model(X_train, y_train):
    """
    Train a logistic regression model on the training data.
    
    Args:
        X_train (pd.DataFrame or np.ndarray): Training features
        y_train (pd.Series or np.ndarray): Training target labels
        
    Returns:
        sklearn.linear_model.LogisticRegression: Trained logistic regression model
    """
    # Initialize logistic regression model
    # Using max_iter=1000 to ensure convergence for complex datasets
    model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained machine learning model (e.g., LogisticRegression)
        X_test (pd.DataFrame or np.ndarray): Testing features
        y_test (pd.Series or np.ndarray): Testing target labels
        
    Returns:
        tuple: A tuple containing:
            - accuracy (float): Model accuracy score
            - classification_report (str): Detailed classification report
    """
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=False)
    
    return accuracy, report
