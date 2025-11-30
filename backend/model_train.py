# backend/model_train.py
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from typing import Tuple, Optional

RANDOM_STATE = 42


def get_model(name: str):
    name = name.lower()
    if name == "logistic":
        # Increased max_iter and using robust settings
        return LogisticRegression(
            max_iter=5000,
            solver='lbfgs',
            random_state=RANDOM_STATE
        )

    if name == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)

    if name == "svm":
        # SVM requires scaling; handled below
        return SVC(probability=True, random_state=RANDOM_STATE)

    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE)

    raise ValueError(f"Unknown model name '{name}'")


def train_and_predict(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    sample_weight: Optional[pd.Series] = None
) -> Tuple[object, pd.Series, pd.Series]:
    """
    Train the chosen model and return (trained_model, y_pred, y_proba)
    Automatically scales the data for Logistic Regression and SVM.
    """
    model = get_model(model_name)

    # ------------------------------
    # SCALE FEATURES for models that need it
    # ------------------------------
    needs_scaling = model_name in ["logistic", "svm"]

    if needs_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # ------------------------------
    # Model Training
    # ------------------------------
    if sample_weight is None:
        model.fit(X_train_scaled, y_train)
    else:
        # model.fit accepts sample_weight in most sklearn models
        model.fit(X_train_scaled, y_train, sample_weight=sample_weight.values)

    # ------------------------------
    # Predictions
    # ------------------------------
    y_pred = pd.Series(model.predict(X_test_scaled), index=X_test.index, name='y_pred')

    # compute probabilities when possible
    try:
        y_proba = pd.Series(model.predict_proba(X_test_scaled)[:, 1],
                            index=X_test.index, name='y_proba')
    except Exception:
        # fallback using decision function or predicted labels
        try:
            decision = model.decision_function(X_test_scaled)
            probs = 1 / (1 + np.exp(-decision))
            y_proba = pd.Series(probs, index=X_test.index, name='y_proba')
        except Exception:
            y_proba = pd.Series(y_pred.astype(float), index=X_test.index, name='y_proba')

    return model, y_pred, y_proba


def evaluate_model(y_test: pd.Series, y_pred: pd.Series):
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)
    return acc, report
