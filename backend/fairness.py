# backend/fairness.py
import pandas as pd
import numpy as np

def compute_group_metrics(y_true, y_pred, sensitive_feature):
    """
    Compute per-group metrics for ANY number of sensitive groups.
    Returns a dict with main metrics and group metrics DataFrame.
    """
    df = pd.DataFrame({
        "y_true": np.array(y_true).astype(int),
        "y_pred": np.array(y_pred).astype(int),
        "group": np.array(sensitive_feature).astype(str)
    })

    groups = df["group"].unique()
    group_metrics = []

    for g in groups:
        g_df = df[df["group"] == g]
        positive_rate = g_df["y_pred"].mean()  # P(y_pred=1 | group=g)

        if g_df[g_df["y_true"] == 1].shape[0] > 0:
            tpr = g_df[g_df["y_true"] == 1]["y_pred"].mean()
            fnr = 1 - tpr
        else:
            tpr = np.nan
            fnr = np.nan

        if g_df[g_df["y_true"] == 0].shape[0] > 0:
            fpr = g_df[g_df["y_true"] == 0]["y_pred"].mean()
        else:
            fpr = np.nan

        if g_df[g_df["y_pred"] == 1].shape[0] > 0:
            precision = g_df[g_df["y_pred"] == 1]["y_true"].mean()
        else:
            precision = np.nan

        group_metrics.append({
            "group": g,
            "count": len(g_df),
            "positive_prediction_rate": positive_rate,
            "true_positive_rate": tpr,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "precision": precision
        })

    group_df = pd.DataFrame(group_metrics)

    dp_rates = group_df["positive_prediction_rate"].dropna()
    demographic_parity_diff = float(dp_rates.max() - dp_rates.min()) if len(dp_rates) > 0 else np.nan

    tpr_values = group_df["true_positive_rate"].dropna()
    equal_opportunity_diff = float(tpr_values.max() - tpr_values.min()) if len(tpr_values) > 0 else np.nan

    fpr_values = group_df["false_positive_rate"].dropna()
    equalized_odds_diff = np.nan
    if len(tpr_values) > 0 and len(fpr_values) > 0:
        equalized_odds_diff = float((tpr_values.max() - tpr_values.min()) + (fpr_values.max() - fpr_values.min()))

    disparate_impact_ratio = np.nan
    if len(dp_rates) > 0 and dp_rates.max() > 0:
        disparate_impact_ratio = float(dp_rates.min() / dp_rates.max())

    prec_values = group_df["precision"].dropna()
    predictive_parity_diff = float(prec_values.max() - prec_values.min()) if len(prec_values) > 0 else np.nan

    return {
        "demographic_parity_difference": demographic_parity_diff,
        "equal_opportunity_difference": equal_opportunity_diff,
        "equalized_odds_difference": equalized_odds_diff,
        "disparate_impact_ratio": disparate_impact_ratio,
        "predictive_parity_difference": predictive_parity_diff,
        "group_metrics": group_df
    }

def probs_to_preds(y_proba, threshold=0.5):
    return (y_proba >= threshold).astype(int)
