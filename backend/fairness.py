import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def compute_group_metrics(y_true, y_pred, sensitive_feature):
    """
    Compute fairness metrics for ANY number of sensitive groups.
    Returns:
        - demographic parity differences
        - equal opportunity differences
        - per-group metrics dataframe
    """
    
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "group": sensitive_feature
    })

    groups = df["group"].unique()
    group_metrics = []

    for g in groups:
        g_df = df[df["group"] == g]

        # positive prediction rate = P(y_pred = 1 | group = g)
        positive_rate = g_df["y_pred"].mean()

        # true positive rate = P(y_pred = 1 | y_true = 1, group = g)
        if g_df[g_df["y_true"] == 1].shape[0] > 0:
            tpr = g_df[g_df["y_true"] == 1]["y_pred"].mean()
        else:
            tpr = np.nan  # No positives in this group → cannot compute

        # false positive rate = P(y_pred = 1 | y_true = 0, group = g)
        if g_df[g_df["y_true"] == 0].shape[0] > 0:
            fpr = g_df[g_df["y_true"] == 0]["y_pred"].mean()
        else:
            fpr = np.nan

        group_metrics.append({
            "group": g,
            "positive_prediction_rate": positive_rate,
            "true_positive_rate": tpr,
            "false_positive_rate": fpr,
            "count": len(g_df)
        })

    group_df = pd.DataFrame(group_metrics)

    # ----- Demographic Parity Difference -----
    dp_rates = group_df["positive_prediction_rate"].dropna()
    demographic_parity_diff = dp_rates.max() - dp_rates.min()

    # ----- Equal Opportunity Difference -----
    tpr_values = group_df["true_positive_rate"].dropna()
    equal_opportunity_diff = tpr_values.max() - tpr_values.min()

    return demographic_parity_diff, equal_opportunity_diff, group_df
