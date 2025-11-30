# backend/data_processing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42

def _standardize_strings(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=['object', 'category']).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip().str.replace(r'\.+$', '', regex=True)
    return df

def _replace_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.replace(['?', 'NA', 'N/A', '', 'nan', 'NaN'], np.nan, inplace=True)
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generic missing value handler:
    - Numeric columns -> median
    - Categorical columns -> mode
    """
    df = df.copy()
    df = _replace_missing_markers(df)

    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().any():
            mode = df[col].mode(dropna=True)
            if len(mode) > 0:
                df[col] = df[col].fillna(mode[0])
            else:
                df[col] = df[col].fillna('Unknown')

    return df

def prepare_dataset(
    df: pd.DataFrame,
    target_col: str,
    sensitive_col: str = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE
):
    """
    Cleans dataset, imputes missing values, encodes categorical features with LabelEncoder,
    splits into train/test.
    Returns a dictionary with train/test splits and some helpers.
    """
    df = df.copy()
    df = _standardize_strings(df)
    df = handle_missing_values(df)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    # Extract target
    y = df[target_col].copy()

    # Normalize target to binary (0/1). Accept many common encodings.
    # If numeric and exactly two unique numeric values, map to 0/1
    if y.dtype.kind in 'biufc':
        unique_vals = sorted(list(pd.Series(y.unique()).astype(float)))
        if len(unique_vals) == 2:
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            y_bin = y.map(mapping).astype(int)
        else:
            raise ValueError("Numeric target with != 2 unique values is not supported.")
    else:
        # make strings lower and strip
        y_str = y.astype(str).str.strip().str.lower()
        # common positive markers
        pos_markers = ['>50k', '>50k.', 'yes', 'true', '1', 'good', 'bad']  # we will handle 'good'/'bad' externally
        # If values are like 'good'/'bad' map to 0/1 (bad=1)
        uniq = sorted(y_str.unique())
        if set(uniq) <= set(['good', 'bad']):
            mapping = {'good': 0, 'bad': 1}
            y_bin = y_str.map(mapping).astype(int)
        elif set(uniq) <= set(['yes', 'no']):
            mapping = {'no': 0, 'yes': 1}
            y_bin = y_str.map(mapping).astype(int)
        elif any(x.startswith('>') for x in uniq) or any('>' in x for x in uniq):
            # try to detect > threshold style
            y_bin = y_str.apply(lambda v: 1 if '>' in v else 0).astype(int)
        else:
            # fallback: label encode but ensure binary
            le_tmp = LabelEncoder()
            y_bin = pd.Series(le_tmp.fit_transform(y_str), index=y.index)
            if len(le_tmp.classes_) != 2:
                raise ValueError("Target column does not map to a binary problem. Please provide a binary target.")
            y_bin = y_bin.astype(int)

    # Remove target from features
    X = df.drop(columns=[target_col]).copy()

    # Sensitive series (kept as string)
    if sensitive_col:
        if sensitive_col not in df.columns:
            raise ValueError(f"Sensitive column '{sensitive_col}' not found in dataframe")
        sensitive = df[sensitive_col].astype(str).copy()
        # Note: we keep sensitive column out of X for training fairness analysis (but we may keep it for downloads)
        X = X.drop(columns=[sensitive_col])
    else:
        sensitive = pd.Series(['all'] * len(df), index=df.index)

    # Encode categorical feature columns with LabelEncoder (required for SMOTE)
    X_enc = X.copy()
    label_encoders = {}
    cat_cols = X_enc.select_dtypes(include=['object', 'category']).columns
    for c in cat_cols:
        le = LabelEncoder()
        X_enc[c] = X_enc[c].astype(str)
        X_enc[c] = le.fit_transform(X_enc[c])
        label_encoders[c] = le

    # Train-test split; stratify if possible
    stratify = y_bin if len(y_bin.unique()) > 1 else None
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X_enc, y_bin, sensitive, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Reset indices for convenience
    return {
        "X_train": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "s_train": s_train.reset_index(drop=True),
        "s_test": s_test.reset_index(drop=True),
        "df_clean": df.reset_index(drop=True),
        "label_encoders": label_encoders
    }

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state=RANDOM_STATE, k_neighbors=5):
    """
    Apply SMOTE to the training set. Returns X_res, y_res (as DataFrame/Series).
    """
    sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)
    return X_res, y_res

def apply_reweighing(y_train: pd.Series, sensitive_train: pd.Series):
    """
    Compute sample weights using the standard reweighing formula:
      w(a,y) = P(A=a) * P(Y=y) / P(A=a, Y=y)
    Returns a pandas Series of sample weights aligned with y_train / sensitive_train.
    """
    y = pd.Series(y_train).reset_index(drop=True)
    a = pd.Series(sensitive_train).astype(str).reset_index(drop=True)

    df = pd.DataFrame({"A": a, "Y": y})

    # global probabilities
    p_y = df['Y'].value_counts(normalize=True).to_dict()
    p_a = df['A'].value_counts(normalize=True).to_dict()

    # joint probabilities
    joint = df.groupby(['A','Y']).size().reset_index(name='count')
    joint['p_ay'] = joint['count'] / len(df)
    joint_map = {(row['A'], row['Y']): row['p_ay'] for _, row in joint.iterrows()}

    weights = []
    for ai, yi in zip(df['A'], df['Y']):
        p_a_i = p_a.get(ai, 0)
        p_y_i = p_y.get(yi, 0)
        p_ay = joint_map.get((ai, yi), 0)
        # avoid division by zero; if joint prob is zero set weight = 1 (fallback)
        if p_ay <= 0:
            w = 1.0
        else:
            w = (p_a_i * p_y_i) / p_ay
        weights.append(w)

    weights = pd.Series(weights, index=y.index, name='sample_weight')
    # Normalize weights to have mean 1 (optional, keeps scale stable)
    if len(weights) > 0 and weights.mean() > 0:
        weights = weights / weights.mean()

    return weights
