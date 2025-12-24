
# Performs feature selection using MI, correlation, and variance analysis

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold

def compute_mutual_information(df):
    X = df.drop(columns=["track_genre"])
    y = df["track_genre"]
    mi = mutual_info_classif(X, y)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    return mi_series


def compute_correlations(df):
    numeric_df = df.drop(columns=["track_genre"])
    corr_matrix = numeric_df.corr()
    return corr_matrix


def low_variance_features(df, threshold=0.0001):
    selector = VarianceThreshold(threshold=threshold)
    X = df.drop(columns=["track_genre"])

    selector.fit(X)
    retained = selector.get_support(indices=True)
    dropped = [col for i, col in enumerate(X.columns) if i not in retained]
    return dropped


def features_to_drop(df):
    # 1. Mutual Information filtering
    mi_scores = compute_mutual_information(df)
    low_mi = list(mi_scores[mi_scores < 0.01].index)

    # 2. Forced drop: musically irrelevant features
    forced_drop = ["key", "mode", "time_signature"]

    # 3. Correlation filtering
    corr = compute_correlations(df)
    highly_corr = []
    for col in corr.columns:
        for row in corr.index:
            if col != row and abs(corr.loc[row, col]) > 0.90:
                highly_corr.append(col)

    # 4. Low variance filtering
    low_var = low_variance_features(df)

    # Combine all drop sources
    drop = set(low_mi + highly_corr + low_var + forced_drop)

    return list(drop)



def apply_feature_selection(df):
    drop_cols = features_to_drop(df)

    df_clean = df.drop(columns=drop_cols)

    return df_clean, drop_cols
