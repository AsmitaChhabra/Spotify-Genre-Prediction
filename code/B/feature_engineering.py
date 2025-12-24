# feature_engineering.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def add_log_features(df):
    if "duration_ms" in df.columns:
        df["log_duration_ms"] = np.log1p(df["duration_ms"])
    if "instrumentalness" in df.columns:
        df["log_instrumentalness"] = np.log1p(df["instrumentalness"])
    return df


def add_binned_features(df):
    if "tempo" in df.columns:
        df["tempo_bin"] = pd.cut(
            df["tempo"],
            bins=[0, 90, 140, 250],
            labels=["slow", "medium", "fast"]
        )

    if "loudness" in df.columns:
        df["loudness_bin"] = pd.cut(
            df["loudness"],
            bins=[-60, -20, -5, 0],
            labels=["quiet", "normal", "loud"]
        )
    return df


def encode_binned_features(df):
    le = LabelEncoder()
    for col in ["tempo_bin", "loudness_bin"]:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def add_interaction_features(df):
    if set(["energy", "danceability"]).issubset(df.columns):
        df["energy_x_dance"] = df["energy"] * df["danceability"]

    if set(["acousticness", "instrumentalness"]).issubset(df.columns):
        df["acoustic_x_instrument"] = df["acousticness"] * df["instrumentalness"]

    return df


def apply_feature_engineering(df, add_poly=False):
    df = add_log_features(df)
    df = add_binned_features(df)
    df = encode_binned_features(df)    # <<--- FIX
    df = add_interaction_features(df)
    return df
