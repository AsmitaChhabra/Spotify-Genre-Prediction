# preprocessing.py
# Complete data loading, cleaning, splitting, and preprocessing pipeline

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from feature_selection import apply_feature_selection
from feature_engineering import apply_feature_engineering



# ============================================================
# 1) Load & Clean Data
# ============================================================
def load_data(filepath="A/spotify_30_percent.csv"):
    df = pd.read_csv(filepath)
    print(f"Initial shape: {df.shape}")

    # Drop unnecessary text/id columns
    drop_cols = ["track_id", "artists", "album_name", "track_name", "Unnamed: 0"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    print(f"After dropping ID/text columns: {df.shape}")

    # Identify numeric columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    print("Numeric columns:", num_cols)

    # Handle missing numeric values
    if df[num_cols].isnull().sum().sum() > 0:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    else:
        print("No missing numeric values found.")

    # Encode target
    label_encoder = LabelEncoder()
    df["track_genre"] = label_encoder.fit_transform(df["track_genre"])

    X = df.drop(columns=["track_genre"])
    y = df["track_genre"]

    return X, y, label_encoder



# ============================================================
# 2) Create 70/15/15 Stratified Split
# ============================================================

def split_data(X, y):
    # Train = 70%, Temp = 30%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Temp → Validation (15%) + Test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print("\nDATA SPLIT SHAPES:")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test



# ============================================================
# 3) Build the Preprocessing Pipeline
# ============================================================

def build_preprocessing_pipeline(X_train):
    # Identify numeric columns for scaling
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # ColumnTransformer → apply StandardScaler to numeric columns
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols)
        ],
        remainder="passthrough"  # Keep any non-numeric columns unchanged
    )

    # Wrap into a pipeline
    pipeline = Pipeline([
        ("preprocess", column_transformer)
    ])

    return pipeline


# ============================================================
# 4) Full Preprocessing Function (FINAL VERSION)
# ============================================================

from feature_engineering import apply_feature_engineering
from feature_selection import apply_feature_selection

def preprocess(filepath="A/spotify_30_percent.csv"):
    # ---------------------------
    # Step 1: Load raw dataset (X features + y target)
    # ---------------------------
    X, y, label_encoder = load_data(filepath)

    # Combine into a single df for FE + FS
    df = X.copy()
    df["track_genre"] = y

    # ---------------------------
    # Step 2: Feature Engineering (must be applied on full df)
    # ---------------------------
    print("\nApplying feature engineering...")
    df = apply_feature_engineering(df, add_poly=False)

    # ---------------------------
    # Step 3: Feature Selection (also needs full df)
    # ---------------------------
    print("Applying feature selection...")
    df, dropped_cols = apply_feature_selection(df)
    print("Dropped columns:", dropped_cols)

    print("\n===== DATAFRAME AFTER FEATURE ENGINEERING + SELECTION =====")
    print(df.head())
    print("\nShape:", df.shape)
    print("Columns:", df.columns.tolist())


    # After selection, split back into X and y
    X = df.drop(columns=["track_genre"])
    y = df["track_genre"]

    # ---------------------------
    # Step 4: Split into train/val/test
    # ---------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # ---------------------------
    # Step 5: Build preprocessing pipeline
    # ---------------------------
    pipeline = build_preprocessing_pipeline(X_train)

    # ---------------------------
    # Step 6: Fit only on training data
    # ---------------------------
    pipeline.fit(X_train)

    # ---------------------------
    # Step 7: Transform all splits
    # ---------------------------
    X_train_trans = pipeline.transform(X_train)
    X_val_trans = pipeline.transform(X_val)
    X_test_trans = pipeline.transform(X_test)

    print("\nSCALING + PREPROCESSING COMPLETE.\n")

    return (
        X_train_trans,
        X_val_trans,
        X_test_trans,
        y_train,
        y_val,
        y_test,
        pipeline,
        label_encoder
    )
