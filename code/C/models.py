# models.py
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

import numpy as np


# ====================================================================
# UTILITY - unified evaluation function
# ====================================================================
def eval_on_val(model, X_val, y_val):
    y_pred = model.predict(X_val)

    prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_val, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_val, y_pred, average="weighted", zero_division=0)

    return prec, rec, f1, y_pred


# ====================================================================
# RANDOM FOREST
# ====================================================================
def train_random_forest(X_train, y_train, X_val, y_val):
    print("\n========== TRAINING RANDOM FOREST ==========\n")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)
    prec, rec, f1, _ = eval_on_val(model, X_val, y_val)

    print("RF Precision:", round(prec, 4))
    print("RF Recall:   ", round(rec, 4))
    print("RF F1 Score: ", round(f1, 4))

    return model, prec, rec, f1


# ====================================================================
# BASELINE XGBOOST
# ====================================================================
def train_xgboost(X_train, y_train, X_val, y_val):
    print("\n========== TRAINING XGBOOST ==========\n")

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=4,
        random_state=42
    )

    model.fit(X_train, y_train)
    prec, rec, f1, _ = eval_on_val(model, X_val, y_val)

    print("XGB Precision:", round(prec, 4))
    print("XGB Recall:   ", round(rec, 4))
    print("XGB F1 Score: ", round(f1, 4))

    return model, prec, rec, f1


# ====================================================================
# CATBOOST
# ====================================================================
def train_catboost(X_train, y_train, X_val, y_val):
    print("\n========== TRAINING CATBOOST (OPTIMIZED FOR MAC) ==========\n")

    X_train = X_train.astype("float32")
    X_val   = X_val.astype("float32")

    model = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.07,
        loss_function="MultiClass",
        eval_metric="TotalF1",
        thread_count=4,
        random_seed=42,
        verbose=False
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )

    model.save_model("catboost_model.cbm")

    print("CatBoost model saved as catboost_model.cbm")

    prec, rec, f1, _ = eval_on_val(model, X_val, y_val)

    print("\n===== CATBOOST VALIDATION METRICS =====")
    print("Precision:", round(prec, 4))
    print("Recall:   ", round(rec, 4))
    print("F1 Score: ", round(f1, 4))

    return model, prec, rec, f1


# ====================================================================
# FINAL XGBOOST (using best_params) + EARLY STOPPING
# ====================================================================
def train_final_xgboost(X_train, y_train, X_val, y_val, best_params):
    print("\n========== TRAINING FINAL XGBOOST MODEL ==========\n")

    model = XGBClassifier(
        **best_params,
        objective="multi:softmax",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=4,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True       # <-- keep this ONLY
    )

    prec, rec, f1, _ = eval_on_val(model, X_val, y_val)

    print("\n===== FINAL XGBOOST VALIDATION METRICS =====")
    print("Precision:", round(prec, 4))
    print("Recall:   ", round(rec, 4))
    print("F1 Score: ", round(f1, 4))

    model.save_model("final_xgboost.model")
    print("\nSaved final model as final_xgboost.model")

    return model, prec, rec, f1

# ====================================================================
# FINAL TEST EVALUATION
# ====================================================================
def evaluate_on_test(model, X_test, y_test):
    print("\n========== TEST SET EVALUATION ==========\n")

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred, digits=4))

    f1_test = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    print("\nFinal Test F1 Score:", round(f1_test, 4))

    return f1_test
