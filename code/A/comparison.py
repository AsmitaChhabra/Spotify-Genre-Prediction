# comparison.py
from models import train_random_forest, train_xgboost, train_catboost
import pandas as pd

def compare_models(X_train, y_train, X_val, y_val):
    results = []

    # Random Forest
    rf_model, p, r, f = train_random_forest(X_train, y_train, X_val, y_val)
    results.append(["RandomForest", p, r, f])

    # XGBoost
    xgb_model, p, r, f = train_xgboost(X_train, y_train, X_val, y_val)
    results.append(["XGBoost", p, r, f])

    # CatBoost
    cb_model, p, r, f = train_catboost(X_train, y_train, X_val, y_val)
    results.append(["CatBoost", p, r, f])

    df_results = pd.DataFrame(results, columns=["Model", "Precision", "Recall", "F1"])
    df_results = df_results.sort_values("F1", ascending=False).reset_index(drop=True)

    print("\n========== MODEL COMPARISON ==========\n")
    print(df_results)

    return df_results, {"rf": rf_model, "xgb": xgb_model, "cb": cb_model}
