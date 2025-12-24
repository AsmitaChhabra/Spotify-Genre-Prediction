# hyperparameter_tuning.py
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer
import numpy as np

def tune_xgboost(X_train, y_train, X_val, y_val):
    """
    Performs randomized search hyperparameter tuning for XGBoost.
    Returns best model and its parameters.
    """

    print("\n===== RUNNING XGBOOST HYPERPARAMETER TUNING =====\n")

    param_dist = {
        "n_estimators": [200, 300, 400, 500],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 1, 5],
    }

    model = XGBClassifier(
        objective="multi:softmax",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )

    scorer = make_scorer(f1_score, average="weighted")

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,           # ← was 20
        scoring=scorer,
        cv=2,                # ← was 3
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    print("\n===== BEST PARAMETERS FOUND =====")
    print(search.best_params_)

    # Train model again on train+val (optional)
    best_model = search.best_estimator_
    best_model.fit(X_train, y_train)

    # Evaluate on validation
    preds = best_model.predict(X_val)
    f1 = f1_score(y_val, preds, average="weighted")

    print("\nValidation F1 Score:", round(f1, 4))

    return best_model, search.best_params_
