# train.py
# Trains Logistic Regression using preprocessed data
# Includes: 
#   1) Validation-based evaluation
#   2) Direct test-set evaluation (no validation)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score


# ============================================================
# 1) VALIDATION-BASED TRAINING
# ============================================================

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Train Logistic Regression on TRAIN
    Evaluate using ONLY Precision, Recall, and F1 on VALIDATION.
    """

    print("\n========== Training Logistic Regression (Validation) ==========\n")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

    print("\n===== VALIDATION METRICS (WEIGHTED) =====")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    return model, prec, rec, f1


