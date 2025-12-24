import json
from preprocessing import preprocess
from models import train_final_xgboost, evaluate_on_test

def main():

    print("\n========== PREPROCESSING ==========\n")
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        pipeline, label_encoder
    ) = preprocess("B/spotify_30_percent_B.csv")  # or C version

    print("\n========== LOADING TUNED PARAMETERS ==========\n")
    with open("best_xgb_params.json", "r") as f:
        best_params = json.load(f)
    print("Loaded best parameters:", best_params)

    print("\n========== TRAINING FINAL MODEL ==========\n")
    model, p, r, f1 = train_final_xgboost(
        X_train, y_train,
        X_val, y_val,
        best_params
    )

    print("\n========== VALIDATION PERFORMANCE ==========\n")
    evaluate_on_test(model, X_test, y_test)


if __name__ == "__main__":
    main()
