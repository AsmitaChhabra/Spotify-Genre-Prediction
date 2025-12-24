# main.py
# Full pipeline: preprocessing → tuning → training → save → load → analysis → SHAP → DR

import os
import json
from EDA import run_eda
from preprocessing import preprocess
from hyperparameter_tuning import tune_xgboost
from models import train_final_xgboost, evaluate_on_test
from feature_importance import plot_feature_importance
from SHAP import run_explainability
from classwise_performance import run_full_analysis
from dimensionality_reduction import run_dimensionality_reduction


# =============================
# FLAGS (Control What Runs)
# =============================
RUN_XGB_TUNING = True
RUN_FINAL_TRAINING = True        # MUST BE TRUE FIRST TIME ON NEW MACHINE
RUN_EXPLAINABILITY = True
RUN_FULL_ANALYSIS = True
RUN_DR = True
RUN_ALL_MODELS = True

RUN_EDA = True


def main():

    # --------------------------
    # PREPROCESSING
    # --------------------------
    print("\n========== RUNNING PREPROCESSING ==========\n")

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        pipeline, label_encoder
    ) = preprocess("A/spotify_30_percent.csv")

    print("\n========== PREPROCESSING COMPLETE ==========")
    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    # --------------------------
    # RUN EDA
    # --------------------------
    if RUN_EDA:
        print("\n========== RUNNING EDA ==========\n")
        run_eda("A/spotify_30_percent.csv")
        print("\n========== EDA COMPLETE ==========\n")
    else:
        print("\nSkipping EDA.\n")

    # ====================================================
    # SAFELY HANDLE PATHS FOR PARAMS + MODEL
    # ====================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(BASE_DIR, "best_xgb_params.json")
    model_path  = os.path.join(BASE_DIR, "final_xgboost.model")

    # ====================================================
    # OPTIONAL: HYPERPARAMETER TUNING
    # ====================================================
    if RUN_XGB_TUNING:
        print("\n========== TUNING XGBOOST ==========\n")

        best_xgb_model, best_params = tune_xgboost(X_train, y_train, X_val, y_val)

        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=4)

        print("\nSaved → best_xgb_params.json\n")

    else:
        # Ensure params file exists
        if not os.path.exists(params_path):
            raise FileNotFoundError(
                "best_xgb_params.json NOT found.\n"
                "→ To generate it, set RUN_XGB_TUNING=True and run once."
            )

        with open(params_path, "r") as f:
            best_params = json.load(f)

        print("\nLoaded best_xgb_params.json:", best_params)

    # ====================================================
    # OPTIONAL: TRAIN FINAL MODEL (First Run)
    # ====================================================
    if RUN_FINAL_TRAINING:
        print("\n========== TRAINING FINAL XGBOOST ==========\n")

        final_model, p, r, f1 = train_final_xgboost(
            X_train, y_train,
            X_val, y_val,
            best_params
        )

        evaluate_on_test(final_model, X_test, y_test)

        # Save model
        final_model.save_model(model_path)
        print(f"\nFinal model saved → {model_path}\n")

    else:
        # ====================================================
        # LOAD MODEL SAFELY IF NOT TRAINING
        # ====================================================
        print("\n========== LOADING SAVED FINAL MODEL ==========\n")

        from xgboost import XGBClassifier
        final_model = XGBClassifier()

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "final_xgboost.model NOT found.\n"
                "→ Set RUN_FINAL_TRAINING=True and run once to generate it."
            )

        final_model.load_model(model_path)
        print("Model loaded successfully.\n")

    # ====================================================
    # EXPLAINABILITY (SHAP + LIME)
    # ====================================================
    if RUN_EXPLAINABILITY:
        print("\n========== RUNNING EXPLAINABILITY ==========\n")

        run_explainability(
            model=final_model,
            X_train_trans=X_train,
            X_test_trans=X_test,
            pipeline=pipeline,
            index=5
        )

        print("\n========== EXPLAINABILITY COMPLETE ==========\n")

    # ====================================================
    # FULL ANALYSIS
    # ====================================================
    if RUN_FULL_ANALYSIS:
        print("\n========== RUNNING FULL ANALYSIS ==========\n")

        run_full_analysis(
            final_model,
            X_val,
            y_val,
            label_encoder
        )

        print("\n========== ANALYSIS COMPLETE ==========\n")
    else:
        print("\nSkipping Full Analysis.\n")

    # ====================================================
    # FEATURE IMPORTANCE
    # ====================================================
    print("\n========== FEATURE IMPORTANCE ==========\n")
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    plot_feature_importance(final_model, feature_names)

    # ====================================================
    # DIMENSIONALITY REDUCTION
    # ====================================================
    if RUN_DR:
        print("\n========== RUNNING DIMENSIONALITY REDUCTION ==========\n")

        import numpy as np
        X_all = np.vstack([X_train, X_val, X_test])
        y_all = np.concatenate([y_train, y_val, y_test])

        run_dimensionality_reduction(X_all, y_all)

        print("\n========== DIMENSIONALITY REDUCTION COMPLETE ==========\n")


if __name__ == "__main__":
    main()
