import shap
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

shap.initjs()
plt.rcParams["font.size"] = 12


# ============================================================
#  GLOBAL SHAP  (Working + Clean + Final)
# ============================================================
def global_shap(model, X_train_trans, X_sample_trans, pipeline, sample_size=3000):
    print("\n========== GLOBAL SHAP ==========\n")

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()

    # Sampling for speed (3000 max)
    if X_sample_trans.shape[0] > sample_size:
        idx = np.random.choice(X_sample_trans.shape[0], sample_size, replace=False)
        X_use = X_sample_trans[idx]
    else:
        X_use = X_sample_trans

    # Build SHAP explainer
    explainer = shap.TreeExplainer(model, model_output="raw")
    shap_values = explainer.shap_values(X_use)

    # Handle multiclass SHAP formats
    pred_class = np.argmax(model.predict_proba(X_use), axis=1)[0]

    if isinstance(shap_values, list):
        shap_vals = shap_values[pred_class]

    elif shap_values.ndim == 3:
        shap_vals = shap_values[:, :, pred_class]

    else:
        shap_vals = shap_values

    # ================================
    # GLOBAL SHAP Beeswarm
    # ================================
    plt.close()
    shap.summary_plot(
        shap_vals,
        X_use,
        feature_names=feature_names,
        show=False,
        plot_size=(14, 9)
    )

    ax = plt.gca()
    fig = plt.gcf()

    fig.set_facecolor("black")
    ax.set_facecolor("black")

    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.tick_params(colors="white")

    plt.title("Global SHAP Summary: Feature Impact Across All Songs", color="white")
    plt.tight_layout()
    plt.show()

    # ================================
    # GLOBAL SHAP Bar Plot
    # ================================
    plt.close()
    shap.summary_plot(
        shap_vals,
        X_use,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        plot_size=(12, 7)
    )

    ax = plt.gca()
    fig = plt.gcf()

    fig.set_facecolor("black")
    ax.set_facecolor("black")

    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.tick_params(colors="white")

    plt.title("Global Feature Importance (Mean |SHAP| Value)", color="white")
    plt.tight_layout()
    plt.show()

    return explainer

# ============================================================
#  LOCAL SHAP WATERFALL (Stable + Works for XGBoost)
# ============================================================\
def local_shap_waterfall(model, explainer, X_sample_trans, pipeline, index=0):
    print("\n========== LOCAL SHAP WATERFALL ==========\n")

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    row = X_sample_trans[index:index+1]

    shap_values = explainer(row)
    pred_class = np.argmax(model.predict_proba(row))
    shap_row = shap_values.values[0][:, pred_class]
    base_value = model.predict(row, output_margin=True)[0][pred_class]

    explanation = shap.Explanation(
        values=shap_row,
        base_values=base_value,
        feature_names=feature_names
    )

    plt.close('all')

    # Let SHAP create the figure
    shap.plots.waterfall(explanation, max_display=15, show=False)

    fig = plt.gcf()  # Grab the SHAP figure
    ax = plt.gca()

    # Apply styling
    fig.set_facecolor("black")
    ax.set_facecolor("black")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.tick_params(colors="white")

    plt.title("Local SHAP Waterfall: How This Song Was Classified", color="white")
    plt.tight_layout()

    plt.show()      # Show the SHAP plot only
    plt.close('all')

# ============================================================
#  LIME (Stable + Clean)
# ============================================================
def lime_local(model, X_train_trans, X_sample_trans, pipeline, index=0):
    print("\n========== LIME LOCAL ==========\n")

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()

    explainer = LimeTabularExplainer(
        training_data=X_train_trans,
        feature_names=feature_names,
        class_names=[str(i) for i in np.unique(model.predict(X_train_trans))],
        mode="classification"
    )

    row = X_sample_trans[index]
    exp = explainer.explain_instance(row, model.predict_proba, num_features=10)

    fig = exp.as_pyplot_figure()
    fig.set_facecolor("black")

    plt.title("Local LIME Explanation for Predicted Class", color="white")
    plt.xlabel("Contribution to Prediction", color="white")
    plt.ylabel("Feature", color="white")

    plt.xticks(color="white")
    plt.yticks(color="white")

    plt.tight_layout()
    plt.show()


def run_explainability(model, X_train_trans, X_test_trans, pipeline, index=0):
    print("\n========== RUNNING EXPLAINABILITY (SHAP + LIME) ==========\n")

    # Get explainer from global SHAP
    explainer = global_shap(model, X_train_trans, X_test_trans, pipeline)

    # Local SHAP waterfall
    local_shap_waterfall(model, explainer, X_test_trans, pipeline, index)

    # LIME local
    lime_local(model, X_train_trans, X_test_trans, pipeline, index)
