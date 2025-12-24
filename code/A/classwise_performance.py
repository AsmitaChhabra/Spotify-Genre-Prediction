# advanced_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix

# Spotify Theme
SPOTIFY_GREEN = "#1DB954"
SPOTIFY_BLACK = "#191414"
SPOTIFY_WHITE = "#FFFFFF"

plt.style.use("dark_background")


# ---------------------------------------------------------
# TOP 30 & BOTTOM 30 GENRES
# ---------------------------------------------------------
def top_bottom_genres(model, X_val, y_val, label_encoder):
    print("\n📌 Generating Top-30 and Bottom-30 Genre Performance...\n")

    y_pred = model.predict(X_val)
    classes = np.unique(y_val)

    rows = []
    for c in classes:
        mask = y_val == c
        if mask.sum() == 0:
            continue

        f1 = f1_score(y_val[mask], y_pred[mask], average="weighted", zero_division=0)
        rows.append((c, f1))

    df = pd.DataFrame(rows, columns=["class_id", "f1"])
    df["genre"] = label_encoder.inverse_transform(df["class_id"])

    df_sorted = df.sort_values("f1", ascending=False)

    # TOP 30 -----------------------------------
    plt.figure(figsize=(10, 10))
    sns.barplot(data=df_sorted.head(30), x="f1", y="genre", color=SPOTIFY_GREEN)
    plt.title("Genres Where the Model Performs Best (Top 30)", color=SPOTIFY_GREEN)
    plt.xlabel("F1 Score")
    plt.tight_layout()
    plt.show(block=True)

    # BOTTOM 30 --------------------------------
    plt.figure(figsize=(10, 10))
    sns.barplot(data=df_sorted.tail(30), x="f1", y="genre", color=SPOTIFY_GREEN)
    plt.title("Genres Where the Model Performs Worst (Bottom 30)", color=SPOTIFY_GREEN)
    plt.xlabel("F1 Score")
    plt.tight_layout()
    plt.show(block=True)


# ---------------------------------------------------------
# TOP CONFUSED GENRE PAIRS
# ---------------------------------------------------------
def top_confused_pairs(model, X_val, y_val, label_encoder, top_n=20):
    print("\n📌 Finding Top Confused Genre Pairs...\n")

    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)

    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((i, j, cm[i, j]))

    df = pd.DataFrame(pairs, columns=["true", "pred", "count"])
    df = df.sort_values("count", ascending=False).head(top_n)

    df["true_genre"] = label_encoder.inverse_transform(df["true"])
    df["pred_genre"] = label_encoder.inverse_transform(df["pred"])

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df,
        x="count",
        y="true_genre",
        hue="pred_genre",
        palette="Greens",
        linewidth=0

        
    )
    plt.title("Genres the Model Most Frequently Confuses", color=SPOTIFY_GREEN)
    plt.xlabel("Confusion Count")
    plt.tight_layout()
    plt.show(block=True)


# ---------------------------------------------------------
# MINI CONFUSION MATRIX (Single Genre)
# ---------------------------------------------------------
def mini_confusion_matrix(model, X_val, y_val, label_encoder, genre_name="pop"):
    print(f"\n📌 Mini Confusion Matrix for genre: {genre_name}\n")

    genre_id = label_encoder.transform([genre_name])[0]

    y_pred = model.predict(X_val)
    mask = (y_val == genre_id)
    preds = y_pred[mask]

    unique_preds, counts = np.unique(preds, return_counts=True)
    pred_genres = label_encoder.inverse_transform(unique_preds)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=counts, y=pred_genres, color=SPOTIFY_GREEN)
    plt.title(f"What the Model Predicts When the True Genre Is '{genre_name}'", color=SPOTIFY_GREEN)
    plt.xlabel("Count")
    plt.tight_layout()
    plt.show(block=True)


# ---------------------------------------------------------
# THRESHOLD SENSITIVITY ANALYSIS
# ---------------------------------------------------------
def threshold_sensitivity(model, X_val, y_val):
    print("\n📈 Running Threshold Sensitivity Analysis...\n")

    probas = model.predict_proba(X_val)

    thresholds = np.arange(0.05, 1.01, 0.05)
    f1_scores = []

    for t in thresholds:
        max_proba = probas.max(axis=1)
        mask = max_proba >= t

        if mask.sum() == 0:
            f1_scores.append(0)
            continue

        preds = probas[mask].argmax(axis=1)
        true = y_val[mask]

        f1 = f1_score(true, preds, average="weighted", zero_division=0)
        f1_scores.append(f1)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, marker="o", color=SPOTIFY_GREEN)
    plt.title("Threshold vs Weighted F1: How Model Performance Changes with Prediction Confidence", color=SPOTIFY_GREEN)
    plt.xlabel("Threshold")
    plt.ylabel("Weighted F1")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)

    return thresholds, f1_scores


# ---------------------------------------------------------
# MASTER FUNCTION
# ---------------------------------------------------------
def run_full_analysis(model, X_val, y_val, label_encoder):

    top_bottom_genres(model, X_val, y_val, label_encoder)
    top_confused_pairs(model, X_val, y_val, label_encoder)
    mini_confusion_matrix(model, X_val, y_val, label_encoder, genre_name="pop")
    threshold_sensitivity(model, X_val, y_val)

    print("\n🎉 ALL ANALYSIS COMPLETE!\n")
