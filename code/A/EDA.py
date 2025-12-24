# eda.py
# Spotify-Themed Exploratory Data Analysis + PCA
# Beautiful dark-mode visuals for academic reporting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif

# -------------------------------------------
# Spotify Color Palette
# -------------------------------------------
SPOTIFY_GREEN = "#1DB954"
SPOTIFY_BLACK = "#191414"
SPOTIFY_WHITE = "#FFFFFF"
SPOTIFY_GRAY = "#B3B3B3"

plt.style.use("dark_background")


# -------------------------------------------
# Load Data
# -------------------------------------------
def load_spotify_data(filepath="spotify_30_percent.csv"):
    df = pd.read_csv(filepath)

    # Drop non-numeric columns
    drop_cols = ["track_id", "artists", "album_name", "track_name", "Unnamed: 0"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Encode target
    label_encoder = LabelEncoder()
    df["track_genre"] = label_encoder.fit_transform(df["track_genre"])

    return df, label_encoder


# -------------------------------------------
# Save individual histograms silently
# -------------------------------------------
def plot_histograms(df, save_folder="plots/"):
    numeric_cols = [c for c in df.select_dtypes(["int64", "float64"]).columns if c != "track_genre"]

    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, color=SPOTIFY_GREEN, alpha=0.8)
        plt.title(f"Distribution of {col}", color=SPOTIFY_GREEN)
        plt.tight_layout()
        plt.savefig(f"{save_folder}{col}_hist.png")
        plt.close()


# -------------------------------------------
# Save individual boxplots silently
# -------------------------------------------
def plot_boxplots(df, save_folder="plots/"):
    numeric_cols = [c for c in df.select_dtypes(["int64", "float64"]).columns if c != "track_genre"]

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col], color=SPOTIFY_GREEN)
        plt.title(f"Boxplot of {col}", color=SPOTIFY_GREEN)
        plt.tight_layout()
        plt.savefig(f"{save_folder}{col}_box.png")
        plt.close()


# -------------------------------------------
# GRID: ALL HISTOGRAMS (POP-UP + SAVE)
# -------------------------------------------
def plot_all_histograms_grid(df, save_folder="plots/", show=False):
    numeric_cols = [
        c for c in df.select_dtypes(["int64", "float64"]).columns
        if c != "track_genre"
    ]

    n = len(numeric_cols)
    cols = 3                                     # fewer columns → better visibility
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(18, 14), dpi=100)        # reduced size for laptops

    for i, col in enumerate(numeric_cols, 1):
        ax = plt.subplot(rows, cols, i)
        sns.histplot(df[col], bins=35, color=SPOTIFY_GREEN)

        ax.set_title(col, fontsize=10, color=SPOTIFY_GREEN, pad=4)
        ax.set_xlabel("", fontsize=8)
        ax.set_ylabel("Count", fontsize=8, color=SPOTIFY_WHITE)
        ax.tick_params(axis="both", labelsize=7, colors=SPOTIFY_WHITE)

    plt.tight_layout(pad=2.0)
    plt.savefig(f"{save_folder}ALL_HISTOGRAMS_GRID.png")

    if show:
        plt.show()
    else:
        plt.close()


# -------------------------------------------
# GRID: ALL BOXPLOTS (POP-UP + SAVE)
# -------------------------------------------
def plot_all_boxplots_grid(df, save_folder="plots/", show=False):
    numeric_cols = [
        c for c in df.select_dtypes(["int64", "float64"]).columns
        if c != "track_genre"
    ]

    n = len(numeric_cols)
    cols = 3                                     # fewer columns → better layout
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(18, 14), dpi=100)

    for i, col in enumerate(numeric_cols, 1):
        ax = plt.subplot(rows, cols, i)
        sns.boxplot(x=df[col], color=SPOTIFY_GREEN)

        ax.set_title(col, fontsize=10, color=SPOTIFY_GREEN, pad=6)
        ax.set_xlabel("", fontsize=8)
        ax.tick_params(axis="x", labelsize=7, colors=SPOTIFY_WHITE)
        ax.set_yticks([])

    plt.tight_layout(pad=2.0)
    plt.savefig(f"{save_folder}ALL_BOXPLOTS_GRID.png")

    if show:
        plt.show()
    else:
        plt.close()


# -------------------------------------------
# Correlation Heatmap (POP-UP + SAVE)
# -------------------------------------------
def plot_correlation_matrix(df, save_folder="plots/", show=False):
    numeric_df = df.drop(columns=["track_genre"])
    corr = numeric_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        cmap="Greens",
        linewidths=0.5,
        annot_kws={"color": SPOTIFY_BLACK},
        linecolor=SPOTIFY_BLACK,
        cbar_kws={"shrink": 0.6}
    )
    plt.title("Spotify Feature Correlation Matrix", color=SPOTIFY_GREEN)
    plt.tight_layout()
    plt.savefig(f"{save_folder}correlation_matrix.png")

    if show:
        plt.show()
    else:
        plt.close()


# -------------------------------------------
# PCA Scatter (POP-UP + SAVE)
# -------------------------------------------
def run_pca(df, save_folder="plots/", show=False):
    features = df.drop(columns=["track_genre"])
    labels = df["track_genre"]

    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(features)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        pca_2d[:, 0], pca_2d[:, 1],
        c=labels, cmap="Greens", s=5, alpha=0.7
    )
    plt.title("PCA 2D Projection of Spotify Dataset", color=SPOTIFY_GREEN)
    plt.xlabel("PC1", color=SPOTIFY_WHITE)
    plt.ylabel("PC2", color=SPOTIFY_WHITE)
    plt.tight_layout()
    plt.savefig(f"{save_folder}pca_2d.png")

    if show:
        plt.show()
    else:
        plt.close()



# -------------------------------------------
# PCA Explained Variance (POP-UP + SAVE)
# -------------------------------------------
def plot_pca_variance(df, save_folder="plots/", show=False):
    features = df.drop(columns=["track_genre"])
    pca = PCA().fit(features)
    explained = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained)+1), explained, marker='o', color=SPOTIFY_GREEN)
    plt.title("PCA Cumulative Variance Explained", color=SPOTIFY_GREEN)
    plt.xlabel("Number of Components", color=SPOTIFY_WHITE)
    plt.ylabel("Cumulative Variance", color=SPOTIFY_WHITE)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_folder}pca_explained_variance.png")

    if show:
        plt.show()
    else:
        plt.close()


# -------------------------------------------
# Mutual Information (POP-UP + SAVE)
# -------------------------------------------
def plot_mutual_information(df, save_folder="plots/", show=False):
    X = df.drop(columns=["track_genre"])
    y = df["track_genre"]

    mi_scores = mutual_info_classif(X, y)
    mi_df = pd.DataFrame({"Feature": X.columns, "MI Score": mi_scores})
    mi_df = mi_df.sort_values("MI Score", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=mi_df, x="MI Score", y="Feature", palette="Greens_r")
    plt.title("Mutual Information Between Features and Genres", color=SPOTIFY_GREEN)
    plt.xlabel("MI Score", color=SPOTIFY_WHITE)
    plt.ylabel("Feature", color=SPOTIFY_WHITE)
    plt.tight_layout()
    plt.savefig(f"{save_folder}mutual_information.png")

    if show:
        plt.show()
    else:
        plt.close()


# -------------------------------------------
# MAIN EDA PIPELINE
# -------------------------------------------
def run_eda(filepath="spotify.csv"):
    df, _ = load_spotify_data(filepath)

    print("Running Spotify-Themed EDA...")

    import os
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Save individual plots
    plot_histograms(df)
    plot_boxplots(df)

    # POP-UP + save: GRID displays
    plot_all_histograms_grid(df, show=True)
    plot_all_boxplots_grid(df, show=True)

    # POP-UP + save: Key plots
    plot_correlation_matrix(df, show=True)
    plot_mutual_information(df, show=True)
    run_pca(df, show=True)
    plot_pca_variance(df, show=True)

    print("EDA complete! All plots saved in /plots folder.")


if __name__ == "__main__":
    run_eda("A/spotify_30_percent.csv")
