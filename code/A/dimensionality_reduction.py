# ============================================
# dimensionality_reduction.py (FULL VERSION)
# PCA Experiments → PCA-2D → PCA-3D → t-SNE → UMAP
# ============================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import trustworthiness

import umap


# Spotify Theme Colors
SPOTIFY_GREEN = "#1DB954"
plt.style.use("dark_background")


# ---------------------------------------------------------
# Helper Plot
# ---------------------------------------------------------
def plot_embedding(X, labels, title):
    plt.figure(figsize=(10, 8))
    plt.scatter(
        X[:, 0], X[:, 1],
        c=labels,
        cmap="tab20",
        s=6,
        alpha=0.8
    )
    plt.title(title, color=SPOTIFY_GREEN)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.show()


# =========================================================
# 1. PCA EXPERIMENT (Accuracy + Training Time)
# =========================================================
def pca_experiment(X_scaled, y, components_list=[3, 4, 5, 7]):
    accuracies = []
    times = []

    for n in components_list:
        print(f"\n🎧 PCA Experiment → {n} Components")

        # Start timer
        t0 = time.time()

        # PCA transform
        pca = PCA(n_components=n, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        # Simple classifier for speed (same for all configs)
        model = LogisticRegression(max_iter=2000)
        model.fit(X_pca, y)
        preds = model.predict(X_pca)

        # Compute accuracy
        acc = accuracy_score(y, preds)

        # Compute elapsed time
        elapsed = time.time() - t0

        accuracies.append(acc)
        times.append(elapsed)

        print(f"Accuracy: {acc:.4f} | Time: {elapsed:.2f} sec")

    # Plot Acc vs Components
    plt.figure(figsize=(10, 5))
    plt.plot(components_list, accuracies, marker="o", color=SPOTIFY_GREEN)
    plt.title("Accuracy vs PCA Components", color=SPOTIFY_GREEN)
    plt.xlabel("PCA Components")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()

    # Plot Time vs Components
    plt.figure(figsize=(10, 5))
    plt.plot(components_list, times, marker="o", color="#FF5733")
    plt.title("Computation Time vs PCA Components", color=SPOTIFY_GREEN)
    plt.xlabel("PCA Components")
    plt.ylabel("Time (sec)")
    plt.tight_layout()
    plt.show()


# =========================================================
# 2. PCA Scree Plot + PCA 2D + PCA 3D
# =========================================================
def run_pca_visualizations(X_scaled, labels):
    print("\n🎧 Running PCA Visualizations...")

    # Scree plot
    pca_full = PCA()
    pca_full.fit(X_scaled)

    plt.figure(figsize=(10, 6))
    plt.plot(
        np.arange(1, len(pca_full.explained_variance_ratio_) + 1),
        pca_full.explained_variance_ratio_,
        marker="o",
        color=SPOTIFY_GREEN
    )
    plt.title("Scree Plot: Variance Explained", color=SPOTIFY_GREEN)
    plt.xlabel("Principal Components")
    plt.ylabel("Variance Explained")
    plt.tight_layout()
    plt.show()

    # PCA 2D
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    plot_embedding(X_pca_2d, labels, "PCA 2D Projection")

    # PCA 3D
    pca_3d = PCA(n_components=3, random_state=42)
    X_pca_3 = pca_3d.fit_transform(X_scaled)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        X_pca_3[:, 0], X_pca_3[:, 1], X_pca_3[:, 2],
        c=labels,
        cmap="tab20",
        s=8,
        alpha=0.7
    )

    ax.set_title("PCA 3D Projection", color=SPOTIFY_GREEN)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.tight_layout()
    plt.show()

    return X_pca_3


# =========================================================
# 3. t-SNE (NO SAMPLING) — full dataset
# =========================================================
def run_tsne(X_pca7, labels):
    print("\n⚡ Running FULL t-SNE (no sampling)...")

    tsne = TSNE(
        n_components=2,
        perplexity=35,
        learning_rate=250,
        max_iter=750,
        random_state=42
    )

    X_tsne = tsne.fit_transform(X_pca7)
    sil = silhouette_score(X_tsne, labels)

    print(f"🌟 t-SNE Silhouette Score = {sil:.4f}")

    plot_embedding(X_tsne, labels, "t-SNE (Full Dataset)")
    return X_tsne


# =========================================================
# 4. UMAP (NO SAMPLING)
# =========================================================
def run_umap(X_pca7, labels, sample_size=12000):
    print("\n✨ Running UMAP (fast mode)...")

    # -------------------------------------------
    # 1. Sampling for speed + stability
    # -------------------------------------------
    n = X_pca7.shape[0]

    if n > sample_size:
        idx = np.random.choice(n, sample_size, replace=False)
        X_sample = X_pca7[idx]
        y_sample = labels[idx]
        print(f"Using sampled subset: {sample_size} points out of {n}")
    else:
        X_sample = X_pca7
        y_sample = labels
        print(f"Dataset small enough — using full {n} points.")

    # -------------------------------------------
    # 2. Configure UMAP
    # -------------------------------------------
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        random_state=42,
        n_jobs=1,
        low_memory=True
    )

    # -------------------------------------------
    # 3. Fit & transform
    # -------------------------------------------
    X_umap = reducer.fit_transform(X_sample)

    # -------------------------------------------
    # 4. Trustworthiness
    # -------------------------------------------
    tw = trustworthiness(X_sample, X_umap, n_neighbors=5)
    print(f"🌟 UMAP Trustworthiness (sampled) = {tw:.4f}")

    # -------------------------------------------
    # 5. Plot
    # -------------------------------------------
    plot_embedding(X_umap, y_sample, f"UMAP Projection ({len(y_sample)} samples)")

    return X_umap, y_sample



# =========================================================
# 5. MASTER WRAPPER
# =========================================================
def run_dimensionality_reduction(X_scaled, labels):

    print("\n🚀 Starting Dimensionality Reduction Pipeline on Full 30k...")

    # PCA 3D / 2D / Scree
    X_pca_3 = run_pca_visualizations(X_scaled, labels)

    # PCA experiment (3,4,5,7)
    pca_experiment(X_scaled, labels)

    # PCA-7 for downstream TSNE/UMAP
    pca7 = PCA(n_components=7, random_state=42).fit_transform(X_scaled)

    # t-SNE (full)
    run_tsne(pca7, labels)

    # UMAP (full)
    run_umap(pca7, labels)

    print("\n🎉 All Dimensionality Reduction Steps Completed!\n")
