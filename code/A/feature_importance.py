# feature_importance_spotify.py

import matplotlib.pyplot as plt
import numpy as np

# Spotify Theme
SPOTIFY_GREEN = "#1DB954"
SPOTIFY_BLACK = "#191414"
SPOTIFY_WHITE = "#FFFFFF"

plt.style.use("dark_background")


def plot_feature_importance(model, feature_names):
    print("\n========== SPOTIFY-THEMED FEATURE IMPORTANCE ==========\n")

    booster = model.get_booster()
    importance_dict = booster.get_score(importance_type='gain')

    # Map f0 → feature_names[0], f1 → feature_names[1], etc.
    scores = np.array([importance_dict.get(f"f{i}", 0) for i in range(len(feature_names))])

    # Sort for beautiful visualization
    sorted_idx = np.argsort(scores)

    plt.figure(figsize=(12, 8))
    plt.barh(
        np.array(feature_names)[sorted_idx],
        scores[sorted_idx],
        color=SPOTIFY_GREEN
    )

    plt.xlabel("Gain", color=SPOTIFY_WHITE)
    plt.title("Feature Importance (XGBoost Gain)", color=SPOTIFY_GREEN, fontsize=16)
    plt.tight_layout()
    plt.show()
