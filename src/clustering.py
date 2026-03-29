import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# =========================================================
# PATH CONFIGURATION
# =========================================================
DATA_PATH = Path("data/processed/cleaned_heart_failure.csv")
OUTPUT_PATH = Path("data/processed/clustered_heart_failure.csv")
FIG_DIR = Path("reports/figures")

FIG_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# PART 11: OUTLIER HANDLING (PREPROCESSING STEP)
# =========================================================
def cap_outliers_iqr(df, columns):
    """
    Caps outliers using IQR method instead of removing them.
    This is important for medical datasets where extreme values
    may still be meaningful.
    """
    df = df.copy()

    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df[col] = df[col].clip(lower, upper)

    return df


# =========================================================
# PART 12: CLUSTERING ANALYSIS (K-MEANS)
# =========================================================
def main():

    # -----------------------------
    # STEP 1: Load dataset
    # -----------------------------
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    # -----------------------------
    # STEP 2: Select features for clustering
    # -----------------------------
    features = [
        "age",
        "creatinine_phosphokinase",
        "ejection_fraction",
        "platelets",
        "serum_creatinine",
        "serum_sodium",
        "time"
    ]

    df_cluster = df[features].copy()

    # -----------------------------
    # STEP 3: Apply outlier handling
    # -----------------------------
    print("\nApplying outlier capping using IQR...")
    df_cluster = cap_outliers_iqr(df_cluster, features)

    # -----------------------------
    # STEP 4: Feature scaling
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    # -----------------------------
    # STEP 5: Determine optimal k
    # -----------------------------
    inertias = []
    silhouette_scores = []

    k_values = range(2, 7)

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        inertias.append(km.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    # -----------------------------
    # STEP 6: Plot Elbow Method
    # -----------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(list(k_values), inertias, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "elbow_method.png")
    plt.close()

    # -----------------------------
    # STEP 7: Plot Silhouette Score
    # -----------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(list(k_values), silhouette_scores, marker="o")
    plt.title("Silhouette Scores")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "silhouette_scores.png")
    plt.close()

    # -----------------------------
    # STEP 8: Select best k
    # -----------------------------
    best_k = list(k_values)[silhouette_scores.index(max(silhouette_scores))]
    print("\nBest number of clusters (k):", best_k)

    # -----------------------------
    # STEP 9: Final KMeans model
    # -----------------------------
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["cluster"] = model.fit_predict(X_scaled)

    # -----------------------------
    # STEP 10: PCA Visualization
    # -----------------------------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=df["cluster"],
        palette="Set2"
    )
    plt.title(f"PCA Cluster Visualization (k={best_k})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cluster_visualization.png")
    plt.close()

    # -----------------------------
    # STEP 11: Cluster summary
    # -----------------------------
    cluster_summary = df.groupby("cluster")[features + ["DEATH_EVENT"]].mean()

    print("\nCluster Summary:")
    print(cluster_summary)

    # -----------------------------
    # STEP 12: Save output
    # -----------------------------
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nClustered dataset saved to:", OUTPUT_PATH)


# =========================================================
# RUN SCRIPT
# =========================================================
if __name__ == "__main__":
    main()