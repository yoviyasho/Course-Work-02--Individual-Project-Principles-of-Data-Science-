import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = Path("data/processed/cleaned_heart_failure.csv")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)

    print("Descriptive statistics:")
    print(df.describe(include="all"))

    # Target distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="DEATH_EVENT")
    plt.title("Death Event Distribution")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "target_distribution.png")
    plt.close()

    # Numerical distributions
    numeric_cols = [
        "age", "creatinine_phosphokinase", "ejection_fraction",
        "platelets", "serum_creatinine", "serum_sodium", "time"
    ]

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{col}_distribution.png")
        plt.close()

    # Boxplots for outliers
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{col}_boxplot.png")
        plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include=["int64", "float64"]).corr(),
                annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png")
    plt.close()

    # Target vs selected important variables
    important_cols = ["ejection_fraction", "serum_creatinine", "time", "age"]
    for col in important_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x="DEATH_EVENT", y=col)
        plt.title(f"{col} vs DEATH_EVENT")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{col}_vs_target.png")
        plt.close()

    print("EDA figures saved to:", FIG_DIR)

if __name__ == "__main__":
    main()