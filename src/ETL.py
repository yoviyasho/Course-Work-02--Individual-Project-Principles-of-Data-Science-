import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/heart_failure_clinical_records_dataset.csv")
PROCESSED_PATH = Path("data/processed/cleaned_heart_failure.csv")

def main():
    df = pd.read_csv(RAW_PATH)

    print("Initial shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    # Remove duplicates if any
    df = df.drop_duplicates()

    # Basic type check
    print("\nData types:")
    print(df.dtypes)

    # Simple feature engineering
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 40, 50, 60, 70, 100],
        labels=["<=40", "41-50", "51-60", "61-70", "70+"]
    )

    # Risk ratio style feature
    df["creatinine_to_sodium_ratio"] = df["serum_creatinine"] / df["serum_sodium"]
    
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("\nSaved cleaned file to:", PROCESSED_PATH)
    print("Final shape:", df.shape)

if __name__ == "__main__":
    main()