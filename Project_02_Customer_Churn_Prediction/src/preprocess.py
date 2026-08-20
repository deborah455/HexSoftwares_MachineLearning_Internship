import pandas as pd
from pathlib import Path


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "customer_churn.csv"
OUTPUT_PATH = BASE_DIR / "data" / "cleaned_customer_churn.csv"


def load_and_clean_data():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Original shape: {df.shape}")

    # Remove customer ID because it is an identifier,
    # not a useful predictive feature.
    df = df.drop(columns=["customerID"])

    # Convert TotalCharges from text to numeric.
    # Invalid/blank values become NaN.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Check missing values created during conversion
    missing_total_charges = df["TotalCharges"].isna().sum()

    print(f"Missing TotalCharges values: {missing_total_charges}")

    # Remove rows with missing TotalCharges
    df = df.dropna(subset=["TotalCharges"])

    # Convert target to numerical values
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nCleaned shape: {df.shape}")

    print("\nClass distribution:")
    print(df["Churn"].value_counts())

    print("\nClass percentages:")
    print(
        (df["Churn"].value_counts(normalize=True) * 100)
        .round(2)
    )

    print("\nRemaining missing values:")
    print(df.isnull().sum().sum())

    print(f"\nCleaned dataset saved to:")
    print(OUTPUT_PATH)

    return df


if __name__ == "__main__":
    load_and_clean_data()