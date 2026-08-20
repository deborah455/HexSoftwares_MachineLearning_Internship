import pandas as pd


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load and clean the SMS spam dataset."""

    df = pd.read_csv(filepath, encoding="latin-1")

    # Keep only the columns required for classification
    df = df[["v1", "v2"]].copy()

    # Rename columns
    df.columns = ["label", "message"]

    # Remove missing values
    df.dropna(inplace=True)

    # Remove duplicate messages
    df.drop_duplicates(inplace=True)

    # Normalize labels
    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    return df


if __name__ == "__main__":
    df = load_and_clean_data("data/spam.csv")

    print("Cleaned dataset shape:", df.shape)
    print("\nClass distribution:")
    print(df["label"].value_counts())

    print("\nSample data:")
    print(df.head())