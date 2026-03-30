"""
Data ingestion and validation for German Credit Dataset.
Downloads from UCI ML Repository if not present locally.
"""

import pandas as pd
import numpy as np
import os
import urllib.request


# German Credit Dataset column names
COLUMN_NAMES = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job",
    "num_dependents", "own_telephone", "foreign_worker", "class"
]

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
LOCAL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "german_credit.csv")


def download_data() -> str:
    """Download German Credit Dataset if not present."""
    os.makedirs(os.path.dirname(LOCAL_PATH), exist_ok=True)

    if not os.path.exists(LOCAL_PATH):
        print("Downloading German Credit Dataset from UCI...")
        urllib.request.urlretrieve(DATA_URL, LOCAL_PATH.replace(".csv", ".data"))

        # read the space-separated file and save as CSV
        df = pd.read_csv(
            LOCAL_PATH.replace(".csv", ".data"),
            sep=r"\s+",
            header=None,
            names=COLUMN_NAMES,
        )
        df.to_csv(LOCAL_PATH, index=False)
        print(f"Saved to {LOCAL_PATH}")

        # cleanup raw file
        raw_path = LOCAL_PATH.replace(".csv", ".data")
        if os.path.exists(raw_path):
            os.remove(raw_path)
    else:
        print(f"Data already exists at {LOCAL_PATH}")

    return LOCAL_PATH


def load_data() -> pd.DataFrame:
    """Load and perform basic validation on the dataset."""
    path = download_data()
    df = pd.read_csv(path)

    # convert target: 1 = Good, 2 = Bad → 1 = Good (0), 2 = Bad (1)
    df["class"] = df["class"].map({1: 0, 2: 1})

    # basic validation
    assert df.shape[0] == 1000, f"Expected 1000 rows, got {df.shape[0]}"
    assert df["class"].isnull().sum() == 0, "Target column has null values"

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target distribution:\n{df['class'].value_counts().to_string()}")

    return df


def get_feature_types(df: pd.DataFrame) -> dict:
    """Identify categorical and numerical features."""
    target = "class"
    categorical = df.select_dtypes(include=["object"]).columns.tolist()
    numerical = [
        col for col in df.select_dtypes(include=["int64", "float64"]).columns
        if col != target
    ]
    return {"categorical": categorical, "numerical": numerical, "target": target}


if __name__ == "__main__":
    df = load_data()
    feat_types = get_feature_types(df)
    print(f"\nCategorical features ({len(feat_types['categorical'])}): {feat_types['categorical']}")
    print(f"Numerical features ({len(feat_types['numerical'])}): {feat_types['numerical']}")
    print(f"\nSample:\n{df.head()}")
