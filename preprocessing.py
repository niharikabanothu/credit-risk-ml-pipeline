"""
Feature engineering and preprocessing pipeline.
Handles encoding, scaling, feature creation, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_data, get_feature_types


class FeatureEngineer:
    """
    Complete preprocessing pipeline for credit risk data.
    
    Handles:
    - Categorical encoding (Label Encoding)
    - Feature scaling (StandardScaler)
    - New feature creation
    - Train/test split with stratification
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features from existing ones."""
        df = df.copy()

        # credit amount per month of duration
        df["credit_per_month"] = df["credit_amount"] / (df["duration"] + 1)

        # age-related risk buckets
        df["age_group"] = pd.cut(
            df["age"], bins=[0, 25, 35, 50, 100], labels=[0, 1, 2, 3]
        ).astype(int)

        # credit burden = credit amount relative to installment commitment
        df["credit_burden"] = df["credit_amount"] * df["installment_commitment"] / 100

        # duration risk (longer loans = higher risk)
        df["duration_risk"] = (df["duration"] > df["duration"].median()).astype(int)

        # amount risk (higher amounts = higher risk)
        df["amount_risk"] = (df["credit_amount"] > df["credit_amount"].median()).astype(int)

        # interaction: duration × amount
        df["duration_amount_interaction"] = df["duration"] * df["credit_amount"]

        print(f"  Created 6 new features. Total features: {df.shape[1]}")
        return df

    def encode_categoricals(self, df: pd.DataFrame, feat_types: dict) -> pd.DataFrame:
        """Label encode all categorical features."""
        df = df.copy()

        for col in feat_types["categorical"]:
            if col not in self.label_encoders:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # handle unseen labels
                df[col] = df[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        print(f"  Encoded {len(feat_types['categorical'])} categorical features")
        return df

    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """
        Full preprocessing pipeline.
        
        Returns: X_train, X_test, y_train, y_test, feature_names
        """
        print("\n" + "=" * 50)
        print("  PREPROCESSING PIPELINE")
        print("=" * 50)

        feat_types = get_feature_types(df)

        # step 1: feature engineering
        df = self.create_features(df)

        # step 2: encode categoricals
        df = self.encode_categoricals(df, feat_types)

        # step 3: separate features and target
        target = feat_types["target"]
        X = df.drop(columns=[target])
        y = df[target].values
        self.feature_names = X.columns.tolist()

        # step 4: train/test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # step 5: scale numerical features
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train), columns=self.feature_names, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test), columns=self.feature_names, index=X_test.index
        )

        print(f"  Train set: {X_train_scaled.shape[0]} samples")
        print(f"  Test set:  {X_test_scaled.shape[0]} samples")
        print(f"  Features:  {X_train_scaled.shape[1]}")
        print(f"  Target balance (train): {np.bincount(y_train)}")

        return X_train_scaled, X_test_scaled, y_train, y_test, self.feature_names


if __name__ == "__main__":
    df = load_data()
    fe = FeatureEngineer()
    X_train, X_test, y_train, y_test, features = fe.prepare_data(df)
    print(f"\nFeature names: {features}")
