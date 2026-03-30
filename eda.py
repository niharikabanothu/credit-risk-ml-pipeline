"""
Exploratory Data Analysis — generates summary statistics, 
distribution plots, and correlation analysis.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_data, get_feature_types


def dataset_summary(df: pd.DataFrame) -> dict:
    """Generate comprehensive dataset summary."""
    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes.value_counts().to_dict(),
        "missing_values": df.isnull().sum().sum(),
        "duplicates": df.duplicated().sum(),
        "target_distribution": df["class"].value_counts().to_dict(),
        "target_balance_ratio": df["class"].value_counts().min() / df["class"].value_counts().max(),
    }

    print("=" * 50)
    print("  DATASET SUMMARY")
    print("=" * 50)
    for key, val in summary.items():
        print(f"  {key}: {val}")

    return summary


def numerical_analysis(df: pd.DataFrame, feat_types: dict, save_dir: str = "data"):
    """Analyze numerical features — stats + distributions."""
    numerical = feat_types["numerical"]

    print(f"\n{'='*50}")
    print(f"  NUMERICAL FEATURES STATISTICS")
    print(f"{'='*50}")
    stats = df[numerical].describe().T
    stats["skew"] = df[numerical].skew()
    stats["kurtosis"] = df[numerical].kurtosis()
    print(stats.to_string())

    # distribution plots
    os.makedirs(save_dir, exist_ok=True)
    n_cols = 3
    n_rows = (len(numerical) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numerical):
        axes[i].hist(df[col], bins=30, edgecolor="black", alpha=0.7, color="#2196F3")
        axes[i].set_title(col, fontsize=11)
        axes[i].set_ylabel("Count")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Numerical Feature Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "numerical_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {save_dir}/numerical_distributions.png")


def correlation_analysis(df: pd.DataFrame, feat_types: dict, save_dir: str = "data"):
    """Generate correlation heatmap for numerical features."""
    numerical = feat_types["numerical"] + [feat_types["target"]]
    corr = df[numerical].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                square=True, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "correlation_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_dir}/correlation_matrix.png")

    # top correlations with target
    target_corr = corr["class"].drop("class").abs().sort_values(ascending=False)
    print(f"\n  Top features correlated with target:")
    for feat, val in target_corr.head(5).items():
        print(f"    {feat}: {val:.4f}")


def class_comparison(df: pd.DataFrame, feat_types: dict, save_dir: str = "data"):
    """Compare feature distributions across target classes."""
    numerical = feat_types["numerical"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    top_features = ["duration", "credit_amount", "age", "installment_commitment", "existing_credits", "residence_since"]

    for i, col in enumerate(top_features):
        if col in df.columns:
            for cls, color, label in [(0, "#4CAF50", "Good"), (1, "#F44336", "Bad")]:
                axes[i].hist(df[df["class"] == cls][col], bins=25, alpha=0.6,
                           color=color, label=label, edgecolor="black")
            axes[i].set_title(col, fontsize=11)
            axes[i].legend()

    plt.suptitle("Feature Distributions by Credit Risk Class", fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "class_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_dir}/class_comparison.png")


def run_full_eda():
    """Run the complete EDA pipeline."""
    df = load_data()
    feat_types = get_feature_types(df)

    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    dataset_summary(df)
    numerical_analysis(df, feat_types, save_dir)
    correlation_analysis(df, feat_types, save_dir)
    class_comparison(df, feat_types, save_dir)

    print(f"\n{'='*50}")
    print(f"  EDA COMPLETE — all plots saved to {save_dir}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    run_full_eda()
