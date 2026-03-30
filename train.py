"""
Model training pipeline with multi-model comparison and hyperparameter tuning.
Trains Logistic Regression, Random Forest, XGBoost, and SVM.
Saves the best model for deployment.
"""

import numpy as np
import pandas as pd
import joblib
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_data
from src.preprocessing import FeatureEngineer


# ---- Model Configurations ----
MODELS = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, 15],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
        },
    },
    "SVM": {
        "model": SVC(probability=True, random_state=42),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
    },
}


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    """Train all models, tune hyperparameters, and return results."""
    results = {}
    best_score = 0
    best_model_name = None
    best_model = None

    for name, config in MODELS.items():
        print(f"\n{'─'*50}")
        print(f"  Training: {name}")
        print(f"{'─'*50}")

        # GridSearchCV with cross-validation
        grid = GridSearchCV(
            config["model"],
            config["params"],
            cv=5,
            scoring="f1",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)

        model = grid.best_estimator_
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        # cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")

        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "best_params": grid.best_params_,
        }

        print(f"  Best params: {grid.best_params_}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")
        print(f"  CV F1:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        if f1 > best_score:
            best_score = f1
            best_model_name = name
            best_model = model

    # feature importance (if available)
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_names)
        importances = importances.sort_values(ascending=False)
        print(f"\n{'='*50}")
        print(f"  TOP 10 FEATURE IMPORTANCES ({best_model_name})")
        print(f"{'='*50}")
        for feat, imp in importances.head(10).items():
            bar = "█" * int(imp * 100)
            print(f"  {feat:<35} {imp:.4f} {bar}")

    return results, best_model_name, best_model


def save_model(model, model_name: str, feature_names: list, results: dict):
    """Save the best model and metadata."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "best_model.pkl")
    metadata = {
        "model": model,
        "model_name": model_name,
        "feature_names": feature_names,
        "metrics": {k: v for k, v in results[model_name].items() if k != "model"},
    }
    joblib.dump(metadata, model_path)
    print(f"\n  Model saved to {model_path}")
    return model_path


def print_summary(results: dict, best_name: str):
    """Print comparison table of all models."""
    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8}")
    print(f"  {'─'*64}")
    for name, r in results.items():
        marker = " ★" if name == best_name else ""
        print(
            f"  {name:<25} {r['accuracy']:<8.4f} {r['precision']:<8.4f} "
            f"{r['recall']:<8.4f} {r['f1']:<8.4f} {r['roc_auc']:<8.4f}{marker}"
        )
    print(f"\n  ★ Best model: {best_name}")


def run_training_pipeline():
    """Execute the full training pipeline."""
    print("=" * 50)
    print("  CREDIT RISK ML PIPELINE — TRAINING")
    print("=" * 50)

    # load and preprocess
    df = load_data()
    fe = FeatureEngineer()
    X_train, X_test, y_train, y_test, features = fe.prepare_data(df)

    # save the feature engineer for inference
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(fe, os.path.join(models_dir, "feature_engineer.pkl"))

    # train and evaluate
    results, best_name, best_model = train_and_evaluate(
        X_train, X_test, y_train, y_test, features
    )

    # summary and save
    print_summary(results, best_name)
    save_model(best_model, best_name, features, results)

    # detailed report for best model
    y_pred = best_model.predict(X_test)
    print(f"\n{'='*50}")
    print(f"  CLASSIFICATION REPORT — {best_name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Good Credit", "Bad Credit"]))

    return results, best_model


if __name__ == "__main__":
    results, model = run_training_pipeline()
