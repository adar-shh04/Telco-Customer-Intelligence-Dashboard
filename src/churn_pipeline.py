"""
Reusable churn-model training and serving utilities.

This module is intentionally the single source of truth for API-facing churn
features. It avoids data leakage by fitting preprocessing steps only on the
training split via an sklearn Pipeline.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COL = "churned"

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "tenure_months",
    "monthly_data_gb",
    "monthly_minutes",
    "ott_usage_hours",
    "arpu",
    "data_allowance_gb",
    "minutes_allowance",
    "satisfaction_score",
    "num_complaints_12m",
    "support_tickets_12m",
    "payment_score",
    "late_payments_12m",
    "campaigns_exposed",
    "total_clicks",
    "total_conversions",
    "monthly_web_sessions",
    "monthly_app_sessions",
    "avg_session_duration_min",
    "self_service_transactions",
]

CATEGORICAL_FEATURES = [
    "plan_type",
    "gender",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_data_path() -> str:
    return os.path.join(project_root(), "data", "raw", "master_dataset.csv")


def default_artifact_path() -> str:
    return os.path.join(project_root(), "artifacts", "churn_model.joblib")


def default_metrics_json_path() -> str:
    return os.path.join(project_root(), "data", "processed", "churn_metrics.json")


def default_metrics_report_path() -> str:
    return os.path.join(project_root(), "reports", "metrics.md")


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Create the preprocessing + model pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def load_training_data(data_path: str | None = None) -> pd.DataFrame:
    """Load the master dataset and validate the required API-facing columns."""
    resolved_path = data_path or default_data_path()
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(
            f"Training data not found at {resolved_path}. Run `python generate_data.py` first."
        )

    df = pd.read_csv(resolved_path)
    missing = [col for col in FEATURE_COLUMNS + [TARGET_COL] if col not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")
    return df


def evaluate_predictions(y_true, y_pred, y_proba, threshold: float) -> dict[str, Any]:
    """Calculate classification metrics for model-card style reporting."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def train_churn_pipeline(
    data_path: str | None = None,
    artifact_path: str | None = None,
    metrics_json_path: str | None = None,
    metrics_report_path: str | None = None,
    test_size: float = 0.2,
    threshold: float = 0.5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train the churn pipeline, save the artifact, and write metrics reports."""
    df = load_training_data(data_path)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    pipeline = build_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = evaluate_predictions(y_test, y_pred, y_proba, threshold)
    metrics.update(
        {
            "training_records": int(len(df)),
            "train_records": int(len(X_train)),
            "test_records": int(len(X_test)),
            "positive_rate": float(y.mean()),
            "model_type": "GradientBoostingClassifier",
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
    )

    artifact = {
        "pipeline": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET_COL,
        "metrics": metrics,
    }

    resolved_artifact_path = artifact_path or default_artifact_path()
    os.makedirs(os.path.dirname(resolved_artifact_path), exist_ok=True)
    joblib.dump(artifact, resolved_artifact_path)

    resolved_metrics_json = metrics_json_path or default_metrics_json_path()
    os.makedirs(os.path.dirname(resolved_metrics_json), exist_ok=True)
    with open(resolved_metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    write_metrics_report(metrics, metrics_report_path or default_metrics_report_path())

    return {
        "artifact_path": resolved_artifact_path,
        "metrics_json_path": resolved_metrics_json,
        "metrics_report_path": metrics_report_path or default_metrics_report_path(),
        "metrics": metrics,
    }


def load_churn_artifact(artifact_path: str | None = None) -> dict[str, Any]:
    """Load a trained churn artifact for serving."""
    resolved_path = artifact_path or default_artifact_path()
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(
            f"Model artifact not found at {resolved_path}. Run `python train_model.py` first."
        )
    return joblib.load(resolved_path)


def write_metrics_report(metrics: dict[str, Any], report_path: str) -> None:
    """Write a compact metrics report that can be cited in the README/resume."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    cm = metrics["confusion_matrix"]
    content = f"""# Churn Model Metrics

Generated: {metrics["created_at"]}

## Data Split

- Training records: {metrics["train_records"]:,}
- Test records: {metrics["test_records"]:,}
- Overall churn rate: {metrics["positive_rate"]:.2%}
- Decision threshold: {metrics["threshold"]:.2f}

## Performance

- ROC-AUC: {metrics["roc_auc"]:.4f}
- PR-AUC: {metrics["pr_auc"]:.4f}
- Brier score: {metrics["brier_score"]:.4f}
- Accuracy: {metrics["accuracy"]:.4f}
- Precision: {metrics["precision"]:.4f}
- Recall: {metrics["recall"]:.4f}
- F1: {metrics["f1"]:.4f}

## Confusion Matrix

|                | Predicted Retained | Predicted Churn |
|----------------|-------------------:|----------------:|
| Actual Retained | {cm["true_negative"]} | {cm["false_positive"]} |
| Actual Churn    | {cm["false_negative"]} | {cm["true_positive"]} |

## Notes

The model is trained through an sklearn Pipeline with train-only fitting for imputation, scaling, and one-hot encoding. This keeps preprocessing consistent between training and API serving.

Leakage audit: an earlier version generated satisfaction, complaints, support tickets, late payments and campaign conversions from the churn label, which inflated ROC-AUC to 0.99. Those generators now use only non-label signals (contract, tech support, payment method, tenure), and `tests/test_data_loader.py` fails if any derived feature changes when the label is flipped.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

