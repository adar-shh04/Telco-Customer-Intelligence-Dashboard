"""
Train and save the API-facing churn model artifact.

Run:
    python train_model.py
"""

from src.churn_pipeline import train_churn_pipeline


def main() -> None:
    result = train_churn_pipeline()
    metrics = result["metrics"]

    print("Churn model artifact saved:")
    print(f"  Artifact: {result['artifact_path']}")
    print(f"  Metrics JSON: {result['metrics_json_path']}")
    print(f"  Metrics report: {result['metrics_report_path']}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()

