# Churn Model Metrics

Generated: 2026-05-02T22:55:38Z

## Data Split

- Training records: 5,634
- Test records: 1,409
- Overall churn rate: 26.54%
- Decision threshold: 0.50

## Performance

- ROC-AUC: 0.9905
- PR-AUC: 0.9753
- Brier score: 0.0338
- Accuracy: 0.9588
- Precision: 0.9438
- Recall: 0.8984
- F1: 0.9205

## Confusion Matrix

|                | Predicted Retained | Predicted Churn |
|----------------|-------------------:|----------------:|
| Actual Retained | 1015 | 20 |
| Actual Churn    | 38 | 336 |

## Notes

The model is trained through an sklearn Pipeline with train-only fitting for imputation, scaling, and one-hot encoding. This keeps preprocessing consistent between training and API serving.
