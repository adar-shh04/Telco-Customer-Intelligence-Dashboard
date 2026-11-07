# Churn Model Metrics

Generated: 2026-11-07T21:08:42Z

## Data Split

- Training records: 5,634
- Test records: 1,409
- Overall churn rate: 26.54%
- Decision threshold: 0.50

## Performance

- ROC-AUC: 0.8288
- PR-AUC: 0.5999
- Brier score: 0.1467
- Accuracy: 0.7899
- Precision: 0.6250
- Recall: 0.5214
- F1: 0.5685

## Confusion Matrix

|                | Predicted Retained | Predicted Churn |
|----------------|-------------------:|----------------:|
| Actual Retained | 918 | 117 |
| Actual Churn    | 179 | 195 |

## Notes

The model is trained through an sklearn Pipeline with train-only fitting for imputation, scaling, and one-hot encoding. This keeps preprocessing consistent between training and API serving.

Leakage audit: an earlier version generated satisfaction, complaints, support tickets, late payments and campaign conversions from the churn label, which inflated ROC-AUC to 0.99. Those generators now use only non-label signals (contract, tech support, payment method, tenure), and `tests/test_data_loader.py` fails if any derived feature changes when the label is flipped.
